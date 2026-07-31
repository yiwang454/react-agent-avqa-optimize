"""Batch-scoped recovery for empty Qwen AV responses.

The Qwen server can return an empty visible completion when several AV calls are
in flight.  Retrying such a request inside the worker that made it makes the
load spike worse and changes the request layout mid-batch.  This module keeps
the initial batch concurrent, then lets its caller re-run only affected
examples with deterministic media-layout fallbacks. Recovery is serial by
default and can be bounded to a small parallelism level when needed.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator, Sequence, TypeVar


@dataclass(frozen=True)
class QwenRequestProfile:
    """A fixed Qwen AV media layout used for one evaluation pass."""

    name: str
    use_audio_in_video: bool
    video_first: bool
    serial: bool
    defer_empty_response_retries: bool = True


PRIMARY_QWEN_PROFILE = QwenRequestProfile(
    name="primary_separate_audio_first",
    use_audio_in_video=False,
    video_first=False,
    serial=False,
)
RETRY_VIDEO_FIRST_QWEN_PROFILE = QwenRequestProfile(
    name="retry_separate_video_first",
    use_audio_in_video=False,
    video_first=True,
    serial=True,
)
RETRY_EMBEDDED_AUDIO_QWEN_PROFILE = QwenRequestProfile(
    name="retry_embedded_audio",
    use_audio_in_video=True,
    # There is no standalone audio item to order in this layout.  Keeping the
    # value true makes the profile explicit and avoids consulting QWEN_VIDEO_FIRST.
    video_first=True,
    serial=True,
)


class _ExcludedScoreSum:
    """A running validation sum that deliberately omits invalid Qwen rollouts."""

    def __init__(self, total: float = 0.0, included_count: int = 0) -> None:
        self.total = total
        self.included_count = included_count

    def __add__(self, other: Any) -> "_ExcludedScoreSum":
        if isinstance(other, QwenValidationScore):
            if other.included:
                return _ExcludedScoreSum(self.total + other.value, self.included_count + 1)
            return _ExcludedScoreSum(self.total, self.included_count)
        return _ExcludedScoreSum(self.total + float(other), self.included_count + 1)

    __radd__ = __add__

    def __truediv__(self, _denominator: float) -> float:
        return self.total / self.included_count if self.included_count else float("-inf")


class QwenValidationScore:
    """Per-example validation score that can be excluded without losing its ID.

    GEPA maps scores to validation IDs positionally.  Removing a middle list
    entry would attach every following score to the wrong validation sample.
    This value preserves that one-to-one mapping while its addition semantics
    omit an exhausted empty-Qwen rollout from GEPA's validation mean.
    """

    def __init__(self, value: float, *, included: bool) -> None:
        self.value = float(value)
        self.included = included

    def __radd__(self, other: Any) -> _ExcludedScoreSum:
        if isinstance(other, _ExcludedScoreSum):
            return other + self
        if isinstance(other, (int, float)):
            return _ExcludedScoreSum(float(other), 0) + self
        return NotImplemented

    def __add__(self, other: Any) -> _ExcludedScoreSum:
        return _ExcludedScoreSum(self.value if self.included else 0.0, int(self.included)) + other

    def _comparison_value(self) -> float:
        return self.value if self.included else float("-inf")

    def __gt__(self, other: Any) -> bool:
        return self._comparison_value() > _score_comparison_value(other)

    def __ge__(self, other: Any) -> bool:
        return self._comparison_value() >= _score_comparison_value(other)

    def __lt__(self, other: Any) -> bool:
        return self._comparison_value() < _score_comparison_value(other)

    def __le__(self, other: Any) -> bool:
        return self._comparison_value() <= _score_comparison_value(other)

    def __eq__(self, other: object) -> bool:
        if not self.included:
            return False
        try:
            return self.value == _score_comparison_value(other)
        except (TypeError, ValueError):
            return False

    def __float__(self) -> float:
        return self.value if self.included else 0.0

    def __repr__(self) -> str:
        return (
            f"QwenValidationScore(value={self.value!r}, included={self.included!r})"
        )


def _score_comparison_value(value: Any) -> float:
    if isinstance(value, QwenValidationScore):
        return value._comparison_value()
    return float(value)


_ACTIVE_PROFILE_LOCK = threading.RLock()
_ACTIVE_PROFILE: QwenRequestProfile | None = None
_SERIAL_QWEN_REQUEST_GATE = threading.BoundedSemaphore(value=1)


def _read_nonnegative_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw_value!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}")
    return value


def _read_positive_int_env(name: str, default: int) -> int:
    value = _read_nonnegative_int_env(name, default)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def active_qwen_request_profile() -> QwenRequestProfile | None:
    """Return the batch profile without changing process environment variables."""
    with _ACTIVE_PROFILE_LOCK:
        return _ACTIVE_PROFILE


@contextmanager
def qwen_request_profile(profile: QwenRequestProfile) -> Iterator[None]:
    """Make a profile visible to DSPy's worker threads for one batch.

    ``contextvars`` do not automatically propagate into DSPy's worker threads,
    so this deliberately uses a small process-local state protected by a lock.
    The outer GEPA launcher runs one experiment at a time, which makes this
    scope unambiguous.  No environment variable is changed.
    """
    global _ACTIVE_PROFILE
    with _ACTIVE_PROFILE_LOCK:
        if _ACTIVE_PROFILE is not None:
            raise RuntimeError(
                "A ReliableQwenExecutor profile is already active. "
                "Run only one GEPA experiment per process."
            )
        _ACTIVE_PROFILE = profile
    try:
        yield
    finally:
        with _ACTIVE_PROFILE_LOCK:
            _ACTIVE_PROFILE = None


@contextmanager
def qwen_request_slot() -> Iterator[None]:
    """Enforce max_inflight=1 for a serial fallback Qwen inference request."""
    profile = active_qwen_request_profile()
    if profile is None or not profile.serial:
        yield
        return
    with _SERIAL_QWEN_REQUEST_GATE:
        yield


def prediction_has_empty_qwen_observation(prediction: Any) -> bool:
    """Return whether a ReAct prediction contains an empty Qwen tool response."""
    return turn_trace_has_empty_qwen_observation(
        getattr(prediction, "turn_trace", None) or []
    )


def turn_trace_has_empty_qwen_observation(turn_trace: Sequence[Any]) -> bool:
    """Return whether a serialized ReAct trace contains an empty Qwen response."""
    for turn in turn_trace:
        if not isinstance(turn, dict):
            continue
        if turn.get("planner_action") != "tool":
            continue
        if str(turn.get("perception_backend") or "").strip().lower() != "qwen":
            continue
        observation = turn.get("tool_observation")
        if not isinstance(observation, str) or not observation.strip():
            return True
    return False


def mark_empty_qwen_prediction(
    prediction: Any,
    *,
    attempted_profiles: Sequence[str],
) -> None:
    """Record an exhausted Qwen recovery error in the rollout trace."""
    mark_empty_qwen_turn_trace(
        getattr(prediction, "turn_trace", None) or [],
        attempted_profiles=attempted_profiles,
    )


def mark_empty_qwen_turn_trace(
    turn_trace: Sequence[Any],
    *,
    attempted_profiles: Sequence[str],
) -> None:
    """Record an exhausted Qwen recovery error in a serialized ReAct trace."""
    for turn in turn_trace:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("perception_backend") or "").strip().lower() != "qwen":
            continue
        observation = turn.get("tool_observation")
        if isinstance(observation, str) and observation.strip():
            continue
        turn["reliable_qwen_error"] = {
            "error_type": "EmptyQwenResponse",
            "message": "Qwen returned no visible content after all batch-level fallbacks.",
            "attempted_profiles": list(attempted_profiles),
        }


TItem = TypeVar("TItem")
TResult = TypeVar("TResult")


@dataclass
class ReliableQwenBatchResult:
    results: list[Any]
    exhausted_indices: list[int]
    attempted_profiles: list[str]


class ReliableQwenExecutor:
    """Own the fixed primary/fallback policy for one GEPA experiment.

    ``max_batch_retries`` counts retries *after* the concurrent primary pass.
    Its default is two: separate audio/video with video first, then embedded
    audio.  Additional retries deliberately repeat embedded audio, as this is
    the final prescribed fallback.
    """

    def __init__(
        self,
        max_batch_retries: int | None = None,
        fallback_max_workers: int | None = None,
    ) -> None:
        self.max_batch_retries = (
            _read_nonnegative_int_env("QWEN_RELIABLE_MAX_BATCH_RETRIES", 2)
            if max_batch_retries is None
            else max_batch_retries
        )
        if self.max_batch_retries < 0:
            raise ValueError("max_batch_retries must be non-negative")
        self.fallback_max_workers = (
            _read_positive_int_env("QWEN_RELIABLE_FALLBACK_NUM_THREADS", 1)
            if fallback_max_workers is None
            else fallback_max_workers
        )
        if self.fallback_max_workers < 1:
            raise ValueError("fallback_max_workers must be >= 1")

    @property
    def primary_profile(self) -> QwenRequestProfile:
        return PRIMARY_QWEN_PROFILE

    def retry_profile(self, retry_number: int) -> QwenRequestProfile:
        if retry_number < 1:
            raise ValueError("retry_number must start at 1")
        if retry_number == 1:
            return RETRY_VIDEO_FIRST_QWEN_PROFILE
        return RETRY_EMBEDDED_AUDIO_QWEN_PROFILE

    @contextmanager
    def primary_batch(self) -> Iterator[None]:
        with qwen_request_profile(self.primary_profile):
            yield

    @contextmanager
    def retry_batch(
        self,
        retry_number: int,
        *,
        allow_parallel_requests: bool = False,
    ) -> Iterator[QwenRequestProfile]:
        profile = self.retry_profile(retry_number)
        # The executor bounds fallback workers itself. Disable the per-request
        # serial gate only for a bounded parallel fallback.
        if allow_parallel_requests:
            profile = replace(profile, serial=False)
        with qwen_request_profile(profile):
            yield profile

    def run_batch(
        self,
        items: Sequence[TItem],
        *,
        run_item: Callable[[TItem], TResult],
        has_empty_qwen_response: Callable[[TResult], bool],
        max_workers: int,
    ) -> ReliableQwenBatchResult:
        """Run a concurrent primary batch and bounded recovery for empty responses."""
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        results: list[TResult | None] = [None] * len(items)
        with self.primary_batch():
            if max_workers == 1:
                for index, item in enumerate(items):
                    results[index] = run_item(item)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(run_item, item): index
                        for index, item in enumerate(items)
                    }
                    for future in as_completed(futures):
                        results[futures[future]] = future.result()

        # Every primary worker has completed before any fallback begins.
        completed_results = [result for result in results]
        if any(result is None for result in completed_results):
            raise RuntimeError("ReliableQwenExecutor primary batch returned an incomplete result set")
        invalid_indices = [
            index
            for index, result in enumerate(completed_results)
            if has_empty_qwen_response(result)
        ]
        attempted_profiles = [self.primary_profile.name]
        for retry_number in range(1, self.max_batch_retries + 1):
            if not invalid_indices:
                break
            fallback_workers = min(self.fallback_max_workers, len(invalid_indices))
            with self.retry_batch(
                retry_number,
                allow_parallel_requests=fallback_workers > 1,
            ) as profile:
                attempted_profiles.append(profile.name)
                if fallback_workers == 1:
                    for index in invalid_indices:
                        completed_results[index] = run_item(items[index])
                else:
                    with ThreadPoolExecutor(max_workers=fallback_workers) as executor:
                        futures = {
                            executor.submit(run_item, items[index]): index
                            for index in invalid_indices
                        }
                        for future in as_completed(futures):
                            completed_results[futures[future]] = future.result()
            invalid_indices = [
                index
                for index in invalid_indices
                if has_empty_qwen_response(completed_results[index])
            ]

        return ReliableQwenBatchResult(
            results=completed_results,
            exhausted_indices=invalid_indices,
            attempted_profiles=attempted_profiles,
        )
