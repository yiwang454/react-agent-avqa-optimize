"""Per-question retry-adjusted latency accounting for ReAct inference.

The sample boundary and component boundary are owned by the high-level inference
path.  Transport implementations only emit attempt lifecycle events.  All state
is context-local so concurrent inference workers cannot mix measurements.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from litellm.integrations.custom_logger import CustomLogger


_ACTIVE_TRACKER: contextvars.ContextVar["LatencyTracker | None"] = contextvars.ContextVar(
    "dspy_avqa_latency_tracker", default=None
)
_ACTIVE_COMPONENT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "dspy_avqa_latency_component", default=None
)


@dataclass
class LatencyTracker:
    first_attempt_started: float | None = None
    active_attempt_started: float | None = None
    active_attempt_component: str | None = None
    last_failed_attempt_ended: float | None = None
    failed_attempt_seconds: float = 0.0
    retry_backoff_seconds: float = 0.0
    successful_model_call_seconds: float = 0.0
    successful_calls: dict[str, int] = field(
        default_factory=lambda: {"planner": 0, "perception": 0}
    )

    def start_attempt(self, component: str) -> None:
        now = time.perf_counter()
        if self.active_attempt_started is not None:
            # A new request means the prior request returned successfully.  This
            # primarily protects callback-driven transports from losing an event.
            self.finish_attempt(success=True, ended=now)
        if self.last_failed_attempt_ended is not None:
            self.retry_backoff_seconds += max(0.0, now - self.last_failed_attempt_ended)
            self.last_failed_attempt_ended = None
        if self.first_attempt_started is None:
            self.first_attempt_started = now
        self.active_attempt_started = now
        self.active_attempt_component = component

    def finish_attempt(self, *, success: bool, ended: float | None = None) -> None:
        if self.active_attempt_started is None:
            return
        now = time.perf_counter() if ended is None else ended
        duration = max(0.0, now - self.active_attempt_started)
        component = self.active_attempt_component or "perception"
        self.active_attempt_started = None
        self.active_attempt_component = None
        if success:
            self.successful_model_call_seconds += duration
            if component in self.successful_calls:
                self.successful_calls[component] += 1
            self.last_failed_attempt_ended = None
        else:
            self.failed_attempt_seconds += duration
            self.last_failed_attempt_ended = now

    def summary(self, *, complete: bool) -> dict[str, Any]:
        ended = time.perf_counter()
        if self.active_attempt_started is not None:
            self.finish_attempt(success=False, ended=ended)
        wall = (
            max(0.0, ended - self.first_attempt_started)
            if self.first_attempt_started is not None
            else None
        )
        adjusted = None
        if complete and wall is not None:
            adjusted = max(
                0.0,
                wall - self.failed_attempt_seconds - self.retry_backoff_seconds,
            )
        return {
            "wall_clock_seconds": _rounded(wall),
            "retry_adjusted_seconds": _rounded(adjusted),
            "successful_model_call_seconds": _rounded(
                self.successful_model_call_seconds
            ),
            "planner_successful_calls": self.successful_calls["planner"],
            "perception_successful_calls": self.successful_calls["perception"],
        }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


@contextlib.contextmanager
def sample_latency() -> Iterator[LatencyTracker]:
    tracker = LatencyTracker()
    token = _ACTIVE_TRACKER.set(tracker)
    try:
        yield tracker
    finally:
        _ACTIVE_TRACKER.reset(token)


@contextlib.contextmanager
def model_component(component: str) -> Iterator[None]:
    if component not in {"planner", "perception"}:
        raise ValueError(f"Unsupported latency component: {component}")
    token = _ACTIVE_COMPONENT.set(component)
    try:
        yield
    except BaseException:
        finish_model_attempt(success=False)
        raise
    finally:
        _ACTIVE_COMPONENT.reset(token)


def start_model_attempt() -> None:
    tracker = _ACTIVE_TRACKER.get()
    component = _ACTIVE_COMPONENT.get()
    if tracker is not None and component is not None:
        tracker.start_attempt(component)


def finish_model_attempt(*, success: bool) -> None:
    tracker = _ACTIVE_TRACKER.get()
    if tracker is not None:
        tracker.finish_attempt(success=success)


class _LiteLLMAttemptObserver(CustomLogger):
    """Use LiteLLM's synchronous input/failure hooks without changing retries."""

    def __init__(self) -> None:
        super().__init__(turn_off_message_logging=True)

    def log_pre_api_call(self, model, messages, kwargs):  # noqa: ANN001, ARG002
        start_model_attempt()

    def log_failure_event(  # noqa: ANN001, ARG002
        self, kwargs, response_obj, start_time, end_time
    ):
        finish_model_attempt(success=False)


_LITELLM_OBSERVER = _LiteLLMAttemptObserver()
_LITELLM_OBSERVER_LOCK = threading.Lock()


def install_litellm_attempt_observer() -> None:
    """Register one process-wide observer for native LiteLLM planner attempts."""
    import litellm

    with _LITELLM_OBSERVER_LOCK:
        input_callbacks = getattr(litellm, "input_callback", None)
        failure_callbacks = getattr(litellm, "failure_callback", None)
        # Lightweight test doubles may expose only completion(). Production
        # LiteLLM exposes both callback lists.
        if input_callbacks is None or failure_callbacks is None:
            return
        if _LITELLM_OBSERVER not in input_callbacks:
            input_callbacks.append(_LITELLM_OBSERVER)
        if _LITELLM_OBSERVER not in failure_callbacks:
            failure_callbacks.append(_LITELLM_OBSERVER)
