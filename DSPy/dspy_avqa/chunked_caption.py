"""Offline chunked-caption generation for long-form ReAct caption caches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .caption_cache import (
    CACHE_SCHEMA_VERSION,
    MANIFEST_NAME,
    load_caption_cache_manifest,
    validate_caption_cache_coverage,
)
from .latency import model_component, sample_latency
from .tools import (
    call_gemini_perception,
    consume_last_perception_metadata,
    cut_video_clip,
    probe_media_duration,
)


TIMESTAMP_MODES = {"local_offset", "model_original"}
PARSER_SCHEMA_VERSION = 7
_EVENT_SECONDS_RE = re.compile(
    r"^\s*(?:[-*]\s*)?\[\[\s*start_s\s*=\s*"
    r"(?P<start>\d+(?:\.\d+)?)\s*[;,]\s*end_s\s*=\s*"
    r"(?P<end>\d+(?:\.\d+)?)\s*\]\]\s*(?P<description>\S.*)\s*$",
    flags=re.IGNORECASE,
)
_EVENT_LOCAL_CLOCK_RE = re.compile(
    r"^\s*(?:[-*]\s*)?\[\[\s*start_local\s*=\s*"
    r"(?P<start>\d{1,3}:[0-5]\d(?:\.\d{1,3})?)\s*[;,]\s*end_local\s*=\s*"
    r"(?P<end>\d{1,3}:[0-5]\d(?:\.\d{1,3})?)\s*\]\]\s*"
    r"(?P<description>\S.*)\s*$",
    flags=re.IGNORECASE,
)
_EVENT_SECONDS_COMPACT_RE = re.compile(
    r"^\s*(?:[-*]\s*)?\[\[\s*(?P<start>\d+(?:\.\d+)?)\s*[;,]\s*"
    r"(?P<end>\d+(?:\.\d+)?)\s*\]\]\s*(?P<description>\S.*)\s*$",
    flags=re.IGNORECASE,
)
_EVENT_LOCAL_CLOCK_COMPACT_RE = re.compile(
    r"^\s*(?:[-*]\s*)?\[\[\s*"
    r"(?P<start>\d{1,3}:[0-5]\d(?:\.\d{1,3})?)\s*[;,]\s*"
    r"(?P<end>\d{1,3}:[0-5]\d(?:\.\d{1,3})?)\s*\]\]\s*"
    r"(?P<description>\S.*)\s*$",
    flags=re.IGNORECASE,
)
_SINGLE_CLOSING_LOCAL_TAG_RE = re.compile(
    r"(?P<tag>\[\[\s*start_local\s*=\s*"
    r"\d{1,3}:[0-5]\d(?:\.\d{1,3})?\s*[;,]\s*end_local\s*=\s*"
    r"\d{1,3}:[0-5]\d(?:\.\d{1,3})?)\](?!\])",
    flags=re.IGNORECASE,
)
_LOCAL_TAG_REWRITE_RE = re.compile(
    r"(?P<prefix>\[\[\s*start_local\s*=\s*)"
    r"(?P<start>\d{1,3}:[0-5]\d(?:\.\d{1,3})?)"
    r"(?P<middle>\s*[;,]\s*end_local\s*=\s*)"
    r"(?P<end>\d{1,3}:[0-5]\d(?:\.\d{1,3})?)"
    r"(?P<suffix>\s*\]\])",
    flags=re.IGNORECASE,
)
_LOCAL_COMPACT_TAG_REWRITE_RE = re.compile(
    r"(?P<prefix>\[\[\s*)"
    r"(?P<start>\d{1,3}:[0-5]\d(?:\.\d{1,3})?)"
    r"(?P<middle>\s*[;,]\s*)"
    r"(?P<end>\d{1,3}:[0-5]\d(?:\.\d{1,3})?)"
    r"(?P<suffix>\s*\]\])",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ChunkRange:
    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class VideoJob:
    video_id: str
    video_path: Path
    annotated_duration_s: float | None
    question_ids: tuple[str, ...]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._")
    if not normalized:
        normalized = "video"
    if normalized != value:
        normalized = f"{normalized}-{_sha256_text(value)[:10]}"
    return normalized


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def plan_chunks(
    duration_s: float,
    *,
    chunk_seconds: float = 60.0,
    min_tail_seconds: float = 30.0,
) -> list[ChunkRange]:
    """Split a duration into target-size chunks and merge tails below the threshold."""
    duration = float(duration_s)
    target = float(chunk_seconds)
    minimum_tail = float(min_tail_seconds)
    if duration <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if target <= 0:
        raise ValueError(f"chunk_seconds must be positive, got {chunk_seconds}")
    if not 0 <= minimum_tail <= target:
        raise ValueError(
            "min_tail_seconds must be between zero and chunk_seconds, got "
            f"{min_tail_seconds} and {chunk_seconds}"
        )
    if duration <= target:
        return [ChunkRange(index=0, start_s=0.0, end_s=duration)]

    full_chunks = int(math.floor(duration / target))
    remainder = duration - full_chunks * target
    if math.isclose(remainder, 0.0, abs_tol=1e-6):
        remainder = 0.0

    boundaries = [(index * target, (index + 1) * target) for index in range(full_chunks)]
    if remainder > 0:
        if remainder <= minimum_tail and boundaries:
            start, _ = boundaries[-1]
            boundaries[-1] = (start, duration)
        else:
            boundaries.append((full_chunks * target, duration))
    return [
        ChunkRange(index=index, start_s=start, end_s=end)
        for index, (start, end) in enumerate(boundaries)
    ]


def load_chunk_prompt_config(
    path: Path,
    *,
    profile_key: str = "CHUNKED_CAPTION_PROMPT",
) -> dict[str, Any]:
    """Load the base instruction and all chunk prompt text from one YAML file."""
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Chunk prompt YAML must contain a mapping: {resolved}")
    normalized_profile_key = str(profile_key or "").strip()
    config = payload.get(normalized_profile_key)
    if not isinstance(config, Mapping):
        raise ValueError(f"Missing {normalized_profile_key} mapping: {resolved}")
    base_key = str(config.get("base_instruction_key") or "").strip()
    base_instruction = str(payload.get(base_key) or "").strip()
    template = str(config.get("template") or "").strip()
    modes = config.get("modes")
    if not base_key or not base_instruction:
        raise ValueError(f"Invalid chunk caption base instruction key in {resolved}")
    if not template:
        raise ValueError(f"Missing CHUNKED_CAPTION_PROMPT.template in {resolved}")
    if not isinstance(modes, Mapping):
        raise ValueError(f"Missing CHUNKED_CAPTION_PROMPT.modes in {resolved}")
    normalized_modes: dict[str, dict[str, str]] = {}
    for mode in sorted(TIMESTAMP_MODES):
        mode_config = modes.get(mode)
        if not isinstance(mode_config, Mapping):
            raise ValueError(f"Missing prompt contract for mode={mode!r} in {resolved}")
        normalized_modes[mode] = {}
        for field in ("time_contract", "line_format", "value_contract"):
            value = str(mode_config.get(field) or "").strip()
            if not value:
                raise ValueError(f"Missing {mode}.{field} in {resolved}")
            normalized_modes[mode][field] = value
    return {
        "source_path": str(resolved),
        "profile_key": normalized_profile_key,
        "base_instruction_key": base_key,
        "base_instruction": base_instruction,
        "template": template,
        "modes": normalized_modes,
    }


def build_chunk_prompt(
    prompt_config: Mapping[str, Any],
    *,
    mode: str,
    chunk: ChunkRange,
) -> str:
    """Render a YAML-owned machine-decodable timestamp prompt."""
    if mode not in TIMESTAMP_MODES:
        raise ValueError(f"Unsupported timestamp mode: {mode!r}")
    modes = prompt_config.get("modes")
    mode_config = modes.get(mode) if isinstance(modes, Mapping) else None
    if not isinstance(mode_config, Mapping):
        raise ValueError(f"Prompt config has no mode={mode!r} contract")
    substitutions = {
        "chunk_duration_s": f"{chunk.duration_s:.3f}",
        "chunk_start_s": f"{chunk.start_s:.3f}",
        "chunk_end_s": f"{chunk.end_s:.3f}",
    }
    try:
        return str(prompt_config["template"]).format(
            base_instruction=str(prompt_config["base_instruction"]).strip(),
            time_contract=str(mode_config["time_contract"]).format(**substitutions),
            line_format=str(mode_config["line_format"]).format(**substitutions),
            value_contract=str(mode_config["value_contract"]).format(**substitutions),
        ).strip()
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid YAML chunk prompt template for mode={mode!r}: {exc}") from exc


def parse_timestamped_events(
    response: str,
    *,
    mode: str,
    chunk: ChunkRange,
    tolerance_s: float = 1.5,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse tagged events; only local-offset mode changes returned timestamps."""
    if mode not in TIMESTAMP_MODES:
        raise ValueError(f"Unsupported timestamp mode: {mode!r}")
    events: list[dict[str, Any]] = []
    warnings: list[str] = []
    event_pattern = _EVENT_LOCAL_CLOCK_RE if mode == "local_offset" else _EVENT_SECONDS_RE
    compact_pattern = (
        _EVENT_LOCAL_CLOCK_COMPACT_RE
        if mode == "local_offset"
        else _EVENT_SECONDS_COMPACT_RE
    )
    compact_warning_added = False

    def parse_value(value: str) -> float:
        if mode != "local_offset":
            return float(value)
        minutes, seconds = value.split(":", 1)
        return int(minutes) * 60 + float(seconds)

    for line_number, raw_line in enumerate(str(response or "").splitlines(), 1):
        line = raw_line.strip()
        if not line or line in {"```", "```text"}:
            continue
        match = event_pattern.match(line)
        used_compact_format = False
        if match is None:
            match = compact_pattern.match(line)
            used_compact_format = match is not None
        if not match:
            if events:
                events[-1]["description"] = f"{events[-1]['description']} {line}".strip()
                warnings.append(
                    f"line {line_number} was appended to the preceding timestamped event"
                )
            else:
                warnings.append(f"line {line_number} did not match the timestamp contract")
            continue
        if used_compact_format and not compact_warning_added:
            warnings.append("response used compact timestamp fields without names")
            compact_warning_added = True
        returned_start = parse_value(match.group("start"))
        returned_end = parse_value(match.group("end"))
        if returned_end < returned_start:
            raise ValueError(
                f"line {line_number} has end_s < start_s: {returned_end} < {returned_start}"
            )
        if mode == "local_offset":
            lower, upper = 0.0, chunk.duration_s
            original_start = chunk.start_s + returned_start
            original_end = chunk.start_s + returned_end
        else:
            lower, upper = chunk.start_s, chunk.end_s
            original_start = returned_start
            original_end = returned_end
        if returned_start < lower - tolerance_s or returned_end > upper + tolerance_s:
            raise ValueError(
                f"line {line_number} timestamps [{returned_start}, {returned_end}] are outside "
                f"the expected [{lower}, {upper}] range"
            )
        if mode == "local_offset":
            if returned_start < lower or returned_end > upper:
                warnings.append(
                    f"line {line_number} was clamped from [{returned_start}, {returned_end}] "
                    f"to the chunk boundary [{lower}, {upper}]"
                )
            original_start = min(max(original_start, chunk.start_s), chunk.end_s)
            original_end = min(max(original_end, original_start), chunk.end_s)
        elif returned_start < lower or returned_end > upper:
            warnings.append(
                f"line {line_number} is outside [{lower}, {upper}] and was retained "
                "without normalization"
            )
        events.append(
            {
                "returned_start_s": returned_start,
                "returned_end_s": returned_end,
                "original_start_s": round(original_start, 3),
                "original_end_s": round(original_end, 3),
                "description": match.group("description").strip(),
                "source_line": line_number,
            }
        )
    if not events:
        raise ValueError("Caption response contained no parseable timestamped event lines")
    for previous, current in zip(events, events[1:]):
        if current["original_start_s"] < previous["original_start_s"]:
            warnings.append(
                "event start timestamps are not monotonic at source line "
                f"{current['source_line']}"
            )
    return events, warnings


def repair_single_closing_local_timestamp_tags(response: str) -> tuple[str, int]:
    """Repair only an unambiguous missing second bracket in a local timestamp tag."""
    return _SINGLE_CLOSING_LOCAL_TAG_RE.subn(r"\g<tag>]]", str(response or ""))


def rewrite_local_timestamps_to_original(response: str, *, chunk: ChunkRange) -> str:
    """Offset timestamp values while preserving every non-timestamp character."""

    def parse_clock(value: str) -> float:
        minutes, seconds = value.split(":", 1)
        return int(minutes) * 60 + float(seconds)

    def format_clock(value: float) -> str:
        milliseconds = max(0, int(round(value * 1000)))
        minutes, remainder = divmod(milliseconds, 60_000)
        seconds, millis = divmod(remainder, 1000)
        return f"{minutes:02d}:{seconds:02d}.{millis:03d}"

    def replace(match: re.Match[str]) -> str:
        start = chunk.start_s + parse_clock(match.group("start"))
        end = chunk.start_s + parse_clock(match.group("end"))
        return "{}{}{}{}{}".format(
            match.group("prefix"),
            format_clock(start),
            match.group("middle"),
            format_clock(end),
            match.group("suffix"),
        )

    rewritten, named_count = _LOCAL_TAG_REWRITE_RE.subn(
        replace,
        str(response or ""),
    )
    rewritten, compact_count = _LOCAL_COMPACT_TAG_REWRITE_RE.subn(
        replace,
        rewritten,
    )
    if named_count + compact_count == 0:
        raise ValueError("Caption response contained no rewritable local timestamp tags")
    return rewritten


def format_original_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def render_chunk_events(events: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        "[{} - {}] {}".format(
            format_original_timestamp(float(event["original_start_s"])),
            format_original_timestamp(float(event["original_end_s"])),
            str(event["description"]).strip(),
        )
        for event in events
    )


def load_video_jobs_from_jsonl(path: Path) -> list[VideoJob]:
    """Load unique recordings while retaining every question ID for cache materialization."""
    records: dict[str, dict[str, Any]] = {}
    seen_questions: set[str] = set()
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        for row_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            supervisions = row.get("supervisions") or []
            if not supervisions or not isinstance(supervisions[0], Mapping):
                raise ValueError(f"No supervision at {path}:{row_number}")
            supervision = supervisions[0]
            custom = supervision.get("custom") or {}
            recording = row.get("recording") or {}
            sources = recording.get("sources") or []
            source_video = sources[0].get("source") if sources and isinstance(sources[0], Mapping) else None
            question_id = str(row.get("id") or "").strip()
            video_id = str(
                supervision.get("recording_id")
                or custom.get("video_id")
                or recording.get("id")
                or ""
            ).strip()
            video_path = Path(str(custom.get("video_path") or source_video or "")).expanduser()
            annotated_duration = row.get("duration", supervision.get("duration", recording.get("duration")))
            if not question_id or not video_id or not str(video_path):
                raise ValueError(f"Missing question/video/path fields at {path}:{row_number}")
            if question_id in seen_questions:
                raise ValueError(f"Duplicate question ID {question_id!r} at {path}:{row_number}")
            seen_questions.add(question_id)
            resolved_path = video_path.resolve()
            existing = records.get(video_id)
            if existing is None:
                records[video_id] = {
                    "video_path": resolved_path,
                    "annotated_duration_s": float(annotated_duration) if annotated_duration is not None else None,
                    "question_ids": [question_id],
                }
            else:
                if existing["video_path"] != resolved_path:
                    raise ValueError(
                        f"Video ID {video_id!r} maps to multiple paths: "
                        f"{existing['video_path']} and {resolved_path}"
                    )
                existing["question_ids"].append(question_id)
    return [
        VideoJob(
            video_id=video_id,
            video_path=payload["video_path"],
            annotated_duration_s=payload["annotated_duration_s"],
            question_ids=tuple(payload["question_ids"]),
        )
        for video_id, payload in records.items()
    ]


def load_direct_video_jobs(paths: Iterable[Path]) -> list[VideoJob]:
    jobs: list[VideoJob] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        video_id = path.stem
        if video_id in seen:
            raise ValueError(f"Duplicate direct video ID: {video_id!r}")
        seen.add(video_id)
        jobs.append(VideoJob(video_id, path, None, ()))
    return jobs


def _runtime_fingerprint() -> dict[str, Any]:
    keys = (
        "GEMINI_API_BACKEND",
        "CAPTIONER_GEMINI_MODEL",
        "CAPTIONER_GEMINI_BASE_URL",
        "CAPTIONER_GEMINI_PROVIDER",
        "CAPTIONER_GEMINI_AUTH_MODE",
        "CAPTIONER_GEMINI_VIDEO_ONLY",
        "CAPTIONER_GEMINI_TIMEOUT",
        "CAPTIONER_GEMINI_MAX_RETRIES",
        "CAPTIONER_GEMINI_RETRY_DELAY_S",
        "CAPTIONER_GEMINI_TEMPERATURE",
        "CAPTIONER_GEMINI_TOP_P",
        "CAPTIONER_GEMINI_TOP_K",
        "CAPTIONER_GEMINI_MAX_TOKENS",
        "CAPTIONER_GEMINI_SEED",
    )
    return {key: os.environ.get(key) for key in keys}


def _chunk_artifact_path(output_dir: Path, video_id: str, index: int) -> Path:
    return output_dir / "video_artifacts" / _safe_component(video_id) / f"chunk_{index:04d}.json"


def _video_artifact_path(output_dir: Path, video_id: str) -> Path:
    return output_dir / "video_artifacts" / _safe_component(video_id) / "video.json"


def _chunk_signature(
    *,
    job: VideoJob,
    duration_s: float,
    chunk: ChunkRange,
    prompt_config: Mapping[str, Any],
    mode: str,
) -> str:
    stat = job.video_path.stat()
    payload = {
        "video_id": job.video_id,
        "video_path": str(job.video_path),
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "duration_s": round(duration_s, 6),
        "chunk": {"index": chunk.index, "start_s": chunk.start_s, "end_s": chunk.end_s},
        "caption_prompt_sha256": _sha256_text(
            build_chunk_prompt(prompt_config, mode=mode, chunk=chunk)
        ),
        "timestamp_mode": mode,
        "runtime": _runtime_fingerprint(),
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _load_resumable_chunk(path: Path, signature: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_sha256") != signature:
        return None
    if payload.get("status") != "complete" and not payload.get("raw_response"):
        return None
    return payload


def _refresh_parsed_chunk(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Reparse stored raw text after parser-only improvements, without a model call."""
    has_current_derived_fields = (
        payload.get("timestamp_mode") != "local_offset"
        or "rewritten_original_timestamp_response" in payload
    )
    if (
        payload.get("parser_schema_version") == PARSER_SCHEMA_VERSION
        and has_current_derived_fields
    ):
        return payload
    chunk = ChunkRange(
        int(payload["chunk_index"]),
        float(payload["actual_chunk_start_s"]),
        float(payload["actual_chunk_end_s"]),
    )
    raw_response = str(payload.get("raw_response") or "")
    contract_response, repair_count = repair_single_closing_local_timestamp_tags(
        raw_response
    )
    events, warnings = parse_timestamped_events(
        contract_response,
        mode=str(payload["timestamp_mode"]),
        chunk=chunk,
    )
    payload["events"] = events
    payload["format_warnings"] = warnings
    if payload["timestamp_mode"] == "local_offset":
        payload["rendered_original_timestamps"] = render_chunk_events(events)
        payload["rewritten_original_timestamp_response"] = (
            rewrite_local_timestamps_to_original(contract_response, chunk=chunk)
        )
    else:
        payload.pop("rendered_original_timestamps", None)
        payload.pop("rewritten_original_timestamp_response", None)
    payload["timestamp_contract_response"] = contract_response
    payload["deterministic_timestamp_repairs"] = {
        "missing_second_closing_bracket": repair_count,
    }
    payload["timestamp_normalization_applied"] = (
        payload["timestamp_mode"] == "local_offset"
    )
    payload["parser_schema_version"] = PARSER_SCHEMA_VERSION
    payload["status"] = "complete"
    payload.pop("error_type", None)
    payload.pop("error", None)
    _atomic_json(path, payload)
    return payload


def _run_chunk(
    *,
    job: VideoJob,
    duration_s: float,
    chunk: ChunkRange,
    prompt_config: Mapping[str, Any],
    mode: str,
    output_dir: Path,
    force: bool,
) -> dict[str, Any]:
    artifact_path = _chunk_artifact_path(output_dir, job.video_id, chunk.index)
    signature = _chunk_signature(
        job=job,
        duration_s=duration_s,
        chunk=chunk,
        prompt_config=prompt_config,
        mode=mode,
    )
    if not force:
        resumed = _load_resumable_chunk(artifact_path, signature)
        if resumed is not None:
            try:
                return _refresh_parsed_chunk(artifact_path, resumed)
            except ValueError:
                # A stored raw response may still violate the current parser contract.
                # Fall through to a fresh model call instead of aborting the whole run.
                pass

    task_started = time.time()
    total_started = time.perf_counter()
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "config_sha256": signature,
        "video_id": job.video_id,
        "video_path": str(job.video_path),
        "timestamp_mode": mode,
        "chunk_index": chunk.index,
        "chunk_start_s": round(chunk.start_s, 6),
        "chunk_end_s": round(chunk.end_s, 6),
        "chunk_duration_s": round(chunk.duration_s, 6),
        "task_started_epoch_s": task_started,
    }
    clip_path: Path | None = None
    response = ""
    latency_summary: dict[str, Any] | None = None
    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f"{_safe_component(job.video_id)}_{chunk.index:04d}_",
            suffix=".mp4",
            dir=artifact_path.parent,
            delete=False,
        ) as clip_stream:
            clip_path = Path(clip_stream.name)
        preparation_started = time.perf_counter()
        actual_start, actual_end = cut_video_clip(
            str(job.video_path),
            str(clip_path),
            chunk.start_s,
            chunk.end_s,
            preserve_audio=True,
            video_duration=duration_s,
        )
        artifact["clip_preparation_seconds"] = _round(time.perf_counter() - preparation_started)
        artifact["actual_chunk_start_s"] = round(actual_start, 6)
        artifact["actual_chunk_end_s"] = round(actual_end, 6)
        effective_chunk = ChunkRange(chunk.index, actual_start, actual_end)
        prompt = build_chunk_prompt(prompt_config, mode=mode, chunk=effective_chunk)
        artifact["caption_prompt"] = prompt
        artifact["caption_prompt_sha256"] = _sha256_text(prompt)
        with sample_latency() as tracker:
            try:
                with model_component("perception"):
                    response = call_gemini_perception(
                        video_path=str(clip_path),
                        audio_path=None,
                        prompt=prompt,
                        system_prompt="",
                        env_prefix="CAPTIONER_GEMINI",
                    )
            except BaseException:
                latency_summary = tracker.summary(complete=False)
                raise
        latency_summary = tracker.summary(complete=True)
        metadata = consume_last_perception_metadata()
        contract_response, repair_count = repair_single_closing_local_timestamp_tags(
            response
        )
        events, warnings = parse_timestamped_events(
            contract_response,
            mode=mode,
            chunk=effective_chunk,
        )
        artifact.update(
            {
                "status": "complete",
                "raw_response": response,
                "timestamp_contract_response": contract_response,
                "deterministic_timestamp_repairs": {
                    "missing_second_closing_bracket": repair_count,
                },
                "events": events,
                "format_warnings": warnings,
                "parser_schema_version": PARSER_SCHEMA_VERSION,
                "timestamp_normalization_applied": mode == "local_offset",
                "model": metadata.get("model"),
                "backend": metadata.get("backend"),
                "token_usage": metadata.get("token_usage"),
                "latency": latency_summary,
            }
        )
        if mode == "local_offset":
            artifact["rendered_original_timestamps"] = render_chunk_events(events)
            artifact["rewritten_original_timestamp_response"] = (
                rewrite_local_timestamps_to_original(
                    contract_response,
                    chunk=effective_chunk,
                )
            )
    except BaseException as exc:
        artifact.update(
            {
                "status": "error",
                "raw_response": response,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "latency": latency_summary,
            }
        )
    finally:
        if clip_path is not None:
            try:
                clip_path.unlink(missing_ok=True)
            except OSError:
                pass
        artifact["task_finished_epoch_s"] = time.time()
        artifact["total_wall_seconds"] = _round(time.perf_counter() - total_started)
        _atomic_json(artifact_path, artifact)
    return artifact


def aggregate_video_latency(chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report the requested assumed-parallel maximum and serial sums."""
    adjusted = [
        float(chunk["latency"]["retry_adjusted_seconds"])
        for chunk in chunks
        if isinstance(chunk.get("latency"), Mapping)
        and chunk["latency"].get("retry_adjusted_seconds") is not None
    ]
    successful_model = [
        float(chunk["latency"]["successful_model_call_seconds"])
        for chunk in chunks
        if isinstance(chunk.get("latency"), Mapping)
        and chunk["latency"].get("successful_model_call_seconds") is not None
    ]
    total_wall = [float(chunk.get("total_wall_seconds") or 0.0) for chunk in chunks]
    started = [float(chunk["task_started_epoch_s"]) for chunk in chunks if chunk.get("task_started_epoch_s")]
    finished = [float(chunk["task_finished_epoch_s"]) for chunk in chunks if chunk.get("task_finished_epoch_s")]
    return {
        "definition": (
            "Video latency assumes all chunks run concurrently: the maximum chunk "
            "retry-adjusted latency. Failed attempts and retry backoff are excluded."
        ),
        "assumed_parallel_retry_adjusted_seconds": _round(max(adjusted) if adjusted else None),
        "sum_chunk_retry_adjusted_seconds": _round(sum(adjusted)) if adjusted else None,
        "max_chunk_successful_model_call_seconds": _round(max(successful_model) if successful_model else None),
        "sum_chunk_successful_model_call_seconds": _round(sum(successful_model)) if successful_model else None,
        "max_chunk_total_wall_seconds": _round(max(total_wall) if total_wall else None),
        "sum_chunk_total_wall_seconds": _round(sum(total_wall)) if total_wall else None,
        "observed_worker_span_seconds": _round(max(finished) - min(started)) if started and finished else None,
        "complete_chunks": sum(chunk.get("status") == "complete" for chunk in chunks),
        "error_chunks": sum(chunk.get("status") != "complete" for chunk in chunks),
    }


def _sum_token_usage(chunks: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    totals: defaultdict[str, float] = defaultdict(float)
    integer_keys: set[str] = set()
    for chunk in chunks:
        usage = chunk.get("token_usage")
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            totals[str(key)] += float(value)
            if isinstance(value, int):
                integer_keys.add(str(key))
    return {
        key: int(value) if key in integer_keys and value.is_integer() else round(value, 3)
        for key, value in sorted(totals.items())
    }


def _assemble_video(
    *,
    job: VideoJob,
    probed_duration_s: float,
    chunks: Sequence[Mapping[str, Any]],
    mode: str,
    base_prompt: str,
    output_dir: Path,
) -> dict[str, Any]:
    ordered = sorted(chunks, key=lambda item: int(item["chunk_index"]))
    rendered_sections = []
    for chunk in ordered:
        if chunk.get("status") != "complete":
            continue
        caption = (
            str(chunk["rewritten_original_timestamp_response"])
            if mode == "local_offset"
            else str(chunk["raw_response"])
        )
        rendered_sections.append(
            "Chunk {index} (original video {start} - {end}):\n{caption}".format(
                index=int(chunk["chunk_index"]) + 1,
                start=format_original_timestamp(float(chunk["actual_chunk_start_s"])),
                end=format_original_timestamp(float(chunk["actual_chunk_end_s"])),
                caption=caption,
            )
        )
    payload = {
        "schema_version": 1,
        "status": "complete" if len(rendered_sections) == len(ordered) else "error",
        "video_id": job.video_id,
        "video_path": str(job.video_path),
        "annotated_duration_s": job.annotated_duration_s,
        "probed_duration_s": round(probed_duration_s, 6),
        "timestamp_mode": mode,
        "base_caption_prompt": base_prompt.strip(),
        "base_caption_prompt_sha256": _sha256_text(base_prompt.strip()),
        "question_ids": list(job.question_ids),
        "chunk_count": len(ordered),
        "chunks": ordered,
        "combined_response": "\n\n".join(rendered_sections),
        "token_usage": _sum_token_usage(ordered),
        "latency": aggregate_video_latency(ordered),
    }
    _atomic_json(_video_artifact_path(output_dir, job.video_id), payload)
    return payload


def run_chunked_caption_jobs(
    jobs: Sequence[VideoJob],
    *,
    output_dir: Path,
    prompt_config: Mapping[str, Any],
    mode: str = "local_offset",
    chunk_seconds: float = 60.0,
    min_tail_seconds: float = 30.0,
    max_workers: int = 5,
    force: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Caption all unique videos, persist resumable chunks, and assemble per-video outputs."""
    if mode not in TIMESTAMP_MODES:
        raise ValueError(f"Unsupported timestamp mode: {mode!r}")
    if max_workers < 1:
        raise ValueError("max_workers must be at least one")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_prompt = str(prompt_config.get("base_instruction") or "").strip()
    if not base_prompt:
        raise ValueError("prompt_config has no base_instruction")
    started = time.perf_counter()
    probed: dict[str, float] = {}
    plans: dict[str, list[ChunkRange]] = {}
    for job in jobs:
        if not job.video_path.is_file():
            raise FileNotFoundError(f"Video not found: {job.video_path}")
        duration = probe_media_duration(str(job.video_path))
        probed[job.video_id] = duration
        plans[job.video_id] = plan_chunks(
            duration,
            chunk_seconds=chunk_seconds,
            min_tail_seconds=min_tail_seconds,
        )

    chunk_results: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    future_to_video: dict[Any, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for job in jobs:
            for chunk in plans[job.video_id]:
                future = executor.submit(
                    _run_chunk,
                    job=job,
                    duration_s=probed[job.video_id],
                    chunk=chunk,
                    prompt_config=prompt_config,
                    mode=mode,
                    output_dir=output_dir,
                    force=force,
                )
                future_to_video[future] = job.video_id
        for future in as_completed(future_to_video):
            video_id = future_to_video[future]
            result = future.result()
            chunk_results[video_id].append(result)
            print(
                f"[{mode}] {video_id} chunk={result['chunk_index']} "
                f"status={result['status']} total={result.get('total_wall_seconds')}s",
                flush=True,
            )

    video_results = [
        _assemble_video(
            job=job,
            probed_duration_s=probed[job.video_id],
            chunks=chunk_results[job.video_id],
            mode=mode,
            base_prompt=base_prompt,
            output_dir=output_dir,
        )
        for job in jobs
    ]
    all_chunks = [chunk for result in video_results for chunk in result["chunks"]]
    artifact_starts = [
        float(chunk["task_started_epoch_s"])
        for chunk in all_chunks
        if chunk.get("task_started_epoch_s") is not None
    ]
    artifact_finishes = [
        float(chunk["task_finished_epoch_s"])
        for chunk in all_chunks
        if chunk.get("task_finished_epoch_s") is not None
    ]
    summary = {
        "schema_version": 1,
        "timestamp_mode": mode,
        "video_count": len(video_results),
        "question_count": sum(len(job.question_ids) for job in jobs),
        "chunk_count": sum(result["chunk_count"] for result in video_results),
        "complete_videos": sum(result["status"] == "complete" for result in video_results),
        "error_videos": sum(result["status"] != "complete" for result in video_results),
        "invocation_wall_seconds": _round(time.perf_counter() - started),
        "artifact_task_span_seconds": _round(
            max(artifact_finishes) - min(artifact_starts)
        )
        if artifact_starts and artifact_finishes
        else None,
        "artifact_task_span_note": (
            "Span from the earliest persisted chunk task start to the latest finish; "
            "it may cross invocations when a run resumes partial artifacts."
        ),
        "videos": [
            {
                "video_id": result["video_id"],
                "status": result["status"],
                "chunk_count": result["chunk_count"],
                "latency": result["latency"],
            }
            for result in video_results
        ],
    }
    _atomic_json(output_dir / "summary.json", summary)
    return video_results, summary


def materialize_question_cache(
    *,
    jobs: Sequence[VideoJob],
    video_results: Sequence[Mapping[str, Any]],
    cache_dir: Path,
    input_jsonl: Path,
    prompt_yaml: Path,
    base_prompt: str,
) -> dict[str, Any]:
    """Materialize deduplicated video captions into the existing question cache schema."""
    by_video = {str(result["video_id"]): result for result in video_results}
    incomplete = [video_id for video_id, result in by_video.items() if result.get("status") != "complete"]
    if incomplete:
        raise ValueError(f"Cannot materialize cache with incomplete videos: {incomplete[:5]}")
    target = cache_dir.expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        question_ids = [question_id for job in jobs for question_id in job.question_ids]
        coverage = validate_caption_cache_coverage(
            target,
            question_ids,
            expected_prompt=base_prompt,
        )
        return load_caption_cache_manifest(target) | {
            "validated_existing_cache": coverage
        }
    target.mkdir(parents=True, exist_ok=True)

    entries: dict[str, dict[str, Any]] = {}
    for job in jobs:
        result = by_video[job.video_id]
        source_file = _video_artifact_path(cache_dir.parent, job.video_id)
        for question_id in job.question_ids:
            entries[question_id] = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "question_id": question_id,
                "response": str(result["combined_response"]),
                "caption_prompt": base_prompt.strip(),
                "source_results_file": str(source_file),
                "source_rank": 1,
                "source_backend": "gemini_chunked",
                "source_model": os.environ.get("CAPTIONER_GEMINI_MODEL", ""),
                "source_token_usage": result.get("token_usage"),
                "source_video_id": job.video_id,
                "chunk_count": result.get("chunk_count"),
                "timestamp_mode": result.get("timestamp_mode"),
                "video_latency": result.get("latency"),
            }

    digest = hashlib.sha256()
    for question_id in sorted(entries):
        entry_text = json.dumps(entries[question_id], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest.update(question_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry_text.encode("utf-8"))
        digest.update(b"\n")
        _atomic_json(target / f"{question_id}.json", entries[question_id])
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "input_jsonl": str(input_jsonl.expanduser().resolve()),
        "prompt_yaml": str(prompt_yaml.expanduser().resolve()),
        "caption_prompt": base_prompt.strip(),
        "prompt_sha256": _sha256_text(base_prompt.strip()),
        "generation_strategy": "chunked_caption_local_timestamp_then_deterministic_offset",
        "source_backend": "gemini_chunked",
        "source_model": os.environ.get("CAPTIONER_GEMINI_MODEL", ""),
        "video_count": len(jobs),
        "question_count": len(entries),
        "content_sha256": digest.hexdigest(),
    }
    _atomic_json(target / MANIFEST_NAME, manifest)
    return manifest
