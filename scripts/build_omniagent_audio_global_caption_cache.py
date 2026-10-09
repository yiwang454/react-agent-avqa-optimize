#!/usr/bin/env python3
"""Complete a question cache from OmniAgent audio_global_caption traces.

Successful trace responses are preserved per question.  Questions without a
successful call are grouped by video; the exact OmniAgent audio-only request is
then made once for each such video and shared by those missing questions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import yaml


REPO_DIR = Path(__file__).resolve().parents[1]
DSPY_DIR = REPO_DIR / "DSPy"
if str(DSPY_DIR) not in sys.path:
    sys.path.insert(0, str(DSPY_DIR))

from dspy_avqa.gemini_api import call_gemini_messages  # noqa: E402


CACHE_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
TOOL_NAME = "audio_global_caption"


@dataclass(frozen=True)
class Question:
    question_id: str
    video_id: str
    video_path: Path


@dataclass(frozen=True)
class TraceCaption:
    question_id: str
    response: str
    row_number: int
    turn_id: Any
    call_index: int
    model: str
    token_usage: Any


@dataclass(frozen=True)
class RuntimeConfig:
    model: str
    provider: str
    auth_mode: str
    base_url: str
    timeout: int
    max_retries: int
    retry_delay_s: float
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    seed: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-yaml", type=Path, required=True)
    parser.add_argument(
        "--prompt-key", default="OMNIAGENT_AUDIO_GLOBAL_CAPTION_PROMPT"
    )
    parser.add_argument("--perception-config-yaml", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=6)
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return (
        normalized
        if normalized == value and normalized
        else f"{normalized or 'item'}-{sha256_text(value)[:10]}"
    )


def cache_entry_path(cache_dir: Path, question_id: str) -> Path:
    """Return the exact question filename required by --audio-caption-dir."""
    if (
        not question_id
        or Path(question_id).name != question_id
        or "/" in question_id
        or "\\" in question_id
    ):
        raise ValueError(f"Unsafe cache question ID: {question_id!r}")
    return cache_dir / f"{question_id}.json"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def load_prompt(path: Path, key: str) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping) or not isinstance(payload.get(key), str):
        raise ValueError(f"Prompt key {key!r} is missing from {path}")
    prompt = payload[key]
    if not prompt.strip():
        raise ValueError(f"Prompt key {key!r} is empty in {path}")
    return prompt


def _mapping(value: Any, *, label: str, path: Path) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping in {path}")
    return value


def load_runtime_config(path: Path) -> tuple[RuntimeConfig, Mapping[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Perception config must be a mapping: {path}")
    task = _mapping(payload.get("task"), label="task", path=path)
    model = _mapping(payload.get("model"), label="model", path=path)
    sampling = _mapping(payload.get("sampling_params"), label="sampling_params", path=path)
    seed_value = sampling.get("gemini_seed", sampling.get("seed"))
    runtime = RuntimeConfig(
        model=str(model.get("gemini_model") or "gemini-2.5-flash"),
        provider=str(model.get("gemini_provider") or os.getenv("GEMINI_PROVIDER") or "apiplus"),
        auth_mode=str(model.get("gemini_auth_mode") or os.getenv("GEMINI_AUTH_MODE") or "bearer"),
        base_url=str(model.get("gemini_base_url") or os.getenv("GEMINI_BASE_URL") or "https://api.apiplus.org").rstrip("/"),
        timeout=int(task.get("timeout", 180)),
        max_retries=int(task.get("max_retries", 6)),
        retry_delay_s=float(task.get("retry_delay_s", 5)),
        temperature=float(sampling.get("temperature", 0.6)),
        top_p=float(sampling.get("top_p", 0.95)),
        top_k=int(sampling.get("top_k", 20)),
        max_tokens=int(sampling.get("max_tokens", 4096)),
        seed=int(seed_value) if seed_value is not None else None,
    )
    if runtime.provider != "apiplus":
        raise ValueError(
            "Exact OmniAgent replication currently requires gemini_provider=apiplus"
        )
    return runtime, payload


def load_questions(path: Path) -> list[Question]:
    questions: list[Question] = []
    seen_ids: set[str] = set()
    video_paths: dict[str, Path] = {}
    with path.open(encoding="utf-8") as stream:
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
            source_path = (
                sources[0].get("source")
                if sources and isinstance(sources[0], Mapping)
                else None
            )
            question_id = str(row.get("id") or "").strip()
            video_id = str(
                supervision.get("recording_id")
                or custom.get("video_id")
                or recording.get("id")
                or ""
            ).strip()
            raw_video_path = str(custom.get("video_path") or source_path or "").strip()
            if not question_id or not video_id or not raw_video_path:
                raise ValueError(f"Missing question/video/path at {path}:{row_number}")
            if question_id in seen_ids:
                raise ValueError(f"Duplicate question ID {question_id!r}")
            video_path = Path(raw_video_path).expanduser().resolve()
            if not video_path.is_file():
                raise FileNotFoundError(f"Video not found for {question_id}: {video_path}")
            prior_path = video_paths.setdefault(video_id, video_path)
            if prior_path != video_path:
                raise ValueError(f"Video {video_id!r} maps to multiple paths")
            seen_ids.add(question_id)
            questions.append(Question(question_id, video_id, video_path))
    return questions


def _trace_question_id(row: Mapping[str, Any]) -> str:
    question_data = row.get("question_data")
    if isinstance(question_data, Mapping):
        question_id = str(question_data.get("question_id") or "").strip()
        if question_id:
            return question_id
    return str(row.get("video_id") or "").strip()


def _observation_answer(turn: Mapping[str, Any]) -> str:
    observation = turn.get("tool_observation")
    if isinstance(observation, Mapping):
        return str(observation.get("answer") or "").strip()
    return ""


def _perception_metadata(turn: Mapping[str, Any], prompt: str) -> tuple[str, Any]:
    matching: list[Mapping[str, Any]] = []
    for call in turn.get("perception_calls") or []:
        if isinstance(call, Mapping) and str(call.get("prompt") or "") == prompt:
            matching.append(call)
    if not matching:
        return "", None
    call = matching[-1]
    return str(call.get("model") or ""), call.get("usage_metadata")


def extract_trace_captions(
    path: Path,
    *,
    expected_question_ids: set[str],
    prompt: str,
) -> tuple[dict[str, TraceCaption], dict[str, Any]]:
    selected: dict[str, TraceCaption] = {}
    seen_rows: list[str] = []
    successful_calls = 0
    failed_calls = 0
    call_counts: Counter[str] = Counter()
    prompt_mismatch_calls = 0
    with path.open(encoding="utf-8") as stream:
        for row_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = _trace_question_id(row)
            if not question_id:
                raise ValueError(f"Trace row has no question ID at {path}:{row_number}")
            seen_rows.append(question_id)
            question_data = row.get("question_data") or {}
            for call_index, turn in enumerate(question_data.get("turn_trace") or [], 1):
                if not isinstance(turn, Mapping) or turn.get("tool_name") != TOOL_NAME:
                    continue
                call_counts[question_id] += 1
                response = _observation_answer(turn)
                if turn.get("tool_error") or not response:
                    failed_calls += 1
                    continue
                successful_calls += 1
                trace_prompt = ""
                for call in turn.get("perception_calls") or []:
                    if isinstance(call, Mapping):
                        trace_prompt = str(call.get("prompt") or "")
                        if trace_prompt == prompt:
                            break
                if trace_prompt and trace_prompt != prompt:
                    prompt_mismatch_calls += 1
                    continue
                model, token_usage = _perception_metadata(turn, prompt)
                selected.setdefault(
                    question_id,
                    TraceCaption(
                        question_id=question_id,
                        response=response,
                        row_number=row_number,
                        turn_id=turn.get("turn_id"),
                        call_index=call_index,
                        model=model,
                        token_usage=token_usage,
                    ),
                )
    duplicates = sorted(key for key, count in Counter(seen_rows).items() if count > 1)
    trace_ids = set(seen_rows)
    if duplicates:
        raise ValueError(f"Duplicate question IDs in trace, e.g. {duplicates[:5]}")
    if trace_ids != expected_question_ids:
        raise ValueError(
            "Trace/input question coverage mismatch: "
            f"missing={sorted(expected_question_ids - trace_ids)[:5]} "
            f"extra={sorted(trace_ids - expected_question_ids)[:5]}"
        )
    if prompt_mismatch_calls:
        raise ValueError(
            f"Found {prompt_mismatch_calls} successful {TOOL_NAME} calls with a different prompt"
        )
    summary = {
        "trace_rows": len(seen_rows),
        "successful_tool_calls": successful_calls,
        "failed_or_empty_tool_calls": failed_calls,
        "questions_with_successful_trace_caption": len(selected),
        "questions_with_repeated_tool_calls": sum(count > 1 for count in call_counts.values()),
        "prompt_mismatch_calls": prompt_mismatch_calls,
    }
    return selected, summary


def derived_existing_audio_path(video_path: Path) -> Path | None:
    # This intentionally mirrors OmniAgent's audio_llm._existing_audio_path.
    candidate = Path(str(video_path).replace(".mp4", ".wav").replace("videos", "audios"))
    return candidate if candidate != video_path and candidate.is_file() else None


def artifact_path(output_dir: Path, video_id: str) -> Path:
    return output_dir / "video_artifacts" / safe_component(video_id) / "video.json"


def runtime_dict(runtime: RuntimeConfig) -> dict[str, Any]:
    return {
        "model": runtime.model,
        "provider": runtime.provider,
        "auth_mode": runtime.auth_mode,
        "base_url": runtime.base_url,
        "timeout": runtime.timeout,
        "max_retries": runtime.max_retries,
        "retry_delay_s": runtime.retry_delay_s,
        "temperature": runtime.temperature,
        "top_p": runtime.top_p,
        "top_k": runtime.top_k,
        "max_tokens": runtime.max_tokens,
        "seed": runtime.seed,
    }


def generation_signature(question: Question, prompt: str, runtime: RuntimeConfig) -> str:
    stat = question.video_path.stat()
    payload = {
        "video_id": question.video_id,
        "video_path": str(question.video_path),
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "prompt": prompt,
        "runtime": runtime_dict(runtime),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _call_exact_audio_request(
    audio_path: Path,
    *,
    prompt: str,
    runtime: RuntimeConfig,
    api_key: str,
) -> tuple[str, Mapping[str, Any]]:
    media_part = {
        "inlineData": {
            "mimeType": mimetypes.guess_type(audio_path.name)[0]
            or "application/octet-stream",
            "data": __import__("base64").b64encode(audio_path.read_bytes()).decode("utf-8"),
        }
    }
    result = call_gemini_messages(
        [{"role": "user", "parts": [media_part, {"text": prompt}]}],
        system_prompt="",
        model=runtime.model,
        api_key=api_key,
        base_url=runtime.base_url,
        provider=runtime.provider,
        auth_mode=runtime.auth_mode,
        timeout=runtime.timeout,
        max_retries=runtime.max_retries,
        retry_delay_s=runtime.retry_delay_s,
        include_thoughts=False,
        return_thinking=False,
        temperature=runtime.temperature,
        gemini_seed=runtime.seed,
        top_p=runtime.top_p,
        top_k=runtime.top_k,
        max_tokens=runtime.max_tokens,
        retry_degenerate_response=False,
    )
    response, token_usage = result
    response = response.strip()
    if not response:
        raise ValueError("Gemini returned an empty audio_global_caption response")
    return response, token_usage


def generate_video_caption(
    question: Question,
    *,
    missing_question_ids: Sequence[str],
    output_dir: Path,
    prompt: str,
    runtime: RuntimeConfig,
    api_key: str,
) -> dict[str, Any]:
    path = artifact_path(output_dir, question.video_id)
    signature = generation_signature(question, prompt, runtime)
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("status") == "complete" and stored.get("config_sha256") == signature:
            return stored

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "config_sha256": signature,
        "video_id": question.video_id,
        "video_path": str(question.video_path),
        "missing_question_ids": list(missing_question_ids),
        "caption_prompt": prompt,
        "caption_prompt_sha256": sha256_text(prompt),
        "model": runtime.model,
        "task_started_epoch_s": time.time(),
    }
    started = time.perf_counter()
    temporary_audio: Path | None = None
    try:
        audio_path = derived_existing_audio_path(question.video_path)
        if audio_path is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as stream:
                temporary_audio = Path(stream.name)
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-i",
                    str(question.video_path), "-vn", str(temporary_audio),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            audio_path = temporary_audio
            payload["audio_input_kind"] = "ffmpeg_extracted_temporary_wav"
        else:
            payload["audio_input_kind"] = "omniagent_derived_existing_wav"
            payload["audio_path"] = str(audio_path)
        response, usage = _call_exact_audio_request(
            audio_path, prompt=prompt, runtime=runtime, api_key=api_key
        )
        payload.update(status="complete", raw_response=response, token_usage=usage)
    except BaseException as exc:
        payload.update(
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
    finally:
        if temporary_audio is not None:
            temporary_audio.unlink(missing_ok=True)
        payload["task_finished_epoch_s"] = time.time()
        payload["total_wall_seconds"] = round(time.perf_counter() - started, 3)
        atomic_json(path, payload)
    return payload


def trace_cache_entry(
    question: Question,
    trace_caption: TraceCaption,
    *,
    trace_jsonl: Path,
    prompt: str,
    default_model: str,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "question_id": question.question_id,
        "response": trace_caption.response,
        "caption_prompt": prompt,
        "caption_prompt_sha256": sha256_text(prompt),
        "source_kind": "omniagent_trace",
        "source_results_file": str(trace_jsonl),
        "source_trace_row": trace_caption.row_number,
        "source_tool_turn_id": trace_caption.turn_id,
        "source_tool_call_index": trace_caption.call_index,
        "source_backend": "gemini_audio_global_caption",
        "source_model": trace_caption.model or default_model,
        "source_token_usage": trace_caption.token_usage,
        "source_video_id": question.video_id,
        "source_video_path": str(question.video_path),
    }


def generated_cache_entry(
    question: Question,
    artifact: Mapping[str, Any],
    *,
    artifact_file: Path,
    prompt: str,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "question_id": question.question_id,
        "response": str(artifact["raw_response"]),
        "caption_prompt": prompt,
        "caption_prompt_sha256": sha256_text(prompt),
        "source_kind": "generated_for_missing_question_video",
        "source_results_file": str(artifact_file),
        "source_backend": "gemini_audio_global_caption",
        "source_model": str(artifact.get("model") or ""),
        "source_token_usage": artifact.get("token_usage"),
        "source_video_id": question.video_id,
        "source_video_path": str(question.video_path),
        "audio_input_kind": artifact.get("audio_input_kind"),
    }


def content_digest(entries: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for question_id in sorted(entries):
        text = json.dumps(
            entries[question_id], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        digest.update(question_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least one")
    trace_jsonl = args.trace_jsonl.expanduser().resolve()
    input_jsonl = args.input_jsonl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prompt_yaml = args.prompt_yaml.expanduser().resolve()
    perception_config_yaml = args.perception_config_yaml.expanduser().resolve()
    for path in (trace_jsonl, input_jsonl, prompt_yaml, perception_config_yaml):
        if not path.is_file():
            raise FileNotFoundError(path)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is required")

    prompt = load_prompt(prompt_yaml, args.prompt_key)
    runtime, source_config = load_runtime_config(perception_config_yaml)
    questions = load_questions(input_jsonl)
    by_id = {question.question_id: question for question in questions}
    trace_captions, extraction_summary = extract_trace_captions(
        trace_jsonl,
        expected_question_ids=set(by_id),
        prompt=prompt,
    )
    missing_questions = [
        question for question in questions if question.question_id not in trace_captions
    ]
    missing_by_video: dict[str, list[Question]] = defaultdict(list)
    for question in missing_questions:
        missing_by_video[question.video_id].append(question)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / "metadata"
    resolved = {
        "schema_version": 1,
        "trace_jsonl": str(trace_jsonl),
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "prompt_yaml": str(prompt_yaml),
        "prompt_key": args.prompt_key,
        "caption_prompt": prompt,
        "caption_prompt_sha256": sha256_text(prompt),
        "perception_config_yaml": str(perception_config_yaml),
        "perception_source_config": source_config,
        "runtime": runtime_dict(runtime),
        "max_workers": args.max_workers,
        "question_count": len(questions),
        "trace_caption_question_count": len(trace_captions),
        "missing_question_count": len(missing_questions),
        "live_video_count": len(missing_by_video),
        "missing_question_policy": (
            "Every question without its own successful trace call is grouped by video; "
            "each such video receives one new exact audio_global_caption call."
        ),
    }
    prior_resolved_path = metadata_dir / "resolved_config.json"
    if prior_resolved_path.is_file():
        prior = json.loads(prior_resolved_path.read_text(encoding="utf-8"))
        stable_keys = (
            "trace_jsonl", "input_jsonl", "prompt_yaml", "prompt_key",
            "caption_prompt_sha256", "perception_config_yaml",
        )
        if any(prior.get(key) != resolved.get(key) for key in stable_keys):
            raise ValueError(f"Output directory belongs to an incompatible run: {output_dir}")
    elif any(output_dir.glob("*.json")):
        raise ValueError(f"Refusing to overwrite unowned JSON cache files in {output_dir}")
    atomic_json(prior_resolved_path, resolved)
    extraction_summary.update(
        {
            "missing_question_count": len(missing_questions),
            "videos_for_missing_questions": len(missing_by_video),
            "missing_question_ids": [question.question_id for question in missing_questions],
        }
    )
    atomic_json(metadata_dir / "trace_extraction_summary.json", extraction_summary)
    print(json.dumps(resolved, ensure_ascii=False, indent=2), flush=True)

    entries: dict[str, dict[str, Any]] = {}
    for question_id, trace_caption in trace_captions.items():
        entry = trace_cache_entry(
            by_id[question_id], trace_caption,
            trace_jsonl=trace_jsonl,
            prompt=prompt,
            default_model=runtime.model,
        )
        entries[question_id] = entry
        atomic_json(cache_entry_path(output_dir, question_id), entry)

    live_results: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for video_id, video_questions in missing_by_video.items():
            representative = video_questions[0]
            future = executor.submit(
                generate_video_caption,
                representative,
                missing_question_ids=[item.question_id for item in video_questions],
                output_dir=output_dir,
                prompt=prompt,
                runtime=runtime,
                api_key=api_key,
            )
            futures[future] = video_id
        for future in as_completed(futures):
            video_id = futures[future]
            artifact = future.result()
            live_results[video_id] = artifact
            print(
                f"[audio_global_caption] video={video_id} status={artifact['status']} "
                f"questions={len(missing_by_video[video_id])} "
                f"wall={artifact.get('total_wall_seconds')}s",
                flush=True,
            )
            if artifact.get("status") == "complete":
                artifact_file = artifact_path(output_dir, video_id)
                for question in missing_by_video[video_id]:
                    entry = generated_cache_entry(
                        question, artifact, artifact_file=artifact_file, prompt=prompt
                    )
                    entries[question.question_id] = entry
                    atomic_json(
                        cache_entry_path(output_dir, question.question_id), entry
                    )

    error_videos = sorted(
        video_id
        for video_id, artifact in live_results.items()
        if artifact.get("status") != "complete"
    )
    summary = {
        "schema_version": 1,
        "question_count": len(questions),
        "trace_caption_question_count": len(trace_captions),
        "generated_caption_question_count": len(entries) - len(trace_captions),
        "cached_question_count": len(entries),
        "missing_question_count": len(questions) - len(entries),
        "requested_live_videos": len(missing_by_video),
        "complete_live_videos": len(live_results) - len(error_videos),
        "error_live_videos": len(error_videos),
        "error_video_ids": error_videos,
        "invocation_wall_seconds": round(time.perf_counter() - started, 3),
    }
    atomic_json(metadata_dir / "summary.json", summary)
    if error_videos:
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 1

    expected_ids = set(by_id)
    if set(entries) != expected_ids:
        raise ValueError(
            f"Internal cache coverage error: missing={sorted(expected_ids - set(entries))[:5]}"
        )
    actual_root_files = {
        path.stem for path in output_dir.glob("*.json") if path.name != MANIFEST_NAME
    }
    expected_root_files = expected_ids
    if actual_root_files != expected_root_files:
        raise ValueError(
            "Unexpected root cache JSON files: "
            f"extra={sorted(actual_root_files - expected_root_files)[:5]}"
        )
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_kind": "omniagent_audio_global_caption",
        "input_jsonl": str(input_jsonl),
        "trace_jsonl": str(trace_jsonl),
        "prompt_yaml": str(prompt_yaml),
        "prompt_key": args.prompt_key,
        "caption_prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "perception_config_yaml": str(perception_config_yaml),
        "source_backend": "gemini_audio_global_caption",
        "source_model": runtime.model,
        "question_count": len(entries),
        "trace_caption_question_count": len(trace_captions),
        "generated_caption_question_count": len(entries) - len(trace_captions),
        "generated_video_count": len(live_results),
        "content_sha256": content_digest(entries),
    }
    atomic_json(output_dir / MANIFEST_NAME, manifest)
    print(json.dumps({"summary": summary, "manifest": manifest}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
