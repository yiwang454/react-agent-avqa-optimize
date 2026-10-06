#!/usr/bin/env python3
"""Generate one raw whole-video caption per recording and materialize a question cache."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Mapping

import yaml


REPO_DIR = Path(__file__).resolve().parents[1]
DSPY_DIR = REPO_DIR / "DSPy"
if str(DSPY_DIR) not in sys.path:
    sys.path.insert(0, str(DSPY_DIR))

from dspy_avqa.caption_cache import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    MANIFEST_NAME,
    load_caption_cache_manifest,
    validate_caption_cache_coverage,
)
from dspy_avqa.chunked_caption import load_video_jobs_from_jsonl, VideoJob  # noqa: E402
from dspy_avqa.experiment_config import redact_sensitive  # noqa: E402
from dspy_avqa.latency import model_component, sample_latency  # noqa: E402
from dspy_avqa.runner import load_gemini_captioner_config_yaml  # noqa: E402
from dspy_avqa.tools import (  # noqa: E402
    call_gemini_perception,
    consume_last_perception_metadata,
    probe_media_duration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-yaml", type=Path, required=True)
    parser.add_argument("--prompt-key", default="QA_PROMPT_OMNI_CAPTIONER_TIMESTAMP3")
    parser.add_argument("--captioner-config-yaml", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--gemini-api-backend", choices=("legacy", "dspy"), default="legacy")
    parser.add_argument("--skip-question-cache", action="store_true")
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return normalized if normalized == value and normalized else f"{normalized or 'video'}-{sha256_text(value)[:10]}"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def load_prompt(path: Path, key: str) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    prompt = str(payload.get(key) or "").strip() if isinstance(payload, Mapping) else ""
    if not prompt:
        raise ValueError(f"Prompt key {key!r} is missing or empty in {path}")
    return prompt


def runtime_fingerprint() -> dict[str, Any]:
    keys = (
        "GEMINI_API_BACKEND", "CAPTIONER_GEMINI_MODEL", "CAPTIONER_GEMINI_BASE_URL",
        "CAPTIONER_GEMINI_PROVIDER", "CAPTIONER_GEMINI_AUTH_MODE",
        "CAPTIONER_GEMINI_VIDEO_ONLY", "CAPTIONER_GEMINI_TIMEOUT",
        "CAPTIONER_GEMINI_MAX_RETRIES", "CAPTIONER_GEMINI_RETRY_DELAY_S",
        "CAPTIONER_GEMINI_TEMPERATURE", "CAPTIONER_GEMINI_TOP_P",
        "CAPTIONER_GEMINI_TOP_K", "CAPTIONER_GEMINI_MAX_TOKENS",
        "CAPTIONER_GEMINI_SEED",
    )
    return {key: os.environ.get(key) for key in keys}


def artifact_path(output_dir: Path, video_id: str) -> Path:
    return output_dir / "video_artifacts" / safe_component(video_id) / "video.json"


def signature(job: VideoJob, prompt: str) -> str:
    stat = job.video_path.stat()
    payload = {
        "video_id": job.video_id,
        "video_path": str(job.video_path),
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "prompt_sha256": sha256_text(prompt),
        "runtime": runtime_fingerprint(),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def run_video(job: VideoJob, *, output_dir: Path, prompt: str) -> dict[str, Any]:
    path = artifact_path(output_dir, job.video_id)
    config_sha = signature(job, prompt)
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("status") == "complete" and stored.get("config_sha256") == config_sha:
            return stored

    started_epoch = time.time()
    started = time.perf_counter()
    response = ""
    latency = None
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "config_sha256": config_sha,
        "video_id": job.video_id,
        "video_path": str(job.video_path),
        "question_ids": list(job.question_ids),
        "caption_prompt": prompt,
        "caption_prompt_sha256": sha256_text(prompt),
        "media_input_kind": "original_video",
        "timestamp_normalization_applied": False,
        "task_started_epoch_s": started_epoch,
    }
    try:
        payload["probed_duration_s"] = round(probe_media_duration(str(job.video_path)), 6)
        with sample_latency() as tracker:
            accepted = False
            try:
                with model_component("perception"):
                    response = call_gemini_perception(
                        video_path=str(job.video_path),
                        audio_path=None,
                        prompt=prompt,
                        system_prompt="",
                        env_prefix="CAPTIONER_GEMINI",
                    )
                accepted = True
            finally:
                latency = tracker.summary(complete=accepted)
        if not response.strip():
            raise ValueError("Caption response is empty")
        metadata = consume_last_perception_metadata()
        payload.update(
            status="complete",
            raw_response=response,
            model=metadata.get("model"),
            backend=metadata.get("backend"),
            token_usage=metadata.get("token_usage"),
            latency=latency,
        )
    except BaseException as exc:
        payload.update(
            status="error",
            raw_response=response,
            error_type=type(exc).__name__,
            error=str(exc),
            latency=latency,
        )
    payload["task_finished_epoch_s"] = time.time()
    payload["total_wall_seconds"] = round(time.perf_counter() - started, 3)
    atomic_json(path, payload)
    return payload


def materialize_cache(
    *, jobs: list[VideoJob], results: list[Mapping[str, Any]], cache_dir: Path,
    input_jsonl: Path, prompt_yaml: Path, prompt_key: str, prompt: str,
) -> dict[str, Any]:
    question_ids = [question_id for job in jobs for question_id in job.question_ids]
    if cache_dir.exists() and any(cache_dir.iterdir()):
        coverage = validate_caption_cache_coverage(cache_dir, question_ids, expected_prompt=prompt)
        return load_caption_cache_manifest(cache_dir) | {"validated_existing_cache": coverage}

    by_video = {str(result["video_id"]): result for result in results}
    entries: dict[str, dict[str, Any]] = {}
    for job in jobs:
        result = by_video[job.video_id]
        for question_id in job.question_ids:
            entries[question_id] = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "question_id": question_id,
                "response": str(result["raw_response"]),
                "caption_prompt": prompt,
                "source_results_file": str(artifact_path(cache_dir.parent, job.video_id)),
                "source_rank": 1,
                "source_backend": "gemini_whole_video",
                "source_model": os.environ.get("CAPTIONER_GEMINI_MODEL", ""),
                "source_token_usage": result.get("token_usage"),
                "source_video_id": job.video_id,
                "chunk_count": 1,
                "timestamp_mode": "model_raw_whole_video",
                "video_latency": result.get("latency"),
            }
    digest = hashlib.sha256()
    for question_id in sorted(entries):
        text = json.dumps(entries[question_id], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest.update(question_id.encode()); digest.update(b"\0"); digest.update(text.encode()); digest.update(b"\n")
        atomic_json(cache_dir / f"{question_id}.json", entries[question_id])
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "input_jsonl": str(input_jsonl),
        "prompt_yaml": str(prompt_yaml),
        "prompt_key": prompt_key,
        "caption_prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "generation_strategy": "whole_video_raw_timestamp3",
        "source_backend": "gemini_whole_video",
        "source_model": os.environ.get("CAPTIONER_GEMINI_MODEL", ""),
        "video_count": len(jobs),
        "question_count": len(entries),
        "content_sha256": digest.hexdigest(),
    }
    atomic_json(cache_dir / MANIFEST_NAME, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least one")
    input_jsonl = args.input_jsonl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prompt_yaml = args.prompt_yaml.expanduser().resolve()
    captioner_config = args.captioner_config_yaml.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CAPTIONER_MODEL"] = "gemini"
    os.environ["GEMINI_API_BACKEND"] = args.gemini_api_backend
    os.environ.setdefault("GEMINI_RESPONSE_ERROR_SENSITIVE", "false")
    source_config = load_gemini_captioner_config_yaml(captioner_config)
    if not (os.environ.get("CAPTIONER_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise EnvironmentError("GEMINI_API_KEY or CAPTIONER_GEMINI_API_KEY is required")
    prompt = load_prompt(prompt_yaml, args.prompt_key)
    jobs = load_video_jobs_from_jsonl(input_jsonl)
    resolved = {
        "input_jsonl": str(input_jsonl), "output_dir": str(output_dir),
        "prompt_yaml": str(prompt_yaml), "prompt_key": args.prompt_key,
        "captioner_config_yaml": str(captioner_config), "max_workers": args.max_workers,
        "video_count": len(jobs), "question_count": sum(len(j.question_ids) for j in jobs),
        "caption_prompt": prompt, "captioner_source_config": redact_sensitive(source_config),
        "runtime": redact_sensitive(runtime_fingerprint()),
    }
    atomic_json(output_dir / "resolved_whole_video_caption_config.json", resolved)
    print(json.dumps(resolved, ensure_ascii=False, indent=2), flush=True)

    invocation_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_id = {
            executor.submit(run_video, job, output_dir=output_dir, prompt=prompt): job.video_id
            for job in jobs
        }
        for future in as_completed(future_to_id):
            result = future.result()
            results.append(result)
            print(
                f"[whole_video] {result['video_id']} status={result['status']} "
                f"total={result.get('total_wall_seconds')}s",
                flush=True,
            )
    results.sort(key=lambda item: str(item["video_id"]))
    summary = {
        "schema_version": 1,
        "video_count": len(results),
        "question_count": sum(len(job.question_ids) for job in jobs),
        "complete_videos": sum(item["status"] == "complete" for item in results),
        "error_videos": sum(item["status"] != "complete" for item in results),
        "invocation_wall_seconds": round(time.perf_counter() - invocation_started, 3),
        "sum_retry_adjusted_seconds": round(sum(
            float((item.get("latency") or {}).get("retry_adjusted_seconds") or 0) for item in results
        ), 3),
        "results": [{"video_id": item["video_id"], "status": item["status"],
                     "latency": item.get("latency"), "error": item.get("error")} for item in results],
    }
    atomic_json(output_dir / "summary.json", summary)
    cache_manifest = None
    if summary["error_videos"] == 0 and not args.skip_question_cache:
        cache_manifest = materialize_cache(
            jobs=jobs, results=results, cache_dir=output_dir / "caption_cache",
            input_jsonl=input_jsonl, prompt_yaml=prompt_yaml,
            prompt_key=args.prompt_key, prompt=prompt,
        )
    print(json.dumps({"summary": summary, "cache_manifest": cache_manifest}, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["error_videos"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
