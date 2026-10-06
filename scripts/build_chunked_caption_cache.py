#!/usr/bin/env python3
"""Generate long-video chunk captions and an optional ReAct question cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
DSPY_DIR = REPO_DIR / "DSPy"
if str(DSPY_DIR) not in sys.path:
    sys.path.insert(0, str(DSPY_DIR))

from dspy_avqa.chunked_caption import (  # noqa: E402
    TIMESTAMP_MODES,
    load_direct_video_jobs,
    load_chunk_prompt_config,
    load_video_jobs_from_jsonl,
    materialize_question_cache,
    run_chunked_caption_jobs,
)
from dspy_avqa.experiment_config import redact_sensitive  # noqa: E402
from dspy_avqa.runner import load_gemini_captioner_config_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Caption each unique recording as concurrent one-minute chunks, normalize "
            "timestamps to the original video, and materialize the existing ReAct cache schema."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=Path)
    source.add_argument("--video-path", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-yaml", type=Path, required=True)
    parser.add_argument("--captioner-config-yaml", type=Path, required=True)
    parser.add_argument(
        "--chunk-prompt-profile",
        default="CHUNKED_CAPTION_PROMPT",
        help="Top-level prompt profile key in --prompt-yaml.",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        help="Optional recording/video ID filter for a dataset-backed smoke run.",
    )
    parser.add_argument(
        "--timestamp-mode",
        choices=sorted(TIMESTAMP_MODES),
        default="local_offset",
    )
    parser.add_argument("--chunk-seconds", type=float, default=60.0)
    parser.add_argument("--min-tail-seconds", type=float, default=30.0)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--gemini-api-backend", choices=("legacy", "dspy"), default="legacy")
    parser.add_argument("--force", action="store_true", help="Ignore matching completed chunk artifacts.")
    parser.add_argument(
        "--skip-question-cache",
        action="store_true",
        help="Keep video artifacts only. Required implicitly for direct --video-path smoke tests.",
    )
    return parser.parse_args()


def _resolved_runtime() -> dict[str, object]:
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
        "CAPTIONER_GEMINI_API_KEY",
        "GEMINI_API_KEY",
    )
    return redact_sensitive({key: os.environ.get(key) for key in keys})


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    prompt_yaml = args.prompt_yaml.expanduser().resolve()
    captioner_config = args.captioner_config_yaml.expanduser().resolve()
    os.environ["CAPTIONER_MODEL"] = "gemini"
    os.environ["GEMINI_API_BACKEND"] = args.gemini_api_backend
    os.environ.setdefault("GEMINI_RESPONSE_ERROR_SENSITIVE", "false")
    source_config = load_gemini_captioner_config_yaml(captioner_config)
    prompt_config = load_chunk_prompt_config(
        prompt_yaml,
        profile_key=args.chunk_prompt_profile,
    )
    base_prompt = str(prompt_config["base_instruction"])
    if not (os.environ.get("CAPTIONER_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise EnvironmentError(
            "GEMINI_API_KEY or CAPTIONER_GEMINI_API_KEY is required; source the repository .env first."
        )

    if args.input_jsonl is not None:
        input_jsonl = args.input_jsonl.expanduser().resolve()
        jobs = load_video_jobs_from_jsonl(input_jsonl)
        if args.video_id:
            requested_ids = set(args.video_id)
            available_ids = {job.video_id for job in jobs}
            missing_ids = sorted(requested_ids - available_ids)
            if missing_ids:
                raise ValueError(f"Requested video IDs are absent from the input: {missing_ids}")
            jobs = [job for job in jobs if job.video_id in requested_ids]
    else:
        if args.video_id:
            raise ValueError("--video-id can only be used with --input-jsonl")
        input_jsonl = None
        jobs = load_direct_video_jobs(args.video_path or [])

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "input_jsonl": str(input_jsonl) if input_jsonl else None,
        "video_paths": [str(job.video_path) for job in jobs] if input_jsonl is None else None,
        "output_dir": str(output_dir),
        "prompt_yaml": str(prompt_yaml),
        "chunk_prompt_profile": args.chunk_prompt_profile,
        "captioner_config_yaml": str(captioner_config),
        "timestamp_mode": args.timestamp_mode,
        "chunk_seconds": args.chunk_seconds,
        "min_tail_seconds": args.min_tail_seconds,
        "max_workers": args.max_workers,
        "gemini_api_backend": args.gemini_api_backend,
        "video_count": len(jobs),
        "selected_video_ids": [job.video_id for job in jobs],
        "question_count": sum(len(job.question_ids) for job in jobs),
        "base_prompt": base_prompt,
        "chunk_prompt_config": prompt_config,
        "captioner_source_config": redact_sensitive(source_config),
        "runtime": _resolved_runtime(),
    }
    (output_dir / "resolved_chunked_caption_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(resolved, ensure_ascii=False, indent=2), flush=True)

    video_results, summary = run_chunked_caption_jobs(
        jobs,
        output_dir=output_dir,
        prompt_config=prompt_config,
        mode=args.timestamp_mode,
        chunk_seconds=args.chunk_seconds,
        min_tail_seconds=args.min_tail_seconds,
        max_workers=args.max_workers,
        force=args.force,
    )
    cache_manifest = None
    if input_jsonl is not None and not args.skip_question_cache:
        if args.timestamp_mode != "local_offset":
            raise ValueError("Production question-cache materialization requires local_offset mode")
        cache_manifest = materialize_question_cache(
            jobs=jobs,
            video_results=video_results,
            cache_dir=output_dir / "caption_cache",
            input_jsonl=input_jsonl,
            prompt_yaml=prompt_yaml,
            base_prompt=base_prompt,
        )
    final = {"summary": summary, "cache_manifest": cache_manifest}
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["error_videos"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
