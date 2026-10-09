#!/usr/bin/env python3
"""Seed reusable whole-video caption artifacts from prior ReAct trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).resolve().parents[1]
DSPY_DIR = REPO_DIR / "DSPy"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(DSPY_DIR) not in sys.path:
    sys.path.insert(0, str(DSPY_DIR))

from dspy_avqa.chunked_caption import load_video_jobs_from_jsonl  # noqa: E402
from dspy_avqa.runner import load_gemini_captioner_config_yaml  # noqa: E402
from scripts.build_whole_video_caption_cache import (  # noqa: E402
    artifact_path,
    atomic_json,
    load_prompt,
    signature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-yaml", type=Path, required=True)
    parser.add_argument("--prompt-key", default="QA_PROMPT_OMNI_CAPTIONER_TIMESTAMP3")
    parser.add_argument("--captioner-config-yaml", type=Path, required=True)
    parser.add_argument("--gemini-api-backend", choices=("legacy", "dspy"), default="legacy")
    return parser.parse_args()


def valid_caption_turn(payload: Any, expected_prompt: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    for turn in payload.get("turn_trace") or []:
        if not isinstance(turn, dict) or turn.get("tool_name") != "ask_caption":
            continue
        response = str(turn.get("tool_observation") or "").strip()
        prompt = str(turn.get("perception_prompt") or "").strip()
        if response and prompt == expected_prompt:
            return turn
    return None


def main() -> int:
    args = parse_args()
    input_jsonl = args.input_jsonl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prompt_yaml = args.prompt_yaml.expanduser().resolve()
    prompt = load_prompt(prompt_yaml, args.prompt_key)

    os.environ["CAPTIONER_MODEL"] = "gemini"
    os.environ["GEMINI_API_BACKEND"] = args.gemini_api_backend
    load_gemini_captioner_config_yaml(args.captioner_config_yaml.expanduser().resolve())

    jobs = load_video_jobs_from_jsonl(input_jsonl)
    jobs_by_video = {job.video_id: job for job in jobs}
    selected: dict[str, tuple[Path, dict[str, Any]]] = {}
    for source_dir in args.source_dir:
        resolved_source = source_dir.expanduser().resolve()
        if not resolved_source.is_dir():
            raise FileNotFoundError(f"Source directory not found: {resolved_source}")
        for path in sorted(resolved_source.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            turn = valid_caption_turn(payload, prompt)
            if turn is None:
                continue
            video_path = str((turn.get("tool_args") or {}).get("video_path") or "")
            video_id = Path(video_path).stem
            if video_id in jobs_by_video:
                selected.setdefault(video_id, (path, turn))

    seeded = 0
    for video_id, (source_path, turn) in selected.items():
        job = jobs_by_video[video_id]
        destination = artifact_path(output_dir, video_id)
        config_sha256 = signature(job, prompt)
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "complete"
                and existing.get("config_sha256") == config_sha256
            ):
                continue
        atomic_json(
            destination,
            {
                "schema_version": 1,
                "status": "complete",
                "config_sha256": config_sha256,
                "video_id": video_id,
                "video_path": str(job.video_path),
                "question_ids": list(job.question_ids),
                "caption_prompt": prompt,
                "caption_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "media_input_kind": "original_video",
                "timestamp_normalization_applied": False,
                "raw_response": str(turn["tool_observation"]),
                "model": turn.get("perception_model"),
                "backend": turn.get("perception_backend"),
                "token_usage": turn.get("perception_token_usage"),
                "latency": None,
                "seeded_from_react_result": str(source_path),
            },
        )
        seeded += 1

    print(
        json.dumps(
            {
                "video_count": len(jobs),
                "reusable_source_videos": len(selected),
                "newly_seeded_artifacts": seeded,
                "videos_left_for_live_captioning": len(jobs) - len(selected),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
