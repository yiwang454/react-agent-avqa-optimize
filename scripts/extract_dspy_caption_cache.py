#!/usr/bin/env python3
"""Extract a deterministic question-scoped caption cache from DSPy results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
DSPY_DIR = REPO_DIR / "DSPy"
if str(DSPY_DIR) not in sys.path:
    sys.path.insert(0, str(DSPY_DIR))

from dspy_avqa.caption_cache import extract_caption_cache  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(
    "/mnt/ceph_rbd/data/avqa_project/daily_omni/"
    "daily_omni_caption_cache_v8_gemini_from3repeats"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract fixed ask_caption responses from one primary DSPy JSONL, "
            "using fallback JSONLs only when the primary response is blank."
        )
    )
    parser.add_argument("--primary-results", type=Path, required=True)
    parser.add_argument("--fallback-results", type=Path, nargs="*", default=[])
    parser.add_argument("--input-jsonl", type=Path, default="/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3.jsonl")
    parser.add_argument("--prompt-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = extract_caption_cache(
        primary_results=args.primary_results,
        fallback_results=args.fallback_results,
        input_jsonl=args.input_jsonl,
        prompt_yaml=args.prompt_yaml,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Caption cache: {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
