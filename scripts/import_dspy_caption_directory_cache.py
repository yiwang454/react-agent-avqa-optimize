#!/usr/bin/env python3
"""Import per-question Qwen captions into the validated DSPy ReAct cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
DSPY_DIR = REPO_DIR / "DSPy"
if str(DSPY_DIR) not in sys.path:
    sys.path.insert(0, str(DSPY_DIR))

from dspy_avqa.caption_cache import import_caption_directory_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse cache-import arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Import per-question caption JSON files into the deterministic "
            "question-scoped cache used by ask_caption."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-prompt-yaml", type=Path, required=True)
    parser.add_argument("--source-template-name", required=True)
    parser.add_argument("--source-backend", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--prompt-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Import the requested caption directory and print its manifest."""
    args = parse_args()
    manifest = import_caption_directory_cache(
        source_dir=args.source_dir,
        source_prompt_yaml=args.source_prompt_yaml,
        source_template_name=args.source_template_name,
        input_jsonl=args.input_jsonl,
        prompt_yaml=args.prompt_yaml,
        output_dir=args.output_dir,
        source_backend=args.source_backend,
        source_model=args.source_model,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Caption cache: {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
