#!/usr/bin/env python3
"""Delete DSPy output JSON files failed by an insufficient-balance tool error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DIRS = [
    Path(
        "/mnt/ceph_rbd/data/avqa_project/daily_omni/"
        "daily_omni_dspy_qwen_v3_qwen_seed_1234_deepseek_seed_7_repeat3"
    ),
    Path(
        "/mnt/ceph_rbd/data/avqa_project/daily_omni/"
        "daily_omni_dspy_qwen_v5_qwen_seed_1234_deepseek_seed_7_repeat1"
    ),
]

DEFAULT_PHRASE = "Insufficient Balance"


def iter_json_files(root: Path) -> list[Path]:
    if not root.exists():
        print(f"[WARN] Missing directory: {root}")
        return []
    if not root.is_dir():
        print(f"[WARN] Not a directory: {root}")
        return []
    return sorted(root.rglob("*.json"))


def has_tool_error_phrase(value: Any, phrase: str) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "tool_error" and isinstance(child, str) and phrase in child:
                return True
            if has_tool_error_phrase(child, phrase):
                return True
    elif isinstance(value, list):
        return any(has_tool_error_phrase(item, phrase) for item in value)
    return False


def file_contains_phrase(path: Path, phrase: str) -> bool:
    try:
        return phrase in path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"[WARN] Cannot read {path}: {exc}")
        return False


def is_bad_file(path: Path, phrase: str, text_fallback: bool) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Cannot parse JSON {path}: {exc}")
        return text_fallback and file_contains_phrase(path, phrase)

    if has_tool_error_phrase(data, phrase):
        return True
    return text_fallback and file_contains_phrase(path, phrase)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find or delete JSON files whose tool_error contains "
            "'Insufficient Balance'. Defaults to dry-run."
        )
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        type=Path,
        default=DEFAULT_DIRS,
        help="Directories to scan recursively. Defaults to the two Qwen repeat output dirs.",
    )
    parser.add_argument(
        "--phrase",
        default=DEFAULT_PHRASE,
        help=f"Error phrase to match inside tool_error. Default: {DEFAULT_PHRASE!r}",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete matched files. Without this flag, only print matches.",
    )
    parser.add_argument(
        "--no-text-fallback",
        action="store_true",
        help="Only trust parsed JSON tool_error fields; do not fallback to raw text search.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text_fallback = not args.no_text_fallback

    matched: list[Path] = []
    scanned = 0
    for root in args.dirs:
        json_files = iter_json_files(root)
        scanned += len(json_files)
        for path in json_files:
            if is_bad_file(path, args.phrase, text_fallback):
                matched.append(path)

    action = "DELETE" if args.delete else "DRY-RUN"
    print(f"[{action}] Scanned {scanned} JSON files; matched {len(matched)} bad files.")

    deleted = 0
    for path in matched:
        if args.delete:
            try:
                path.unlink()
                deleted += 1
                print(f"deleted {path}")
            except OSError as exc:
                print(f"[WARN] Cannot delete {path}: {exc}")
        else:
            print(path)

    if args.delete:
        print(f"[DELETE] Deleted {deleted}/{len(matched)} matched files.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
