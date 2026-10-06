#!/usr/bin/env python3
"""Audit and optionally invalidate cached samples that called one clip tool."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _uses_tool(question_data: Any, tool_name: str) -> bool:
    if not isinstance(question_data, dict):
        return False
    return any(
        isinstance(turn, dict) and turn.get("tool_name") == tool_name
        for turn in question_data.get("turn_trace") or []
    )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sample_id_from_row(row: dict[str, Any]) -> str:
    question_data = row.get("question_data")
    if isinstance(question_data, dict) and question_data.get("question_id") is not None:
        return str(question_data["question_id"])
    return str(row.get("video_id") or "")


def audit(output_dir: Path, tool_name: str) -> tuple[list[Path], set[str]]:
    cache_paths: list[Path] = []
    sample_ids: set[str] = set()
    for path in sorted(output_dir.glob("*.json")):
        try:
            question_data = _load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARN unreadable cache file {path}: {exc}")
            continue
        if _uses_tool(question_data, tool_name):
            cache_paths.append(path)
            sample_ids.add(str(question_data.get("question_id") or path.stem))
    return cache_paths, sample_ids


def _filter_output_jsonl(
    source: Path,
    destination: Path,
    *,
    tool_name: str,
    sample_ids: set[str],
) -> tuple[int, int]:
    kept: list[str] = []
    removed = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
            question_data = row.get("question_data") if isinstance(row, dict) else None
            if _sample_id_from_row(row) in sample_ids or _uses_tool(question_data, tool_name):
                removed += 1
            else:
                kept.append(json.dumps(row, ensure_ascii=False) + "\n")
    destination.write_text("".join(kept), encoding="utf-8")
    return removed, len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "List cached samples whose turn_trace called a clip tool. With --apply, "
            "move their per-sample JSON files into a quarantine directory and remove "
            "their rows from output_test.jsonl so the normal run script regenerates them."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tool",
        required=True,
        choices=("omni_clip_caption", "omni_clip_perception"),
    )
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        parser.error(f"output directory does not exist: {output_dir}")
    output_jsonl = (args.output_jsonl or output_dir / "output_test.jsonl").resolve()
    cache_paths, sample_ids = audit(output_dir, args.tool)

    print(f"Output directory: {output_dir}")
    print(f"Tool: {args.tool}")
    print(f"Matched per-sample caches: {len(cache_paths)}")
    for sample_id in sorted(sample_ids):
        print(sample_id)

    if not args.apply:
        print("Audit only; no files changed. Re-run with --apply after reviewing the IDs.")
        return
    if not cache_paths:
        print("Nothing to invalidate.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = output_dir / f"invalidated_{args.tool}_{stamp}"
    quarantine.mkdir()
    for path in cache_paths:
        shutil.move(str(path), quarantine / path.name)

    if output_jsonl.exists():
        backup = quarantine / output_jsonl.name
        shutil.copy2(output_jsonl, backup)
        removed, kept = _filter_output_jsonl(
            backup,
            output_jsonl,
            tool_name=args.tool,
            sample_ids=sample_ids,
        )
        print(f"Rewrote {output_jsonl}: removed {removed} rows, kept {kept} rows.")
    else:
        print(f"WARN aggregate output not found: {output_jsonl}")
    print(f"Moved {len(cache_paths)} cache files to recoverable quarantine: {quarantine}")


if __name__ == "__main__":
    main()
