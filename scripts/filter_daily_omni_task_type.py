#!/usr/bin/env python3
"""Filter a Daily-Omni JSONL split to one native task type."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def task_type(cut: dict, source: Path, line_number: int) -> str:
    supervisions = cut.get("supervisions")
    if not isinstance(supervisions, list) or not supervisions or not isinstance(supervisions[0], dict):
        raise ValueError(f"{source}:{line_number}: expected a first supervision object")
    custom = supervisions[0].get("custom")
    value = custom.get("task_type", custom.get("Type")) if isinstance(custom, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}:{line_number}: missing custom.task_type/custom.Type")
    return value.strip()


def filter_jsonl_by_task_type(input_jsonl: Path, output_jsonl: Path, requested_type: str) -> tuple[int, int]:
    if input_jsonl.resolve() == output_jsonl.resolve():
        raise ValueError("input_jsonl and output_jsonl must be different files")
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_jsonl}")
    requested_type = requested_type.strip()
    if not requested_type:
        raise ValueError("task_type must not be empty")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_jsonl.with_name(output_jsonl.name + ".tmp")
    total_rows = selected_rows = 0
    try:
        with input_jsonl.open(encoding="utf-8") as source, temporary_output.open("w", encoding="utf-8") as destination:
            for line_number, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    continue
                total_rows += 1
                cut = json.loads(raw_line)
                if not isinstance(cut, dict):
                    raise ValueError(f"{input_jsonl}:{line_number}: expected a JSON object")
                if task_type(cut, input_jsonl, line_number) != requested_type:
                    continue
                destination.write(json.dumps(cut, ensure_ascii=False) + "\n")
                selected_rows += 1
        if not selected_rows:
            raise ValueError(f"No rows with task type {requested_type!r} found in {input_jsonl}")
        os.replace(temporary_output, output_jsonl)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    return total_rows, selected_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--task-type", default="Event Sequence")
    args = parser.parse_args()
    total_rows, selected_rows = filter_jsonl_by_task_type(args.input_jsonl, args.output_jsonl, args.task_type)
    print(f"Wrote {selected_rows}/{total_rows} rows with task_type={args.task_type!r} to {args.output_jsonl}")


if __name__ == "__main__":
    main()
