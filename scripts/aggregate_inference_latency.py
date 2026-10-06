#!/usr/bin/env python3
"""Validate and aggregate per-question inference latency from an output JSONL."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence


DURATION_KEYS = (
    "wall_clock_seconds",
    "retry_adjusted_seconds",
    "successful_model_call_seconds",
)
CALL_COUNT_KEYS = (
    "planner_successful_calls",
    "perception_successful_calls",
)
LATENCY_KEYS = frozenset((*DURATION_KEYS, *CALL_COUNT_KEYS))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate question_data.latency and aggregate latency statistics "
            "across a completed inference JSONL."
        )
    )
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Summary path (default: latency_summary.json beside the input).",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=0,
        help="Require this many rows; 0 disables the row-count check.",
    )
    return parser.parse_args()


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated percentile, matching NumPy's default."""
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _rounded(value: float | int) -> float | int:
    return value if isinstance(value, int) else round(value, 3)


def _summarize(values: list[float | int], *, total_rows: int) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "null_count": total_rows,
            "sum": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    numeric = [float(value) for value in values]
    total = sum(values)
    return {
        "count": len(values),
        "null_count": total_rows - len(values),
        "sum": _rounded(total),
        "mean": round(statistics.fmean(numeric), 3),
        "median": round(statistics.median(numeric), 3),
        "p90": round(_percentile(numeric, 0.90), 3),
        "p95": round(_percentile(numeric, 0.95), 3),
        "min": _rounded(min(values)),
        "max": _rounded(max(values)),
    }


def _validate_nonnegative_number(value: Any, *, key: str, sample_id: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Invalid {key} for {sample_id}: {value!r}")


def aggregate_latency(input_jsonl: Path, *, expected_count: int = 0) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if expected_count < 0:
        raise ValueError("expected_count must be nonnegative")
    if expected_count and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, found {len(rows)}")

    sample_ids = [str(row.get("video_id") or "") for row in rows]
    if not all(sample_ids):
        raise ValueError("At least one output row has an empty video_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Output contains duplicate video_id values")

    values: dict[str, list[float | int]] = {key: [] for key in LATENCY_KEYS}
    for sample_id, row in zip(sample_ids, rows):
        question_data = row.get("question_data")
        latency = question_data.get("latency") if isinstance(question_data, dict) else None
        if not isinstance(latency, dict) or set(latency) != LATENCY_KEYS:
            raise ValueError(f"Invalid latency schema for {sample_id}: {latency!r}")

        for key in DURATION_KEYS:
            value = latency[key]
            if key == "retry_adjusted_seconds" and value is None:
                continue
            _validate_nonnegative_number(value, key=key, sample_id=sample_id)
            values[key].append(value)

        for key in CALL_COUNT_KEYS:
            value = latency[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid {key} for {sample_id}: {value!r}")
            values[key].append(value)

    return {
        "input_jsonl": str(input_jsonl.resolve()),
        "row_count": len(rows),
        "unique_id_count": len(set(sample_ids)),
        "duration_seconds": {
            key: _summarize(values[key], total_rows=len(rows))
            for key in DURATION_KEYS
        },
        "successful_call_counts": {
            key: _summarize(values[key], total_rows=len(rows))
            for key in CALL_COUNT_KEYS
        },
    }


def main() -> None:
    args = _parse_args()
    output_json = args.output_json or args.input_jsonl.with_name("latency_summary.json")
    try:
        summary = aggregate_latency(
            args.input_jsonl,
            expected_count=args.expected_count,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Latency aggregation failed: {exc}") from exc

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote latency summary: {output_json}")


if __name__ == "__main__":
    main()
