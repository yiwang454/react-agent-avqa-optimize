#!/usr/bin/env python3
"""Gate final inference on a selected GEPA candidate improving over candidate 0."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-validation-examples", type=int, default=125)
    return parser.parse_args()


def load_candidates(path: Path, expected_examples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            candidate_index = row.get("candidate_index")
            score = row.get("full_valset_score")
            if not isinstance(candidate_index, int):
                raise ValueError(f"line {line_number} has no integer candidate_index")
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
                raise ValueError(f"candidate {candidate_index} has no finite full_valset_score")
            correct_count = float(score) * expected_examples
            if not math.isclose(correct_count, round(correct_count), abs_tol=1e-7):
                raise ValueError(
                    f"candidate {candidate_index} score {score} is not a valid "
                    f"accuracy over {expected_examples} examples"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"no candidates found in {path}")
    return rows


def build_report(
    rows: list[dict[str, Any]], candidates_path: Path, expected_examples: int
) -> dict[str, Any]:
    by_index = {row["candidate_index"]: row for row in rows}
    if len(by_index) != len(rows):
        raise ValueError("duplicate candidate_index values")
    if 0 not in by_index:
        raise ValueError("candidate 0 is missing")

    selected = [row for row in rows if row.get("whether_selected_as_best") is True]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one selected candidate, found {len(selected)}")

    baseline = by_index[0]
    selected_row = selected[0]
    best_row = max(rows, key=lambda row: (float(row["full_valset_score"]), -row["candidate_index"]))
    baseline_score = float(baseline["full_valset_score"])
    selected_score = float(selected_row["full_valset_score"])
    improved = selected_score > baseline_score
    return {
        "candidates_jsonl": str(candidates_path),
        "candidate_count": len(rows),
        "validation_examples_per_candidate": expected_examples,
        "baseline_candidate_index": 0,
        "baseline_validation_accuracy": baseline_score,
        "baseline_validation_correct": round(baseline_score * expected_examples),
        "selected_candidate_index": selected_row["candidate_index"],
        "selected_validation_accuracy": selected_score,
        "selected_validation_correct": round(selected_score * expected_examples),
        "highest_scoring_candidate_index": best_row["candidate_index"],
        "highest_validation_accuracy": float(best_row["full_valset_score"]),
        "strict_improvement": improved,
        "decision": "run_inference" if improved else "skip_inference",
    }


def main() -> int:
    args = parse_args()
    try:
        rows = load_candidates(args.candidates_jsonl, args.expected_validation_examples)
        report = build_report(rows, args.candidates_jsonl, args.expected_validation_examples)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Validation gate failed: {exc}")
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(args.output_json)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["strict_improvement"] else 10


if __name__ == "__main__":
    raise SystemExit(main())
