#!/usr/bin/env python3
"""Aggregate the per-question metrics persisted by a modality-merge run."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_jsonl", type=Path)
    parser.add_argument("--expected-count", type=int, default=125)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _number(mapping: dict[str, Any], *keys: str) -> float:
    value: Any = mapping
    for key in keys:
        value = value.get(key, {}) if isinstance(value, dict) else {}
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.results_jsonl.read_text().splitlines() if line.strip()]
    if len(rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} rows, found {len(rows)}")

    correct = 0
    parseable = 0
    invalid = 0
    budget_reached = 0
    tool_counts: Counter[str] = Counter()
    final_status: Counter[str] = Counter()
    live_input = 0.0
    live_output = 0.0
    live_cost = 0.0
    latency_total = 0.0
    latency_count = 0
    for row in rows:
        question = row.get("question_data") or {}
        metrics = question.get("metrics") or {}
        evaluation = metrics.get("evaluation") or {}
        correct += int(evaluation.get("correct") is True)
        parseable += int(evaluation.get("parseable") is True)
        invalid += int(_number(metrics, "tool_calls", "invalid_total"))
        budget_reached += int((metrics.get("budget") or {}).get("reached") is True)
        tool_counts.update((metrics.get("tool_calls") or {}).get("executed_by_tool") or {})
        final_status[str((metrics.get("final_answer") or {}).get("status") or "missing")] += 1
        live_input += _number(metrics, "tokens", "live_total", "input_tokens")
        live_output += _number(metrics, "tokens", "live_total", "output_tokens")
        live_cost += _number(metrics, "estimated_cost_usd", "live_marginal_total")
        latency = question.get("latency") or {}
        seconds = latency.get("retry_adjusted_seconds")
        if isinstance(seconds, (int, float)):
            latency_total += float(seconds)
            latency_count += 1

    count = len(rows)
    summary = {
        "questions": count,
        "correct": correct,
        "accuracy": correct / count if count else 0,
        "parseable": parseable,
        "average_live_input_tokens": live_input / count if count else 0,
        "average_live_output_tokens": live_output / count if count else 0,
        "average_live_cost_usd": live_cost / count if count else 0,
        "average_retry_adjusted_latency_seconds": (
            latency_total / latency_count if latency_count else None
        ),
        "latency_coverage": latency_count,
        "tool_calls": dict(sorted(tool_counts.items())),
        "invalid_calls": invalid,
        "questions_reaching_budget": budget_reached,
        "final_answer_status": dict(sorted(final_status.items())),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
