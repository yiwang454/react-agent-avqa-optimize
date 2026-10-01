#!/usr/bin/env python3
"""Aggregate accuracy and tool-use traces for original OmniAgent outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


TAG_ANSWER_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)
PREFIX_ANSWER_RE = re.compile(r"^\s*([A-D])(?:\s|[.:,;]|$)", re.IGNORECASE)
OPTION_LABEL_RE = re.compile(r"(?im)(?<![A-Z0-9])([A-D])\s*[\).:]")


def pct(numerator: float, denominator: float) -> float:
    return 0.0 if not denominator else numerator * 100.0 / denominator


def extract_answer(response: Any) -> str | None:
    text = str(response or "")
    match = TAG_ANSWER_RE.search(text) or PREFIX_ANSWER_RE.match(text)
    return match.group(1).upper() if match else None


def is_abcd_multiple_choice(query: str) -> bool:
    """Requested proxy: query explicitly contains the full A/B/C/D choice set."""
    return set(OPTION_LABEL_RE.findall(query.upper())) == {"A", "B", "C", "D"}


def legalize_range(start: float, end: float, duration: float) -> tuple[float, float]:
    """Mirror the video clipping helper's behavior at a duration boundary."""
    requested_length = end - start
    if start >= duration:
        return max(0.0, duration - requested_length), duration
    if end <= 0.0:
        return 0.0, min(requested_length, duration)
    return max(start, 0.0), min(end, duration)


def load_durations(path: Path) -> dict[str, float]:
    durations: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                durations[str(item["id"])] = float(item["duration"])
    return durations


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(name: str, result_path: Path, durations: dict[str, float]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "name": name,
        "questions": 0,
        "correct": 0,
        "unparseable": 0,
        "tools": Counter(),
        "tool_samples": defaultdict(set),
        "original_calls": Counter(),
        "non_original_calls": Counter(),
        "query_audit": [],
        "clip_rows": [],
        "actions": Counter(),
    }
    with result_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            question_data = item["question_data"]
            question_id = str(question_data["question_id"])
            stats["questions"] += 1
            prediction = extract_answer(question_data.get("response"))
            if prediction is None:
                stats["unparseable"] += 1
            elif prediction == str(question_data.get("answer", "")).upper().strip():
                stats["correct"] += 1

            for turn in question_data.get("turn_trace", []):
                action = str(turn.get("planner_action"))
                stats["actions"][action] += 1
                if action != "tool":
                    continue
                tool_name = str(turn.get("tool_name"))
                stats["tools"][tool_name] += 1
                stats["tool_samples"][tool_name].add(question_id)
                args = turn.get("tool_args") or {}
                query = args.get("question")
                query_field = "question"
                if not isinstance(query, str):
                    query = args.get("query")
                    query_field = "query"
                if isinstance(query, str):
                    original = is_abcd_multiple_choice(query)
                    stats["original_calls" if original else "non_original_calls"][tool_name] += 1
                    stats["query_audit"].append(
                        {
                            "experiment": name,
                            "question_id": question_id,
                            "turn_id": turn.get("turn_id"),
                            "tool_name": tool_name,
                            "query_field": query_field,
                            "is_ask_original_qa": original,
                            "tool_query": query,
                        }
                    )

                time_range = args.get("time_range")
                if isinstance(time_range, list) and len(time_range) == 2:
                    if question_id not in durations:
                        raise KeyError(f"Missing source duration for {question_id}")
                    start, end = map(float, time_range)
                    effective_start, effective_end = legalize_range(start, end, durations[question_id])
                    clip_length = effective_end - effective_start
                    stats["clip_rows"].append(
                        {
                            "experiment": name,
                            "question_id": question_id,
                            "turn_id": turn.get("turn_id"),
                            "tool_name": tool_name,
                            "video_duration_seconds": durations[question_id],
                            "requested_start_seconds": start,
                            "requested_end_seconds": end,
                            "effective_start_seconds": effective_start,
                            "effective_end_seconds": effective_end,
                            "clip_duration_seconds": clip_length,
                            "clip_percent_of_video": pct(clip_length, durations[question_id]),
                            "range_was_reanchored": (start, end) != (effective_start, effective_end),
                        }
                    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment", nargs=2, metavar=("NAME", "RESULTS_JSONL"), action="append", required=True
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    durations = load_durations(args.cuts_jsonl)
    experiments = [analyze(name, Path(path), durations) for name, path in args.experiment]

    accuracy_rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []
    for stats in experiments:
        accuracy_rows.append(
            {
                "experiment": stats["name"],
                "questions": stats["questions"],
                "correct": stats["correct"],
                "accuracy_percent": round(pct(stats["correct"], stats["questions"]), 6),
                "unparseable_responses": stats["unparseable"],
                "tool_calls": sum(stats["tools"].values()),
            }
        )
        total_calls = sum(stats["tools"].values())
        for tool_name, calls in sorted(stats["tools"].items()):
            original = stats["original_calls"][tool_name]
            non_original = stats["non_original_calls"][tool_name]
            query_calls = original + non_original
            tool_rows.append(
                {
                    "experiment": stats["name"],
                    "tool_name": tool_name,
                    "call_count": calls,
                    "call_share_percent": round(pct(calls, total_calls), 6),
                    "samples_called": len(stats["tool_samples"][tool_name]),
                    "benchmark_coverage_percent": round(
                        pct(len(stats["tool_samples"][tool_name]), stats["questions"]), 6
                    ),
                    "ask_original_qa_calls": original if query_calls else "",
                    "not_ask_original_qa_calls": non_original if query_calls else "",
                    "ask_original_qa_share_percent": round(pct(original, query_calls), 6)
                    if query_calls
                    else "",
                }
            )
        query_rows.extend(stats["query_audit"])
        clip_rows.extend(stats["clip_rows"])

    write_csv(args.output_dir / "accuracy_summary.csv", accuracy_rows, list(accuracy_rows[0]))
    write_csv(args.output_dir / "tool_call_summary.csv", tool_rows, list(tool_rows[0]))
    write_csv(
        args.output_dir / "tool_query_classification.csv",
        query_rows,
        ["experiment", "question_id", "turn_id", "tool_name", "query_field", "is_ask_original_qa", "tool_query"],
    )
    if clip_rows:
        write_csv(args.output_dir / "video_clip_calls.csv", clip_rows, list(clip_rows[0]))

    report = [
        "# Original OmniAgent repeats: accuracy and tool calls",
        "",
        "Latency is intentionally excluded: no matching complete-run latency was supplied for repeat 2.",
        "",
        "## Accuracy",
        "",
        "| Run | Correct / 1,197 | Accuracy |",
        "| --- | ---: | ---: |",
    ]
    for stats in experiments:
        report.append(
            f"| {stats['name']} | {stats['correct']} / {stats['questions']} | "
            f"{pct(stats['correct'], stats['questions']):.3f}% |"
        )
    report.extend(
        [
            "",
            "Accuracy is parsed from `<answer>A</answer>` (with a leading A/B/C/D fallback).",
            "",
            "## Tool calling",
            "",
            "Call share is out of the run's recorded tool calls. Coverage is distinct question samples invoking the tool divided by 1,197.",
            "",
            "| Run | Tool | Calls (share) | Samples (coverage) | Original-QA / not-original-QA* |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in tool_rows:
        query_ratio = "—"
        if row["ask_original_qa_calls"] != "":
            query_ratio = (
                f"{row['ask_original_qa_calls']}/{row['not_ask_original_qa_calls']} "
                f"({row['ask_original_qa_share_percent']:.2f}% / "
                f"{100.0 - row['ask_original_qa_share_percent']:.2f}%)"
            )
        report.append(
            f"| {row['experiment']} | {row['tool_name']} | {row['call_count']} ({row['call_share_percent']:.2f}%) | "
            f"{row['samples_called']} ({row['benchmark_coverage_percent']:.2f}%) | {query_ratio} |"
        )
    report.extend(
        [
            "",
            "*Only calls with a `question` or `query` argument are classified. The requested original-QA proxy requires the full explicitly labeled A/B/C/D option set; calls without an input question (caption, ASR, metadata, and event-list tools) are `—`.",
            "",
            "## Non-caption query style, aggregated",
            "",
            "| Run | Calls with `question` / `query` | Original-QA style | Targeted/non-original-QA style |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for stats in experiments:
        original = sum(stats["original_calls"].values())
        non_original = sum(stats["non_original_calls"].values())
        total_queries = original + non_original
        report.append(
            f"| {stats['name']} | {total_queries} | {original} ({pct(original, total_queries):.3f}%) | "
            f"{non_original} ({pct(non_original, total_queries):.3f}%) |"
        )
    if clip_rows:
        report.extend(
            [
                "",
                "## Video clips",
                "",
                "| Run | Calls | Samples | Mean effective duration | Mean effective video share | Re-anchored ranges |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for stats in experiments:
            rows = stats["clip_rows"]
            if not rows:
                continue
            lengths = [float(row["clip_duration_seconds"]) for row in rows]
            shares = [float(row["clip_percent_of_video"]) for row in rows]
            report.append(
                f"| {stats['name']} | {len(rows)} | {len({row['question_id'] for row in rows})} | "
                f"{mean(lengths):.3f}s (median {median(lengths):.3f}s) | {mean(shares):.3f}% | "
                f"{sum(bool(row['range_was_reanchored']) for row in rows)} |"
            )
    report.extend(
        [
            "",
            "## Files",
            "",
            "- `accuracy_summary.csv`: answer totals and accuracy.",
            "- `tool_call_summary.csv`: per-tool calls, share, coverage, and query classification counts.",
            "- `tool_query_classification.csv`: every classified tool question/query.",
            "- `video_clip_calls.csv`: every video clip range and effective duration.",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
