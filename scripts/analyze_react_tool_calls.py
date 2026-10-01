#!/usr/bin/env python3
"""Aggregate tool-use traces from a DSPy AVQA ReAct rollout.

The script is intentionally dependency-free: the experiment result already has
the per-turn trace, while the source cuts JSONL provides the duration of each
benchmark video.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


# An A-D multiple-choice tool question is the user's requested operational
# definition of "ask original QA".  This accepts A., A), A:, etc., and only
# labels it MCQ when every one of A/B/C/D is present.
OPTION_LABEL_RE = re.compile(r"(?im)(?<![A-Z0-9])([A-D])\s*[\).:]")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percent(numerator: float, denominator: float) -> float:
    return 0.0 if not denominator else 100.0 * numerator / denominator


def is_abcd_multiple_choice(question: str) -> bool:
    """Whether a tool question explicitly presents the full A/B/C/D choice set."""
    return set(OPTION_LABEL_RE.findall(question.upper())) == {"A", "B", "C", "D"}


def legalize_range(start: float, end: float, video_duration: float) -> tuple[float, float]:
    """Mirror tools.legalize_time_range for a trace that crossed video bounds."""
    requested_duration = end - start
    if start >= video_duration:
        return max(0.0, video_duration - requested_duration), video_duration
    if end <= 0.0:
        return 0.0, min(requested_duration, video_duration)
    return max(start, 0.0), min(end, video_duration)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument(
        "--cuts-jsonl",
        type=Path,
        required=True,
        help="Input cuts JSONL; its `id` and `duration` fields define video duration.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    results = read_jsonl(args.results_jsonl)
    source_cuts = read_jsonl(args.cuts_jsonl)
    durations = {str(cut["id"]): float(cut["duration"]) for cut in source_cuts}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = len(results)
    tool_calls: Counter[str] = Counter()
    tool_samples: dict[str, set[str]] = defaultdict(set)
    per_sample_calls: dict[str, Counter[str]] = defaultdict(Counter)
    question_rows: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []
    total_tool_turns = 0
    cache_hits = 0
    call_order = 0

    for result in results:
        question_data = result["question_data"]
        question_id = str(question_data["question_id"])
        if question_id not in durations:
            raise KeyError(f"No source-cut duration for {question_id}")
        original_question = str(question_data.get("question", ""))

        for turn in question_data.get("turn_trace", []):
            if turn.get("planner_action") != "tool":
                continue
            call_order += 1
            total_tool_turns += 1
            tool_name = str(turn.get("tool_name"))
            tool_calls[tool_name] += 1
            tool_samples[tool_name].add(question_id)
            per_sample_calls[tool_name][question_id] += 1
            if turn.get("caption_cache_hit") is True:
                cache_hits += 1

            tool_args = turn.get("tool_args") or {}
            if tool_name == "ask_perception":
                perceptual_question = str(tool_args.get("perceptual_question", ""))
                question_rows.append(
                    {
                        "question_id": question_id,
                        "turn_id": turn.get("turn_id"),
                        "is_ask_original_qa": is_abcd_multiple_choice(perceptual_question),
                        "original_question": original_question,
                        "perceptual_question": perceptual_question,
                    }
                )
            elif tool_name == "omni_clip_caption":
                requested_range = tool_args.get("time_range")
                if not isinstance(requested_range, list) or len(requested_range) != 2:
                    raise ValueError(f"Invalid clip range for {question_id}: {requested_range!r}")
                start, end = map(float, requested_range)
                video_duration = durations[question_id]
                effective_start, effective_end = legalize_range(start, end, video_duration)
                clip_duration = effective_end - effective_start
                clip_rows.append(
                    {
                        "question_id": question_id,
                        "turn_id": turn.get("turn_id"),
                        "video_duration_seconds": video_duration,
                        "requested_start_seconds": start,
                        "requested_end_seconds": end,
                        "effective_start_seconds": effective_start,
                        "effective_end_seconds": effective_end,
                        "clip_duration_seconds": clip_duration,
                        "clip_percent_of_video": percent(clip_duration, video_duration),
                        "range_was_reanchored": (start, end) != (effective_start, effective_end),
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    tool_order = ["ask_caption", "ask_perception", "omni_clip_caption"]
    for tool_name in tool_order:
        calls = tool_calls[tool_name]
        samples = len(tool_samples[tool_name])
        summary_rows.append(
            {
                "tool_name": tool_name,
                "call_count": calls,
                "call_share_percent": round(percent(calls, total_tool_turns), 4),
                "samples_called": samples,
                "sample_coverage_percent": round(percent(samples, total_samples), 4),
                "mean_calls_per_all_samples": round(calls / total_samples, 6),
                "mean_calls_per_called_sample": round(calls / samples, 6) if samples else 0.0,
            }
        )

    write_csv(
        args.output_dir / "tool_call_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    write_csv(
        args.output_dir / "ask_perception_question_type.csv",
        question_rows,
        [
            "question_id",
            "turn_id",
            "is_ask_original_qa",
            "original_question",
            "perceptual_question",
        ],
    )
    write_csv(
        args.output_dir / "omni_clip_caption_calls.csv",
        clip_rows,
        [
            "question_id",
            "turn_id",
            "video_duration_seconds",
            "requested_start_seconds",
            "requested_end_seconds",
            "effective_start_seconds",
            "effective_end_seconds",
            "clip_duration_seconds",
            "clip_percent_of_video",
            "range_was_reanchored",
        ],
    )

    original_qa_calls = sum(row["is_ask_original_qa"] for row in question_rows)
    original_qa_samples = {
        row["question_id"] for row in question_rows if row["is_ask_original_qa"]
    }
    perception_samples = tool_samples["ask_perception"]
    clip_lengths = [float(row["clip_duration_seconds"]) for row in clip_rows]
    clip_percentages = [float(row["clip_percent_of_video"]) for row in clip_rows]
    reanchored = sum(bool(row["range_was_reanchored"]) for row in clip_rows)

    report = [
        "# ReAct tool-call aggregation",
        "",
        f"- Results: `{args.results_jsonl}`",
        f"- Source cuts: `{args.cuts_jsonl}`",
        f"- Benchmark samples: **{total_samples}**",
        f"- Total recorded tool calls: **{total_tool_turns}**",
        f"- Caption-cache hits: **{cache_hits}**",
        "",
        "## Tool usage",
        "",
        "| Tool | Calls | Share of all calls | Samples called | Benchmark coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    report.extend(
        "| {tool_name} | {call_count} | {call_share_percent:.2f}% | {samples_called} | {sample_coverage_percent:.2f}% |".format(**row)
        for row in summary_rows
    )
    report.extend(
        [
            "",
            "## `omni_clip_caption`",
            "",
            f"- Calls: **{len(clip_rows)}**, across **{len(tool_samples['omni_clip_caption'])}** samples.",
            f"- Mean effective clip duration: **{mean(clip_lengths):.3f} s** (median {median(clip_lengths):.3f} s; range {min(clip_lengths):.3f}–{max(clip_lengths):.3f} s).",
            f"- Mean effective clip/video ratio: **{mean(clip_percentages):.3f}%** (median {median(clip_percentages):.3f}%; range {min(clip_percentages):.3f}–{max(clip_percentages):.3f}%).",
            f"- {reanchored} requested ranges crossed a video boundary; their effective ranges were re-anchored using the same logic as the tool implementation.",
            "",
            "## `ask_perception`: original-QA style",
            "",
            "Definition: an `ask_perception` question is counted as **ask original QA** when it explicitly contains all four labeled answer choices A/B/C/D (accepting `A.`, `A)`, or `A:` forms). This implements the requested A/B/C/D-answer proxy; it does not require exact wording identity with the benchmark question.",
            "",
            f"- Ask original QA: **{original_qa_calls}/{len(question_rows)} = {percent(original_qa_calls, len(question_rows)):.3f}%** of `ask_perception` calls.",
            f"- Not ask original QA: **{len(question_rows) - original_qa_calls}/{len(question_rows)} = {percent(len(question_rows) - original_qa_calls, len(question_rows)):.3f}%**.",
            f"- At least one original-QA-style perception call appears in **{len(original_qa_samples)}/{len(perception_samples)} = {percent(len(original_qa_samples), len(perception_samples)):.3f}%** of samples that invoked `ask_perception` ({percent(len(original_qa_samples), total_samples):.3f}% of the benchmark).",
            "",
            "## Audit files",
            "",
            "- `tool_call_summary.csv`: aggregate call and sample coverage numbers.",
            "- `omni_clip_caption_calls.csv`: every requested/effective clip range and its percentage.",
            "- `ask_perception_question_type.csv`: every perception question with the binary classification and source question.",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
