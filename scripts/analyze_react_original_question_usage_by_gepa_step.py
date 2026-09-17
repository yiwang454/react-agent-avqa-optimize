#!/usr/bin/env python3
"""Audit original-question reuse in GEPA's persisted Val-set best-output frontier.

``track_best_outputs`` is sparse after iteration zero: it retains an output only
when a task receives a new best/frontier response.  Thus, each reported step is
the reconstructed per-task frontier, rather than a fresh complete evaluation of
one program at that step.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from analyze_react_original_question_usage import (
    PerceptionCall,
    build_summary,
    contains_words,
    extract_explicit_option,
    normalized_words,
    summarize_tool_usage,
)


SNAPSHOT_RE = re.compile(r"iter_(\d+)_prog_(\d+)\.json$")


def question_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        item = json.loads(raw)
        supervision = (item.get("supervisions") or [{}])[0]
        question_id = str(supervision.get("id") or item.get("id") or "").strip()
        question = str(supervision.get("text") or "").strip()
        if not question_id or not question:
            raise ValueError(f"Missing id/question at {path}:{line_number}")
        rows.append((question_id, question))
    return rows


def snapshots(root: Path) -> dict[int, list[tuple[int, int, Path, dict[str, Any]]]]:
    records: dict[int, list[tuple[int, int, Path, dict[str, Any]]]] = defaultdict(list)
    for path in sorted(root.glob("task_*/iter_*_prog_*.json")):
        task_name = path.parent.name
        try:
            task_index = int(task_name.removeprefix("task_"))
        except ValueError as error:
            raise ValueError(f"Unexpected task directory: {path.parent}") from error
        match = SNAPSHOT_RE.search(path.name)
        if not match:
            continue
        iteration, program = (int(value) for value in match.groups())
        with path.open(encoding="utf-8") as stream:
            output = json.load(stream)
        records[iteration].append((task_index, program, path, output))
    if not records:
        raise ValueError(f"No iter_N_prog_M snapshots below {root}")
    return records


def calls_for_output(
    question_id: str, original_question: str, output: dict[str, Any], edge_word_count: int
) -> list[PerceptionCall]:
    original_words = normalized_words(original_question)
    result: list[PerceptionCall] = []
    for turn_index, turn in enumerate(output.get("turn_trace") or [], 1):
        if not isinstance(turn, dict) or turn.get("tool_name") != "ask_perception":
            continue
        perception_question = str(
            (turn.get("tool_args") or {}).get("perceptual_question") or ""
        ).strip()
        perception_words = normalized_words(perception_question)
        option, answer_text = extract_explicit_option(turn.get("tool_observation"))
        enough_words = len(original_words) >= edge_word_count
        result.append(
            PerceptionCall(
                question_id=question_id,
                turn_id=int(turn.get("turn_id") or turn_index),
                original_question=original_question,
                perception_question=perception_question,
                observation_answer=answer_text,
                exact_match=contains_words(original_words, perception_words),
                first_edge_match=enough_words
                and contains_words(original_words[:edge_word_count], perception_words),
                last_edge_match=enough_words
                and contains_words(original_words[-edge_word_count:], perception_words),
                option_answer_match=bool(option),
            )
        )
    return result


def metric_calls_by_program(path: Path | None) -> list[int]:
    if path is None:
        return []
    with path.open("rb") as stream:
        state = pickle.load(stream)
    values = state.get("num_metric_calls_by_discovery")
    return [int(value) for value in values] if isinstance(values, list) else []


def rate(count: int, denominator: int) -> str:
    return f"{count}/{denominator} ({100 * count / denominator:.2f}%)" if denominator else "0/0 (0.00%)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valset-jsonl", type=Path, required=True)
    parser.add_argument("--best-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gepa-state", type=Path)
    parser.add_argument("--selected-program", type=int)
    parser.add_argument("--edge-word-count", type=int, default=3)
    args = parser.parse_args()
    if args.edge_word_count < 1:
        raise SystemExit("--edge-word-count must be >= 1")

    questions = question_rows(args.valset_jsonl)
    records = snapshots(args.best_output_dir)
    metric_calls = metric_calls_by_program(args.gepa_state)
    frontier: dict[int, tuple[int, int, Path, dict[str, Any]]] = {}
    trend: list[dict[str, Any]] = []
    tool_usage_trend: list[dict[str, Any]] = []
    all_calls: list[dict[str, Any]] = []

    for iteration in sorted(records):
        for task_index, program, path, output in records[iteration]:
            if task_index >= len(questions):
                raise ValueError(f"Task {task_index} exceeds {len(questions)} Val examples")
            frontier[task_index] = (iteration, program, path, output)
        if len(frontier) != len(questions):
            raise ValueError(
                f"Iteration {iteration} has only {len(frontier)}/{len(questions)} baseline tasks"
            )

        current_calls: list[PerceptionCall] = []
        for task_index, (source_iteration, source_program, source_path, output) in sorted(frontier.items()):
            question_id, original_question = questions[task_index]
            calls = calls_for_output(question_id, original_question, output, args.edge_word_count)
            current_calls.extend(calls)
            for call in calls:
                row = asdict(call)
                row.update(
                    {
                        "snapshot_gepa_iteration": iteration,
                        "source_best_output_iteration": source_iteration,
                        "source_program": source_program,
                        "source_path": str(source_path),
                        "edge_match": call.edge_match,
                        "detected": call.detected,
                    }
                )
                all_calls.append(row)
        summary = build_summary([item[0] for item in questions], current_calls)
        tool_usage = summarize_tool_usage(
            [item[0] for item in questions],
            (
                (questions[task_index][0], output.get("turn_trace"))
                for task_index, (_, _, _, output) in sorted(frontier.items())
            ),
        )
        tool_usage_trend.append({"gepa_iteration": iteration, **tool_usage})
        calls_by_tool = tool_usage["calls_by_tool"]
        samples_by_tool = tool_usage["samples_by_tool"]
        known_tool_calls = sum(
            calls_by_tool.get(name, 0)
            for name in ("ask_caption", "ask_perception", "omni_clip_caption")
        )
        metrics = summary["metrics"]
        programs = sorted({record[1] for record in records[iteration]})
        trend.append(
            {
                "gepa_iteration": iteration,
                "programs_introducing_best_output": ",".join(map(str, programs)),
                "metric_calls_at_program_discovery": ",".join(
                    str(metric_calls[program]) if program < len(metric_calls) else ""
                    for program in programs
                ),
                "new_best_output_records": len(records[iteration]),
                "tasks_updated_after_baseline": sum(
                    source[0] > 0 for source in frontier.values()
                ),
                "samples_with_ask_perception": summary["samples_with_perception"],
                "ask_perception_calls": summary["total_perception_calls"],
                "total_tool_calls": tool_usage["total_tool_calls"],
                "ask_caption_calls": calls_by_tool.get("ask_caption", 0),
                "omni_clip_caption_calls": calls_by_tool.get("omni_clip_caption", 0),
                "samples_with_omni_clip_caption": samples_by_tool.get(
                    "omni_clip_caption", 0
                ),
                "other_tool_calls": tool_usage["total_tool_calls"]
                - known_tool_calls,
                "budget_exempt_calls": tool_usage["budget_exempt_calls"],
                "exact_match_calls": metrics["exact_original_question"]["perception_calls"]["count"],
                "first_edge_match_calls": metrics["first_edge"]["perception_calls"]["count"],
                "last_edge_match_calls": metrics["last_edge"]["perception_calls"]["count"],
                "edge_match_calls": metrics["first_or_last_edge"]["perception_calls"]["count"],
                "combined_match_calls": metrics["combined_edge_or_abcd"]["perception_calls"]["count"],
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(trend[0])
    with (output_dir / "trend.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trend)
    call_fields = list(all_calls[0]) if all_calls else []
    with (output_dir / "perception_calls_by_step.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=call_fields)
        writer.writeheader()
        writer.writerows(all_calls)

    inventory = {
        "scope": "reconstructed per-task Val-set best-output frontier",
        "valset_jsonl": str(args.valset_jsonl.resolve()),
        "baseline_output_records": len(records.get(0, [])),
        "post_baseline_improvement_records": sum(len(value) for key, value in records.items() if key > 0),
        "iterations_with_persisted_best_output_changes": sorted(records),
        "all_candidate_rollouts_persisted": False,
        "selected_program": args.selected_program,
    }
    (output_dir / "record_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "tool_usage_by_step.json").write_text(
        json.dumps(tool_usage_trend, indent=2) + "\n", encoding="utf-8"
    )
    tool_table = [
        "| GEPA iteration | Total tool calls | `ask_caption` | `ask_perception` | `omni_clip_caption` | Samples using clip caption | Other | Budget-exempt |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in trend:
        tool_table.append(
            "| {gepa_iteration} | {total_tool_calls} | {ask_caption_calls} | "
            "{ask_perception_calls} | {omni_clip_caption_calls} | "
            "{samples_with_omni_clip_caption}/{total} | {other_tool_calls} | "
            "{budget_exempt_calls} |".format(total=len(questions), **row)
        )
    table = [
        "| GEPA iteration | Program adding best output | Metric calls at discovery | Updated tasks vs c0 | Samples using `ask_perception` | Calls | Exact | First 3 words | Last 3 words | First or last 3 words | Broad combined |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in trend:
        denominator = row["ask_perception_calls"]
        table.append(
            "| {gepa_iteration} | {programs_introducing_best_output} | {metric_calls_at_program_discovery} | "
            "{tasks_updated_after_baseline}/{total} | {samples}/{total} ({sample_pct:.2f}%) | {calls} | "
            "{exact} | {first} | {last} | {edge} | {combined} |".format(
                total=len(questions),
                samples=row["samples_with_ask_perception"],
                sample_pct=100 * row["samples_with_ask_perception"] / len(questions),
                calls=denominator,
                exact=rate(row["exact_match_calls"], denominator),
                first=rate(row["first_edge_match_calls"], denominator),
                last=rate(row["last_edge_match_calls"], denominator),
                edge=rate(row["edge_match_calls"], denominator),
                combined=rate(row["combined_match_calls"], denominator),
                **row,
            )
        )
    selected = f" The compiled final program is candidate {args.selected_program}." if args.selected_program is not None else ""
    report = "\n".join(
        [
            "# Original-question usage across GEPA best-output steps",
            "",
            "## Scope and limitation",
            "",
            "This reconstructs the persisted per-task Val125 best-output frontier. "
            "GEPA saved all iteration-0 outputs, then only outputs that improved an individual task; "
            "therefore this is not a complete re-evaluation of one deployable program at every step."
            + selected,
            "",
            "Matching uses the companion analyzer: exact normalized-question containment is the strict "
            "lower bound; first/last-three-word matches and the broad combined value are lexical heuristics.",
            "",
            "## Tool usage",
            "",
            *tool_table,
            "",
            "## Trend",
            "",
            *table,
            "",
            "Artifacts: `trend.csv`, `tool_usage_by_step.json`, "
            "`perception_calls_by_step.csv`, and `record_inventory.json`.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
