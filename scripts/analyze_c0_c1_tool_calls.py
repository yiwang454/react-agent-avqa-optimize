#!/usr/bin/env python3
"""Audit tool-call counts and sequences for the completed C0/C1 DailyOmni runs.

The report intentionally treats C0 max-turn 4, C0 max-turn 6, and the legacy
C1 seed-18 selected-c0 inference as separate complete runs.  It reads the raw
``turn_trace`` in each output JSONL and fails on missing/duplicate question
IDs, unexpected tools, an invalid first tool, or a budget overrun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/mnt/ceph_rbd/data/avqa_project/daily_omni")
RUNS = (
    (
        "C0 max-turn 4",
        4,
        DATA_ROOT
        / "daily_omni_dspy_free_react_gpt4_1_gemini_first_cached_then_live_caption_maxturns4"
        / "output_test.jsonl",
    ),
    (
        "C0 max-turn 6",
        6,
        DATA_ROOT
        / "daily_omni_dspy_free_react_gpt4_1_gemini_first_cached_then_live_caption_maxturns6"
        / "output_test.jsonl",
    ),
    (
        "Prior C1 seed18 (selected candidate 0)",
        4,
        DATA_ROOT
        / "daily_omni_dspy_free_react_gpt4_1_gemini_first_cached_then_live_caption_gepa_planner_maxturns4_seed18_calls2500"
        / "output_test.jsonl",
    ),
)
EXPECTED_QUESTION_COUNT = 1197
TOOLS = {"ask_caption", "ask_perception"}


def pct(numerator: int, denominator: int) -> str:
    return "—" if denominator == 0 else f"{100 * numerator / denominator:.2f}%"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class RunStats:
    name: str
    max_turns: int
    path: Path
    question_ids: set[str] = field(default_factory=set)
    tool_counts: Counter[str] = field(default_factory=Counter)
    tool_count_per_question: Counter[int] = field(default_factory=Counter)
    sequences: Counter[str] = field(default_factory=Counter)
    caption_modes: Counter[str] = field(default_factory=Counter)
    caption_backends: Counter[str] = field(default_factory=Counter)
    perception_backends: Counter[str] = field(default_factory=Counter)
    final_modes: Counter[str] = field(default_factory=Counter)
    error_types: Counter[str] = field(default_factory=Counter)
    first_tool_enforced: int = 0
    first_requested_not_caption: int = 0
    budget_exempt_tools: int = 0
    questions_with_perception: int = 0

    @property
    def rows(self) -> int:
        return len(self.question_ids)

    @property
    def total_tools(self) -> int:
        return sum(self.tool_counts.values())


def sequence_label(tools: list[dict[str, Any]]) -> str:
    return " → ".join(
        "C" if tool["tool_name"] == "ask_caption" else "P" for tool in tools
    )


def analyze_run(name: str, max_turns: int, path: Path) -> RunStats:
    if not path.is_file():
        raise FileNotFoundError(path)
    stats = RunStats(name=name, max_turns=max_turns, path=path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            question_data = record.get("question_data") or {}
            question_id = str(question_data.get("question_id") or "")
            if not question_id:
                raise ValueError(f"{path}:{line_number}: missing question_id")
            if question_id in stats.question_ids:
                raise ValueError(f"{path}:{line_number}: duplicate question_id {question_id}")
            stats.question_ids.add(question_id)

            trace = question_data.get("turn_trace")
            if not isinstance(trace, list) or not trace:
                raise ValueError(f"{path}:{line_number}: missing or empty turn_trace")
            tools = [turn for turn in trace if turn.get("planner_action") == "tool"]
            finals = [turn for turn in trace if turn.get("planner_action") == "final"]
            terminal_action = trace[-1].get("planner_action")
            if terminal_action not in {"final", "error"}:
                raise ValueError(f"{path}:{line_number}: unexpected terminal action {terminal_action!r}")
            if terminal_action == "final" and len(finals) != 1:
                raise ValueError(f"{path}:{line_number}: expected one terminal final action")
            if terminal_action == "error":
                if finals:
                    raise ValueError(f"{path}:{line_number}: error trace unexpectedly contains final action")
                stats.final_modes["terminal error"] += 1
                error = question_data.get("planner_error") or trace[-1].get("planner_error") or {}
                stats.error_types[str(error.get("error_type") or "<missing>")] += 1
            elif not tools or tools[0].get("tool_name") != "ask_caption":
                raise ValueError(f"{path}:{line_number}: first executed tool is not ask_caption")
            if len(tools) > max_turns:
                raise ValueError(f"{path}:{line_number}: {len(tools)} tools exceed max_turns={max_turns}")

            if tools:
                first = tools[0]
                if first.get("tool_name") != "ask_caption":
                    raise ValueError(f"{path}:{line_number}: first executed tool is not ask_caption")
                stats.first_tool_enforced += bool(first.get("first_tool_enforced"))
                requested = first.get("planner_requested_tool_name")
                if first.get("planner_requested_action") != "tool" or requested != "ask_caption":
                    stats.first_requested_not_caption += 1

            for tool in tools:
                tool_name = tool.get("tool_name")
                if tool_name not in TOOLS:
                    raise ValueError(f"{path}:{line_number}: unexpected tool {tool_name!r}")
                stats.tool_counts[tool_name] += 1
                stats.budget_exempt_tools += bool(tool.get("budget_exempt"))
                backend = str(tool.get("perception_backend") or "<missing>")
                if tool_name == "ask_caption":
                    stats.caption_modes[str(tool.get("caption_source_mode") or "<missing>")] += 1
                    stats.caption_backends[backend] += 1
                else:
                    stats.perception_backends[backend] += 1

            stats.questions_with_perception += any(
                tool.get("tool_name") == "ask_perception" for tool in tools
            )

            stats.tool_count_per_question[len(tools)] += 1
            stats.sequences[sequence_label(tools) if tools else "no tool"] += 1
            if terminal_action == "final":
                stats.final_modes[
                    "forced fallback" if "final_fallback" in (finals[0].get("planner_calls") or {}) else "planner final"
                ] += 1

    if stats.rows != EXPECTED_QUESTION_COUNT:
        raise ValueError(f"{path}: expected {EXPECTED_QUESTION_COUNT} rows, found {stats.rows}")
    return stats


def counter_text(counter: Counter[str]) -> str:
    return ", ".join(f"{key}: {value}" for key, value in sorted(counter.items())) or "—"


def render(stats_list: list[RunStats]) -> str:
    lines = [
        "# C0/C1 ReAct tool-call audit",
        "",
        "Generated by `scripts/analyze_c0_c1_tool_calls.py` from the three completed raw `output_test.jsonl` files listed below. The replacement C1 runs (seeds 2/42) are excluded: their gates recorded `decision: skip_inference`, so no complete final output exists.",
        "",
        "## Scope and counting rules",
        "",
        "- A **tool call** is a `turn_trace` item with `planner_action: tool`; the terminal `final` action is not a tool call.",
        "- `C` means `ask_caption`; `P` means `ask_perception`. Percentages in the two tool columns use total tool calls as denominator; the sequence table uses 1,197 questions as denominator for each run.",
        "- All three outputs contain exactly 1,197 unique question IDs. Non-error rows have exactly one terminal final action; any terminal error row is counted explicitly below. All executed tools are `ask_caption` / `ask_perception`, every non-error trace starts with `ask_caption`, and no row exceeds its `max_turns`.",
        "- In this workflow, the first `ask_caption` is cache-backed. A later caption, if any, would be live Gemini because these runs use `caption_cache_scope: first_call_only`.",
        "",
        "## Run provenance",
        "",
        "| Run | `max_turns` | Raw output | SHA-256 |",
        "| --- | ---: | --- | --- |",
    ]
    for stats in stats_list:
        lines.append(
            f"| {stats.name} | {stats.max_turns} | [{stats.path.name}]({stats.path}) | `{sha256(stats.path)}` |"
        )

    lines.extend([
        "",
        "## Tool counts and proportions",
        "",
        "| Run | Total tool calls | Calls / question | `ask_caption` | `ask_perception` | Questions with `P` | Cache captions | Live later captions | Planner final / forced fallback / terminal error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for stats in stats_list:
        captions = stats.tool_counts["ask_caption"]
        perceptions = stats.tool_counts["ask_perception"]
        lines.append(
            f"| {stats.name} | {stats.total_tools:,} | {stats.total_tools / stats.rows:.3f} | "
            f"{captions:,} ({pct(captions, stats.total_tools)}) | "
            f"{perceptions:,} ({pct(perceptions, stats.total_tools)}) | "
            f"{stats.questions_with_perception:,} ({pct(stats.questions_with_perception, stats.rows)}) | "
            f"{stats.caption_modes['cache']:,} | {stats.caption_modes['live']:,} | "
            f"{stats.final_modes['planner final']:,} / {stats.final_modes['forced fallback']:,} / {stats.final_modes['terminal error']:,} |"
        )

    lines.extend([
        "",
        "Every successful row used exactly one cached caption; the one remaining C0 max-turn-4 row terminated with a Gemini policy error before any tool call. No run made a second caption call, so the “later captions may be live” capability was available but unused in the successful outputs. `ask_perception` is a call count, while “Questions with `P`” avoids counting repeated perception calls twice.",
        "",
        "## Per-question tool sequences",
        "",
        "| Run | `C → final` | `C → P → final` | Terminal error before a tool | Other exact tool sequence(s) | Tool-count distribution |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for stats in stats_list:
        c_only = stats.sequences["C"]
        c_p = stats.sequences["C → P"]
        no_tool = stats.sequences["no tool"]
        other_sequences = "; ".join(
            f"`{sequence}`: {count} ({pct(count, stats.rows)})"
            for sequence, count in sorted(stats.sequences.items())
            if sequence not in {"C", "C → P", "no tool"}
        ) or "—"
        distribution = ", ".join(
            f"{count} calls: {rows}" for count, rows in sorted(stats.tool_count_per_question.items())
        )
        lines.append(
            f"| {stats.name} | {c_only:,} ({pct(c_only, stats.rows)}) | "
            f"{c_p:,} ({pct(c_p, stats.rows)}) | {no_tool:,} ({pct(no_tool, stats.rows)}) | "
            f"{other_sequences} | {distribution} |"
        )

    lines.extend([
        "",
        "`max_turns` is an upper bound on tool calls, not a required number of calls and not a count including the final answer. The observed maxima were three tool calls for C0 max-turn-4 and prior C1, and four for C0 max-turn-6; the forced final fallback was never used. The terminal error occurred before any tool call and is not evidence of a valid zero-tool policy.",
        "",
        "## Workflow-enforcement and backend checks",
        "",
        "| Run | First tool force-inserted | First planner request not explicitly `ask_caption` | Budget-exempt tool observations | Caption backend | Perception backend |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ])
    for stats in stats_list:
        lines.append(
            f"| {stats.name} | {stats.first_tool_enforced:,} | {stats.first_requested_not_caption:,} | "
            f"{stats.budget_exempt_tools:,} | {counter_text(stats.caption_backends)} | "
            f"{counter_text(stats.perception_backends)} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The prompt configuration requires `ask_caption` as the first tool. From the second tool call onward it allows either tool to repeat, but the planner schema requires `ask_caption` with empty arguments so the runner supplies its default factual whole-video request; `ask_perception` remains limited to observable audio/visual questions. Therefore “free” means free tool selection after the initial caption within those rules and the run-specific budget; it does not mean arbitrary tools, arbitrary caption prompts, or unbounded turns.",
        "",
        "For the completed C0/C1 outputs audited here, later `ask_caption` was never used. Repeated `ask_perception` was rare but present: 2/1,197 C0 max-turn-4 questions, 12/1,197 C0 max-turn-6 questions, and 3/1,197 prior-C1 questions. Most successful traces were `C → final` or `C → P → final`; this empirical distribution should not be generalized to a different planner prompt or a future C1 rerun.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("c0_c1_tool_call_analysis.md"),
        help="Markdown report path (default: scripts/c0_c1_tool_call_analysis.md)",
    )
    args = parser.parse_args()
    stats_list = [analyze_run(*run) for run in RUNS]
    report = render(stats_list)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
