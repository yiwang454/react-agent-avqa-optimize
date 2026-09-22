#!/usr/bin/env python3
"""Compare GPT-5.4 no-GEPA and G0-planner GEPA outputs at question level.

The script uses the repository's canonical DSPy answer parser, validates that
both final outputs contain the same 1,197 question IDs and gold answers, and
exports every correct-to-wrong (C->W) regression with the two full ReAct
traces.  A reviewer can therefore audit qualitative labels against the exact
evidence that each planner saw.

Example:
  python scripts/analyze_gpt54_gepa_regressions.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from visualize_dspy_output import extract_pred_answer, question_data


DATA_ROOT = Path("/mnt/ceph_rbd/data/avqa_project/daily_omni")
BASELINE = DATA_ROOT / "daily_omni_dspy_GPT_v8GeminiCaptionInTask_planner_gpt-5.4_seed_1234/output_test.jsonl"
GEPA = DATA_ROOT / "daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_planner_workflow_prompt_gpt5_4_medium_planner_gpt-5.4_seed_1234_gepa_seed18/output_test.jsonl"
DEFAULT_OUT = DATA_ROOT / "analysis/gpt54_g0_planner_regressions"


def load(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            qid = str(question_data(record).get("question_id") or "")
            if not qid or qid in rows:
                raise ValueError(f"{path}:{line_number}: missing or duplicate question_id {qid!r}")
            rows[qid] = record
    if len(rows) != 1197:
        raise ValueError(f"{path}: expected 1,197 rows, found {len(rows)}")
    return rows


def answer(record: dict[str, Any]) -> str:
    return str(question_data(record).get("answer") or "").strip().upper()


def trace_summary(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep exactly the information needed to audit a planner decision."""
    result = []
    for turn in question_data(record).get("turn_trace") or []:
        if not isinstance(turn, dict):
            continue
        item = {
            key: turn.get(key)
            for key in (
                "turn_id", "planner_action", "tool_name", "tool_args",
                "tool_observation", "tool_error", "final_answer",
                "reasoning_summary", "planner_raw",
            )
            if turn.get(key) not in (None, "", {}, [])
        }
        result.append(item)
    return result


def compact(record: dict[str, Any]) -> dict[str, Any]:
    qd = question_data(record)
    return {
        "prediction": extract_pred_answer(record),
        "turn_count": len(qd.get("turn_trace") or []),
        "trace": trace_summary(record),
    }


def md_escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def tool_field(case: dict[str, Any], condition: str, field: str) -> str:
    """Return the sole perception-call field, or an empty string if absent."""
    for turn in case[condition]["trace"]:
        if turn.get("tool_name") == "ask_perception":
            if field == "question":
                args = turn.get("tool_args") or {}
                return str(args.get("perceptual_question") or "")
            if field == "observation":
                return str(turn.get("tool_observation") or "")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--gepa", type=Path, default=GEPA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    baseline = load(args.baseline)
    gepa = load(args.gepa)
    if set(baseline) != set(gepa):
        raise ValueError("The final outputs do not have identical question-ID sets")

    transitions: Counter[str] = Counter()
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    regressions: list[dict[str, Any]] = []
    for qid in sorted(baseline):
        base, opt = baseline[qid], gepa[qid]
        gold = answer(base)
        if gold != answer(opt):
            raise ValueError(f"{qid}: gold mismatch ({gold!r} != {answer(opt)!r})")
        base_pred, opt_pred = extract_pred_answer(base), extract_pred_answer(opt)
        transition = ("C" if base_pred == gold else "W") + "->" + ("C" if opt_pred == gold else "W")
        transitions[transition] += 1
        task_type = str(question_data(base).get("task_type") or "<missing>")
        by_task[task_type][transition] += 1
        if transition == "C->W":
            qd = question_data(base)
            regressions.append({
                "question_id": qid,
                "task_type": task_type,
                "question": qd.get("question"),
                "options": qd.get("options"),
                "gold": gold,
                "baseline": compact(base),
                "gepa": compact(opt),
                # To be filled during human qualitative coding.  Keeping this
                # blank avoids presenting an un-audited automatic inference as
                # an error cause.
                "primary_cause": None,
                "secondary_causes": [],
                "coding_note": None,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "correct_to_wrong_cases.json"
    json_path.write_text(json.dumps(regressions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# GPT-5.4 G0 planner GEPA: correct-to-wrong audit", "",
        "## Inputs and validation", "",
        f"- no-GEPA: `{args.baseline}`", f"- GEPA: `{args.gepa}`",
        "- Both files: 1,197 unique aligned question IDs; gold answers match exactly.",
        "- Parser: `visualize_dspy_output.extract_pred_answer` (the canonical DSPy parser).", "",
        "## Outcome transition matrix", "",
        "| no-GEPA → GEPA | Questions |", "| --- | ---: |",
    ]
    for transition in ("C->C", "C->W", "W->C", "W->W"):
        lines.append(f"| {transition} | {transitions[transition]} |")
    lines.extend([
        "", f"GEPA changes net accuracy by `W->C − C->W = {transitions['W->C']} − {transitions['C->W']} = {transitions['W->C'] - transitions['C->W']}` questions.",
        "", "## C->W regressions by task type", "", "| Task type | C->W | W->C | Net |", "| --- | ---: | ---: | ---: |",
    ])
    for task, counts in sorted(by_task.items(), key=lambda pair: (-pair[1]["C->W"], pair[0])):
        lines.append(f"| {md_escape(task)} | {counts['C->W']} | {counts['W->C']} | {counts['W->C'] - counts['C->W']} |")
    lines.extend([
        "", "## Case-level audit file", "",
        f"All {len(regressions)} C->W cases, including full tool observations and planner traces from both conditions, are in [`correct_to_wrong_cases.json`]({json_path.name}).",
        "", "The `primary_cause`, `secondary_causes`, and `coding_note` fields are intentionally blank until human coding; this prevents the analysis from conflating a keyword heuristic with a qualitative causal explanation.",
    ])
    report_path = args.output_dir / "summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # The full JSON is deliberately lossless and can be large.  This companion
    # index is optimized for reading and human annotation, with both targeted
    # perception questions and observations shown side by side.
    index_lines = ["# GPT-5.4 GEPA correct-to-wrong case index", ""]
    for ordinal, case in enumerate(regressions, 1):
        index_lines.extend([
            f"## {ordinal}. `{case['question_id']}` — {case['task_type']}", "",
            f"- Gold: `{case['gold']}`; no-GEPA: `{case['baseline']['prediction']}`; GEPA: `{case['gepa']['prediction']}`",
            f"- Question: {case['question']}",
            f"- Options: {json.dumps(case['options'], ensure_ascii=False)}", "",
            "**No-GEPA perception query**", "", tool_field(case, "baseline", "question"), "",
            "**GEPA perception query**", "", tool_field(case, "gepa", "question"), "",
            "**GEPA perception observation**", "", tool_field(case, "gepa", "observation"), "",
        ])
    index_path = args.output_dir / "case_index.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {json_path}")
    print(f"wrote {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
