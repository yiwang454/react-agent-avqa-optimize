#!/usr/bin/env python3
"""Analyze GPT-4.1 ReAct seed18 GEPA history and paired final outputs.

The analysis has two deliberately separate comparisons:

1. prior seed18 candidate 0 (the unchanged initial workflow template) versus
   candidate 7, a worse GEPA endpoint with a completed 1,197-row inference;
2. the strict three-run majority vote of Experiment E / Gemini Direct None
   versus that same candidate-0 ReAct full inference.

All ReAct predictions use the repository's canonical parser. Gemini Direct
predictions use the strict long-answer parser from ``eval_results_geminiLong``.
The script validates exact question-ID and gold-answer alignment before
exporting transitions and lossless traces for qualitative coding.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from visualize_dspy_output import extract_pred_answer, question_data


DATA_ROOT = Path("/mnt/ceph_rbd/data/avqa_project/daily_omni")
REPO_ROOT = Path(__file__).resolve().parents[1]
DIRECT_EVAL_DIR = Path(
    "/mnt/ceph_rbd/workspace/avqa_project/avqa_reasoning_datasets/daily_omni"
)
sys.path.insert(0, str(DIRECT_EVAL_DIR))
from eval_results_geminiLong import extract_last_option  # noqa: E402


SOURCE_RUN = DATA_ROOT / (
    "daily_omni_dspy_free_react_gpt4_1_gemini_first_cached_then_live_caption_"
    "gepa_planner_maxturns4_seed18_calls2500"
)
CANDIDATE0 = SOURCE_RUN / "output_test.jsonl"
CANDIDATE7 = DATA_ROOT / (
    "daily_omni_dspy_free_react_gpt4_1_gemini_first_cached_then_live_caption_"
    "gepa_planner_maxturns4_seed18_calls2500_candidate7_full_v3/output_test.jsonl"
)
GEMINI_RUNS = (
    DATA_ROOT / "results/daily_omni_seed27_gemini-2.5-flash_QA_PROMPT_TEMPLATE_0.0/output_test.json",
    DATA_ROOT / "results/daily_omni_seed27_repeat2_gemini-2.5-flash_QA_PROMPT_TEMPLATE_0.0/output_test.json",
    DATA_ROOT / "results/daily_omni_seed27_repeat3_gemini-2.5-flash_QA_PROMPT_TEMPLATE_0.0/output_test.json",
)
CANDIDATE_HISTORY = SOURCE_RUN / "planner_workflow_prompt_candidates.jsonl"
QUESTION_TYPE_METRICS = REPO_ROOT / (
    "scripts/experiment_records/"
    "daily_omni_dspy_free_react_gpt4_1_gemini_first_cached_then_live_caption_"
    "gepa_planner_maxturns4_seed18_calls2500_live/question_type_metrics.csv"
)
DEFAULT_OUT = REPO_ROOT / "scripts/experiment_records/gpt4_1_gepa_failure_analysis"
TASK_TYPES = (
    "AV Event Alignment",
    "Comparative",
    "Context understanding",
    "Event Sequence",
    "Inference",
    "Reasoning",
)


def load_react(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            qid = str(question_data(record).get("question_id") or "")
            if not qid or qid in rows:
                raise ValueError(f"{path}:{line_number}: missing/duplicate ID {qid!r}")
            rows[qid] = record
    if len(rows) != 1197:
        raise ValueError(f"{path}: expected 1,197 rows, found {len(rows)}")
    return rows


def load_gemini(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a list")
    rows: dict[str, dict[str, Any]] = {}
    for outer in payload:
        questions = outer.get("questions") if isinstance(outer, dict) else None
        if not isinstance(questions, list) or len(questions) != 1:
            raise ValueError(f"{path}: expected exactly one nested question per row")
        question = questions[0]
        qid = str(question.get("question_id") or "")
        if not qid or qid in rows:
            raise ValueError(f"{path}: missing/duplicate ID {qid!r}")
        rows[qid] = question
    if len(rows) != 1197:
        raise ValueError(f"{path}: expected 1,197 rows, found {len(rows)}")
    return rows


def gold(record: dict[str, Any]) -> str:
    return str(question_data(record).get("answer") or "").strip().upper()


def trace_summary(record: dict[str, Any]) -> list[dict[str, Any]]:
    keep = (
        "turn_id",
        "planner_action",
        "tool_name",
        "tool_args",
        "tool_observation",
        "tool_error",
        "final_answer",
        "reasoning_summary",
        "planner_raw",
    )
    result: list[dict[str, Any]] = []
    for turn in question_data(record).get("turn_trace") or []:
        if not isinstance(turn, dict):
            continue
        result.append(
            {key: turn.get(key) for key in keep if turn.get(key) not in (None, "", {}, [])}
        )
    return result


def compact_react(record: dict[str, Any]) -> dict[str, Any]:
    qd = question_data(record)
    return {
        "prediction": extract_pred_answer(record),
        "response": qd.get("response"),
        "turn_count": len(qd.get("turn_trace") or []),
        "trace": trace_summary(record),
    }


def validate_alignment(*groups: dict[str, dict[str, Any]]) -> list[str]:
    ids = set(groups[0])
    for group in groups[1:]:
        if set(group) != ids:
            raise ValueError("input files do not contain identical question-ID sets")
    for qid in ids:
        expected = gold(groups[0][qid])
        if not expected or any(gold(group[qid]) != expected for group in groups[1:]):
            raise ValueError(f"{qid}: missing or mismatched gold answer")
    return sorted(ids)


def transition(before_correct: bool, after_correct: bool) -> str:
    return ("C" if before_correct else "W") + "->" + ("C" if after_correct else "W")


def base_case(record: dict[str, Any], qid: str) -> dict[str, Any]:
    qd = question_data(record)
    return {
        "question_id": qid,
        "task_type": str(qd.get("task_type") or "<missing>"),
        "question": qd.get("question"),
        "options": qd.get("options"),
        "gold": gold(record),
    }


def analyze_react_pair(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> tuple[Counter[str], dict[str, Counter[str]], list[dict[str, Any]]]:
    ids = validate_alignment(before, after)
    totals: Counter[str] = Counter()
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    changed: list[dict[str, Any]] = []
    for qid in ids:
        before_row, after_row = before[qid], after[qid]
        answer = gold(before_row)
        before_pred = extract_pred_answer(before_row)
        after_pred = extract_pred_answer(after_row)
        label = transition(before_pred == answer, after_pred == answer)
        task = str(question_data(before_row).get("task_type") or "<missing>")
        totals[label] += 1
        by_task[task][label] += 1
        if label in {"C->W", "W->C"}:
            case = base_case(before_row, qid)
            case.update(
                {
                    "comparison": "candidate0_to_candidate7",
                    "transition": label,
                    "before_name": "ReAct candidate 0 / initial workflow",
                    "after_name": "ReAct GEPA candidate 7",
                    "before": compact_react(before_row),
                    "after": compact_react(after_row),
                }
            )
            changed.append(case)
    return totals, by_task, changed


def direct_prediction(question: dict[str, Any]) -> str:
    response = str(question.get("response") or "")
    if "</think>" in response:
        response = response.split("</think>", 1)[1].strip()
    return extract_last_option(response)


def majority_vote(predictions: list[str]) -> tuple[str, bool]:
    counts = Counter(pred for pred in predictions if pred)
    if not counts:
        return "", False
    top = counts.most_common()
    if top[0][1] < 2 or len(top) > 1 and top[1][1] == top[0][1]:
        return "", False
    return top[0][0], True


def analyze_direct_vs_react(
    direct_runs: list[dict[str, dict[str, Any]]], react: dict[str, dict[str, Any]]
) -> tuple[Counter[str], dict[str, Counter[str]], list[dict[str, Any]], Counter[str]]:
    ids = validate_alignment(react, *direct_runs)
    totals: Counter[str] = Counter()
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    vote_status: Counter[str] = Counter()
    changed: list[dict[str, Any]] = []
    for qid in ids:
        react_row = react[qid]
        answer = gold(react_row)
        direct_preds = [direct_prediction(run[qid]) for run in direct_runs]
        majority_pred, has_majority = majority_vote(direct_preds)
        vote_status["strict_majority" if has_majority else "no_strict_majority"] += 1
        react_pred = extract_pred_answer(react_row)
        label = transition(majority_pred == answer, react_pred == answer)
        task = str(question_data(react_row).get("task_type") or "<missing>")
        totals[label] += 1
        by_task[task][label] += 1
        if label in {"C->W", "W->C"}:
            case = base_case(react_row, qid)
            case.update(
                {
                    "comparison": "gemini_majority_to_candidate0",
                    "transition": label,
                    "before_name": "Gemini Direct None / 3-run strict majority",
                    "after_name": "ReAct candidate 0 / initial workflow",
                    "before": {
                        "prediction": majority_pred,
                        "has_strict_majority": has_majority,
                        "run_predictions": direct_preds,
                        "run_responses": [run[qid].get("response") for run in direct_runs],
                    },
                    "after": compact_react(react_row),
                }
            )
            changed.append(case)
    return totals, by_task, changed, vote_status


def load_history(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or rows[0].get("candidate_index") != 0:
        raise ValueError(f"{path}: missing candidate 0")
    return rows


def load_native_type_metrics(path: Path) -> dict[int, dict[str, tuple[int, float]]]:
    result: dict[int, dict[str, tuple[int, float]]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["taxonomy"] != "native":
                continue
            result[int(row["candidate_index"])][row["question_type"]] = (
                int(row["num_examples"]),
                float(row["score_mean"]),
            )
    return result


def write_history_artifacts(
    out: Path, history: list[dict[str, Any]], metrics: dict[int, dict[str, tuple[int, float]]]
) -> None:
    csv_path = out / "candidate_history_by_task.csv"
    fields = ["candidate_index", "parents", "val_correct", "val_accuracy"]
    for task in TASK_TYPES:
        fields.extend([f"{task}_correct", f"{task}_accuracy", f"{task}_delta_vs_c0_pp"])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        baseline = metrics[0]
        for item in history:
            idx = int(item["candidate_index"])
            row: dict[str, Any] = {
                "candidate_index": idx,
                "parents": json.dumps(item.get("parents")),
                "val_correct": round(125 * float(item["full_valset_score"])),
                "val_accuracy": 100 * float(item["full_valset_score"]),
            }
            for task in TASK_TYPES:
                n, score = metrics[idx][task]
                row[f"{task}_correct"] = round(n * score)
                row[f"{task}_accuracy"] = 100 * score
                row[f"{task}_delta_vs_c0_pp"] = 100 * (score - baseline[task][1])
            writer.writerow(row)

    lines = [
        "# Seed18 candidate instruction history and parent diffs",
        "",
        "Each diff is against the candidate's recorded parent. Candidate 0 is the initial workflow template.",
        "",
    ]
    by_index = {int(item["candidate_index"]): item for item in history}
    for item in history:
        idx = int(item["candidate_index"])
        parents = [p for p in item.get("parents") or [] if p is not None]
        parent = int(parents[0]) if parents else None
        lines.extend(
            [
                f"## Candidate {idx}",
                "",
                f"- Parent: `{parent if parent is not None else 'none'}`",
                f"- Val125: `{round(125 * float(item['full_valset_score']))}/125 = {100 * float(item['full_valset_score']):.1f}%`",
                f"- Prompt characters: `{len(item['prompt_text'])}`",
                "",
            ]
        )
        if parent is None:
            lines.extend(["```text", item["prompt_text"], "```", ""])
            continue
        diff = difflib.unified_diff(
            by_index[parent]["prompt_text"].splitlines(),
            item["prompt_text"].splitlines(),
            fromfile=f"candidate_{parent}",
            tofile=f"candidate_{idx}",
            lineterm="",
        )
        lines.extend(["```diff", *diff, "```", ""])
    (out / "candidate_instruction_diffs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_lines(by_task: dict[str, Counter[str]]) -> list[str]:
    result = [
        "| Task type | C->C | C->W | W->C | W->W | Net |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in sorted(by_task):
        counts = by_task[task]
        result.append(
            f"| {task} | {counts['C->C']} | {counts['C->W']} | "
            f"{counts['W->C']} | {counts['W->W']} | {counts['W->C'] - counts['C->W']} |"
        )
    return result


def write_summary(
    out: Path,
    react_totals: Counter[str],
    react_by_task: dict[str, Counter[str]],
    direct_totals: Counter[str],
    direct_by_task: dict[str, Counter[str]],
    vote_status: Counter[str],
) -> None:
    lines = [
        "# GPT-4.1 ReAct seed18 GEPA paired outcome analysis",
        "",
        "## Validated inputs",
        "",
        f"- ReAct None v1 / prior seed18 candidate 0: `{CANDIDATE0}`",
        f"- ReAct GEPA candidate 7: `{CANDIDATE7}`",
        "- Gemini Direct None: three strict-parser runs listed in `analysis_manifest.json`.",
        "- Every input contains the same 1,197 unique question IDs and matching gold answers.",
        "- ReAct parser: `visualize_dspy_output.extract_pred_answer`.",
        "- Gemini parser: `eval_results_geminiLong.extract_last_option`.",
        "",
        "## Candidate 0 -> candidate 7",
        "",
        "| Transition | Questions |",
        "| --- | ---: |",
    ]
    for label in ("C->C", "C->W", "W->C", "W->W"):
        lines.append(f"| {label} | {react_totals[label]} |")
    lines.extend(
        [
            "",
            f"Net change: `{react_totals['W->C']} - {react_totals['C->W']} = {react_totals['W->C'] - react_totals['C->W']}` questions.",
            "",
            *table_lines(react_by_task),
            "",
            "## Gemini Direct 3-run majority -> ReAct candidate 0",
            "",
            f"Strict-majority answers: `{vote_status['strict_majority']}`; no strict majority: `{vote_status['no_strict_majority']}`.",
            "",
            "| Transition | Questions |",
            "| --- | ---: |",
        ]
    )
    for label in ("C->C", "C->W", "W->C", "W->W"):
        lines.append(f"| {label} | {direct_totals[label]} |")
    lines.extend(
        [
            "",
            f"Net change from Gemini majority to ReAct: `{direct_totals['W->C']} - {direct_totals['C->W']} = {direct_totals['W->C'] - direct_totals['C->W']}` questions.",
            "",
            *table_lines(direct_by_task),
            "",
            "The changed-case JSON files contain the complete ReAct traces. Gemini Direct stores only final responses, so any causal label about Direct errors must remain `unobservable`; the benefit analysis is trace-supported from the ReAct side.",
        ]
    )
    (out / "paired_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tool_behavior(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flows: Counter[str] = Counter()
    perception_calls = 0
    caption_calls = 0
    recorded_decisions = 0
    no_perception = 0
    repeated_perception = 0
    for record in rows.values():
        turns = question_data(record).get("turn_trace") or []
        tools = [str(turn.get("tool_name") or turn.get("planner_action") or "?") for turn in turns]
        flows[" -> ".join(tools)] += 1
        perception_count = sum(turn.get("tool_name") == "ask_perception" for turn in turns)
        perception_calls += perception_count
        caption_calls += sum(turn.get("tool_name") == "ask_caption" for turn in turns)
        recorded_decisions += len(turns)
        no_perception += perception_count == 0
        repeated_perception += perception_count > 1
    return {
        "questions": len(rows),
        "caption_calls": caption_calls,
        "perception_calls": perception_calls,
        "all_tool_calls": caption_calls + perception_calls,
        "no_perception_questions": no_perception,
        "repeated_perception_questions": repeated_perception,
        "mean_recorded_decisions": recorded_decisions / len(rows),
        "flow_counts": dict(flows.most_common()),
    }


def changed_tool_behavior(
    cases: list[dict[str, Any]],
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in ("C->W", "W->C"):
        ids = [case["question_id"] for case in cases if case["transition"] == label]
        result[label] = {
            "before": tool_behavior({qid: before[qid] for qid in ids}),
            "after": tool_behavior({qid: after[qid] for qid in ids}),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    candidate0 = load_react(CANDIDATE0)
    candidate7 = load_react(CANDIDATE7)
    direct_runs = [load_gemini(path) for path in GEMINI_RUNS]

    react_totals, react_by_task, react_changed = analyze_react_pair(candidate0, candidate7)
    direct_totals, direct_by_task, direct_changed, vote_status = analyze_direct_vs_react(
        direct_runs, candidate0
    )

    dump_json(out / "candidate0_vs_candidate7_changed_cases.json", react_changed)
    dump_json(out / "gemini_majority_vs_candidate0_changed_cases.json", direct_changed)
    history = load_history(CANDIDATE_HISTORY)
    metrics = load_native_type_metrics(QUESTION_TYPE_METRICS)
    write_history_artifacts(out, history, metrics)
    write_summary(out, react_totals, react_by_task, direct_totals, direct_by_task, vote_status)
    dump_json(
        out / "tool_behavior.json",
        {
            "candidate0": tool_behavior(candidate0),
            "candidate7": tool_behavior(candidate7),
            "changed_cases": changed_tool_behavior(react_changed, candidate0, candidate7),
        },
    )
    dump_json(
        out / "analysis_manifest.json",
        {
            "candidate0": str(CANDIDATE0),
            "candidate7": str(CANDIDATE7),
            "gemini_direct_runs": [str(path) for path in GEMINI_RUNS],
            "candidate_history": str(CANDIDATE_HISTORY),
            "question_type_metrics": str(QUESTION_TYPE_METRICS),
            "candidate0_vs_candidate7": dict(react_totals),
            "gemini_majority_vs_candidate0": dict(direct_totals),
            "gemini_vote_status": dict(vote_status),
        },
    )
    print(f"wrote analysis to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
