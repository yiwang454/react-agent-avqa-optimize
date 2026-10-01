#!/usr/bin/env python3
"""Export paired Gemini Direct +think versus ReAct candidate-0 changes.

The Direct prediction is parsed only from the visible response; separately
stored Gemini thought text is retained for qualitative coding but never enters
answer parsing.  Inputs must contain exactly the same 1,197 question IDs and
matching gold answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from visualize_dspy_output import extract_pred_answer, question_data


DATA_ROOT = Path("/mnt/ceph_rbd/data/avqa_project/daily_omni")
REPO_ROOT = Path(__file__).resolve().parents[1]
DIRECT_EVAL_DIR = Path(
    "/mnt/ceph_rbd/workspace/avqa_project/avqa_reasoning_datasets/daily_omni"
)
sys.path.insert(0, str(DIRECT_EVAL_DIR))
from eval_results_geminiLong import extract_last_option  # noqa: E402


GEMINI_THINK = DATA_ROOT / (
    "mllm_instruct_opt/"
    "daily_gemini2.5flash_single_mllm_legacy_thinking_store_qa_baseline_full_v3/"
    "output_test.jsonl"
)
REACT_CANDIDATE0 = DATA_ROOT / (
    "daily_omni_dspy_free_react_gpt4_1_gemini_first_cached_then_live_caption_"
    "gepa_planner_maxturns4_seed18_calls2500/output_test.jsonl"
)
DEFAULT_OUT = REPO_ROOT / (
    "scripts/experiment_records/gemini_think_vs_react_candidate0_analysis"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1197:
        raise ValueError(f"{path}: expected 1,197 rows, found {len(rows)}")
    return rows


def index_rows(
    rows: list[dict[str, Any]], getter: Any, source: Path
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(getter(row).get("question_id") or "")
        if not qid or qid in indexed:
            raise ValueError(f"{source}: missing or duplicate question ID {qid!r}")
        indexed[qid] = row
    return indexed


def direct_qd(row: dict[str, Any]) -> dict[str, Any]:
    nested = row.get("question_data")
    return nested if isinstance(nested, dict) else row


def gold(qd: dict[str, Any]) -> str:
    return str(qd.get("answer") or "").strip().upper()


def direct_prediction(row: dict[str, Any]) -> str:
    qd = direct_qd(row)
    # Thought is persisted separately.  Parsing the visible response prevents
    # option letters in chain-of-thought from contaminating the score.
    return extract_last_option(str(qd.get("response") or row.get("response_text") or ""))


def trace_summary(row: dict[str, Any]) -> list[dict[str, Any]]:
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
    for turn in question_data(row).get("turn_trace") or []:
        if isinstance(turn, dict):
            result.append(
                {key: turn.get(key) for key in keep if turn.get(key) not in (None, "", {}, [])}
            )
    return result


def transition(before_correct: bool, after_correct: bool) -> str:
    return ("C" if before_correct else "W") + "->" + ("C" if after_correct else "W")


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    direct_rows = load_jsonl(GEMINI_THINK)
    react_rows = load_jsonl(REACT_CANDIDATE0)
    direct = index_rows(direct_rows, direct_qd, GEMINI_THINK)
    react = index_rows(react_rows, question_data, REACT_CANDIDATE0)
    if set(direct) != set(react):
        raise ValueError("Gemini and ReAct question-ID sets differ")

    totals: Counter[str] = Counter()
    parser_audit: Counter[str] = Counter()
    changed: list[dict[str, Any]] = []
    for qid in sorted(direct):
        direct_row = direct[qid]
        react_row = react[qid]
        dqd = direct_qd(direct_row)
        rqd = question_data(react_row)
        direct_gold = gold(dqd)
        react_gold = gold(rqd)
        if not direct_gold or direct_gold != react_gold:
            raise ValueError(f"{qid}: missing or mismatched gold answer")

        direct_pred = direct_prediction(direct_row)
        stored_pred = str(dqd.get("predicted_answer") or "").strip().upper()
        react_pred = extract_pred_answer(react_row)
        parser_audit["strict_parser_valid"] += bool(direct_pred)
        parser_audit["strict_parser_invalid"] += not bool(direct_pred)
        parser_audit["stored_prediction_valid"] += stored_pred in {"A", "B", "C", "D"}
        parser_audit["stored_prediction_invalid"] += stored_pred not in {"A", "B", "C", "D"}
        parser_audit["strict_vs_stored_mismatch"] += direct_pred != stored_pred
        parser_audit["nonempty_thinking"] += bool(str(dqd.get("think") or "").strip())

        label = transition(direct_pred == direct_gold, react_pred == react_gold)
        totals[label] += 1
        if label not in {"C->W", "W->C"}:
            continue
        changed.append(
            {
                "comparison": "gemini_think_to_react_candidate0",
                "transition": label,
                "question_id": qid,
                "task_type": str(rqd.get("task_type") or "<missing>"),
                "question": rqd.get("question"),
                "options": rqd.get("options"),
                "gold": react_gold,
                "before_name": "Gemini Direct +think",
                "after_name": "ReAct candidate 0 / initial workflow",
                "before": {
                    "prediction": direct_pred,
                    "visible_response": dqd.get("response"),
                    "thinking": dqd.get("think"),
                    "stored_prediction": stored_pred,
                    "status": direct_row.get("status"),
                },
                "after": {
                    "prediction": react_pred,
                    "response": rqd.get("response"),
                    "turn_count": len(rqd.get("turn_trace") or []),
                    "trace": trace_summary(react_row),
                },
            }
        )

    dump_json(output_dir / "changed_cases.json", changed)
    manifest = {
        "gemini_direct_think": str(GEMINI_THINK),
        "react_candidate0": str(REACT_CANDIDATE0),
        "rows": len(direct),
        "transition_counts": dict(totals),
        "gemini_correct": totals["C->C"] + totals["C->W"],
        "react_correct": totals["C->C"] + totals["W->C"],
        "parser_audit": dict(parser_audit),
        "scoring_rule": "eval_results_geminiLong.extract_last_option on visible response only",
    }
    dump_json(output_dir / "analysis_manifest.json", manifest)

    lines = [
        "# Gemini Direct +think versus ReAct candidate 0",
        "",
        "All 1,197 unique question IDs and gold answers align exactly. Gemini is",
        "scored from the visible response with `eval_results_geminiLong.extract_last_option`;",
        "the separately stored thought is used only for qualitative analysis.",
        "",
        "| Transition (Gemini -> ReAct) | Questions |",
        "| --- | ---: |",
    ]
    for label in ("C->C", "C->W", "W->C", "W->W"):
        lines.append(f"| {label} | {totals[label]} |")
    lines.extend(
        [
            "",
            f"Gemini Direct +think: `{manifest['gemini_correct']}/1197 = {100 * manifest['gemini_correct'] / 1197:.2f}%`; "
            f"ReAct candidate 0: `{manifest['react_correct']}/1197 = {100 * manifest['react_correct'] / 1197:.2f}%`.",
            f"Net ReAct change: `{totals['W->C']} - {totals['C->W']} = {totals['W->C'] - totals['C->W']}` questions.",
            "",
            "## Parser audit",
            "",
            f"- Strict visible-response parser: `{parser_audit['strict_parser_valid']}` valid, `{parser_audit['strict_parser_invalid']}` invalid.",
            f"- Stored runner prediction: `{parser_audit['stored_prediction_valid']}` valid, `{parser_audit['stored_prediction_invalid']}` invalid.",
            f"- Strict versus stored prediction differences: `{parser_audit['strict_vs_stored_mismatch']}`.",
            f"- Nonempty separately stored Gemini thoughts: `{parser_audit['nonempty_thinking']}/1197`.",
            "",
            "No by-task-category analysis is included in this comparison.",
        ]
    )
    (output_dir / "paired_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(changed)} changed cases and summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
