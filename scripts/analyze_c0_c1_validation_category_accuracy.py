#!/usr/bin/env python3
"""Report C0 and C1 Val125 accuracy by ask_perception question category.

C0 has a complete full-inference output, so each Val125 question can be put
into one exact group: no ask_perception call, or category 1/2/3/4.

The historical C1 GEPA state saved every correct trajectory plus eleven
incorrect trajectories.  A trajectory that was not saved is known to be
incorrect, but its tool calls are unavailable; it is therefore reported as
``not_saved_trajectory`` rather than treated as a no-call question.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from react_eval import extract_pred_answer, get_question_data, get_turn_trace


DATA_ROOT = Path("/mnt/ceph_rbd/data/avqa_project/daily_omni")
JUDGE_ROOT = DATA_ROOT / "ask_perception_question_judge_gpt5_5"
VALSET_PATH = DATA_ROOT / "daily_omni_cuts_selectedVal125.jsonl"
C0_MANIFEST_PATH = JUDGE_ROOT / "c0" / "manifest.json"
C0_JUDGMENTS_PATH = JUDGE_ROOT / "c0" / "output_test.json"
C1_ROOT = JUDGE_ROOT / "c1_gepa_val125"
OUTPUT_PATH = JUDGE_ROOT / "c0_c1_validation_category_accuracy.csv"


def read_val_ids() -> set[str]:
    ids = {
        str(json.loads(line)["id"])
        for line in VALSET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(ids) != 125:
        raise ValueError(f"expected 125 unique Val125 IDs, found {len(ids)}")
    return ids


def add(grouped: dict[str, list[int]], group: str, correct: bool) -> None:
    grouped[group][0] += int(correct)
    grouped[group][1] += 1


def category_for_calls(calls: list[dict[str, Any]]) -> str:
    if not calls:
        return "no_ask_perception"
    if len(calls) != 1:
        raise ValueError(f"expected at most one ask_perception call, got {len(calls)}")
    category = calls[0].get("category")
    if category not in (1, 2, 3, 4):
        raise ValueError(f"invalid category: {category!r}")
    return f"category_{category}"


def result_rows(run: str, scope: str, grouped: dict[str, list[int]], overall_total: int, overall_correct: int) -> list[dict[str, object]]:
    rows = []
    for group in ("no_ask_perception", "category_1", "category_2", "category_3", "category_4", "not_saved_trajectory"):
        correct, total = grouped[group]
        rows.append({
            "run": run,
            "scope": scope,
            "question_group": group,
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else None,
            "overall_correct": overall_correct,
            "overall_total": overall_total,
            "overall_accuracy": overall_correct / overall_total,
        })
    return rows


def c0_rows(val_ids: set[str]) -> list[dict[str, object]]:
    manifest = json.loads(C0_MANIFEST_PATH.read_text(encoding="utf-8"))
    by_id = {
        question["question_id"]: question["ask_perception_judgments"]
        for video in json.loads(C0_JUDGMENTS_PATH.read_text(encoding="utf-8"))
        for question in video["questions"]
    }
    grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    overall_correct = 0
    observed_ids: set[str] = set()
    with Path(manifest["source_output"]).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            question = get_question_data(row)
            question_id = str(question["question_id"])
            if question_id not in val_ids:
                continue
            calls = by_id[question_id]
            group = category_for_calls(calls)
            prediction, _ = extract_pred_answer(get_turn_trace(row, question))
            correct = prediction == str(question["answer"]).strip().upper()
            add(grouped, group, correct)
            overall_correct += int(correct)
            observed_ids.add(question_id)
    if observed_ids != val_ids:
        raise ValueError("C0 full output does not cover exactly the Val125 IDs")
    return result_rows("C0 full benchmark", "complete Val125 trajectories", grouped, 125, overall_correct)


def c1_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for output_path in sorted(C1_ROOT.glob("candidate_*/output_test.json")):
        candidate = output_path.parent.name.removeprefix("candidate_")
        questions = [
            question
            for video in json.loads(output_path.read_text(encoding="utf-8"))
            for question in video["questions"]
        ]
        if len(questions) != 125:
            raise ValueError(f"{output_path}: expected 125 questions, got {len(questions)}")
        grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        overall_correct = 0
        for question in questions:
            correct = question["validation_score"] == 1.0
            overall_correct += int(correct)
            if question["trajectory_status"] != "saved":
                add(grouped, "not_saved_trajectory", correct)
            else:
                add(grouped, category_for_calls(question["ask_perception_judgments"]), correct)
        rows.extend(result_rows(
            f"C1 candidate {candidate}",
            "saved trajectories only; unsaved group has unknown call status",
            grouped,
            125,
            overall_correct,
        ))
    if len(rows) != 12 * 6:
        raise ValueError(f"expected 12 C1 candidates, found {len(rows) // 6}")
    return rows


def main() -> None:
    rows = c0_rows(read_val_ids()) + c1_rows()
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
