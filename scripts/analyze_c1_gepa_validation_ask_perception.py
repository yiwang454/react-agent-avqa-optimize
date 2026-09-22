#!/usr/bin/env python3
"""Judge saved C1 seed18 GEPA Val125 trajectories, one candidate at a time.

Only unpickle the trusted, local GEPA state from the cited experiment run.
The state preserves 125 scores per candidate but not 125 predictions per
candidate; missing predictions are marked rather than treated as no tool call.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "DSPy"))  # required to unpickle local DSPy predictions

from analyze_ask_perception_questions import (  # noqa: E402
    DATA_ROOT,
    MODEL,
    extract_calls,
    judge_row,
    load_checkpoint,
    regex_category_one,
    write_final,
)


RUN_DIR = DATA_ROOT / "daily_omni_dspy_free_react_gpt4_1_gemini_first_cached_then_live_caption_gepa_planner_maxturns4_seed18_calls2500"
STATE_PATH = RUN_DIR / "gepa_logs/gepa_state.bin"
METADATA_PATH = RUN_DIR / "compiled_gepa_metadata.json"
CANDIDATES_PATH = RUN_DIR / "planner_workflow_prompt_candidates.jsonl"
DEFAULT_OUTPUT = DATA_ROOT / "ask_perception_question_judge_gpt5_5/c1_gepa_val125"


def validation_score_value(score: Any) -> float:
    """Support historical float scores and current QwenValidationScore objects."""
    return float(getattr(score, "value", score))


def validation_score_included(score: Any) -> bool:
    return bool(getattr(score, "included", True))


def load_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    val_path = Path(metadata["valset_jsonl"])
    val_rows = [json.loads(line) for line in val_path.open(encoding="utf-8") if line.strip()]
    candidates = [json.loads(line) for line in CANDIDATES_PATH.open(encoding="utf-8") if line.strip()]
    with STATE_PATH.open("rb") as handle:
        state = pickle.load(handle)
    if not isinstance(state, dict):
        raise TypeError("GEPA state must be a mapping")
    if len(val_rows) != 125 or len(candidates) != len(state["prog_candidate_val_subscores"]):
        raise ValueError("Val125 or candidate count differs from saved GEPA state")
    if set(state["best_outputs_valset"]) != set(range(125)):
        raise ValueError("Saved-output validation indices differ from Val125")
    if sorted(x["candidate_index"] for x in candidates) != list(range(len(candidates))):
        raise ValueError("Candidate index sequence is incomplete")
    return state, val_rows, candidates, metadata


def candidate_rows(state: dict[str, Any], val_rows: list[dict[str, Any]], candidate_index: int) -> list[dict[str, Any]]:
    predictions = {}
    for val_index, candidate_predictions in state["best_outputs_valset"].items():
        for index, prediction in candidate_predictions:
            if index == candidate_index:
                if val_index in predictions:
                    raise ValueError(f"duplicate candidate {candidate_index} prediction at Val index {val_index}")
                predictions[val_index] = prediction
    subscores = state["prog_candidate_val_subscores"][candidate_index]
    if set(subscores) != set(range(125)):
        raise ValueError(f"candidate {candidate_index}: missing validation scores")

    rows = []
    for index, val in enumerate(val_rows):
        supervision = val["supervisions"][0]
        custom = supervision["custom"]
        qid = str(supervision["id"])
        if qid != str(val["id"]):
            raise ValueError(f"validation ID mismatch at index {index}")
        prediction = predictions.get(index)
        trace = list(getattr(prediction, "turn_trace", []) or []) if prediction is not None else []
        score = subscores[index]
        if not validation_score_included(score):
            raise ValueError(f"candidate {candidate_index}: excluded Val score at index {index}")
        rows.append({
            "video_id": qid,
            "metadata": {
                "video_id": qid,
                "recording_id": supervision.get("recording_id"),
                "val_index": index,
            },
            "question_data": {
                "question_id": qid,
                "task_type": custom.get("task_type"),
                "question": supervision["text"],
                "options": custom["options"],
                "answer": custom["answer"],
                "response": getattr(prediction, "answer", None) if prediction is not None else None,
                "turn_trace": trace,
                "trajectory_status": "saved" if prediction is not None else "not_saved",
                "validation_index": index,
                "validation_score": validation_score_value(score),
            },
        })
    if len({row["question_data"]["question_id"] for row in rows}) != 125:
        raise ValueError("Val125 contains duplicate question IDs")
    return rows


def classify_candidate(candidate_index: int, rows: list[dict[str, Any]], candidate: dict[str, Any], args: argparse.Namespace, key: str) -> bool:
    name = f"candidate_{candidate_index:02d}"
    saved_count = sum(row["question_data"]["trajectory_status"] == "saved" for row in rows)
    calls = sum(len(extract_calls(row)) for row in rows)
    regex_count = sum(bool(regex_category_one(call["perceptual_question"])) for row in rows for call in extract_calls(row))
    print(f"{name}: saved trajectories={saved_count}/125; ask_perception calls={calls}; regex={regex_count}; GPT={calls-regex_count}", flush=True)
    if args.dry_run:
        return True

    target = args.output_root / name
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "judgments_checkpoint.jsonl"
    saved = load_checkpoint(checkpoint, STATE_PATH)
    pending = []
    for row in rows:
        qid = row["question_data"]["question_id"]
        previous = saved.get(qid)
        prior_calls = previous["ask_perception_judgments"] if previous else []
        calls_for_row = extract_calls(row)
        if previous is not None and len(prior_calls) == len(calls_for_row) and all(x["status"] == "ok" for x in prior_calls):
            continue
        pending.append((row, prior_calls))
    if args.limit is not None:
        pending = pending[:args.limit]

    with checkpoint.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(judge_row, row, prior, key, args.retries, STATE_PATH) for row, prior in pending]
            for processed, future in enumerate(as_completed(futures), 1):
                item = future.result()
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                saved[item["question_id"]] = item
                if processed % 25 == 0:
                    print(f"{name}: processed {processed}; saved {len(saved)}/125", flush=True)

    write_final(target, STATE_PATH, name, rows, saved)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "candidate_index": candidate_index,
        "discovery_eval_count": candidate.get("discovery_eval_count"),
        "full_valset_score": candidate["full_valset_score"],
        "val_correct": round(125 * candidate["full_valset_score"]),
        "saved_trajectory_count": saved_count,
        "missing_trajectory_count": 125 - saved_count,
        "missing_trajectory_question_ids": [row["question_data"]["question_id"] for row in rows if row["question_data"]["trajectory_status"] == "not_saved"],
        "valset_jsonl": str(Path(json.loads(METADATA_PATH.read_text())["valset_jsonl"])),
        "candidates_jsonl": str(CANDIDATES_PATH),
        "state_file": str(STATE_PATH),
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors = sum(call["status"] != "ok" for item in saved.values() for call in item["ask_perception_judgments"])
    print(f"{name}: saved {len(saved)}/125; errors={errors}", flush=True)
    return len(saved) == 125 and errors == 0


def write_trend(output_root: Path, candidate_count: int) -> None:
    rows = []
    for index in range(candidate_count):
        target = output_root / f"candidate_{index:02d}"
        manifest_path = target / "manifest.json"
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["completed_question_count"] != 125 or any(k not in ("1", "2", "3", "4") for k in manifest["counts"]):
            return
        output = json.loads((target / "output_test.json").read_text(encoding="utf-8"))
        questions = [question for video in output for question in video["questions"]]
        if len(questions) != 125 or any(call["status"] != "ok" for q in questions for call in q["ask_perception_judgments"]):
            return
        saved_rows = [q for q in questions if q["trajectory_status"] == "saved"]
        no_call = sum(not q["ask_perception_judgments"] for q in saved_rows)
        calls = manifest["ask_perception_call_count"]
        row = {
            "candidate_index": index,
            "discovery_eval_count": manifest["discovery_eval_count"],
            "full_val_correct": manifest["val_correct"],
            "full_val_total": 125,
            "full_val_accuracy": manifest["full_valset_score"],
            "saved_trajectories": len(saved_rows),
            "missing_trajectories": 125 - len(saved_rows),
            "saved_trajectory_coverage": len(saved_rows) / 125,
            "saved_trajectories_no_call": no_call,
            "observed_ask_perception_calls": calls,
        }
        for category in range(1, 5):
            count = int(manifest["counts"].get(str(category), 0))
            row[f"category_{category}_count"] = count
            row[f"category_{category}_share_of_observed_calls"] = count / calls if calls else None
        rows.append(row)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "trend.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "trend.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved trend across {candidate_count} validation candidates to {output_root / 'trend.csv'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=int, action="append", help="Candidate index to classify; repeat for several (default: all)")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or args.workers < 1 or args.retries < 0:
        parser.error("--limit and --workers must be positive; --retries must be nonnegative")
    key = os.environ.get("ELM_API_KEY", "").strip()
    if not args.dry_run and not key:
        parser.error("ELM_API_KEY is required for live judging")

    state, val_rows, candidates, _ = load_inputs()
    selected = args.candidate or list(range(len(candidates)))
    if any(index < 0 or index >= len(candidates) for index in selected):
        parser.error(f"--candidate must be between 0 and {len(candidates)-1}")
    complete = True
    for index in selected:
        rows = candidate_rows(state, val_rows, index)
        complete = classify_candidate(index, rows, candidates[index], args, key) and complete
    if not args.dry_run:
        write_trend(args.output_root, len(candidates))
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
