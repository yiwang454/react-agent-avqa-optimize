#!/usr/bin/env python3
"""Classify saved ReAct ask_perception questions with the ELM GPT backend.

Run through scripts/run_ask_perception_question_judge.sh to load the same
ELM_API_KEY and Python environment used by the ReAct experiment launchers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/mnt/ceph_rbd/data/avqa_project/daily_omni")
RUNS = {
    "c0": "daily_omni_dspy_free_react_gpt4_1_gemini_first_cached_then_live_caption_maxturns4",
    "gpt41_none": "daily_omni_dspy_GPT_v8GeminiCaptionInTask_planner_gpt-4.1_seed_1234",
    "gpt41_g0_planner": "daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_planner_workflow_prompt_planner_gpt-4.1_seed_1234_gepa_seed18",
    "gpt41_g0_captioner": "daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_captioner_default_caption_instruction_gpt4_1_0721_planner_gpt-4.1_seed_1234_gepa_seed18",
    "gpt41_g0_planner_captioner": "daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_planner_workflow_prompt_and_captioner_default_caption_instruction_gpt4_1_0721_planner_gpt-4.1_seed_1234_gepa_seed18",
    # The seed18 result is described as "prior C1 selected-c0" and counted
    # under C0 in the current experiment ledger. Select it explicitly only.
    "c1_prior_seed18": "daily_omni_dspy_free_react_gpt4_1_gemini_first_cached_then_live_caption_gepa_planner_maxturns4_seed18_calls2500",
}
DEFAULT_RUNS = tuple(name for name in RUNS if name != "c1_prior_seed18")
MODEL = "openai/gpt-5.5"  # same LiteLLM/OpenAI route as dspy_avqa.context

# Category 1 means asking the perception model to choose or return the
# original multiple-choice answer. Merely mentioning option content is not
# enough: a two-option paraphrase is category 2 under the requested rubric.
OPTION_ANSWER_PATTERNS = (
    re.compile(r"\b(?:which|what|select|choose|pick|identify|determine|give|return|provide|state|output|reply|respond)(?:\s+\w+){0,5}\s+(?:option|choice|answer\s+choice|letter)\b", re.I),
    re.compile(r"\b(?:correct|best|matching|right)\s+(?:option|choice|answer\s+choice|letter)\b", re.I),
    re.compile(r"\b(?:answer|respond|reply|output|return)\s+(?:with|as|using|in)\s+(?:the\s+)?(?:option|choice|letter|[A-F](?:\s*[,/]|\s+or\s+)[A-F])\b", re.I),
    re.compile(r"\b(?:option|choice)\s+[A-F]\s*(?:,|or|versus|vs\.?|/|and)\s*(?:option\s+|choice\s+)?[A-F]\b", re.I),
    re.compile(r"\b(?:A|B|C|D|E|F)\s*[,/]\s*(?:A|B|C|D|E|F)\s*[,/]\s*(?:A|B|C|D|E|F)\b", re.I),
)

SYSTEM_PROMPT = """You classify a ReAct planner's ask_perception question relative to its ORIGINAL multiple-choice QA. Classify the intent of the tool QUESTION, not the tool observation or the correctness of the final answer. Use exactly one category:
1: The tool question asks the perception model to choose/return an original answer option or option letter. This includes presenting the original options and asking which is correct/matches. Mere mention of option-like content without requesting an option/letter is not enough.
2: The tool question still asks for the original answer target, but paraphrases it while narrowing it or adding specifications, such as asking about only two of four option descriptions, or asking for the timestamp of the event in the original question. The requested observation directly resolves the original question.
3: The tool question breaks the original task into a smaller prerequisite or intermediate evidence question. The requested fact alone does not directly answer the original QA; further combination or comparison is needed.
4: Other cases, including unrelated, generic, malformed, or questions that do not fit 1-3.
Priority: 1 before 2; 2 before 3. A question can mention two option descriptions and still be 2 if it does not ask for an option label. A focused timestamp of the original event is 2. Explain the decisive relationship in one brief sentence, grounded in the original QA and tool question. Do not infer facts from the video.
Return only a JSON object with integer category (1-4) and string evidence."""


def regex_category_one(question: str) -> str | None:
    for pattern in OPTION_ANSWER_PATTERNS:
        match = pattern.search(question)
        if match:
            return match.group(0)
    # A rendered multiple-choice list plus an explicit selection request.
    labels = re.findall(r"(?:^|\s)([A-F])[.)]\s+", question)
    if len(set(labels)) >= 2 and re.search(
        r"\b(?:which\s+of\s+(?:the\s+)?following|which\s+(?:one|statement|event|answer)\s+(?:is|matches|occurred)|select|choose|pick)\b",
        question, re.I,
    ):
        return "multiple labeled choices with a selection request"
    return None


def source_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("question_data"), dict):
                raise ValueError(f"{path}:{lineno}: expected question_data object")
            rows.append(row)
    if len(rows) != 1197:
        raise ValueError(f"{path}: expected complete 1197-row run; got {len(rows)}")
    ids = [str(row["question_data"].get("question_id") or "") for row in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: missing or duplicate question_id")
    return rows


def extract_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    qd = row["question_data"]
    calls = []
    for index, turn in enumerate(qd.get("turn_trace") or [], 1):
        if not isinstance(turn, dict) or turn.get("tool_name") != "ask_perception":
            continue
        args = turn.get("tool_args") or {}
        question = args.get("perceptual_question") if isinstance(args, dict) else None
        calls.append({
            "turn_id": turn.get("turn_id", index),
            "perceptual_question": question if isinstance(question, str) else "",
        })
    return calls


def judge(question: str, original: dict[str, Any], api_key: str, retries: int) -> dict[str, Any]:
    match = regex_category_one(question)
    if match:
        return {
            "category": 1,
            "evidence": f"The tool question requests an original answer option: {match}.",
            "method": "regex",
            "status": "ok",
        }
    if not question.strip():
        return {"category": None, "evidence": "", "method": None, "status": "missing_perceptual_question"}

    # This is the same native LiteLLM OpenAI-compatible route used by
    # dspy_avqa.context._configure_native_litellm for provider=elm_gpt.
    import litellm

    litellm.drop_params = True
    payload = {
        "original_question": original["question"],
        "original_options": original["options"],
        "ask_perception_question": question,
    }
    last_error = ""
    for _ in range(retries + 1):
        try:
            response = litellm.completion(
                model=MODEL,
                api_key=api_key,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=512,
                timeout=180,
                reasoning_effort="none",
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            if not raw.strip():
                raise ValueError(f"empty judge content (finish_reason={response.choices[0].finish_reason})")
            parsed = json.loads(raw)
            category = parsed.get("category")
            evidence = parsed.get("evidence")
            if isinstance(category, bool) or category not in (1, 2, 3, 4):
                raise ValueError(f"invalid category: {category!r}")
            if not isinstance(evidence, str) or not evidence.strip():
                raise ValueError("empty evidence")
            usage = response.usage
            return {
                "category": category,
                "evidence": evidence.strip(),
                "method": "gpt-5.5",
                "status": "ok",
                "judge_response": raw,
                "judge_usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return {"category": None, "evidence": "", "method": "gpt-5.5", "status": "judge_error", "error": last_error}


def load_checkpoint(path: Path, source_path: Path) -> dict[str, dict[str, Any]]:
    saved: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("source_output") != str(source_path):
                    raise ValueError(f"{path}:{lineno}: checkpoint source mismatch")
                saved[str(item["question_id"])] = item
    return saved


def write_final(path: Path, source: Path, name: str, rows: list[dict[str, Any]], saved: dict[str, dict[str, Any]]) -> None:
    grouped: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    calls_count = 0
    for row in rows:
        qd = row["question_data"]
        qid = str(qd["question_id"])
        item = saved.get(qid)
        if item is None:
            continue
        video_id = str(row.get("video_id") or qd.get("recording_id") or "")
        if not video_id:
            raise ValueError(f"missing video_id for {qid}")
        entry = grouped.setdefault(video_id, {
            "video_id": video_id,
            "metadata": row.get("metadata") or {},
            "questions": [],
        })
        question_record = {
            "question_id": qid,
            "task_type": qd.get("task_type"),
            "question": qd.get("question"),
            "options": qd.get("options"),
            "answer": qd.get("answer"),
            "response": qd.get("response"),
            "ask_perception_judgments": item["ask_perception_judgments"],
        }
        if "trajectory_status" in qd:
            question_record["trajectory_status"] = qd["trajectory_status"]
        if "validation_score" in qd:
            question_record["validation_score"] = qd["validation_score"]
        if "validation_index" in qd:
            question_record["validation_index"] = qd["validation_index"]
        entry["questions"].append(question_record)
        for call in item["ask_perception_judgments"]:
            calls_count += 1
            counts[str(call.get("category") or call["status"])] += 1
    output = path / "output_test.json"
    temp = output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(list(grouped.values()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(output)
    manifest = {
        "experiment": name,
        "source_output": str(source),
        "judge_model": MODEL,
        "source_question_count": len(rows),
        "completed_question_count": len(saved),
        "ask_perception_call_count": calls_count,
        "counts": dict(counts),
        "output": str(output),
    }
    (path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def judge_row(row: dict[str, Any], prior_calls: list[dict[str, Any]], api_key: str, retries: int, source: Path) -> dict[str, Any]:
    qd = row["question_data"]
    calls = extract_calls(row)
    judgments = []
    for index, call in enumerate(calls):
        prior = prior_calls[index] if index < len(prior_calls) else None
        if prior and prior["status"] == "ok" and prior["perceptual_question"] == call["perceptual_question"]:
            judgments.append(prior)
            continue
        result = judge(call["perceptual_question"], qd, api_key, retries)
        judgments.append({**call, **result})
    return {
        "source_output": str(source),
        "video_id": row.get("video_id"),
        "question_id": str(qd["question_id"]),
        "ask_perception_judgments": judgments,
    }


def run_one(name: str, args: argparse.Namespace, api_key: str) -> int:
    source = DATA_ROOT / RUNS[name] / "output_test.jsonl"
    rows = source_rows(source)
    calls_total = sum(len(extract_calls(row)) for row in rows)
    print(f"{name}: {len(rows)} source questions, {calls_total} ask_perception calls, source={source}", flush=True)
    if args.dry_run:
        regex_count = sum(bool(regex_category_one(call["perceptual_question"])) for row in rows for call in extract_calls(row))
        print(f"{name}: regex category-1 calls={regex_count}; GPT calls={calls_total - regex_count}", flush=True)
        return 0

    target = args.output_root / name
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "judgments_checkpoint.jsonl"
    saved = load_checkpoint(checkpoint, source)
    pending = []
    for row in rows:
        qid = str(row["question_data"]["question_id"])
        calls = extract_calls(row)
        previous = saved.get(qid)
        prior_calls = previous["ask_perception_judgments"] if previous else []
        if previous is not None and len(prior_calls) == len(calls) and all(call["status"] == "ok" for call in prior_calls):
            continue
        pending.append((row, prior_calls))
    if args.limit is not None:
        pending = pending[:args.limit]

    with checkpoint.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(judge_row, row, prior_calls, api_key, args.retries, source)
                for row, prior_calls in pending
            ]
            for processed, future in enumerate(as_completed(futures), 1):
                item = future.result()
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                saved[item["question_id"]] = item
                if processed % 25 == 0:
                    print(f"{name}: processed {processed} new/retried questions; saved {len(saved)}/{len(rows)}", flush=True)
    write_final(target, source, name, rows, saved)
    errors = sum(call["status"] != "ok" for item in saved.values() for call in item["ask_perception_judgments"])
    print(f"{name}: saved {len(saved)}/{len(rows)} questions; judgment errors={errors}; output={target / 'output_test.json'}", flush=True)
    return 0 if len(saved) == len(rows) and errors == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(RUNS), action="append", help="Repeat for specific experiments; default is C0 and the four GPT-4.1 runs. Prior C1 seed18 requires explicit selection.")
    parser.add_argument("--output-root", type=Path, default=DATA_ROOT / "ask_perception_question_judge_gpt5_5")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and count regex/GPT calls without using the API")
    parser.add_argument("--limit", type=int, help="Process at most this many new/retried questions per experiment")
    parser.add_argument("--retries", type=int, default=2, help="Additional attempts for API/format errors")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent question judgments (default: 8)")
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or args.retries < 0 or args.workers < 1:
        parser.error("--limit and --workers must be positive; --retries must be nonnegative")
    import os

    key = os.environ.get("ELM_API_KEY", "").strip()
    if not args.dry_run and not key:
        parser.error("ELM_API_KEY is required for live judging")
    return max(run_one(name, args, key) for name in args.experiment or DEFAULT_RUNS)


if __name__ == "__main__":
    sys.exit(main())
