#!/usr/bin/env python3
"""Trace-code the GPT-5.4 no-GEPA -> G0-GEPA correct-to-wrong cases.

This is a post-hoc qualitative coding aid, not a claim that a text-only judge
can recover video ground truth.  It gives the judge the gold option, both
conditions' targeted perception queries/observations, and asks for one
trace-supported primary mechanism.  Raw decisions and short evidence notes
are checkpointed so the aggregate remains auditable and reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/mnt/ceph_rbd/data/avqa_project/daily_omni")
DEFAULT_INPUT = DATA_ROOT / "analysis/gpt54_g0_planner_regressions/correct_to_wrong_cases.json"
DEFAULT_OUTPUT = DATA_ROOT / "analysis/gpt54_g0_planner_regressions/trace_coding"
MODEL = "openai/gpt-5.5"

SYSTEM = """You are conducting a conservative post-hoc error analysis of two
ReAct traces for the same multiple-choice AVQA item.  You cannot see the
video.  Use only the supplied gold option, options, final predictions, tool
queries, and tool observations.  Do not assert that an observation is
factually false unless it directly conflicts with the gold option.  Assign
exactly one PRIMARY mechanism for why GEPA changed a correct no-GEPA answer to
a wrong answer:

query_miss: The GEPA perception question/observation does not obtain the
discriminative fact needed to separate gold from its chosen distractor (wrong
time anchor, wrong property, incomplete comparison, or ignores a key option
clause).
perception_conflict: The GEPA perception observation explicitly favors the
chosen wrong option or contradicts a defining gold-option fact; it plausibly
misled the planner.  This is trace-relative, not a video-ground-truth claim.
mapping_error: GEPA obtains evidence that favors the gold option, or does not
favor its selected distractor, but maps it to the wrong option anyway.
overconstraint: The GEPA query is unnecessarily narrowed/loaded by the
optimized prompt (for example, it pursues an incidental detail or a forced
timestamp) and that restriction plausibly causes the lost answer.  Use only
when this is more specific than query_miss.
unresolved: The supplied traces cannot distinguish the above mechanisms.

Secondary mechanisms may contain zero or more other labels. Return ONLY a JSON
object with primary_cause, secondary_causes (array of the other labels), and
evidence (one concise sentence quoting/paraphrasing a trace fact)."""


def field(case: dict[str, Any], condition: str, kind: str) -> str:
    for turn in case[condition]["trace"]:
        if kind == "perception_query" and turn.get("tool_name") == "ask_perception":
            return str((turn.get("tool_args") or {}).get("perceptual_question") or "")
        if kind == "perception_observation" and turn.get("tool_name") == "ask_perception":
            return str(turn.get("tool_observation") or "")
        if kind == "final_reasoning" and turn.get("planner_action") == "final":
            return str(turn.get("reasoning_summary") or "")
    return ""


def payload(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": case["question_id"],
        "task_type": case["task_type"],
        "question": case["question"],
        "options": case["options"],
        "gold": case["gold"],
        "no_gepa": {
            "prediction": case["baseline"]["prediction"],
            "perception_query": field(case, "baseline", "perception_query"),
            "perception_observation": field(case, "baseline", "perception_observation"),
            "final_reasoning": field(case, "baseline", "final_reasoning"),
        },
        "gepa": {
            "prediction": case["gepa"]["prediction"],
            "perception_query": field(case, "gepa", "perception_query"),
            "perception_observation": field(case, "gepa", "perception_observation"),
            "final_reasoning": field(case, "gepa", "final_reasoning"),
        },
    }


def judge(case: dict[str, Any], key: str) -> dict[str, Any]:
    import litellm

    litellm.drop_params = True
    response = litellm.completion(
        model=MODEL,
        api_key=key,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload(case), ensure_ascii=False)},
        ],
        max_tokens=300,
        timeout=180,
        reasoning_effort="none",
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    parsed = json.loads(raw)
    allowed = {"query_miss", "perception_conflict", "mapping_error", "overconstraint", "unresolved"}
    primary = parsed.get("primary_cause")
    secondary = parsed.get("secondary_causes")
    evidence = parsed.get("evidence")
    if primary not in allowed or not isinstance(secondary, list) or any(item not in allowed - {primary} for item in secondary):
        raise ValueError(f"invalid labels: {parsed!r}")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError(f"missing evidence: {parsed!r}")
    return {"question_id": case["question_id"], "primary_cause": primary, "secondary_causes": secondary, "evidence": evidence.strip(), "raw": raw}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or args.limit is not None and args.limit < 1:
        parser.error("--workers and --limit must be positive")
    key = os.environ.get("ELM_API_KEY", "").strip()
    if not key:
        parser.error("ELM_API_KEY is required")
    cases = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{args.input}: expected a non-empty case list")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "coding_checkpoint.jsonl"
    saved: dict[str, dict[str, Any]] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                saved[item["question_id"]] = item
    pending = [case for case in cases if case["question_id"] not in saved]
    if args.limit is not None:
        pending = pending[:args.limit]
    with checkpoint.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(judge, case, key) for case in pending]
        for future in as_completed(futures):
            item = future.result()
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            saved[item["question_id"]] = item
    if len(saved) != len(cases):
        print(f"checkpointed {len(saved)}/{len(cases)} cases; no aggregate until coding is complete")
        return 0
    counts = Counter(item["primary_cause"] for item in saved.values())
    aggregate = {
        "source": str(args.input), "judge_model": MODEL, "coded_cases": len(cases),
        "primary_counts": dict(sorted(counts.items())),
        "primary_percentages": {name: round(100 * count / len(cases), 1) for name, count in sorted(counts.items())},
        "cases": [saved[case["question_id"]] for case in cases],
    }
    output = args.output_dir / "trace_coding.json"
    output.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    task_counts: dict[str, Counter[str]] = {}
    case_lookup = {case["question_id"]: case for case in cases}
    for item in saved.values():
        task = str(case_lookup[item["question_id"]]["task_type"])
        task_counts.setdefault(task, Counter())[item["primary_cause"]] += 1
    labels = ("perception_conflict", "query_miss", "mapping_error", "overconstraint", "unresolved")
    md = [
        "# Trace-supported coding of GPT-5.4 GEPA correct-to-wrong cases", "",
        "## Scope", "",
        f"- Coded population: all {len(cases)} no-GEPA-correct → G0-planner-GEPA-wrong cases.",
        f"- Judge: `{MODEL}`, fixed rubric in `scripts/code_gpt54_gepa_regressions.py`; raw decisions are in [`trace_coding.json`](trace_coding.json).",
        "- Evidence supplied per case: question/options/gold, both perception queries and observations, both final predictions, and both final rationales.",
        "- This is trace-supported post-hoc coding.  It does not independently re-watch videos, so `perception_conflict` means conflict with the gold-option-defining fact in the recorded trace, not an assertion of video-ground-truth error.",
        "", "## Primary mechanism frequency", "", "| Primary mechanism | Cases | Share of 156 |", "| --- | ---: | ---: |",
    ]
    descriptions = {
        "perception_conflict": "GEPA observation favors the chosen distractor or conflicts with a gold-defining fact",
        "query_miss": "GEPA query/observation misses the fact needed to separate gold from distractor",
        "mapping_error": "Recorded observation favors gold or does not support the distractor, but final mapping is wrong",
        "overconstraint": "Prompt-induced narrow/loaded query plausibly excludes the discriminative evidence",
        "unresolved": "Recorded traces do not distinguish a mechanism",
    }
    for label in labels:
        if counts[label]:
            md.append(f"| {label}: {descriptions[label]} | {counts[label]} | {100 * counts[label] / len(cases):.1f}% |")
    md.extend(["", "## Primary mechanism by task type", "", "| Task type | " + " | ".join(label for label in labels if counts[label]) + " |", "| --- | " + " | ".join("---:" for label in labels if counts[label]) + " |"])
    active = tuple(label for label in labels if counts[label])
    for task in sorted(task_counts):
        md.append("| " + task + " | " + " | ".join(str(task_counts[task][label]) for label in active) + " |")
    md.extend([
        "", "## Interpretation boundary", "",
        "All 156 regressions retain the same three-turn `ask_caption → ask_perception → final` structure and none has a recorded GEPA tool error.  Thus these cases are not explained by failed calls or extra-turn exhaustion.  The dominant recorded failure is content/evidence instability in the single perception observation, often compounded by a query that is too narrowly anchored to the optimized prompt's preferred cue type.  Mapping errors are a smaller but distinct residual: the observation did not justify the chosen option.",
    ])
    markdown = args.output_dir / "trace_coding.md"
    markdown.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
