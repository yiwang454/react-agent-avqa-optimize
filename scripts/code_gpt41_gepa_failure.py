#!/usr/bin/env python3
"""Trace-code changed outcomes in the GPT-4.1 GEPA failure analysis.

This is conservative post-hoc coding. The judge sees question/options/gold,
both final predictions, and every available ReAct tool trace. It never sees
the video. Gemini Direct persisted only final responses, so its internal error
cause is explicitly treated as unobservable.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "scripts/experiment_records/gpt4_1_gepa_failure_analysis"
MODEL = "openai/gpt-5.5"
FAILURE_LABELS = {
    "premature_stop_or_caption_overreliance",
    "query_miss",
    "perception_conflict",
    "mapping_error",
    "overconstraint",
    "unresolved",
}
BENEFIT_LABELS = {
    "caption_based_correction",
    "better_query_targeting",
    "helpful_perception_evidence",
    "better_evidence_mapping",
    "robust_evidence_integration",
    "unresolved",
}

SYSTEM = """You are conducting a conservative post-hoc analysis of a changed
multiple-choice AVQA outcome. You cannot see the video. Use only the supplied
gold option, question/options, final predictions, and recorded ReAct traces.
The transition is either C->W (the AFTER system regressed) or W->C (the AFTER
system improved).

For C->W, assign exactly one primary_cause:
- premature_stop_or_caption_overreliance: AFTER finalized from the caption or
  partial evidence without obtaining a still-needed discriminative fact.
- query_miss: AFTER's perception query/observation misses the fact needed to
  separate gold from its chosen distractor, including wrong time/property or
  incomplete option comparison.
- perception_conflict: AFTER's perception observation explicitly favors its
  wrong option or conflicts with a defining gold-option fact. This is
  trace-relative, not a video-ground-truth assertion.
- mapping_error: AFTER obtains evidence favoring gold, or not favoring its
  chosen distractor, but maps that evidence to the wrong option.
- overconstraint: AFTER's instruction/query is unnecessarily narrow, loaded,
  or forced to a specific cue and plausibly excludes the needed evidence.
- unresolved: the available traces cannot distinguish a mechanism.

For W->C, assign exactly one primary_cause:
- caption_based_correction: AFTER becomes correct from caption evidence without
  needing a useful perception observation.
- better_query_targeting: AFTER asks a more discriminative or correctly
  anchored perception question than BEFORE.
- helpful_perception_evidence: AFTER receives observation content that directly
  supports gold or rules out the previous wrong option.
- better_evidence_mapping: similar evidence is available but AFTER maps it to
  the answer options more correctly.
- robust_evidence_integration: AFTER succeeds by reconciling caption and
  perception evidence, where no single source alone explains the improvement.
- unresolved: the traces do not support one of the above mechanisms.

When BEFORE is Gemini Direct majority, only three final responses are known.
Never invent its hidden reasoning or tool evidence; explain the transition
from the observable AFTER ReAct trace. Return ONLY one JSON object with:
primary_cause, secondary_causes (zero or more other allowed labels for that
direction), confidence (high/medium/low), and evidence (one concise sentence
grounded in the supplied record)."""


def load_cases(analysis_dir: Path) -> list[dict[str, Any]]:
    files = (
        analysis_dir / "candidate0_vs_candidate7_changed_cases.json",
        analysis_dir / "gemini_majority_vs_candidate0_changed_cases.json",
    )
    cases: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected a list")
        cases.extend(payload)
    if not cases:
        raise ValueError("no changed cases")
    keys = [(case["comparison"], case["question_id"]) for case in cases]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate comparison/question keys")
    return cases


def case_key(case: dict[str, Any]) -> str:
    return f"{case['comparison']}::{case['question_id']}"


def judge(case: dict[str, Any], api_key: str) -> dict[str, Any]:
    import litellm

    litellm.drop_params = True
    payload = {
        key: case[key]
        for key in (
            "comparison",
            "transition",
            "before_name",
            "after_name",
            "question_id",
            "task_type",
            "question",
            "options",
            "gold",
            "before",
            "after",
        )
    }
    response = litellm.completion(
        model=MODEL,
        api_key=api_key,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        max_tokens=500,
        timeout=180,
        reasoning_effort="none",
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    parsed = json.loads(raw)
    allowed = FAILURE_LABELS if case["transition"] == "C->W" else BENEFIT_LABELS
    primary = parsed.get("primary_cause")
    secondary = parsed.get("secondary_causes")
    confidence = parsed.get("confidence")
    evidence = parsed.get("evidence")
    if primary not in allowed:
        raise ValueError(f"invalid primary label: {parsed!r}")
    if not isinstance(secondary, list) or any(
        label not in allowed - {primary} for label in secondary
    ):
        raise ValueError(f"invalid secondary labels: {parsed!r}")
    if confidence not in {"high", "medium", "low"}:
        raise ValueError(f"invalid confidence: {parsed!r}")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError(f"missing evidence: {parsed!r}")
    return {
        "case_key": case_key(case),
        "comparison": case["comparison"],
        "question_id": case["question_id"],
        "task_type": case["task_type"],
        "transition": case["transition"],
        "primary_cause": primary,
        "secondary_causes": secondary,
        "confidence": confidence,
        "evidence": evidence.strip(),
        "raw": raw,
    }


def read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    saved: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return saved
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            saved[item["case_key"]] = item
    return saved


def write_aggregate(
    output_dir: Path, cases: list[dict[str, Any]], saved: dict[str, dict[str, Any]]
) -> None:
    ordered = [saved[case_key(case)] for case in cases]
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    task_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    confidence_counts: Counter[str] = Counter()
    for item in ordered:
        group = (item["comparison"], item["transition"])
        counts[group][item["primary_cause"]] += 1
        task_counts[(item["comparison"], item["transition"], item["task_type"])][
            item["primary_cause"]
        ] += 1
        confidence_counts[item["confidence"]] += 1

    aggregate = {
        "judge_model": MODEL,
        "coded_cases": len(ordered),
        "confidence_counts": dict(confidence_counts),
        "groups": {
            f"{comparison}::{direction}": {
                "cases": sum(group_counts.values()),
                "primary_counts": dict(sorted(group_counts.items())),
                "primary_percentages": {
                    label: round(100 * count / sum(group_counts.values()), 1)
                    for label, count in sorted(group_counts.items())
                },
            }
            for (comparison, direction), group_counts in sorted(counts.items())
        },
        "cases": ordered,
    }
    (output_dir / "trace_coding.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    case_lookup = {case_key(case): case for case in cases}
    lines = [
        "# Trace-supported post-hoc coding",
        "",
        f"Judge: `{MODEL}`. Coded population: all `{len(ordered)}` changed-outcome cases from both comparisons.",
        "",
        "The judge did not see video. `perception_conflict` is trace-relative, and Gemini Direct error mechanisms are unobservable because those artifacts persist only final responses.",
        "",
    ]
    for (comparison, direction), group_counts in sorted(counts.items()):
        total = sum(group_counts.values())
        lines.extend(
            [
                f"## {comparison}: {direction}",
                "",
                "| Primary mechanism | Cases | Share | Representative IDs |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        group_items = [
            item
            for item in ordered
            if item["comparison"] == comparison and item["transition"] == direction
        ]
        for label, count in group_counts.most_common():
            ids = [item["question_id"] for item in group_items if item["primary_cause"] == label][
                :3
            ]
            lines.append(
                f"| {label} | {count} | {100 * count / total:.1f}% | "
                + ", ".join(f"`{qid}`" for qid in ids)
                + " |"
            )
        lines.extend(
            [
                "",
                "### By task type",
                "",
                "| Task type | Changed cases | Primary counts |",
                "| --- | ---: | --- |",
            ]
        )
        task_names = sorted(
            task
            for comp, trans, task in task_counts
            if comp == comparison and trans == direction
        )
        for task in task_names:
            values = task_counts[(comparison, direction, task)]
            rendered = "; ".join(f"{label}={count}" for label, count in values.most_common())
            lines.append(f"| {task} | {sum(values.values())} | {rendered} |")
        lines.append("")

    lines.extend(["## Case evidence", ""])
    for ordinal, item in enumerate(ordered, 1):
        case = case_lookup[item["case_key"]]
        lines.extend(
            [
                f"### {ordinal}. `{item['question_id']}` — {item['comparison']} {item['transition']}",
                "",
                f"- Task: {item['task_type']}",
                f"- Gold: `{case['gold']}`; before: `{case['before']['prediction']}`; after: `{case['after']['prediction']}`",
                f"- Coding: `{item['primary_cause']}` ({item['confidence']})",
                f"- Evidence: {item['evidence']}",
                "",
            ]
        )
    (output_dir / "trace_coding.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or args.limit is not None and args.limit < 1:
        parser.error("--workers and --limit must be positive")
    api_key = os.environ.get("ELM_API_KEY", "").strip()
    if not api_key:
        parser.error("ELM_API_KEY is required")

    analysis_dir = args.analysis_dir.expanduser().resolve()
    output_dir = (args.output_dir or analysis_dir / "trace_coding").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(analysis_dir)
    checkpoint = output_dir / "coding_checkpoint.jsonl"
    saved = read_checkpoint(checkpoint)
    pending = [case for case in cases if case_key(case) not in saved]
    if args.limit is not None:
        pending = pending[: args.limit]

    with checkpoint.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = {pool.submit(judge, case, api_key): case for case in pending}
        for future in as_completed(futures):
            item = future.result()
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            saved[item["case_key"]] = item
            print(f"coded {len(saved)}/{len(cases)}: {item['case_key']}")

    if len(saved) != len(cases):
        print(f"checkpointed {len(saved)}/{len(cases)}; aggregate waits for complete coding")
        return 0
    write_aggregate(output_dir, cases, saved)
    print(f"wrote coding aggregate to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
