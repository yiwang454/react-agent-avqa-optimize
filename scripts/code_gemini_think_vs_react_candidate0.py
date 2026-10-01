#!/usr/bin/env python3
"""Code all changed Gemini +think/ReAct outcomes and write a concise report."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = REPO_ROOT / (
    "scripts/experiment_records/gemini_think_vs_react_candidate0_analysis"
)
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

SYSTEM = """You are conducting conservative post-hoc analysis of a paired
multiple-choice audio-visual QA outcome. Gemini Direct +think is BEFORE and
ReAct candidate 0 is AFTER. You cannot inspect the video. You may use the
question, options, gold answer, Gemini's separately persisted thought and
visible response, and ReAct's recorded caption/perception trace.

For C->W (Gemini correct, ReAct wrong), assign exactly one primary_cause:
- premature_stop_or_caption_overreliance: ReAct finalized from caption or
  incomplete evidence despite a material unresolved distinction.
- query_miss: ReAct's perception query/observation did not obtain the fact
  needed to distinguish gold from the chosen distractor.
- perception_conflict: ReAct's recorded perception observation explicitly
  conflicts with a gold-defining fact or favors its wrong choice. This is a
  trace-relative description, not independently verified video truth.
- mapping_error: ReAct had evidence supporting gold or undermining its chosen
  distractor but mapped the evidence to the wrong option.
- overconstraint: ReAct used an unnecessarily narrow/loaded question that
  plausibly excluded or distorted the needed evidence.
- unresolved: the available evidence cannot distinguish a mechanism.

For W->C (Gemini wrong, ReAct correct), assign exactly one primary_cause:
- caption_based_correction: ReAct becomes correct mainly from caption evidence.
- better_query_targeting: ReAct asks a more discriminative or better anchored
  question than the approach visible in Gemini's thought.
- helpful_perception_evidence: ReAct's observation directly supports gold or
  rules out Gemini's wrong answer.
- better_evidence_mapping: ReAct maps similar available evidence to the answer
  options more correctly than Gemini.
- robust_evidence_integration: ReAct succeeds by reconciling caption and
  perception evidence, with no single source alone explaining the improvement.
- unresolved: none of the mechanisms is supported clearly enough.

Do not treat Gemini thought or ReAct observations as verified video truth.
Return only one JSON object with primary_cause, secondary_causes, confidence
(high/medium/low), evidence (one concise English sentence), and summary_cn
(one short Chinese sentence describing what Gemini and ReAct did differently
in this concrete example)."""


def load_cases(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: expected a nonempty list")
    keys = [str(item.get("question_id") or "") for item in value]
    if not all(keys) or len(keys) != len(set(keys)):
        raise ValueError(f"{path}: missing or duplicate question IDs")
    return value


def case_key(case: dict[str, Any]) -> str:
    return f"{case['comparison']}::{case['question_id']}"


def judge(case: dict[str, Any], api_key: str) -> dict[str, Any]:
    import litellm

    litellm.drop_params = True
    response = litellm.completion(
        model=MODEL,
        api_key=api_key,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(case, ensure_ascii=False)},
        ],
        max_tokens=700,
        timeout=180,
        reasoning_effort="none",
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    parsed = json.loads(raw)
    allowed = FAILURE_LABELS if case["transition"] == "C->W" else BENEFIT_LABELS
    primary = parsed.get("primary_cause")
    secondary_raw = parsed.get("secondary_causes")
    confidence = parsed.get("confidence")
    evidence = parsed.get("evidence")
    summary_cn = parsed.get("summary_cn")
    if primary not in allowed:
        raise ValueError(f"invalid primary label: {parsed!r}")
    # Secondary labels are optional context.  Some otherwise valid responses
    # paraphrase one; retain only exact rubric members so aggregate coding is
    # fixed-vocabulary without discarding a valid primary/evidence judgment.
    secondary = (
        [item for item in secondary_raw if item in allowed - {primary}]
        if isinstance(secondary_raw, list)
        else []
    )
    if confidence not in {"high", "medium", "low"}:
        raise ValueError(f"invalid confidence: {parsed!r}")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError(f"missing evidence: {parsed!r}")
    if not isinstance(summary_cn, str) or not summary_cn.strip():
        raise ValueError(f"missing summary_cn: {parsed!r}")
    return {
        "case_key": case_key(case),
        "comparison": case["comparison"],
        "question_id": case["question_id"],
        "transition": case["transition"],
        "primary_cause": primary,
        "secondary_causes": secondary,
        "confidence": confidence,
        "evidence": evidence.strip(),
        "summary_cn": summary_cn.strip(),
        "raw": raw,
    }


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                result[item["case_key"]] = item
    return result


def confidence_rank(value: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}[value]


def choose_examples(items: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    matches = [item for item in items if item["primary_cause"] == label]
    return sorted(matches, key=lambda item: (confidence_rank(item["confidence"]), item["question_id"]))[:3]


def md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_outputs(
    analysis_dir: Path,
    cases: list[dict[str, Any]],
    coded: dict[str, dict[str, Any]],
) -> None:
    ordered = [coded[case_key(case)] for case in cases]
    lookup = {case_key(case): case for case in cases}
    groups: dict[str, Counter[str]] = {"C->W": Counter(), "W->C": Counter()}
    confidence = Counter(item["confidence"] for item in ordered)
    for item in ordered:
        groups[item["transition"]][item["primary_cause"]] += 1

    aggregate = {
        "judge_model": MODEL,
        "coded_cases": len(ordered),
        "confidence_counts": dict(confidence),
        "groups": {
            direction: {
                "cases": sum(counts.values()),
                "primary_counts": dict(counts.most_common()),
                "primary_percentages": {
                    label: round(100 * count / sum(counts.values()), 1)
                    for label, count in counts.most_common()
                },
            }
            for direction, counts in groups.items()
        },
        "cases": ordered,
    }
    dump = json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n"
    (analysis_dir / "trace_coding.json").write_text(dump, encoding="utf-8")

    paired = json.loads((analysis_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    transitions = paired["transition_counts"]
    lines = [
        "# Gemini Direct +think versus ReAct candidate 0: endpoint analysis",
        "",
        "## Result summary",
        "",
        f"The single complete Gemini Direct +think run scores `{paired['gemini_correct']}/1197 = {100 * paired['gemini_correct'] / 1197:.2f}%`; "
        f"ReAct candidate 0 scores `{paired['react_correct']}/1197 = {100 * paired['react_correct'] / 1197:.2f}%`.",
        "",
        "| Transition (Gemini -> ReAct) | Questions |",
        "| --- | ---: |",
    ]
    for direction in ("C->C", "C->W", "W->C", "W->W"):
        lines.append(f"| {direction} | {transitions.get(direction, 0)} |")
    lines.extend(
        [
            "",
            f"ReAct has `{transitions['W->C']}` unique wins but `{transitions['C->W']}` unique losses, "
            f"for a net change of `{transitions['W->C'] - transitions['C->W']}` questions.",
            "",
            "## Method boundary",
            "",
            "All changed cases were coded from the question/options/gold, Gemini's persisted thought and visible answer, and ReAct's recorded caption/perception trace. The judge did not see the video. Accordingly, these are **trace-supported post-hoc mechanisms**, not independently verified video-ground-truth or causal claims. A `perception_conflict` means the recorded observation conflicts with the gold-defining fact or supports the distractor.",
            "",
            f"Coding completeness: `{len(ordered)}/{len(cases)}` changed cases; confidence: "
            + ", ".join(f"`{key}={value}`" for key, value in confidence.most_common())
            + ".",
            "",
        ]
    )

    titles = {
        "C->W": "Gemini Direct +think correct, ReAct wrong",
        "W->C": "Gemini Direct +think wrong, ReAct correct",
    }
    for direction in ("C->W", "W->C"):
        counts = groups[direction]
        total = sum(counts.values())
        lines.extend(
            [
                f"## gemini_think_to_react_candidate0: {direction}",
                "",
                f"{titles[direction]}，共 `{total}` 题。",
                "",
                "| Primary mechanism | Cases | Share | Representative IDs |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for label, count in counts.most_common():
            examples = choose_examples(ordered, label)
            ids = ", ".join(f"`{item['question_id']}`" for item in examples)
            lines.append(f"| {label} | {count} | {100 * count / total:.1f}% | {ids} |")
        lines.append("")

        for rank, (label, count) in enumerate(counts.most_common(2), 1):
            lines.extend(
                [
                    f"### Top {rank}: `{label}` — 3 examples",
                    "",
                    "| Question ID | Gold | Gemini | ReAct | 简要概括 |",
                    "| --- | ---: | ---: | ---: | --- |",
                ]
            )
            for item in choose_examples(ordered, label):
                case = lookup[item["case_key"]]
                lines.append(
                    f"| `{item['question_id']}` | `{case['gold']}` | "
                    f"`{case['before']['prediction'] or '<invalid>'}` | "
                    f"`{case['after']['prediction'] or '<invalid>'}` | "
                    f"{md_cell(item['summary_cn'])} |"
                )
            lines.append("")

    lines.extend(
        [
            "## Parser audit",
            "",
            f"Gemini answers use the strict final-option parser on visible response only: "
            f"`{paired['parser_audit']['strict_parser_valid']}` valid and "
            f"`{paired['parser_audit']['strict_parser_invalid']}` invalid. The run's stored parser had "
            f"`{paired['parser_audit']['stored_prediction_invalid']}` invalid predictions, and differs from the strict parser on "
            f"`{paired['parser_audit']['strict_vs_stored_mismatch']}` rows. Separately stored thought is nonempty on "
            f"`{paired['parser_audit']['nonempty_thinking']}/1197` rows and never enters answer parsing.",
            "The five remaining strict-parser-invalid responses are counted as wrong without manual answer recovery.",
            "",
            "Per request, this report does not include a by-task-category breakdown.",
            "",
            "## Reproducibility",
            "",
            "- Exact input paths, transition counts, and parser audit: [`analysis_manifest.json`](analysis_manifest.json)",
            "- All 250 paired changed cases with Gemini thought and ReAct trace: [`changed_cases.json`](changed_cases.json)",
            "- Complete per-case coding and evidence: [`trace_coding.md`](trace_coding.md)",
            "- Extraction script: [`analyze_gemini_think_vs_react_candidate0.py`](../../analyze_gemini_think_vs_react_candidate0.py)",
            "- Coding script and fixed rubric: [`code_gemini_think_vs_react_candidate0.py`](../../code_gemini_think_vs_react_candidate0.py)",
        ]
    )
    (analysis_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    evidence_lines = ["# Complete trace-supported coding", ""]
    for index, item in enumerate(ordered, 1):
        case = lookup[item["case_key"]]
        evidence_lines.extend(
            [
                f"## {index}. `{item['question_id']}` — {item['transition']}",
                "",
                f"- Gold: `{case['gold']}`; Gemini: `{case['before']['prediction'] or '<invalid>'}`; ReAct: `{case['after']['prediction'] or '<invalid>'}`",
                f"- Primary mechanism: `{item['primary_cause']}` ({item['confidence']})",
                f"- Summary: {item['summary_cn']}",
                f"- Evidence: {item['evidence']}",
                "",
            ]
        )
    (analysis_dir / "trace_coding.md").write_text(
        "\n".join(evidence_lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or args.limit is not None and args.limit < 1:
        parser.error("--workers and --limit must be positive")
    api_key = os.environ.get("ELM_API_KEY", "").strip()
    if not api_key:
        parser.error("ELM_API_KEY is required")

    analysis_dir = args.analysis_dir.expanduser().resolve()
    cases = load_cases(analysis_dir / "changed_cases.json")
    checkpoint = analysis_dir / "coding_checkpoint.jsonl"
    saved = load_checkpoint(checkpoint)
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
        print(f"checkpointed {len(saved)}/{len(cases)}; aggregate waits for completion")
        return 0
    write_outputs(analysis_dir, cases, saved)
    print(f"wrote complete coding and report to {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
