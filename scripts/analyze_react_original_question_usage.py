#!/usr/bin/env python3
# ruff: noqa: D101,D102,D103,T201
"""Measure whether ReAct perception calls reuse the original AVQA question.

The report keeps the matching signals separate:

* exact: the normalized original question occurs verbatim in the perception question;
* first edge: the first N normalized words occur in the perception question;
* last edge: the last N normalized words occur in the perception question;
* edge union: either the first or last edge matches;
* option answer: the perception observation explicitly returns an A/B/C/D option.

The edge and option-answer signals are heuristics.  The generated review sample is
drawn from edge-only calls so its precision can be checked manually.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL
)
OPTION_PATTERNS = (
    re.compile(r"^\s*([A-D])\s*$", re.IGNORECASE),
    re.compile(r"^\s*[\(\[]?([A-D])[\)\]]?\s*[.：:\-]", re.IGNORECASE),
    re.compile(
        r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?(?:the\s+)?(?:correct\s+)?"
        r"(?:answer|option)(?:\*\*)?\s*(?:is|:)?\s*(?:\*\*)?"
        r"[\(\[]?([A-D])\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class PerceptionCall:
    question_id: str
    turn_id: int
    original_question: str
    perception_question: str
    observation_answer: str
    exact_match: bool
    first_edge_match: bool
    last_edge_match: bool
    option_answer_match: bool

    @property
    def edge_match(self) -> bool:
        return self.first_edge_match or self.last_edge_match

    @property
    def detected(self) -> bool:
        return self.edge_match or self.option_answer_match


def normalized_words(text: Any) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(str(text or ""))]


def contains_words(needle: list[str], haystack: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[index : index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def observation_answer_text(observation: Any) -> str:
    text = str(observation or "").strip()
    candidates: list[str] = []
    for raw_json in FENCED_JSON_RE.findall(text):
        try:
            value = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("answer") is not None:
            candidates.append(str(value["answer"]).strip())
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict) and value.get("answer") is not None:
        candidates.append(str(value["answer"]).strip())
    candidates.append(text)
    return next((candidate for candidate in candidates if candidate), "")


def extract_explicit_option(observation: Any) -> tuple[str, str]:
    text = str(observation or "").strip()
    candidates: list[str] = []
    answer_text = observation_answer_text(text)
    if answer_text and answer_text != text:
        candidates.append(answer_text)
    candidates.append(text)
    for candidate in candidates:
        for pattern in OPTION_PATTERNS:
            if match := pattern.search(candidate):
                return match.group(1).upper(), answer_text
    return "", answer_text


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for row_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{row_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{row_number}")
            yield record


def summarize_tool_usage(
    question_ids: list[str], traces: Iterable[tuple[str, Any]]
) -> dict[str, Any]:
    call_counts: Counter[str] = Counter()
    sample_ids_by_tool: dict[str, set[str]] = defaultdict(set)
    samples_with_any_tool: set[str] = set()
    budget_exempt_calls = 0
    for question_id, trace in traces:
        for turn in trace or []:
            if not isinstance(turn, dict):
                continue
            tool_name = str(turn.get("tool_name") or "").strip()
            if not tool_name:
                continue
            call_counts[tool_name] += 1
            sample_ids_by_tool[tool_name].add(question_id)
            samples_with_any_tool.add(question_id)
            budget_exempt_calls += bool(turn.get("budget_exempt"))
    return {
        "total_tool_calls": sum(call_counts.values()),
        "samples_with_any_tool": len(samples_with_any_tool),
        "samples_without_any_tool": len(question_ids) - len(samples_with_any_tool),
        "budget_exempt_calls": budget_exempt_calls,
        "calls_by_tool": dict(sorted(call_counts.items())),
        "samples_by_tool": {
            name: len(sample_ids_by_tool[name]) for name in sorted(sample_ids_by_tool)
        },
    }


def analyze(
    path: Path, edge_word_count: int
) -> tuple[list[str], list[PerceptionCall], dict[str, Any]]:
    question_ids: list[str] = []
    calls: list[PerceptionCall] = []
    traces: list[tuple[str, Any]] = []
    for record in iter_records(path):
        question_data = record.get("question_data") or {}
        question_id = str(question_data.get("question_id") or "").strip()
        if not question_id:
            raise ValueError("Every result row must have question_data.question_id")
        question_ids.append(question_id)
        traces.append((question_id, question_data.get("turn_trace")))
        original_question = str(question_data.get("question") or "").strip()
        original_words = normalized_words(original_question)
        for turn_index, turn in enumerate(question_data.get("turn_trace") or [], 1):
            if not isinstance(turn, dict) or turn.get("tool_name") != "ask_perception":
                continue
            tool_args = turn.get("tool_args") or {}
            perception_question = str(
                tool_args.get("perceptual_question") or ""
            ).strip()
            perception_words = normalized_words(perception_question)
            option, answer_text = extract_explicit_option(turn.get("tool_observation"))
            enough_words = len(original_words) >= edge_word_count
            calls.append(
                PerceptionCall(
                    question_id=question_id,
                    turn_id=int(turn.get("turn_id") or turn_index),
                    original_question=original_question,
                    perception_question=perception_question,
                    observation_answer=answer_text,
                    exact_match=contains_words(original_words, perception_words),
                    first_edge_match=enough_words
                    and contains_words(
                        original_words[:edge_word_count], perception_words
                    ),
                    last_edge_match=enough_words
                    and contains_words(
                        original_words[-edge_word_count:], perception_words
                    ),
                    option_answer_match=bool(option),
                )
            )
    if len(question_ids) != len(set(question_ids)):
        duplicates = [
            item for item, count in Counter(question_ids).items() if count > 1
        ]
        raise ValueError(f"Duplicate question IDs: {duplicates[:5]}")
    return question_ids, calls, summarize_tool_usage(question_ids, traces)


def metric(count: int, denominator: int) -> dict[str, int | float]:
    return {
        "count": count,
        "denominator": denominator,
        "percent": 100.0 * count / denominator if denominator else 0.0,
    }


def build_summary(
    question_ids: list[str], calls: list[PerceptionCall]
) -> dict[str, Any]:
    calls_by_question: dict[str, list[PerceptionCall]] = {
        question_id: [] for question_id in question_ids
    }
    for call in calls:
        calls_by_question[call.question_id].append(call)
    perception_question_ids = {call.question_id for call in calls}

    predicates = {
        "exact_original_question": lambda call: call.exact_match,
        "first_edge": lambda call: call.first_edge_match,
        "last_edge": lambda call: call.last_edge_match,
        "first_or_last_edge": lambda call: call.edge_match,
        "explicit_abcd_observation": lambda call: call.option_answer_match,
        "combined_edge_or_abcd": lambda call: call.detected,
    }
    metrics: dict[str, Any] = {}
    for name, predicate in predicates.items():
        call_count = sum(predicate(call) for call in calls)
        sample_count = sum(
            any(predicate(call) for call in sample_calls)
            for sample_calls in calls_by_question.values()
        )
        metrics[name] = {
            "perception_calls": metric(call_count, len(calls)),
            "all_samples": metric(sample_count, len(question_ids)),
            "samples_with_perception": metric(
                sample_count, len(perception_question_ids)
            ),
        }

    exact_question_ids = {call.question_id for call in calls if call.exact_match}
    edge_question_ids = {call.question_id for call in calls if call.edge_match}
    option_question_ids = {
        call.question_id for call in calls if call.option_answer_match
    }
    return {
        "total_samples": len(question_ids),
        "samples_with_perception": len(perception_question_ids),
        "samples_without_perception": len(question_ids) - len(perception_question_ids),
        "total_perception_calls": len(calls),
        "samples_with_multiple_perception_calls": sum(
            len(sample_calls) > 1 for sample_calls in calls_by_question.values()
        ),
        "metrics": metrics,
        "exclusive_sample_counts": {
            "edge_without_exact": len(edge_question_ids - exact_question_ids),
            "abcd_additions_beyond_edge": len(option_question_ids - edge_question_ids),
        },
        "exclusive_call_counts": {
            "edge_without_exact": sum(
                call.edge_match and not call.exact_match for call in calls
            ),
            "abcd_additions_beyond_edge": sum(
                call.option_answer_match and not call.edge_match for call in calls
            ),
        },
    }


def percent_text(item: dict[str, Any]) -> str:
    return f"{item['percent']:.2f}% ({item['count']}/{item['denominator']})"


def render_markdown(source: Path, edge_word_count: int, summary: dict[str, Any]) -> str:
    rows = []
    labels = {
        "exact_original_question": "Exact normalized original question",
        "first_edge": f"First {edge_word_count} words (includes exact)",
        "last_edge": f"Last {edge_word_count} words (includes exact)",
        "first_or_last_edge": f"First or last {edge_word_count} words (includes exact)",
        "explicit_abcd_observation": "Perception observation explicitly returns A/B/C/D",
        "combined_edge_or_abcd": "Combined: edge match or explicit A/B/C/D",
    }
    for key, label in labels.items():
        value = summary["metrics"][key]
        rows.append(
            f"| {label} | {percent_text(value['all_samples'])} | "
            f"{percent_text(value['samples_with_perception'])} | "
            f"{percent_text(value['perception_calls'])} |"
        )
    tool_usage = summary["tool_usage"]
    tool_rows = [
        f"| `{tool_name}` | {call_count} | "
        f"{tool_usage['samples_by_tool'].get(tool_name, 0)} |"
        for tool_name, call_count in tool_usage["calls_by_tool"].items()
    ]
    return "\n".join(
        [
            "# ReAct original-question usage audit",
            "",
            f"Source: `{source}`",
            "",
            f"- Total samples: {summary['total_samples']}",
            f"- Samples calling `ask_perception`: {summary['samples_with_perception']}",
            f"- Total `ask_perception` calls: {summary['total_perception_calls']}",
            "",
            "## Tool usage",
            "",
            f"Total tool calls: {tool_usage['total_tool_calls']}; "
            f"budget-exempt calls: {tool_usage['budget_exempt_calls']}.",
            "",
            "| Tool | Calls | Samples using tool |",
            "| --- | ---: | ---: |",
            *tool_rows,
            "",
            "## Perception-question reuse",
            "",
            "| Detection signal | All samples | Samples with perception | Perception calls |",
            "| --- | ---: | ---: | ---: |",
            *rows,
            "",
            "The combined row is the broad estimate. Exact matching is a lower bound. The edge and",
            "A/B/C/D signals are heuristics and should be interpreted with the manual precision sample.",
            "The A/B/C/D detector only accepts an explicit answer/option label or a leading option label;",
            "it does not treat arbitrary letters elsewhere in an observation as a match.",
            "",
            f"The edge-only pool contains {summary['exclusive_call_counts']['edge_without_exact']} calls "
            f"across {summary['exclusive_sample_counts']['edge_without_exact']} samples. The generated "
            "`edge_heuristic_review_sample.json` is a deterministic "
            f"{summary['review_sample_size']}-call sample from this pool; see "
            "`manual_edge_review.md` after human review. The broad combined estimate adds "
            f"{summary['exclusive_sample_counts']['abcd_additions_beyond_edge']} samples detected only "
            "by an explicit A/B/C/D observation beyond the lexical edge detector.",
            "",
        ]
    )


def write_calls_csv(path: Path, calls: list[PerceptionCall]) -> None:
    fieldnames = [
        "question_id",
        "turn_id",
        "exact_match",
        "first_edge_match",
        "last_edge_match",
        "edge_match",
        "option_answer_match",
        "detected",
        "original_question",
        "perception_question",
        "observation_answer",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for call in calls:
            row = asdict(call)
            row["edge_match"] = call.edge_match
            row["detected"] = call.detected
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edge-word-count", type=int, default=3)
    parser.add_argument("--review-sample-size", type=int, default=10)
    parser.add_argument("--review-seed", type=int, default=18)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.edge_word_count < 1 or args.review_sample_size < 0:
        raise SystemExit(
            "--edge-word-count must be >= 1 and --review-sample-size must be >= 0"
        )
    source = args.input_jsonl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    question_ids, calls, tool_usage = analyze(source, args.edge_word_count)
    edge_only = [call for call in calls if call.edge_match and not call.exact_match]
    summary = build_summary(question_ids, calls)
    summary.update(
        {
            "source": str(source),
            "edge_word_count": args.edge_word_count,
            "review_seed": args.review_seed,
            "review_sample_size": min(args.review_sample_size, len(edge_only)),
            "tool_usage": tool_usage,
        }
    )
    rng = random.Random(args.review_seed)
    review_sample = rng.sample(edge_only, min(args.review_sample_size, len(edge_only)))

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        render_markdown(source, args.edge_word_count, summary), encoding="utf-8"
    )
    (output_dir / "edge_heuristic_review_sample.json").write_text(
        json.dumps(
            [asdict(call) for call in review_sample], ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    write_calls_csv(output_dir / "perception_calls.csv", calls)
    print(render_markdown(source, args.edge_word_count, summary))
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
