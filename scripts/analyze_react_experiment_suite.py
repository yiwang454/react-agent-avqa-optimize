#!/usr/bin/env python3
"""Aggregate accuracy, usage, cost, latency, and tool calls for ReAct experiments.

Usage example:
  python scripts/analyze_react_experiment_suite.py \
    --cuts-jsonl /path/to/cuts.jsonl --output-dir /path/to/analysis \
    --experiment baseline /path/to/output_test.jsonl 13856.46
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any


OPENAI_INPUT_PER_MTOK = 2.00
OPENAI_OUTPUT_PER_MTOK = 8.00
GEMINI_TEXT_VIDEO_INPUT_PER_MTOK = 0.30
GEMINI_AUDIO_INPUT_PER_MTOK = 1.00
GEMINI_OUTPUT_PER_MTOK = 2.50
OPTION_LABEL_RE = re.compile(r"(?im)(?<![A-Z0-9])([A-D])\s*[\).:]")
ANSWER_RE = re.compile(r"^\s*([A-D])(?:\s|[.:,;]|$)", re.IGNORECASE)


def pct(numerator: float, denominator: float) -> float:
    return 0.0 if not denominator else numerator * 100.0 / denominator


def fmt_tokens(tokens: float) -> str:
    return f"{tokens / 1_000:.1f}k"


def parse_answer(value: Any) -> str | None:
    match = ANSWER_RE.match(str(value or ""))
    return match.group(1).upper() if match else None


def is_abcd_multiple_choice(question: str) -> bool:
    return set(OPTION_LABEL_RE.findall(question.upper())) == {"A", "B", "C", "D"}


def legalize_range(start: float, end: float, duration: float) -> tuple[float, float]:
    """Match the clip utility's clipping behavior at a video boundary."""
    requested_length = end - start
    if start >= duration:
        return max(0.0, duration - requested_length), duration
    if end <= 0.0:
        return 0.0, min(requested_length, duration)
    return max(start, 0.0), min(end, duration)


@dataclass
class GeminiTokens:
    input_tokens: int = 0
    output_tokens: int = 0
    # Input-token and audio-token observations only from calls that expose
    # promptTokensDetails. The audio fraction estimates detail-less calls.
    observed_input_by_tool: Counter[str] = field(default_factory=Counter)
    observed_audio_by_tool: Counter[str] = field(default_factory=Counter)
    input_by_tool: Counter[str] = field(default_factory=Counter)

    def add(self, tool_name: str, usage: dict[str, Any]) -> None:
        input_tokens = int(usage.get("promptTokenCount") or 0)
        total_tokens = int(usage.get("totalTokenCount") or 0)
        self.input_tokens += input_tokens
        self.output_tokens += total_tokens - input_tokens
        self.input_by_tool[tool_name] += input_tokens
        details = usage.get("promptTokensDetails") or []
        if details:
            self.observed_input_by_tool[tool_name] += input_tokens
            self.observed_audio_by_tool[tool_name] += sum(
                int(item.get("tokenCount") or 0)
                for item in details
                if str(item.get("modality", "")).upper() == "AUDIO"
            )

    @property
    def estimated_audio_input_tokens(self) -> float:
        estimated = 0.0
        for tool_name, all_input in self.input_by_tool.items():
            observed_input = self.observed_input_by_tool[tool_name]
            observed_audio = self.observed_audio_by_tool[tool_name]
            estimated += all_input * observed_audio / observed_input if observed_input else 0.0
        return estimated

    @property
    def standard_cost(self) -> float:
        audio = self.estimated_audio_input_tokens
        non_audio = self.input_tokens - audio
        return (
            non_audio * GEMINI_TEXT_VIDEO_INPUT_PER_MTOK
            + audio * GEMINI_AUDIO_INPUT_PER_MTOK
            + self.output_tokens * GEMINI_OUTPUT_PER_MTOK
        ) / 1_000_000


@dataclass
class ExperimentStats:
    name: str
    path: Path
    elapsed_seconds: float
    questions: int = 0
    correct: int = 0
    unparseable_responses: int = 0
    planner_input: int = 0
    planner_output: int = 0
    gemini_live: GeminiTokens = field(default_factory=GeminiTokens)
    gemini_cache_source: GeminiTokens = field(default_factory=GeminiTokens)
    tool_calls: Counter[str] = field(default_factory=Counter)
    tool_samples: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    cache_hits: int = 0
    original_qa_calls: Counter[str] = field(default_factory=Counter)
    non_original_qa_calls: Counter[str] = field(default_factory=Counter)
    original_qa_samples: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    clip_rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return pct(self.correct, self.questions)

    @property
    def seconds_per_question(self) -> float:
        return self.elapsed_seconds / self.questions

    @property
    def live_input(self) -> int:
        return self.planner_input + self.gemini_live.input_tokens

    @property
    def live_output(self) -> int:
        return self.planner_output + self.gemini_live.output_tokens

    @property
    def live_standard_cost(self) -> float:
        return (
            self.planner_input * OPENAI_INPUT_PER_MTOK
            + self.planner_output * OPENAI_OUTPUT_PER_MTOK
        ) / 1_000_000 + self.gemini_live.standard_cost

    @property
    def cache_inclusive_input(self) -> int:
        return self.live_input + self.gemini_cache_source.input_tokens

    @property
    def cache_inclusive_output(self) -> int:
        return self.live_output + self.gemini_cache_source.output_tokens

    @property
    def cache_inclusive_standard_cost(self) -> float:
        return self.live_standard_cost + self.gemini_cache_source.standard_cost


def load_durations(path: Path) -> dict[str, float]:
    durations: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cut = json.loads(line)
                durations[str(cut["id"])] = float(cut["duration"])
    return durations


def analyze_experiment(
    name: str, result_path: Path, elapsed_seconds: float, durations: dict[str, float]
) -> ExperimentStats:
    stats = ExperimentStats(name=name, path=result_path, elapsed_seconds=elapsed_seconds)
    with result_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            result = json.loads(line)
            question = result["question_data"]
            question_id = str(question["question_id"])
            stats.questions += 1
            prediction = parse_answer(question.get("response"))
            answer = str(question.get("answer", "")).strip().upper()
            if prediction is None:
                stats.unparseable_responses += 1
            elif prediction == answer:
                stats.correct += 1

            for turn in question.get("turn_trace", []):
                for planner_call in (turn.get("planner_calls") or {}).get("action_decision", []):
                    usage = planner_call.get("usage") or {}
                    stats.planner_input += int(usage.get("prompt_tokens") or 0)
                    stats.planner_output += int(usage.get("completion_tokens") or 0)

                if turn.get("planner_action") != "tool":
                    continue
                tool_name = str(turn.get("tool_name"))
                stats.tool_calls[tool_name] += 1
                stats.tool_samples[tool_name].add(question_id)
                if turn.get("caption_cache_hit") is True:
                    stats.cache_hits += 1

                usage = turn.get("perception_token_usage")
                if isinstance(usage, dict):
                    stats.gemini_live.add(tool_name, usage)
                cache_source_usage = turn.get("caption_cache_source_token_usage")
                if isinstance(cache_source_usage, dict):
                    stats.gemini_cache_source.add(tool_name, cache_source_usage)

                perceptual_question = (turn.get("tool_args") or {}).get("perceptual_question")
                # Caption tools have a caption_instruction rather than a question.
                if isinstance(perceptual_question, str):
                    if is_abcd_multiple_choice(perceptual_question):
                        stats.original_qa_calls[tool_name] += 1
                        stats.original_qa_samples[tool_name].add(question_id)
                    else:
                        stats.non_original_qa_calls[tool_name] += 1

                time_range = (turn.get("tool_args") or {}).get("time_range")
                if "clip" in tool_name and isinstance(time_range, list) and len(time_range) == 2:
                    if question_id not in durations:
                        raise KeyError(f"No source duration for {question_id}")
                    start, end = map(float, time_range)
                    effective_start, effective_end = legalize_range(start, end, durations[question_id])
                    stats.clip_rows.append(
                        {
                            "experiment": name,
                            "question_id": question_id,
                            "turn_id": turn.get("turn_id"),
                            "video_duration_seconds": durations[question_id],
                            "requested_start_seconds": start,
                            "requested_end_seconds": end,
                            "effective_start_seconds": effective_start,
                            "effective_end_seconds": effective_end,
                            "clip_duration_seconds": effective_end - effective_start,
                            "clip_percent_of_video": pct(effective_end - effective_start, durations[question_id]),
                            "range_was_reanchored": (start, end) != (effective_start, effective_end),
                        }
                    )
    return stats


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        nargs=3,
        metavar=("NAME", "RESULTS_JSONL", "ELAPSED_SECONDS"),
        action="append",
        required=True,
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    durations = load_durations(args.cuts_jsonl)
    experiments = [
        analyze_experiment(name, Path(path), float(elapsed), durations)
        for name, path, elapsed in args.experiment
    ]

    main_rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []
    for stats in experiments:
        main_rows.append(
            {
                "experiment": stats.name,
                "questions": stats.questions,
                "correct": stats.correct,
                "accuracy_percent": round(stats.accuracy, 6),
                "elapsed_seconds": stats.elapsed_seconds,
                "seconds_per_question": round(stats.seconds_per_question, 6),
                "live_input_tokens": stats.live_input,
                "live_output_tokens": stats.live_output,
                "planner_input_tokens": stats.planner_input,
                "planner_output_tokens": stats.planner_output,
                "gemini_live_input_tokens": stats.gemini_live.input_tokens,
                "gemini_live_output_tokens": stats.gemini_live.output_tokens,
                "gemini_estimated_audio_input_tokens": round(stats.gemini_live.estimated_audio_input_tokens, 3),
                "live_standard_cost_usd": round(stats.live_standard_cost, 8),
                "live_standard_cost_per_question_usd": round(stats.live_standard_cost / stats.questions, 8),
                "cache_source_gemini_input_tokens": stats.gemini_cache_source.input_tokens,
                "cache_source_gemini_output_tokens": stats.gemini_cache_source.output_tokens,
                "cache_inclusive_input_tokens": stats.cache_inclusive_input,
                "cache_inclusive_output_tokens": stats.cache_inclusive_output,
                "cache_inclusive_standard_cost_per_question_usd": round(
                    stats.cache_inclusive_standard_cost / stats.questions, 8
                ),
                "caption_cache_hits": stats.cache_hits,
                "unparseable_responses": stats.unparseable_responses,
            }
        )
        total_calls = sum(stats.tool_calls.values())
        for tool_name, calls in sorted(stats.tool_calls.items()):
            samples = len(stats.tool_samples[tool_name])
            original = stats.original_qa_calls[tool_name]
            not_original = stats.non_original_qa_calls[tool_name]
            perceptual_calls = original + not_original
            tool_rows.append(
                {
                    "experiment": stats.name,
                    "tool_name": tool_name,
                    "call_count": calls,
                    "call_share_percent": round(pct(calls, total_calls), 6),
                    "samples_called": samples,
                    "benchmark_coverage_percent": round(pct(samples, stats.questions), 6),
                    "ask_original_qa_calls": original if perceptual_calls else "",
                    "not_ask_original_qa_calls": not_original if perceptual_calls else "",
                    "ask_original_qa_share_percent": round(pct(original, perceptual_calls), 6) if perceptual_calls else "",
                }
            )
        clip_rows.extend(stats.clip_rows)

    write_csv(args.output_dir / "experiment_summary.csv", main_rows, list(main_rows[0]))
    write_csv(args.output_dir / "tool_call_summary.csv", tool_rows, list(tool_rows[0]))
    if clip_rows:
        write_csv(args.output_dir / "clip_calls.csv", clip_rows, list(clip_rows[0]))

    report = [
        "# Daily Omni ReAct ablation metrics",
        "",
        "## Main results (live run)",
        "",
        "`System In/Out` is the per-question average of all persisted planner tokens and live Gemini tool tokens. It excludes the first `ask_caption` cache read, because it made no new Gemini request in these runs. `Latency/question` is the full-run wall-clock throughput (`elapsed / 1,197`), so it includes concurrency and retry waits rather than a serial request latency.",
        "",
        "| System | In/Out tok. | Latency/question | Cost/question | Acc. |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for stats in experiments:
        report.append(
            f"| {stats.name} | {fmt_tokens(stats.live_input / stats.questions)}/{fmt_tokens(stats.live_output / stats.questions)} | "
            f"{stats.seconds_per_question:.2f}s | ${stats.live_standard_cost / stats.questions:.4f} | {stats.accuracy:.2f} |"
        )

    report.extend(
        [
            "",
            "## Accuracy and token/cost breakdown",
            "",
            "| System | Correct / 1,197 | Planner In/Out | Live Gemini In/Out | Estimated Gemini audio input | Run cost |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for stats in experiments:
        report.append(
            f"| {stats.name} | {stats.correct} | {fmt_tokens(stats.planner_input)}/{fmt_tokens(stats.planner_output)} | "
            f"{fmt_tokens(stats.gemini_live.input_tokens)}/{fmt_tokens(stats.gemini_live.output_tokens)} | "
            f"{fmt_tokens(stats.gemini_live.estimated_audio_input_tokens)} | ${stats.live_standard_cost:.4f} |"
        )

    report.extend(
        [
            "",
            "Cost uses standard paid API list rates: [o3](https://developers.openai.com/api/docs/models/o3) $2.00/M input and $8.00/M output; [Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/pricing) $0.30/M text/image/video input, $1.00/M audio input, and $2.50/M output. Gemini calls without modality details receive the observed audio fraction for their tool type. This is a comparable list-price estimate, not a gateway invoice, and excludes unpersisted failed retries and cache storage fees.",
            "",
            "## Tool calling",
            "",
            "Call share is out of that experiment's recorded tool calls. Coverage is the number of distinct question samples that invoked the tool divided by 1,197.",
            "",
            "| System | Tool | Calls (share) | Samples (coverage) | Original-QA / not-original-QA* |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in tool_rows:
        question_ratio = "—"
        if row["ask_original_qa_calls"] != "":
            question_ratio = (
                f"{row['ask_original_qa_calls']}/{row['not_ask_original_qa_calls']} "
                f"({row['ask_original_qa_share_percent']:.2f}% / "
                f"{100.0 - row['ask_original_qa_share_percent']:.2f}%)"
            )
        report.append(
            f"| {row['experiment']} | {row['tool_name']} | {row['call_count']} ({row['call_share_percent']:.2f}%) | "
            f"{row['samples_called']} ({row['benchmark_coverage_percent']:.2f}%) | {question_ratio} |"
        )
    report.extend(
        [
            "",
            "*For non-caption tools, an original-QA-style call is a perceptual question explicitly listing all four A/B/C/D choices (A./A)/A: supported), matching the requested proxy. Caption tools do not have a perceptual question and are therefore `—`.",
        ]
    )

    if clip_rows:
        report.extend(
            [
                "",
                "## Video clip calls",
                "",
                "| System | Clip calls | Samples | Mean effective length | Mean effective video share | Re-anchored ranges |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for stats in experiments:
            rows = stats.clip_rows
            if not rows:
                continue
            lengths = [float(row["clip_duration_seconds"]) for row in rows]
            shares = [float(row["clip_percent_of_video"]) for row in rows]
            report.append(
                f"| {stats.name} | {len(rows)} | {len({row['question_id'] for row in rows})} | "
                f"{mean(lengths):.3f}s (median {median(lengths):.3f}s) | "
                f"{mean(shares):.3f}% | {sum(bool(row['range_was_reanchored']) for row in rows)} |"
            )

    report.extend(
        [
            "",
            "## Caption-cache provenance",
            "",
            "All runs made 1,197 `caption_cache` reads. The cache source token totals are shown below only for an end-to-end accounting scenario; they are excluded from the main live-run table because the caption generation occurred before these three experiments. Adding them gives the listed cache-inclusive token/cost estimate, but not an additional run latency.",
            "",
            "| System | Cached-caption Gemini In/Out / question | Cache-inclusive System In/Out / question | Cache-inclusive cost/question |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for stats in experiments:
        report.append(
            f"| {stats.name} | {fmt_tokens(stats.gemini_cache_source.input_tokens / stats.questions)}/{fmt_tokens(stats.gemini_cache_source.output_tokens / stats.questions)} | "
            f"{fmt_tokens(stats.cache_inclusive_input / stats.questions)}/{fmt_tokens(stats.cache_inclusive_output / stats.questions)} | "
            f"${stats.cache_inclusive_standard_cost / stats.questions:.4f} |"
        )
    report.extend(
        [
            "",
            "## Files",
            "",
            "- `experiment_summary.csv`: accuracy, latency, token totals, and cost estimates.",
            "- `tool_call_summary.csv`: tool shares, benchmark coverage, and original-QA proxy counts.",
            "- `clip_calls.csv`: each clip range and effective duration (when a clip tool is used).",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
