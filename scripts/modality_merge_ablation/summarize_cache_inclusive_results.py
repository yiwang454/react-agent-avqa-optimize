#!/usr/bin/env python3
"""Rebuild Stage B tables with caption-generation token and cost attribution.

Every cached caption tool call receives its source generation usage.  Gemini
usage absent from legacy audio-caption trace entries is estimated from the
known generated entries in the same audio-caption cache.  Reported latency
remains the measured question latency; the cache sources do not contain
comparable per-call latency for all configurations.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


DATA = Path("/mnt/ceph_rbd/data/avqa_project/daily_omni")
V2 = DATA / "modality_merge_ablation_v2_selectedVal125_repeats"
V3 = DATA / "modality_merge_ablation_v3_selectedVal125_repeats"
B11 = DATA / "modality_merge_ablation_b11_selectedVal125_repeats"
RERUN = DATA / "modality_merge_ablation_v2_selectedVal125_outlier_reruns"
AUDIO_CACHE = DATA / "daily_omni_audio_global_caption_cache_omniagent_repeat1_completed"
IDS = ["B0", "B1", "B2", "B3", "B4", "B5", "B5-flex", "B6", "B7", "B8", "B9", "B10", "B11"]
REPLACEMENTS = {
    ("B0", 3): RERUN / "b0-repeat3-rerun/output_test.jsonl",
    ("B1", 2): RERUN / "b1-repeat2-rerun/output_test.jsonl",
}


def result_path(experiment_id: str, repeat: int) -> Path:
    if (experiment_id, repeat) in REPLACEMENTS:
        return REPLACEMENTS[(experiment_id, repeat)]
    root = B11 if experiment_id == "B11" else (
        V3 if experiment_id in {"B7", "B8", "B9", "B10"} else V2
    )
    return root / f"repeat{repeat}" / experiment_id.lower() / "output_test.jsonl"


def audio_tokens(usage: dict) -> int | None:
    details = usage.get("promptTokensDetails")
    if not isinstance(details, list) or not details:
        return None
    return sum(
        int(item.get("tokenCount") or 0)
        for item in details
        if str(item.get("modality") or "").upper() == "AUDIO"
    )


def calibration(durations: dict[str, float]) -> dict:
    inputs: dict[float, list[int]] = defaultdict(list)
    fractions: dict[float, list[float]] = defaultdict(list)
    output_per_char: list[float] = []
    for question_id, duration in durations.items():
        payload = json.loads((AUDIO_CACHE / f"{question_id}.json").read_text())
        usage = payload.get("source_token_usage")
        if not isinstance(usage, dict):
            continue
        prompt = int(usage.get("promptTokenCount") or 0)
        output = int(usage.get("totalTokenCount") or 0) - prompt
        audio = audio_tokens(usage)
        if not prompt or audio is None:
            continue
        inputs[duration].append(prompt)
        fractions[duration].append(audio / prompt)
        if payload.get("response"):
            output_per_char.append(output / len(payload["response"]))
    if set(inputs) != set(durations.values()) or not output_per_char:
        raise ValueError("Audio-caption calibration lacks a duration stratum")
    return {
        "input_median_by_duration": {key: statistics.median(value) for key, value in inputs.items()},
        "audio_fraction_by_duration": {key: statistics.median(value) for key, value in fractions.items()},
        "output_tokens_per_character": statistics.median(output_per_char),
        "known_cache_entries_by_duration": {key: len(value) for key, value in inputs.items()},
    }


def omni_audio_fractions(durations: dict[str, float]) -> dict[float, float]:
    cache = DATA / "daily_omni_caption_cache_v8_gemini_from3repeats"
    observed: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for question_id, duration in durations.items():
        usage = json.loads((cache / f"{question_id}.json").read_text())["source_token_usage"]
        audio = audio_tokens(usage)
        if audio is not None:
            observed[duration][0] += audio
            observed[duration][1] += int(usage["promptTokenCount"])
    if set(observed) != set(durations.values()):
        raise ValueError("Omni-caption audio-fraction calibration lacks a duration stratum")
    return {duration: audio / prompt for duration, (audio, prompt) in observed.items()}


def source_usage(turn: dict, duration: float, audio_calibration: dict,
                 omni_fractions: dict[float, float]) -> tuple[int, int, float, bool]:
    usage = turn.get("caption_cache_source_token_usage")
    estimated = not isinstance(usage, dict)
    if estimated:
        if turn.get("tool_name") != "audio_global_caption":
            raise ValueError("Missing source usage outside legacy audio-caption cache")
        input_count = round(audio_calibration["input_median_by_duration"][duration])
        response = turn.get("tool_observation") or {}
        if isinstance(response, dict):
            response = response.get("answer") or ""
        output_count = round(len(str(response)) * audio_calibration["output_tokens_per_character"])
        fraction = audio_calibration["audio_fraction_by_duration"][duration]
        audio_count = input_count * fraction
    else:
        input_count = int(usage.get("promptTokenCount") or 0)
        output_count = int(usage.get("totalTokenCount") or 0) - input_count
        if output_count < 0:
            raise ValueError("Gemini totalTokenCount is smaller than promptTokenCount")
        detail_audio = audio_tokens(usage)
        if detail_audio is not None:
            audio_count = detail_audio
        else:
            fraction = (
                audio_calibration["audio_fraction_by_duration"][duration]
                if turn.get("tool_name") == "audio_global_caption"
                else omni_fractions[duration]
            )
            audio_count = input_count * fraction
    cost = (
        (input_count - audio_count) * 0.30
        + audio_count * 1.00
        + output_count * 2.50
    ) / 1_000_000
    return input_count, output_count, cost, estimated


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def summarize_run(path: Path, durations: dict[str, float], audio_calibration: dict,
                  omni_fractions: dict[float, float]) -> dict:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if len(rows) != 125:
        raise ValueError(f"Expected 125 questions in {path}, found {len(rows)}")
    totals = Counter()
    latencies = []
    status = Counter()
    for row in rows:
        question = row["question_data"]
        question_id = question["question_id"]
        metrics = question["metrics"]
        totals["correct"] += bool(metrics["evaluation"]["correct"])
        totals["parseable"] += bool(metrics["evaluation"]["parseable"])
        totals["calls"] += int(metrics["tool_calls"]["budgeted_total"])
        totals["invalid"] += int(metrics["tool_calls"]["invalid_total"])
        totals["reached_budget"] += bool(metrics["budget"]["reached"])
        status[metrics["final_answer"]["status"]] += 1
        live = metrics["tokens"]["live_total"]
        totals["live_input"] += int(live["input_tokens"])
        totals["live_output"] += int(live["output_tokens"])
        totals["live_cost"] += float(metrics["estimated_cost_usd"]["live_marginal_total"])
        latency = (question.get("latency") or {}).get("retry_adjusted_seconds")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
        for turn in question.get("turn_trace") or []:
            if turn.get("caption_cache_hit") is not True:
                continue
            if turn.get("tool_name") not in {"audio_global_caption", "ask_caption"}:
                raise ValueError(f"Unexpected cached tool: {turn.get('tool_name')}")
            input_count, output_count, cost, estimated = source_usage(
                turn, durations[question_id], audio_calibration, omni_fractions
            )
            totals["cache_calls"] += 1
            totals["cache_usage_estimated"] += estimated
            totals["cache_input"] += input_count
            totals["cache_output"] += output_count
            totals["cache_cost"] += cost
    count = len(rows)
    return {
        "source": str(path), "questions": count, "correct": totals["correct"],
        "accuracy_pct": totals["correct"] / count * 100,
        "parseable": totals["parseable"],
        "calls": totals["calls"], "calls_per_question": totals["calls"] / count,
        "invalid_calls": totals["invalid"], "reached_budget": totals["reached_budget"],
        "final_status": dict(status),
        "cache_calls": totals["cache_calls"],
        "cache_usage_estimated": totals["cache_usage_estimated"],
        "live_input_per_question": totals["live_input"] / count,
        "live_output_per_question": totals["live_output"] / count,
        "live_cost_per_question": totals["live_cost"] / count,
        "cache_input_per_question": totals["cache_input"] / count,
        "cache_output_per_question": totals["cache_output"] / count,
        "cache_cost_per_question": totals["cache_cost"] / count,
        "input_per_question": (totals["live_input"] + totals["cache_input"]) / count,
        "output_per_question": (totals["live_output"] + totals["cache_output"]) / count,
        "cost_per_question": (totals["live_cost"] + totals["cache_cost"]) / count,
        "measured_latency_seconds": statistics.mean(latencies) if latencies else None,
        "latency_coverage": len(latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/figures/modality_merge_ablation"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    durations = {
        row["id"]: float(row["duration"])
        for line in (DATA / "daily_omni_cuts_selectedVal125.jsonl").open()
        if (row := json.loads(line))
    }
    audio_calibration = calibration(durations)
    omni_fractions = omni_audio_fractions(durations)
    individual = []
    aggregate = []
    for experiment_id in IDS:
        runs = []
        for repeat in (1, 2, 3):
            run = summarize_run(
                result_path(experiment_id, repeat), durations,
                audio_calibration, omni_fractions,
            )
            run["id"] = experiment_id
            run["repeat"] = repeat
            individual.append(run)
            runs.append(run)
        aggregate.append({
            "id": experiment_id,
            **{
                field: mean_std([float(run[field]) for run in runs])
                for field in (
                    "accuracy_pct", "calls_per_question", "input_per_question",
                    "output_per_question", "cost_per_question", "measured_latency_seconds",
                )
            },
            "cache_calls": sum(run["cache_calls"] for run in runs),
            "cache_usage_estimated": sum(run["cache_usage_estimated"] for run in runs),
            "parseable": sum(run["parseable"] for run in runs),
            "final_status": dict(sum((Counter(run["final_status"]) for run in runs), Counter())),
        })
    audit = {
        "method": "One source generation call attributed to each cached caption invocation",
        "latency_note": "Measured cache-read rollout latency; source generation latency unavailable for all omni captions and legacy audio-caption traces",
        "audio_calibration": audio_calibration,
        "omni_audio_fraction_by_duration": omni_fractions,
        "individual": individual,
        "aggregate": aggregate,
    }
    (args.output_dir / "cache_inclusive_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = args.output_dir / "ablation_tradeoff_data_cache_inclusive.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "accuracy_mean_pct", "total_tokens_mean",
                         "cost_usd_mean", "measured_latency_seconds_mean"])
        for row in aggregate:
            writer.writerow([row["id"], row["accuracy_pct"]["mean"],
                             row["input_per_question"]["mean"] + row["output_per_question"]["mean"],
                             row["cost_per_question"]["mean"],
                             row["measured_latency_seconds"]["mean"]])
    print(args.output_dir / "cache_inclusive_audit.json")
    print(csv_path)


if __name__ == "__main__":
    main()
