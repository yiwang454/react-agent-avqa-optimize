#!/usr/bin/env python3
"""Compare Qwen3-Omni audio ingestion and media ordering under concurrency."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import random
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI


DEFAULT_SOURCE_DIR = Path(
    "/mnt/ceph_rbd/data/avqa_project/daily_omni/"
    "daily_omni_dspy_qwen_v8GeminiCaptionInTask_qwen_seed_1234_"
    "deepseek_seed_7_repeat1"
)
DEFAULT_CONFIG = Path(
    "DSPy/dspy_avqa/yamls/config_localqwen_api_instruct.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/ceph_rbd/data/avqa_project/daily_omni/"
    "qwen_av_order_concurrency_tests"
)
VARIANTS = (
    {
        "name": "embedded_audio",
        "use_audio_in_video": True,
        "media_order": ("video",),
    },
    {
        "name": "separate_audio_first",
        "use_audio_in_video": False,
        "media_order": ("audio", "video"),
    },
    {
        "name": "separate_video_first",
        "use_audio_in_video": False,
        "media_order": ("video", "audio"),
    },
)
PRINT_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run no-retry Qwen3-Omni requests on historical empty ReACT calls "
            "using three audio/video payload layouts."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--config-yaml", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples-per-stratum", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--schedule",
        choices=("grouped", "shuffled"),
        default="grouped",
        help=(
            "grouped keeps the original sample/variant/repetition batching; "
            "shuffled runs all requests in a globally shuffled order and "
            "tries to keep the same sample out of the same concurrent batch."
        ),
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Seed for --schedule shuffled. Defaults to sampling_params.seed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def is_empty(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip()


def prompt_bucket(prompt_tokens: int) -> str:
    if prompt_tokens < 5000:
        return "low"
    if prompt_tokens < 10000:
        return "mid"
    return "high"


def history_behavior(completion_tokens: int, max_tokens: int) -> str:
    if completion_tokens >= max_tokens - 96:
        return "max_token_empty"
    return "short_empty"


def load_historical_empty_calls(
    source_dir: Path,
    *,
    max_tokens: int,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for path in sorted(source_dir.glob("*.json")):
        if path.name.startswith("output_test"):
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                sample = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        for turn in sample.get("turn_trace", []):
            if turn.get("perception_backend") != "qwen":
                continue
            if turn.get("tool_name") not in {"ask_perception", "ask_caption"}:
                continue
            if not is_empty(turn.get("tool_observation")):
                continue

            tool_args = turn.get("tool_args") or {}
            usage = turn.get("perception_token_usage") or {}
            video_path = Path(str(tool_args.get("video_path", "")))
            audio_path = Path(str(tool_args.get("audio_path", "")))
            prompt = turn.get("perception_prompt")
            if (
                not isinstance(prompt, str)
                or not video_path.is_file()
                or not audio_path.is_file()
            ):
                continue

            prompt_tokens = int(usage.get("prompt_tokens") or -1)
            completion_tokens = int(usage.get("completion_tokens") or -1)
            calls.append(
                {
                    "source_file": path.name,
                    "turn_id": turn.get("turn_id"),
                    "tool_name": turn.get("tool_name"),
                    "video_id": turn.get("video_id"),
                    "video_path": str(video_path),
                    "audio_path": str(audio_path),
                    "system_prompt": turn.get("perception_system_prompt") or "",
                    "prompt": prompt,
                    "historical_prompt_tokens": prompt_tokens,
                    "historical_completion_tokens": completion_tokens,
                    "history_behavior": history_behavior(
                        completion_tokens, max_tokens
                    ),
                    "prompt_bucket": prompt_bucket(prompt_tokens),
                }
            )
    return calls


def select_stratified(
    calls: list[dict[str, Any]],
    *,
    samples_per_stratum: int,
) -> list[dict[str, Any]]:
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        key = (call["history_behavior"], call["prompt_bucket"])
        by_stratum[key].append(call)

    priorities = {
        ("short_empty", "low"): ("00FBAdjlF4g-2.json",),
        ("max_token_empty", "mid"): ("h7ljgWAgb0E-2.json",),
    }
    selected: list[dict[str, Any]] = []
    expected_strata = (
        ("short_empty", "low"),
        ("short_empty", "mid"),
        ("short_empty", "high"),
        ("max_token_empty", "low"),
        ("max_token_empty", "mid"),
        ("max_token_empty", "high"),
    )
    used_video_ids: set[str] = set()

    for key in expected_strata:
        candidates = sorted(
            by_stratum[key],
            key=lambda item: item["source_file"],
        )
        priority_names = priorities.get(key, ())
        candidates.sort(
            key=lambda item: (
                item["source_file"] not in priority_names,
                priority_names.index(item["source_file"])
                if item["source_file"] in priority_names
                else 0,
                item["source_file"],
            )
        )
        stratum_selected = []
        for candidate in candidates:
            video_id = str(candidate["video_id"])
            if video_id in used_video_ids:
                continue
            stratum_selected.append(candidate)
            used_video_ids.add(video_id)
            if len(stratum_selected) == samples_per_stratum:
                break
        if len(stratum_selected) != samples_per_stratum:
            raise ValueError(
                f"Stratum {key} has only {len(stratum_selected)} usable unique videos; "
                f"requested {samples_per_stratum}"
            )
        selected.extend(stratum_selected)
    return selected


def data_url(path_string: str) -> str:
    path = Path(path_string)
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text", item.get("content"))
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)
    return str(value)


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    if isinstance(usage, dict):
        return usage
    return {}


def build_messages(
    sample: dict[str, Any],
    media: dict[str, dict[str, str]],
    variant: dict[str, Any],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for media_type in variant["media_order"]:
        if media_type == "video":
            content.append(
                {"type": "video_url", "video_url": {"url": media["video"]}}
            )
        else:
            content.append(
                {"type": "audio_url", "audio_url": {"url": media["audio"]}}
            )
    content.append({"type": "text", "text": sample["prompt"]})

    messages: list[dict[str, Any]] = []
    if sample["system_prompt"]:
        messages.append({"role": "system", "content": sample["system_prompt"]})
    messages.append({"role": "user", "content": content})
    return messages


def run_request(
    *,
    sample: dict[str, Any],
    media: dict[str, dict[str, str]],
    variant: dict[str, Any],
    repetition: int,
    request_index: int,
    base_url: str,
    api_key: str,
    model: str,
    task: dict[str, Any],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    result = {
        "request_index": request_index,
        "repetition": repetition,
        "variant": variant["name"],
        "use_audio_in_video": variant["use_audio_in_video"],
        "media_order": list(variant["media_order"]),
        "source_file": sample["source_file"],
        "turn_id": sample["turn_id"],
        "video_id": sample["video_id"],
        "history_behavior": sample["history_behavior"],
        "prompt_bucket": sample["prompt_bucket"],
        "historical_prompt_tokens": sample["historical_prompt_tokens"],
        "historical_completion_tokens": sample["historical_completion_tokens"],
    }
    try:
        response = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        ).chat.completions.create(
            model=model,
            messages=build_messages(sample, media, variant),
            stream=False,
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            max_tokens=int(sampling["max_tokens"]),
            seed=int(sampling["seed"]),
            timeout=int(task["timeout"]),
            extra_body={
                "top_k": int(sampling["top_k"]),
                "mm_processor_kwargs": {
                    "fps": float(sampling["fps"]),
                    "max_frames": int(sampling["max_frames"]),
                    "use_audio_in_video": variant["use_audio_in_video"],
                },
                "chat_template_kwargs": {
                    "enable_thinking": bool(task["enable_thinking"])
                },
            },
        )
        message = response.choices[0].message
        content = text_content(getattr(message, "content", None))
        reasoning = text_content(getattr(message, "reasoning_content", None))
        usage = usage_dict(response)
        result.update(
            {
                "status": "ok",
                "response_id": getattr(response, "id", None),
                "finish_reason": response.choices[0].finish_reason,
                "content": content,
                "content_chars": len(content),
                "stripped_content_chars": len(content.strip()),
                "content_empty_after_strip": not bool(content.strip()),
                "whitespace_only": bool(content) and not bool(content.strip()),
                "reasoning": reasoning,
                "reasoning_chars": len(reasoning),
                "usage": usage,
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "content_empty_after_strip": None,
            }
        )
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant in VARIANTS:
        name = variant["name"]
        rows = [row for row in results if row["variant"] == name]
        ok_rows = [row for row in rows if row["status"] == "ok"]
        empty_rows = [
            row for row in ok_rows if row["content_empty_after_strip"]
        ]
        completion_tokens = [
            int(row["usage"]["completion_tokens"])
            for row in ok_rows
            if row.get("usage", {}).get("completion_tokens") is not None
        ]
        summary[name] = {
            "requests": len(rows),
            "ok": len(ok_rows),
            "errors": len(rows) - len(ok_rows),
            "empty": len(empty_rows),
            "empty_rate": round(len(empty_rows) / len(ok_rows), 6)
            if ok_rows
            else None,
            "whitespace_only": sum(row["whitespace_only"] for row in ok_rows),
            "finish_reasons": dict(
                Counter(str(row.get("finish_reason")) for row in ok_rows)
            ),
            "completion_tokens": {
                "min": min(completion_tokens) if completion_tokens else None,
                "median": statistics.median(completion_tokens)
                if completion_tokens
                else None,
                "max": max(completion_tokens) if completion_tokens else None,
            },
            "mean_elapsed_s": round(
                statistics.mean(row["elapsed_s"] for row in rows), 3
            )
            if rows
            else None,
        }
    return summary


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def endpoint_available(base_url: str, api_key: str) -> bool:
    try:
        OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=10,
        ).models.list()
    except Exception:
        return False
    return True


def build_request_tasks(
    selected: list[dict[str, Any]],
    *,
    repetitions: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(selected):
        offset = sample_index % len(VARIANTS)
        variant_order = VARIANTS[offset:] + VARIANTS[:offset]
        for variant_position, variant in enumerate(variant_order):
            for repetition in range(1, repetitions + 1):
                tasks.append(
                    {
                        "sample": sample,
                        "sample_index": sample_index,
                        "variant": variant,
                        "variant_position": variant_position,
                        "repetition": repetition,
                    }
                )
    return tasks


def shuffled_batches(
    tasks: list[dict[str, Any]],
    *,
    concurrency: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    remaining = list(tasks)
    random.Random(seed).shuffle(remaining)
    batches: list[list[dict[str, Any]]] = []

    while remaining:
        batch: list[dict[str, Any]] = []
        used_sources: set[str] = set()
        used_source_variants: set[tuple[str, str]] = set()

        while remaining and len(batch) < concurrency:
            chosen_index = None
            for index, task in enumerate(remaining):
                source_file = task["sample"]["source_file"]
                source_variant = (source_file, task["variant"]["name"])
                if (
                    source_file not in used_sources
                    and source_variant not in used_source_variants
                ):
                    chosen_index = index
                    break
            if chosen_index is None:
                for index, task in enumerate(remaining):
                    source_variant = (
                        task["sample"]["source_file"],
                        task["variant"]["name"],
                    )
                    if source_variant not in used_source_variants:
                        chosen_index = index
                        break
            if chosen_index is None:
                chosen_index = 0

            task = remaining.pop(chosen_index)
            source_file = task["sample"]["source_file"]
            batch.append(task)
            used_sources.add(source_file)
            used_source_variants.add((source_file, task["variant"]["name"]))

        batches.append(batch)

    return batches


def run_concurrent_batch(
    *,
    batch: list[dict[str, Any]],
    batch_index: int,
    request_index_start: int,
    results: list[dict[str, Any]],
    results_path: Path,
    media_by_source: dict[str, dict[str, str]],
    base_url: str,
    api_key: str,
    model: str,
    task: dict[str, Any],
    sampling: dict[str, Any],
    concurrency: int,
) -> tuple[int, list[dict[str, Any]]]:
    batch_results: list[dict[str, Any]] = []
    request_index = request_index_start
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for request_task in batch:
            request_index += 1
            request_task["request_index"] = request_index
            sample = request_task["sample"]
            futures.append(
                executor.submit(
                    run_request,
                    sample=sample,
                    media=media_by_source[sample["source_file"]],
                    variant=request_task["variant"],
                    repetition=request_task["repetition"],
                    request_index=request_index,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    task=task,
                    sampling=sampling,
                )
            )
        for future in as_completed(futures):
            row = future.result()
            row["batch_index"] = batch_index
            results.append(row)
            batch_results.append(row)
            append_jsonl(results_path, row)
            with PRINT_LOCK:
                print(
                    f"  batch={batch_index} {row['variant']} "
                    f"rep={row['repetition']} sample={row['source_file']} "
                    f"status={row['status']} "
                    f"empty={row.get('content_empty_after_strip')} "
                    f"completion_tokens="
                    f"{row.get('usage', {}).get('completion_tokens')} "
                    f"elapsed={row['elapsed_s']}s",
                    flush=True,
                )
    return request_index, batch_results


def run_grouped_schedule(
    *,
    selected: list[dict[str, Any]],
    results: list[dict[str, Any]],
    results_path: Path,
    media_by_source: dict[str, dict[str, str]],
    base_url: str,
    api_key: str,
    model: str,
    task: dict[str, Any],
    sampling: dict[str, Any],
    repetitions: int,
    concurrency: int,
) -> str | None:
    request_index = 0
    batch_index = 0
    for sample_index, sample in enumerate(selected):
        offset = sample_index % len(VARIANTS)
        variant_order = VARIANTS[offset:] + VARIANTS[:offset]
        print(
            f"Sample {sample_index + 1}/{len(selected)} "
            f"{sample['source_file']}, group order: "
            + ", ".join(variant["name"] for variant in variant_order),
            flush=True,
        )
        for variant_position, variant in enumerate(variant_order):
            repetitions_order = list(range(1, repetitions + 1))
            random.Random(
                int(sampling["seed"]) + sample_index * 100 + variant_position
            ).shuffle(repetitions_order)
            batch_index += 1
            batch = [
                {
                    "sample": sample,
                    "sample_index": sample_index,
                    "variant": variant,
                    "variant_position": variant_position,
                    "repetition": repetition,
                }
                for repetition in repetitions_order
            ]
            request_index, group_results = run_concurrent_batch(
                batch=batch,
                batch_index=batch_index,
                request_index_start=request_index,
                results=results,
                results_path=results_path,
                media_by_source=media_by_source,
                base_url=base_url,
                api_key=api_key,
                model=model,
                task=task,
                sampling=sampling,
                concurrency=concurrency,
            )
            group_empty = sum(
                row.get("content_empty_after_strip") is True
                for row in group_results
            )
            group_errors = sum(row["status"] == "error" for row in group_results)
            print(
                f"Group complete: {variant['name']} "
                f"sample={sample['source_file']} "
                f"empty={group_empty}/{len(group_results)} errors={group_errors}",
                flush=True,
            )
            if group_errors == len(group_results) and not endpoint_available(
                base_url, api_key
            ):
                return (
                    f"Endpoint stopped responding after {variant['name']} "
                    f"for {sample['source_file']}"
                )
            if task.get("qwen_delay_s"):
                time.sleep(float(task["qwen_delay_s"]))
    return None


def run_shuffled_schedule(
    *,
    selected: list[dict[str, Any]],
    results: list[dict[str, Any]],
    results_path: Path,
    media_by_source: dict[str, dict[str, str]],
    base_url: str,
    api_key: str,
    model: str,
    task: dict[str, Any],
    sampling: dict[str, Any],
    repetitions: int,
    concurrency: int,
    shuffle_seed: int,
) -> str | None:
    tasks = build_request_tasks(selected, repetitions=repetitions)
    batches = shuffled_batches(
        tasks,
        concurrency=concurrency,
        seed=shuffle_seed,
    )
    request_index = 0
    for batch_index, batch in enumerate(batches, start=1):
        batch_label = ", ".join(
            f"{item['sample']['source_file']}:{item['variant']['name']}:"
            f"r{item['repetition']}"
            for item in batch
        )
        print(
            f"Batch {batch_index}/{len(batches)} shuffled: {batch_label}",
            flush=True,
        )
        request_index, batch_results = run_concurrent_batch(
            batch=batch,
            batch_index=batch_index,
            request_index_start=request_index,
            results=results,
            results_path=results_path,
            media_by_source=media_by_source,
            base_url=base_url,
            api_key=api_key,
            model=model,
            task=task,
            sampling=sampling,
            concurrency=concurrency,
        )
        batch_empty = sum(
            row.get("content_empty_after_strip") is True
            for row in batch_results
        )
        batch_errors = sum(row["status"] == "error" for row in batch_results)
        repeated_sources = len(
            {row["source_file"] for row in batch_results}
        ) != len(batch_results)
        print(
            f"Batch complete: {batch_index} empty={batch_empty}/"
            f"{len(batch_results)} errors={batch_errors} "
            f"repeated_source={repeated_sources}",
            flush=True,
        )
        if batch_errors == len(batch_results) and not endpoint_available(
            base_url, api_key
        ):
            return f"Endpoint stopped responding after shuffled batch {batch_index}"
        if task.get("qwen_delay_s"):
            time.sleep(float(task["qwen_delay_s"]))
    return None


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config_yaml)
    task = config["task"]
    model_config = config["model"]
    sampling = config["sampling_params"]
    base_url = args.base_url or model_config["qwen_base_url"]
    model = model_config["qwen_model"]
    api_key = model_config.get("qwen_api_key") or "EMPTY"
    max_tokens = int(sampling["max_tokens"])
    shuffle_seed = (
        args.shuffle_seed
        if args.shuffle_seed is not None
        else int(sampling["seed"])
    )

    historical_calls = load_historical_empty_calls(
        args.source_dir,
        max_tokens=max_tokens,
    )
    selected = select_stratified(
        historical_calls,
        samples_per_stratum=args.samples_per_stratum,
    )
    print(
        f"Selected {len(selected)} of {len(historical_calls)} historical empty "
        f"Qwen calls from {args.source_dir}",
        flush=True,
    )
    for sample in selected:
        print(
            f"  {sample['source_file']} {sample['history_behavior']}/"
            f"{sample['prompt_bucket']} "
            f"history_tokens={sample['historical_completion_tokens']}",
            flush=True,
        )
    if args.dry_run:
        return

    if not endpoint_available(base_url, api_key):
        raise RuntimeError(f"Qwen endpoint is unavailable: {base_url}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    results_path = output_dir / "results.jsonl"
    manifest_path = output_dir / "manifest.json"

    print("Encoding selected media once...", flush=True)
    media_by_source = {
        sample["source_file"]: {
            "video": data_url(sample["video_path"]),
            "audio": data_url(sample["audio_path"]),
        }
        for sample in selected
    }
    manifest = {
        "created_at": timestamp,
        "source_dir": str(args.source_dir),
        "config_yaml": str(args.config_yaml),
        "base_url": base_url,
        "model": model,
        "no_retry": True,
        "effective_application_retries": 0,
        "effective_transport_retries": 0,
        "stream": False,
        "repetitions": args.repetitions,
        "concurrency": args.concurrency,
        "schedule": args.schedule,
        "shuffle_seed": shuffle_seed if args.schedule == "shuffled" else None,
        "group_delay_s": float(task.get("qwen_delay_s", 0.0)),
        "task": task,
        "sampling_params": sampling,
        "variants": list(VARIANTS),
        "selected_samples": selected,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    results: list[dict[str, Any]] = []
    if args.schedule == "grouped":
        aborted_reason = run_grouped_schedule(
            selected=selected,
            results=results,
            results_path=results_path,
            media_by_source=media_by_source,
            base_url=base_url,
            api_key=api_key,
            model=model,
            task=task,
            sampling=sampling,
            repetitions=args.repetitions,
            concurrency=args.concurrency,
        )
    else:
        aborted_reason = run_shuffled_schedule(
            selected=selected,
            results=results,
            results_path=results_path,
            media_by_source=media_by_source,
            base_url=base_url,
            api_key=api_key,
            model=model,
            task=task,
            sampling=sampling,
            repetitions=args.repetitions,
            concurrency=args.concurrency,
            shuffle_seed=shuffle_seed,
        )
    if aborted_reason:
        print(f"Aborting: {aborted_reason}", flush=True)

    final_summary = summarize(results)
    manifest["summary"] = final_summary
    manifest["status"] = "aborted" if aborted_reason else "complete"
    manifest["aborted_reason"] = aborted_reason
    manifest["finished_at"] = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Results: {results_path}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
