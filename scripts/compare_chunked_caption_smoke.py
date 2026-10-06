#!/usr/bin/env python3
"""Render a side-by-side report for local-offset and model-original smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-output-dir", type=Path, required=True)
    parser.add_argument("--original-output-dir", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _load_videos(root: Path) -> dict[str, dict[str, Any]]:
    videos: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "video_artifacts").glob("*/video.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        videos[str(payload["video_id"])] = payload
    return videos


def _mode_metrics(video: dict[str, Any]) -> dict[str, Any]:
    chunks = video.get("chunks") or []
    event_durations = [
        float(event["original_end_s"]) - float(event["original_start_s"])
        for chunk in chunks
        for event in (chunk.get("events") or [])
    ]
    covered_seconds = 0.0
    total_seconds = 0.0
    for chunk in chunks:
        chunk_start = float(chunk.get("actual_chunk_start_s") or 0.0)
        chunk_end = float(chunk.get("actual_chunk_end_s") or chunk_start)
        total_seconds += max(0.0, chunk_end - chunk_start)
        cursor = chunk_start
        for event in sorted(
            chunk.get("events") or [], key=lambda item: float(item["original_start_s"])
        ):
            start = max(cursor, float(event["original_start_s"]))
            end = max(start, float(event["original_end_s"]))
            covered_seconds += max(0.0, end - start)
            cursor = max(cursor, end)
    return {
        "status": video.get("status"),
        "chunks": len(chunks),
        "events": sum(len(chunk.get("events") or []) for chunk in chunks),
        "warnings": sum(len(chunk.get("format_warnings") or []) for chunk in chunks),
        "error_chunks": sum(chunk.get("status") != "complete" for chunk in chunks),
        "timeline_coverage_percent": round(100 * covered_seconds / total_seconds, 2)
        if total_seconds
        else None,
        "sub_half_second_events": sum(duration < 0.5 for duration in event_durations),
        "assumed_parallel_retry_adjusted_seconds": (video.get("latency") or {}).get(
            "assumed_parallel_retry_adjusted_seconds"
        ),
        "sum_chunk_retry_adjusted_seconds": (video.get("latency") or {}).get(
            "sum_chunk_retry_adjusted_seconds"
        ),
        "observed_worker_span_seconds": (video.get("latency") or {}).get(
            "observed_worker_span_seconds"
        ),
    }


def main() -> int:
    args = parse_args()
    local = _load_videos(args.local_output_dir.expanduser().resolve())
    original = _load_videos(args.original_output_dir.expanduser().resolve())
    if set(local) != set(original):
        raise ValueError(
            f"Video mismatch: local_only={sorted(set(local) - set(original))}, "
            f"original_only={sorted(set(original) - set(local))}"
        )
    comparison = {
        video_id: {
            "local_offset": _mode_metrics(local[video_id]),
            "model_original": _mode_metrics(original[video_id]),
        }
        for video_id in sorted(local)
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Chunked caption timestamp smoke test",
        "",
        "`local_offset` asks Gemini for chunk-local tagged clock times and shifts them "
        "deterministically. `model_original` asks Gemini to emit original-video seconds and "
        "retains that raw response without timestamp normalization.",
        "",
        "| Video | Mode | Status | Chunks | Events | Format warnings | Error chunks | "
        "Timeline coverage | <0.5s events | Parallel adjusted s (max) | "
        "Adjusted s (sum) | Observed worker span s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for video_id in sorted(comparison):
        for mode in ("local_offset", "model_original"):
            metric = comparison[video_id][mode]
            lines.append(
                "| {video} | {mode} | {status} | {chunks} | {events} | {warnings} | "
                "{errors} | {coverage}% | {short_events} | {parallel} | {summed} | {span} |".format(
                    video=video_id,
                    mode=mode,
                    status=metric["status"],
                    chunks=metric["chunks"],
                    events=metric["events"],
                    warnings=metric["warnings"],
                    errors=metric["error_chunks"],
                    coverage=metric["timeline_coverage_percent"],
                    short_events=metric["sub_half_second_events"],
                    parallel=metric["assumed_parallel_retry_adjusted_seconds"],
                    summed=metric["sum_chunk_retry_adjusted_seconds"],
                    span=metric["observed_worker_span_seconds"],
                )
            )
    for video_id in sorted(comparison):
        lines.extend(["", f"## {video_id}"])
        for mode, videos in (("local_offset", local), ("model_original", original)):
            video = videos[video_id]
            lines.extend(["", f"### {mode}", ""])
            for chunk in sorted(video.get("chunks") or [], key=lambda value: value["chunk_index"]):
                lines.extend(
                    [
                        f"#### Chunk {int(chunk['chunk_index']) + 1}: "
                        f"{chunk.get('actual_chunk_start_s')}–{chunk.get('actual_chunk_end_s')} s",
                        "",
                        f"Status: `{chunk.get('status')}`; warnings: "
                        f"`{len(chunk.get('format_warnings') or [])}`; retry-adjusted latency: "
                        f"`{(chunk.get('latency') or {}).get('retry_adjusted_seconds')}` s.",
                        "",
                        "Raw response:",
                        "",
                        "```text",
                        str(chunk.get("raw_response") or chunk.get("error") or ""),
                        "```",
                        "",
                        (
                            "Normalized original-video timestamps:"
                            if mode == "local_offset"
                            else "Timestamp post-processing: not applied; raw response retained."
                        ),
                        "",
                        "```text",
                        (
                            str(chunk.get("rendered_original_timestamps") or "")
                            if mode == "local_offset"
                            else "(not applied)"
                        ),
                        "```",
                    ]
                )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_markdown.expanduser().resolve())
    print(args.output_json.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
