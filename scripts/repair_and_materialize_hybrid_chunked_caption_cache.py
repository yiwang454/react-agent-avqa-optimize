#!/usr/bin/env python3
"""Repair stored chunk captions and use whole-video captions for unresolved videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_DIR = Path(__file__).resolve().parents[1]
DSPY_DIR = REPO_DIR / "DSPy"
if str(DSPY_DIR) not in sys.path:
    sys.path.insert(0, str(DSPY_DIR))

from dspy_avqa.caption_cache import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    MANIFEST_NAME,
    load_caption_cache_manifest,
    validate_caption_cache_coverage,
)
from dspy_avqa.chunked_caption import (  # noqa: E402
    _assemble_video,
    _atomic_json,
    _refresh_parsed_chunk,
    _sha256_text,
    _video_artifact_path,
    load_chunk_prompt_config,
    load_video_jobs_from_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--chunk-output-dir", type=Path, required=True)
    parser.add_argument("--whole-caption-cache-dir", type=Path, required=True)
    parser.add_argument("--prompt-yaml", type=Path, required=True)
    parser.add_argument(
        "--chunk-prompt-profile",
        default="CHUNKED_CAPTION_PROMPT_SPLIT_AVS",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def cache_entry_path(cache_dir: Path, question_id: str) -> Path:
    if Path(question_id).name != question_id or "/" in question_id or "\\" in question_id:
        raise ValueError(f"Unsafe question ID: {question_id!r}")
    return cache_dir / f"{question_id}.json"


def materialize_hybrid_cache(
    *,
    jobs: list[Any],
    video_results: list[Mapping[str, Any]],
    fallback_video_ids: set[str],
    cache_dir: Path,
    whole_cache_dir: Path,
    input_jsonl: Path,
    prompt_yaml: Path,
    base_prompt: str,
) -> dict[str, Any]:
    if cache_dir.exists() and any(cache_dir.iterdir()):
        return load_caption_cache_manifest(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    by_video = {str(result["video_id"]): result for result in video_results}
    entries: dict[str, dict[str, Any]] = {}
    fallback_question_count = 0

    for job in jobs:
        video_result = by_video[job.video_id]
        use_fallback = job.video_id in fallback_video_ids
        if not use_fallback and video_result.get("status") != "complete":
            raise ValueError(f"Non-fallback video is incomplete: {job.video_id}")
        for question_id in job.question_ids:
            if use_fallback:
                source_path = cache_entry_path(whole_cache_dir, question_id)
                source = load_json(source_path)
                response = str(source.get("response") or "").strip()
                if not response:
                    raise ValueError(f"Empty whole-video fallback caption: {source_path}")
                entry = {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "question_id": question_id,
                    "response": response,
                    "caption_prompt": base_prompt,
                    "source_results_file": str(source_path),
                    "source_rank": 1,
                    "source_backend": str(source.get("source_backend") or "gemini_whole_video"),
                    "source_model": str(source.get("source_model") or ""),
                    "source_token_usage": source.get("source_token_usage"),
                    "source_video_id": job.video_id,
                    "chunk_count": 1,
                    "timestamp_mode": str(source.get("timestamp_mode") or "model_raw_whole_video"),
                    "video_latency": source.get("video_latency"),
                    "caption_strategy": "whole_video_timestamp3_fallback",
                    "fallback_from_chunked": True,
                    "source_caption_prompt": str(source.get("caption_prompt") or ""),
                }
                fallback_question_count += 1
            else:
                entry = {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "question_id": question_id,
                    "response": str(video_result["combined_response"]),
                    "caption_prompt": base_prompt,
                    "source_results_file": str(
                        _video_artifact_path(cache_dir.parent, job.video_id)
                    ),
                    "source_rank": 1,
                    "source_backend": "gemini_chunked",
                    "source_model": os.environ.get("CAPTIONER_GEMINI_MODEL", "gemini-2.5-flash"),
                    "source_token_usage": video_result.get("token_usage"),
                    "source_video_id": job.video_id,
                    "chunk_count": video_result.get("chunk_count"),
                    "timestamp_mode": "local_labels_with_original_timestamp_values",
                    "video_latency": video_result.get("latency"),
                    "caption_strategy": "split_avs_plan_a_rewritten_original_timestamps",
                    "fallback_from_chunked": False,
                    "source_caption_prompt": base_prompt,
                }
            entries[question_id] = entry

    digest = hashlib.sha256()
    for question_id in sorted(entries):
        entry_text = json.dumps(
            entries[question_id],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(question_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry_text.encode("utf-8"))
        digest.update(b"\n")
        _atomic_json(cache_entry_path(cache_dir, question_id), entries[question_id])

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "input_jsonl": str(input_jsonl),
        "prompt_yaml": str(prompt_yaml),
        "caption_prompt": base_prompt,
        "prompt_sha256": _sha256_text(base_prompt),
        "generation_strategy": "split_avs_plan_a_with_whole_timestamp3_video_fallback",
        "source_backend": "gemini_chunked_hybrid",
        "source_model": os.environ.get("CAPTIONER_GEMINI_MODEL", "gemini-2.5-flash"),
        "video_count": len(jobs),
        "chunked_video_count": len(jobs) - len(fallback_video_ids),
        "fallback_video_count": len(fallback_video_ids),
        "fallback_video_ids": sorted(fallback_video_ids),
        "question_count": len(entries),
        "fallback_question_count": fallback_question_count,
        "content_sha256": digest.hexdigest(),
    }
    _atomic_json(cache_dir / MANIFEST_NAME, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    input_jsonl = args.input_jsonl.expanduser().resolve()
    output_dir = args.chunk_output_dir.expanduser().resolve()
    whole_cache_dir = args.whole_caption_cache_dir.expanduser().resolve()
    prompt_yaml = args.prompt_yaml.expanduser().resolve()
    prompt_config = load_chunk_prompt_config(
        prompt_yaml,
        profile_key=args.chunk_prompt_profile,
    )
    base_prompt = str(prompt_config["base_instruction"]).strip()
    jobs = load_video_jobs_from_jsonl(input_jsonl)
    repaired_chunks: list[dict[str, Any]] = []
    unresolved_chunks: list[dict[str, Any]] = []
    video_results: list[dict[str, Any]] = []

    for job in jobs:
        video_path = _video_artifact_path(output_dir, job.video_id)
        previous_video = load_json(video_path)
        refreshed_chunks: list[dict[str, Any]] = []
        for chunk_path in sorted(video_path.parent.glob("chunk_*.json")):
            chunk = load_json(chunk_path)
            previous_status = str(chunk.get("status") or "")
            try:
                refreshed = _refresh_parsed_chunk(chunk_path, chunk)
            except ValueError as exc:
                refreshed = chunk
                unresolved_chunks.append(
                    {
                        "video_id": job.video_id,
                        "chunk_index": chunk.get("chunk_index"),
                        "error": str(exc),
                        "artifact": str(chunk_path),
                    }
                )
            else:
                repair_count = int(
                    (refreshed.get("deterministic_timestamp_repairs") or {}).get(
                        "missing_second_closing_bracket", 0
                    )
                )
                if repair_count:
                    repaired_chunks.append(
                        {
                            "video_id": job.video_id,
                            "chunk_index": refreshed.get("chunk_index"),
                            "repair_count": repair_count,
                            "newly_repaired_in_this_invocation": previous_status != "complete",
                            "artifact": str(chunk_path),
                        }
                    )
            refreshed_chunks.append(refreshed)
        if len(refreshed_chunks) != int(previous_video.get("chunk_count") or 0):
            raise ValueError(
                f"Chunk artifact count mismatch for {job.video_id}: "
                f"expected {previous_video.get('chunk_count')}, found {len(refreshed_chunks)}"
            )
        video_results.append(
            _assemble_video(
                job=job,
                probed_duration_s=float(previous_video["probed_duration_s"]),
                chunks=refreshed_chunks,
                mode="local_offset",
                base_prompt=base_prompt,
                output_dir=output_dir,
            )
        )

    fallback_video_ids = {str(item["video_id"]) for item in unresolved_chunks}
    manifest = materialize_hybrid_cache(
        jobs=jobs,
        video_results=video_results,
        fallback_video_ids=fallback_video_ids,
        cache_dir=output_dir / "caption_cache",
        whole_cache_dir=whole_cache_dir,
        input_jsonl=input_jsonl,
        prompt_yaml=prompt_yaml,
        base_prompt=base_prompt,
    )
    question_ids = [question_id for job in jobs for question_id in job.question_ids]
    validation = validate_caption_cache_coverage(
        output_dir / "caption_cache",
        question_ids,
        expected_prompt=base_prompt,
    )
    report = {
        "schema_version": 1,
        "repaired_chunk_count": len(repaired_chunks),
        "repaired_chunks": repaired_chunks,
        "unresolved_chunk_count": len(unresolved_chunks),
        "unresolved_chunks": unresolved_chunks,
        "fallback_video_count": len(fallback_video_ids),
        "fallback_video_ids": sorted(fallback_video_ids),
        "cache_manifest": manifest,
        "cache_validation": validation,
    }
    _atomic_json(output_dir / "hybrid_materialization_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
