#!/usr/bin/env python3
"""Run raw-format chunked and whole-video TIMESTAMP3 caption comparisons."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "DSPy"))
from dspy_avqa.chunked_caption import plan_chunks, ChunkRange, aggregate_video_latency
from dspy_avqa.latency import sample_latency, model_component
from dspy_avqa.runner import load_gemini_captioner_config_yaml
from dspy_avqa.tools import (call_gemini_perception, consume_last_perception_metadata,
                             cut_video_clip, probe_media_duration)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video-path", type=Path, action="append", required=True)
    parser.add_argument("--captioner-config-yaml", type=Path, required=True)
    args = parser.parse_args()
    # Exclusive creation protects every existing experiment, including this one.
    args.output_root.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((REPO / "DSPy/dspy_avqa/yamls/omni_caption_prompt.yaml").read_text())
    base = config["QA_PROMPT_OMNI_CAPTIONER_TIMESTAMP3"].strip()
    contract = config["CHUNKED_CAPTION_PROMPT"]["modes"]["model_original"]["time_contract"]
    os.environ["GEMINI_API_BACKEND"] = "legacy"
    os.environ["CAPTIONER_MODEL"] = "gemini"
    load_gemini_captioner_config_yaml(args.captioner_config_yaml)
    durations = {str(p): probe_media_duration(str(p)) for p in args.video_path}
    manifest = {"base_prompt": base, "chunk_time_contract": contract,
                "captioner_config_yaml": str(args.captioner_config_yaml),
                "video_durations": durations, "max_workers": 5,
                "normalization": False, "parser": "none; raw outputs retained",
                "runtime": {k: os.environ.get(k) for k in (
                    "CAPTIONER_GEMINI_MODEL", "CAPTIONER_GEMINI_TEMPERATURE",
                    "CAPTIONER_GEMINI_MAX_TOKENS", "CAPTIONER_GEMINI_TIMEOUT",
                    "CAPTIONER_GEMINI_MAX_RETRIES", "CAPTIONER_GEMINI_RETRY_DELAY_S")}}
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def run(mode, video, chunk):
        out = args.output_root / mode / video.stem
        out.mkdir(parents=True, exist_ok=True)
        prompt = base
        if mode == "coarse_chunks":
            prompt += "\n\n" + contract.format(chunk_start_s=f"{chunk.start_s:.3f}",
                                               chunk_end_s=f"{chunk.end_s:.3f}")
        record = {"video_id": video.stem, "mode": mode, "chunk_index": chunk.index,
                  "chunk_start_s": chunk.start_s, "chunk_end_s": chunk.end_s,
                  "caption_prompt": prompt, "caption_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                  "task_started_epoch_s": time.time(), "status": "error"}
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="coarse_caption_") as tmp:
                media = video
                if mode == "coarse_chunks":
                    media = Path(tmp) / "clip.mp4"
                    cut_video_clip(str(video), str(media), chunk.start_s, chunk.end_s,
                                   preserve_audio=True, video_duration=durations[str(video)])
                record["media_input_kind"] = "original_video" if mode == "whole_video" else "audio_preserving_clip"
                with sample_latency() as tracker:
                    accepted = False
                    try:
                        with model_component("perception"):
                            response = call_gemini_perception(video_path=str(media), audio_path=None,
                                prompt=prompt, system_prompt="", env_prefix="CAPTIONER_GEMINI")
                        accepted = True
                    finally:
                        record["latency"] = tracker.summary(complete=accepted)
                record.update(raw_response=response, metadata=consume_last_perception_metadata(),
                              status="complete" if response.strip() else "error")
        except Exception as exc:
            record["error"] = str(exc)
        record["task_finished_epoch_s"] = time.time()
        record["total_wall_seconds"] = round(time.perf_counter() - started, 3)
        (out / f"chunk_{chunk.index:04d}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
        print(mode, video.stem, chunk.index, record["status"], flush=True)
        return record

    for mode in ("coarse_chunks", "whole_video"):
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            for video in args.video_path:
                chunks = plan_chunks(durations[str(video)]) if mode == "coarse_chunks" else [ChunkRange(0, 0, durations[str(video)])]
                futures.extend(pool.submit(run, mode, video, chunk) for chunk in chunks)
            records = [future.result() for future in concurrent.futures.as_completed(futures)]
        lines = [f"# {mode}: raw TIMESTAMP3 captions", ""]
        summaries = []
        for video in args.video_path:
            selected = sorted([r for r in records if r["video_id"] == video.stem], key=lambda r: r["chunk_index"])
            summaries.append({"video_id": video.stem, "latency": aggregate_video_latency(selected)})
            lines.extend([f"## {video.stem}", ""])
            for r in selected:
                lines.extend([f"### Chunk {r['chunk_index']+1}: {r['chunk_start_s']}–{r['chunk_end_s']} s",
                              "", r.get("raw_response", r.get("error", "")), ""])
        (args.output_root / mode / "captions.md").write_text("\n".join(lines))
        (args.output_root / mode / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
