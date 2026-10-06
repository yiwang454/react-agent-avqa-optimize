from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from dspy_avqa.caption_cache import validate_caption_cache_coverage
from dspy_avqa.chunked_caption import VideoJob


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/build_whole_video_caption_cache.py"
SPEC = importlib.util.spec_from_file_location("build_whole_video_caption_cache", SCRIPT)
assert SPEC and SPEC.loader
whole = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(whole)


def test_whole_video_artifact_resumes_and_materializes_cache(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    job = VideoJob("video", video, 12.0, ("video-1", "video-2"))
    output = tmp_path / "output"
    prompt = "Caption the whole video with timestamps."
    calls = []

    monkeypatch.setenv("CAPTIONER_GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setattr(whole, "probe_media_duration", lambda path: 12.0)
    monkeypatch.setattr(
        whole,
        "call_gemini_perception",
        lambda **kwargs: calls.append(kwargs) or "[00:00 - 00:12] A complete event.",
    )
    monkeypatch.setattr(
        whole,
        "consume_last_perception_metadata",
        lambda: {
            "model": "gemini-2.5-flash",
            "backend": "gemini",
            "token_usage": {"totalTokenCount": 10},
        },
    )
    result = whole.run_video(job, output_dir=output, prompt=prompt)
    assert result["status"] == "complete"
    assert result["timestamp_normalization_applied"] is False
    assert len(calls) == 1

    resumed = whole.run_video(job, output_dir=output, prompt=prompt)
    assert resumed == result
    assert len(calls) == 1

    input_jsonl = tmp_path / "input.jsonl"
    input_jsonl.write_text("{}\n", encoding="utf-8")
    prompt_yaml = tmp_path / "prompt.yaml"
    prompt_yaml.write_text("prompt: value\n", encoding="utf-8")
    manifest = whole.materialize_cache(
        jobs=[job],
        results=[result],
        cache_dir=output / "caption_cache",
        input_jsonl=input_jsonl,
        prompt_yaml=prompt_yaml,
        prompt_key="QA_PROMPT_OMNI_CAPTIONER_TIMESTAMP3",
        prompt=prompt,
    )
    assert manifest["video_count"] == 1
    assert manifest["question_count"] == 2
    coverage = validate_caption_cache_coverage(
        output / "caption_cache",
        ["video-1", "video-2"],
        expected_prompt=prompt,
    )
    assert coverage["validated"] == 2
    entry = json.loads((output / "caption_cache/video-1.json").read_text())
    assert entry["response"] == "[00:00 - 00:12] A complete event."
    assert entry["source_video_id"] == "video"

    existing = whole.materialize_cache(
        jobs=[job],
        results=[result],
        cache_dir=output / "caption_cache",
        input_jsonl=input_jsonl,
        prompt_yaml=prompt_yaml,
        prompt_key="QA_PROMPT_OMNI_CAPTIONER_TIMESTAMP3",
        prompt=prompt,
    )
    assert existing["validated_existing_cache"]["validated"] == 2
