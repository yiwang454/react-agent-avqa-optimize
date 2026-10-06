from __future__ import annotations

import json
from pathlib import Path

import pytest

from dspy_avqa.caption_cache import validate_caption_cache_coverage
from dspy_avqa.chunked_caption import (
    ChunkRange,
    VideoJob,
    aggregate_video_latency,
    build_chunk_prompt,
    load_chunk_prompt_config,
    materialize_question_cache,
    parse_timestamped_events,
    plan_chunks,
    repair_single_closing_local_timestamp_tags,
    render_chunk_events,
    rewrite_local_timestamps_to_original,
)
from dspy_avqa import chunked_caption
from dspy_avqa import tools


PROMPT_YAML = (
    Path(__file__).resolve().parents[2]
    / "DSPy/dspy_avqa/yamls/omni_caption_prompt.yaml"
)
PROMPT_CONFIG = load_chunk_prompt_config(PROMPT_YAML)
SPLIT_AVS_PROMPT_CONFIG = load_chunk_prompt_config(
    PROMPT_YAML,
    profile_key="CHUNKED_CAPTION_PROMPT_SPLIT_AVS",
)
BASE_PROMPT = PROMPT_CONFIG["base_instruction"]


def test_plan_chunks_merges_tail_below_thirty_seconds():
    assert plan_chunks(304.137, chunk_seconds=60, min_tail_seconds=30) == [
        ChunkRange(0, 0, 60),
        ChunkRange(1, 60, 120),
        ChunkRange(2, 120, 180),
        ChunkRange(3, 180, 240),
        ChunkRange(4, 240, 304.137),
    ]
    assert plan_chunks(330, chunk_seconds=60, min_tail_seconds=30)[-1] == ChunkRange(
        4, 240, 330
    )
    assert plan_chunks(330.001, chunk_seconds=60, min_tail_seconds=30)[-1] == ChunkRange(
        5, 300, 330.001
    )
    assert plan_chunks(329.9, chunk_seconds=60, min_tail_seconds=30)[-1] == ChunkRange(
        4, 240, 329.9
    )


def test_local_prompt_preserves_base_and_hides_original_offset():
    prompt = build_chunk_prompt(
        PROMPT_CONFIG,
        mode="local_offset",
        chunk=ChunkRange(1, 60, 120),
    )
    assert prompt.startswith(BASE_PROMPT)
    assert "first frame is 00:00.000" in prompt
    assert "Do not calculate or mention the chunk's position" in prompt
    assert "[[start_local=MM:SS.mmm;end_local=MM:SS.mmm]]" in prompt
    assert "sixteen seconds must be 00:16.000, never 0.016" in prompt
    assert "60.000 through 120.000" not in prompt
    assert "]] <caption>" in prompt
    assert "Visual: <what is visible>" not in prompt
    assert "Output segment lines only" in prompt


def test_split_avs_profile_retains_prior_three_field_contract():
    prompt = build_chunk_prompt(
        SPLIT_AVS_PROMPT_CONFIG,
        mode="local_offset",
        chunk=ChunkRange(1, 60, 120),
    )
    assert prompt.startswith(BASE_PROMPT)
    assert "Visual: <what is visible>" in prompt
    assert "Audio: <sounds or music>" in prompt
    assert "Speech: <speaker and words, if clear>" in prompt


def test_parse_local_events_offsets_and_renders_original_timestamps():
    events, warnings = parse_timestamped_events(
        "[[start_local=00:01.250;end_local=00:03.500]] A door opens as a bell rings.\n"
        "[[start_local=00:10.000;end_local=00:11.000]] A person enters and says hello.",
        mode="local_offset",
        chunk=ChunkRange(2, 120, 180),
    )
    assert warnings == []
    assert events[0]["original_start_s"] == 121.25
    assert events[1]["original_end_s"] == 131.0
    assert render_chunk_events(events).startswith("[00:02:01.250 - 00:02:03.500]")


def test_missing_second_closing_bracket_repair_is_narrow_and_parseable():
    raw = (
        "[[start_local=00:01.250;end_local=00:03.500] "
        "Visual: a door opens.; Audio: a bell.; Speech: none.]"
    )
    repaired, count = repair_single_closing_local_timestamp_tags(raw)
    assert count == 1
    assert repaired.startswith(
        "[[start_local=00:01.250;end_local=00:03.500]] Visual:"
    )
    events, warnings = parse_timestamped_events(
        repaired,
        mode="local_offset",
        chunk=ChunkRange(2, 120, 180),
    )
    assert len(events) == 1
    assert warnings == []


def test_rewrite_local_timestamps_preserves_caption_text_and_structure():
    raw = (
        "[[start_local=00:01.250;end_local=00:03.500]] "
        "Visual: a door opens.; Audio: a bell.; Speech: hello.]\n"
        "continuation text is preserved exactly"
    )
    rewritten = rewrite_local_timestamps_to_original(
        raw,
        chunk=ChunkRange(2, 120, 180),
    )
    assert rewritten == (
        "[[start_local=02:01.250;end_local=02:03.500]] "
        "Visual: a door opens.; Audio: a bell.; Speech: hello.]\n"
        "continuation text is preserved exactly"
    )


def test_rewrite_compact_local_timestamps_preserves_compact_structure():
    assert rewrite_local_timestamps_to_original(
        "[[00:02.000;00:04.000]] A compact caption.",
        chunk=ChunkRange(1, 60, 120),
    ) == "[[01:02.000;01:04.000]] A compact caption."


def test_model_original_rejects_local_chunk_timestamps():
    with pytest.raises(ValueError, match="outside the expected"):
        parse_timestamped_events(
            "[[start_s=1;end_s=3]] An event occurs.",
            mode="model_original",
            chunk=ChunkRange(2, 120, 180),
        )


def test_untagged_continuation_is_preserved_on_preceding_event():
    events, warnings = parse_timestamped_events(
        "[[start_local=00:00.000;end_local=00:05.000]] Visual: chopping vegetables.\n"
        "Audio: chopping sounds. Speech: none.",
        mode="local_offset",
        chunk=ChunkRange(0, 0, 60),
    )
    assert events[0]["description"] == (
        "Visual: chopping vegetables. Audio: chopping sounds. Speech: none."
    )
    assert warnings == ["line 2 was appended to the preceding timestamped event"]


def test_model_original_retains_returned_timestamps_without_normalization():
    events, warnings = parse_timestamped_events(
        "[[start_s=59.5;end_s=61.0]] A continuous caption.",
        mode="model_original",
        chunk=ChunkRange(1, 60, 120),
    )
    assert events[0]["original_start_s"] == 59.5
    assert events[0]["original_end_s"] == 61.0
    assert warnings == [
        "line 1 is outside [60, 120] and was retained without normalization"
    ]


@pytest.mark.parametrize(
    ("mode", "response", "chunk", "expected_start"),
    [
        (
            "local_offset",
            "[[00:02.000;00:04.000]] A compact local caption.",
            ChunkRange(1, 60, 120),
            62.0,
        ),
        (
            "model_original",
            "[[62.000;64.000]] A compact original-time caption.",
            ChunkRange(1, 60, 120),
            62.0,
        ),
    ],
)
def test_compact_timestamp_fields_are_parsed_with_one_warning(
    mode, response, chunk, expected_start
):
    events, warnings = parse_timestamped_events(response, mode=mode, chunk=chunk)
    assert events[0]["original_start_s"] == expected_start
    assert warnings == ["response used compact timestamp fields without names"]


def test_small_boundary_rounding_drift_is_clamped_and_reported():
    events, warnings = parse_timestamped_events(
        "[[start_local=00:52.000;end_local=01:01.000]] Visual: final action",
        mode="local_offset",
        chunk=ChunkRange(0, 0, 60),
    )
    assert events[0]["original_end_s"] == 60
    assert warnings == [
        "line 1 was clamped from [52.0, 61.0] to the chunk boundary [0.0, 60]"
    ]


def test_video_latency_uses_max_adjusted_and_also_reports_sum():
    chunks = [
        {
            "status": "complete",
            "task_started_epoch_s": 10,
            "task_finished_epoch_s": 18,
            "total_wall_seconds": 8,
            "latency": {
                "retry_adjusted_seconds": 5,
                "successful_model_call_seconds": 4,
            },
        },
        {
            "status": "complete",
            "task_started_epoch_s": 11,
            "task_finished_epoch_s": 20,
            "total_wall_seconds": 9,
            "latency": {
                "retry_adjusted_seconds": 7,
                "successful_model_call_seconds": 6,
            },
        },
    ]
    result = aggregate_video_latency(chunks)
    assert result["assumed_parallel_retry_adjusted_seconds"] == 7
    assert result["sum_chunk_retry_adjusted_seconds"] == 12
    assert result["observed_worker_span_seconds"] == 10


def test_materialized_cache_matches_existing_runtime_schema(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    input_jsonl = tmp_path / "input.jsonl"
    input_jsonl.write_text('{}\n', encoding="utf-8")
    prompt_yaml = tmp_path / "prompt.yaml"
    prompt_yaml.write_text("captioner: {}\n", encoding="utf-8")
    output = tmp_path / "run"
    monkeypatch.setenv("CAPTIONER_GEMINI_MODEL", "gemini-2.5-flash")
    job = VideoJob("video", video_path, 61.0, ("video-1", "video-2"))
    video_result = {
        "video_id": "video",
        "status": "complete",
        "combined_response": "[00:00:00.000 - 00:00:01.000] event",
        "token_usage": {"totalTokenCount": 10},
        "chunk_count": 2,
        "timestamp_mode": "local_offset",
        "latency": {"assumed_parallel_retry_adjusted_seconds": 2.0},
    }
    manifest = materialize_question_cache(
        jobs=[job],
        video_results=[video_result],
        cache_dir=output / "caption_cache",
        input_jsonl=input_jsonl,
        prompt_yaml=prompt_yaml,
        base_prompt=BASE_PROMPT,
    )
    assert manifest["video_count"] == 1
    assert manifest["question_count"] == 2
    coverage = validate_caption_cache_coverage(
        output / "caption_cache",
        ["video-1", "video-2"],
        expected_prompt=BASE_PROMPT,
    )
    assert coverage["validated"] == 2
    entry = json.loads((output / "caption_cache" / "video-1.json").read_text())
    assert entry["source_video_id"] == "video"


def test_audio_preserving_clip_path_uses_optional_audio_mapping(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command

        class Result:
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(tools.subprocess, "run", fake_run)
    result = tools.cut_video_clip(
        "input.mp4",
        str(tmp_path / "output.mp4"),
        60,
        120,
        preserve_audio=True,
        video_duration=300,
    )
    assert result == (60, 120)
    command = captured["command"]
    assert command[command.index("-map") + 1] == "0:v:0"
    assert "0:a?" in command
    assert "-c:a" in command


def test_unparseable_resumed_raw_response_falls_back_to_fresh_call(
    monkeypatch, tmp_path
):
    job = VideoJob("video", tmp_path / "video.mp4", 60.0, ())
    job.video_path.write_bytes(b"video")
    stale = {
        "status": "error",
        "timestamp_mode": "local_offset",
        "chunk_index": 0,
        "actual_chunk_start_s": 0,
        "actual_chunk_end_s": 60,
        "raw_response": "[[broken timestamp] event",
    }
    monkeypatch.setattr(chunked_caption, "_load_resumable_chunk", lambda *_: stale)
    monkeypatch.setattr(
        chunked_caption,
        "cut_video_clip",
        lambda *args, **kwargs: (0.0, 60.0),
    )
    monkeypatch.setattr(
        chunked_caption,
        "call_gemini_perception",
        lambda **kwargs: (
            "[[start_local=00:00.000;end_local=01:00.000]] "
            "A scene with sound."
        ),
    )
    monkeypatch.setattr(chunked_caption, "consume_last_perception_metadata", lambda: {})
    result = chunked_caption._run_chunk(
        job=job,
        duration_s=60,
        chunk=ChunkRange(0, 0, 60),
        prompt_config=PROMPT_CONFIG,
        mode="local_offset",
        output_dir=tmp_path / "output",
        force=False,
    )
    assert result["status"] == "complete"
