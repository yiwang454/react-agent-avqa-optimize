from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "build_omniagent_audio_global_caption_cache.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_omniagent_audio_global_caption_cache", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_fixed_prompt_preserves_omniagent_trailing_space() -> None:
    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "DSPy"
        / "dspy_avqa"
        / "yamls"
        / "omniagent_audio_global_caption_prompt.yaml"
    )
    prompt = MODULE.load_prompt(
        prompt_path, "OMNIAGENT_AUDIO_GLOBAL_CAPTION_PROMPT"
    )
    assert prompt == (
        "Provide a high-level summary of the audio. "
        "Focus on the main topics, key events, and the overall atmosphere, "
    )


def test_trace_extraction_keeps_first_successful_response(tmp_path: Path) -> None:
    prompt = "exact prompt "
    rows = [
        {
            "video_id": "q1",
            "question_data": {
                "question_id": "q1",
                "turn_trace": [
                    {
                        "turn_id": 1,
                        "tool_name": "audio_global_caption",
                        "tool_observation": {"answer": "first"},
                        "tool_error": None,
                        "perception_calls": [
                            {
                                "prompt": prompt,
                                "model": "gemini-2.5-flash",
                                "usage_metadata": {"totalTokenCount": 9},
                            }
                        ],
                    },
                    {
                        "turn_id": 2,
                        "tool_name": "audio_global_caption",
                        "tool_observation": {"answer": "second"},
                        "tool_error": None,
                        "perception_calls": [{"prompt": prompt}],
                    },
                ],
            },
        },
        {
            "video_id": "q2",
            "question_data": {"question_id": "q2", "turn_trace": []},
        },
    ]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    selected, summary = MODULE.extract_trace_captions(
        trace, expected_question_ids={"q1", "q2"}, prompt=prompt
    )

    assert selected["q1"].response == "first"
    assert selected["q1"].token_usage == {"totalTokenCount": 9}
    assert summary["successful_tool_calls"] == 2
    assert summary["questions_with_successful_trace_caption"] == 1
    assert summary["questions_with_repeated_tool_calls"] == 1


def test_runtime_yaml_matches_omniagent_generation_settings() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "DSPy"
        / "dspy_avqa"
        / "yamls"
        / "config_omniagent_gemini25_flash_perception.yaml"
    )
    runtime, _ = MODULE.load_runtime_config(config_path)
    assert runtime.model == "gemini-2.5-flash"
    assert runtime.temperature == 0.6
    assert runtime.top_p == 0.95
    assert runtime.top_k == 20
    assert runtime.max_tokens == 4096
    assert runtime.max_retries == 6


def test_cache_entry_path_preserves_leading_underscore(tmp_path: Path) -> None:
    assert MODULE.cache_entry_path(tmp_path, "_video-1").name == "_video-1.json"
