import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "DSPy" / "dspy_avqa" / "caption_cache.py"
SPEC = importlib.util.spec_from_file_location("caption_cache_under_test", MODULE_PATH)
caption_cache = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = caption_cache
assert SPEC.loader is not None
SPEC.loader.exec_module(caption_cache)
extract_caption_cache = caption_cache.extract_caption_cache
load_cached_caption = caption_cache.load_cached_caption
validate_caption_cache_coverage = caption_cache.validate_caption_cache_coverage


PROMPT = "fixed caption prompt"


def _write_prompt_yaml(path: Path) -> None:
    path.write_text(
        """captioner:
  default_caption_instruction: |-
    fixed caption prompt
  caption_prompt_template: |
    {caption_instruction}
""",
        encoding="utf-8",
    )


def _write_input(path: Path) -> None:
    rows = [{"id": "video-1"}, {"id": "video-2"}]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _result_row(question_id: str, response: str) -> dict:
    return {
        "video_id": question_id,
        "question_data": {
            "question_id": question_id,
            "turn_trace": [
                {
                    "tool_name": "ask_caption",
                    "tool_observation": response,
                    "perception_prompt": PROMPT,
                    "perception_backend": "gemini",
                    "perception_model": "gemini-2.5-flash",
                    "perception_token_usage": {"total": 10},
                }
            ],
        },
    }


def _write_results(path: Path, responses: dict[str, str]) -> None:
    path.write_text(
        "".join(
            json.dumps(_result_row(question_id, response)) + "\n"
            for question_id, response in responses.items()
        ),
        encoding="utf-8",
    )


def test_extract_uses_primary_then_fallback_and_writes_deterministic_manifest(tmp_path):
    input_jsonl = tmp_path / "input.jsonl"
    prompt_yaml = tmp_path / "prompt.yaml"
    primary = tmp_path / "repeat1.jsonl"
    fallback = tmp_path / "repeat2.jsonl"
    _write_input(input_jsonl)
    _write_prompt_yaml(prompt_yaml)
    _write_results(primary, {"video-1": "primary caption", "video-2": ""})
    _write_results(fallback, {"video-1": "unused fallback", "video-2": "fallback caption"})

    first = extract_caption_cache(
        primary_results=primary,
        fallback_results=[fallback],
        input_jsonl=input_jsonl,
        prompt_yaml=prompt_yaml,
        output_dir=tmp_path / "cache1",
    )
    second = extract_caption_cache(
        primary_results=primary,
        fallback_results=[fallback],
        input_jsonl=input_jsonl,
        prompt_yaml=prompt_yaml,
        output_dir=tmp_path / "cache2",
    )

    assert first == second
    assert first["source_counts"] == {"1": 1, "2": 1}
    assert json.loads((tmp_path / "cache1/video-1.json").read_text())["response"] == "primary caption"
    fallback_entry = json.loads((tmp_path / "cache1/video-2.json").read_text())
    assert fallback_entry["response"] == "fallback caption"
    assert fallback_entry["source_rank"] == 2
    coverage = validate_caption_cache_coverage(
        tmp_path / "cache1",
        ["video-1", "video-2"],
        expected_prompt=PROMPT,
    )
    assert coverage["validated"] == 2


def test_cache_lookup_fails_for_missing_empty_or_prompt_mismatch(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    entry = {
        "question_id": "video-1",
        "response": "caption",
        "caption_prompt": PROMPT,
        "source_results_file": "repeat1.jsonl",
        "source_rank": 1,
        "source_backend": "gemini",
        "source_model": "gemini-2.5-flash",
        "source_token_usage": None,
    }
    (cache_dir / "video-1.json").write_text(json.dumps(entry), encoding="utf-8")

    assert load_cached_caption(
        cache_dir,
        "video-1",
        expected_prompt=PROMPT,
    ).response == "caption"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_cached_caption(cache_dir, "missing", expected_prompt=PROMPT)
    with pytest.raises(ValueError, match="prompt mismatch"):
        load_cached_caption(cache_dir, "video-1", expected_prompt="different")
    entry["response"] = ""
    (cache_dir / "video-1.json").write_text(json.dumps(entry), encoding="utf-8")
    with pytest.raises(ValueError, match="response is empty"):
        load_cached_caption(cache_dir, "video-1", expected_prompt=PROMPT)


def test_extractor_rejects_prompt_mismatch_and_nonempty_output_dir(tmp_path):
    input_jsonl = tmp_path / "input.jsonl"
    prompt_yaml = tmp_path / "prompt.yaml"
    results = tmp_path / "results.jsonl"
    _write_input(input_jsonl)
    _write_prompt_yaml(prompt_yaml)
    _write_results(results, {"video-1": "one", "video-2": "two"})
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    rows[0]["question_data"]["turn_trace"][0]["perception_prompt"] = "wrong"
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prompt mismatch"):
        extract_caption_cache(
            primary_results=results,
            fallback_results=[],
            input_jsonl=input_jsonl,
            prompt_yaml=prompt_yaml,
            output_dir=tmp_path / "cache",
        )

    _write_results(results, {"video-1": "one", "video-2": "two"})
    output = tmp_path / "nonempty"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        extract_caption_cache(
            primary_results=results,
            fallback_results=[],
            input_jsonl=input_jsonl,
            prompt_yaml=prompt_yaml,
            output_dir=output,
        )


def test_extractor_rejects_duplicate_ids_and_multiple_caption_turns(tmp_path):
    prompt_yaml = tmp_path / "prompt.yaml"
    _write_prompt_yaml(prompt_yaml)
    duplicate_input = tmp_path / "duplicate_input.jsonl"
    duplicate_input.write_text(
        json.dumps({"id": "video-1"}) + "\n" + json.dumps({"id": "video-1"}) + "\n",
        encoding="utf-8",
    )
    results = tmp_path / "results.jsonl"
    _write_results(results, {"video-1": "caption"})

    with pytest.raises(ValueError, match="Duplicate input IDs"):
        extract_caption_cache(
            primary_results=results,
            fallback_results=[],
            input_jsonl=duplicate_input,
            prompt_yaml=prompt_yaml,
            output_dir=tmp_path / "duplicate_cache",
        )

    input_jsonl = tmp_path / "input.jsonl"
    _write_input(input_jsonl)
    rows = [
        _result_row("video-1", "one"),
        _result_row("video-2", "two"),
    ]
    rows[0]["question_data"]["turn_trace"].append(
        dict(rows[0]["question_data"]["turn_trace"][0])
    )
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one ask_caption"):
        extract_caption_cache(
            primary_results=results,
            fallback_results=[],
            input_jsonl=input_jsonl,
            prompt_yaml=prompt_yaml,
            output_dir=tmp_path / "multiple_turn_cache",
        )


def test_coverage_rejects_tampered_cache_content(tmp_path):
    input_jsonl = tmp_path / "input.jsonl"
    prompt_yaml = tmp_path / "prompt.yaml"
    results = tmp_path / "results.jsonl"
    cache_dir = tmp_path / "cache"
    _write_input(input_jsonl)
    _write_prompt_yaml(prompt_yaml)
    _write_results(results, {"video-1": "one", "video-2": "two"})
    extract_caption_cache(
        primary_results=results,
        fallback_results=[],
        input_jsonl=input_jsonl,
        prompt_yaml=prompt_yaml,
        output_dir=cache_dir,
    )
    entry_path = cache_dir / "video-1.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    entry["response"] = "tampered"
    entry_path.write_text(json.dumps(entry), encoding="utf-8")

    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_caption_cache_coverage(
            cache_dir,
            ["video-1", "video-2"],
            expected_prompt=PROMPT,
        )


def test_question_scoped_cache_is_thread_safe_for_multi_question_video(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for question_id, response in (
        ("shared-video-1", "caption for question one"),
        ("shared-video-2", "caption for question two"),
    ):
        (cache_dir / f"{question_id}.json").write_text(
            json.dumps(
                {
                    "question_id": question_id,
                    "response": response,
                    "caption_prompt": PROMPT,
                    "source_results_file": "repeat1.jsonl",
                    "source_rank": 1,
                    "source_backend": "gemini",
                    "source_model": "gemini-2.5-flash",
                    "source_token_usage": None,
                }
            ),
            encoding="utf-8",
        )

    question_ids = ["shared-video-1", "shared-video-2"] * 20
    with ThreadPoolExecutor(max_workers=8) as executor:
        responses = list(
            executor.map(
                lambda question_id: load_cached_caption(
                    cache_dir,
                    question_id,
                    expected_prompt=PROMPT,
                ).response,
                question_ids,
            )
        )

    assert responses == [
        (
            "caption for question one"
            if question_id == "shared-video-1"
            else "caption for question two"
        )
        for question_id in question_ids
    ]
