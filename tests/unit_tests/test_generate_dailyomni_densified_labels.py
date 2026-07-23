import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "DSPy"
    / "generate_dailyomni_densified_labels.py"
)
SPEC = importlib.util.spec_from_file_location(
    "generate_dailyomni_densified_labels",
    SCRIPT_PATH,
)
generator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generator
assert SPEC.loader is not None
SPEC.loader.exec_module(generator)


def _cut(cut_id="video-1", video_id="video", answer="B"):
    return {
        "id": cut_id,
        "supervisions": [
            {
                "recording_id": video_id,
                "text": "What happened?",
                "custom": {
                    "video_id": video_id,
                    "options": ["A. First", "B. Second"],
                    "answer": answer,
                },
            }
        ],
        "recording": {"id": video_id},
    }


def _write_captions(root, video_id="video"):
    video_dir = root / "Videos" / video_id
    video_dir.mkdir(parents=True)
    for field_name, filename in generator.CAPTION_FILENAMES.items():
        (video_dir / filename).write_text(f"{field_name} evidence", encoding="utf-8")


def _write_prompt(path):
    path.write_text(
        """
densified_label_prompt: |-
  Question: {question}
  Options: {options}
  Gold: {gold_answer}
  AV: {av_alignment_captions}
  Visual: {video_consistent_captions}
  Audio: {audio_revised_captions}
""".lstrip(),
        encoding="utf-8",
    )


class FakeResponses:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            status="completed",
            output=[],
            output_parsed=outcome,
            incomplete_details=None,
        )


class FakeClient:
    def __init__(self, outcomes):
        self.responses = FakeResponses(outcomes)


def _label(key="key", target="target"):
    return generator.DensifiedLabel(
        key_evidence=key,
        ideal_perception_target=target,
    )


def test_preflight_loads_caption_bundle_once_and_renders_prompt(tmp_path, monkeypatch):
    _write_captions(tmp_path)
    calls = []
    original_reader = generator._read_caption_file

    def counting_reader(path):
        calls.append(path)
        return original_reader(path)

    monkeypatch.setattr(generator, "_read_caption_file", counting_reader)
    records = generator.preflight_dataset(
        [_cut("video-1"), _cut("video-2")],
        tmp_path,
    )

    assert len(calls) == 3
    assert records[0].captions is records[1].captions
    prompt = generator.render_prompt(
        'Return {"key_evidence":"..."}|Q={question}|O={options}|G={gold_answer}|'
        "A={av_alignment_captions}"
        "|V={video_consistent_captions}|U={audio_revised_captions}",
        records[0],
    )
    assert '{"key_evidence":"..."}' in prompt
    assert "Q=What happened?" in prompt
    assert '"B. Second"' in prompt
    assert "G=B" in prompt
    assert "av_alignment_captions evidence" in prompt


def test_preflight_collects_missing_caption_errors_before_api_use(tmp_path):
    with pytest.raises(generator.PreflightError) as exc_info:
        generator.preflight_dataset([_cut()], tmp_path)

    message = str(exc_info.value)
    assert "video-1" in message
    assert "av_alignment_captions.txt" in message


def test_preflight_allows_an_empty_annotation_source(tmp_path):
    _write_captions(tmp_path)
    caption_path = (
        tmp_path
        / "Videos"
        / "video"
        / generator.CAPTION_FILENAMES["video_consistent_captions"]
    )
    caption_path.write_text("", encoding="utf-8")

    records = generator.preflight_dataset([_cut()], tmp_path)

    assert records[0].captions.video_consistent_captions == ""


def test_model_call_uses_fixed_gpt54_high_structured_output_and_retries():
    client = FakeClient([RuntimeError("rate limited"), _label()])
    slept = []

    result = generator.call_model_with_retries(
        client,
        "prompt",
        cut_id="video-1",
        max_attempts=2,
        sleep_fn=slept.append,
        random_fn=lambda: 0.0,
    )

    assert result == _label()
    assert slept == [1.0]
    assert len(client.responses.calls) == 2
    request = client.responses.calls[0]
    assert request["model"] == "gpt-5.4"
    assert request["reasoning"] == {"effort": "high"}
    assert request["store"] is False
    assert request["text_format"] is generator.DensifiedLabel
    assert "temperature" not in request


def test_batch_skips_valid_output_replaces_invalid_and_writes_only_two_keys(tmp_path):
    root = tmp_path / "daily"
    output_dir = tmp_path / "labels"
    _write_captions(root)
    records = generator.preflight_dataset(
        [_cut("video-1"), _cut("video-2")],
        root,
    )
    output_dir.mkdir()
    generator.write_label_atomic(output_dir / "video-1.json", _label("old", "old target"))
    (output_dir / "video-2.json").write_text('{"key_evidence": ""}', encoding="utf-8")
    client = FakeClient([_label("new", "new target")])

    summary = generator.generate_batch(
        records,
        "Q={question} {options} {gold_answer} {av_alignment_captions} "
        "{video_consistent_captions} {audio_revised_captions}",
        output_dir,
        client=client,
        max_workers=1,
        max_attempts=1,
        overwrite=False,
    )

    assert summary == generator.GenerationSummary(
        total=2,
        generated=1,
        skipped=1,
        failed=0,
        valid_outputs=2,
    )
    assert json.loads((output_dir / "video-1.json").read_text()) == {
        "key_evidence": "old",
        "ideal_perception_target": "old target",
    }
    generated = json.loads((output_dir / "video-2.json").read_text())
    assert generated == {
        "key_evidence": "new",
        "ideal_perception_target": "new target",
    }
    assert set(generated) == generator.LABEL_KEYS
    assert not list(output_dir.glob("*.tmp"))

    overwrite_client = FakeClient(
        [_label("forced one", "target one"), _label("forced two", "target two")]
    )
    overwrite_summary = generator.generate_batch(
        records,
        "Q={question} {options} {gold_answer} {av_alignment_captions} "
        "{video_consistent_captions} {audio_revised_captions}",
        output_dir,
        client=overwrite_client,
        max_workers=1,
        max_attempts=1,
        overwrite=True,
    )

    assert overwrite_summary.generated == 2
    assert overwrite_summary.skipped == 0
    assert json.loads((output_dir / "video-1.json").read_text())["key_evidence"] == (
        "forced one"
    )


def test_run_returns_nonzero_for_partial_api_failure_and_can_resume(tmp_path):
    root = tmp_path / "daily"
    output_dir = tmp_path / "labels"
    input_jsonl = tmp_path / "train.jsonl"
    prompt_yaml = tmp_path / "prompt.yaml"
    _write_captions(root)
    input_jsonl.write_text(
        json.dumps(_cut("video-1")) + "\n" + json.dumps(_cut("video-2")) + "\n",
        encoding="utf-8",
    )
    _write_prompt(prompt_yaml)
    args = generator.build_argument_parser().parse_args(
        [
            "--input-jsonl",
            str(input_jsonl),
            "--daily-omni-root",
            str(root),
            "--prompt-yaml",
            str(prompt_yaml),
            "--output-dir",
            str(output_dir),
            "--limit",
            "2",
            "--max-workers",
            "1",
            "--max-retries",
            "1",
        ]
    )

    first_client = FakeClient([_label("first", "target"), RuntimeError("API down")])
    assert generator.run(args, client=first_client) == 1
    assert len(list(output_dir.glob("*.json"))) == 1

    second_client = FakeClient([_label("second", "target")])
    assert generator.run(args, client=second_client) == 0
    assert len(second_client.responses.calls) == 1
    assert len(list(output_dir.glob("*.json"))) == 2


def test_dry_run_can_print_fully_rendered_first_prompt_without_api_key(
    tmp_path,
    monkeypatch,
    capsys,
):
    root = tmp_path / "daily"
    output_dir = tmp_path / "labels"
    input_jsonl = tmp_path / "train.jsonl"
    prompt_yaml = tmp_path / "prompt.yaml"
    _write_captions(root)
    input_jsonl.write_text(json.dumps(_cut()) + "\n", encoding="utf-8")
    _write_prompt(prompt_yaml)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PLANNER_API_KEY", raising=False)
    monkeypatch.delenv("ELM_API_KEY", raising=False)

    exit_code = generator.main(
        [
            "--input-jsonl",
            str(input_jsonl),
            "--daily-omni-root",
            str(root),
            "--prompt-yaml",
            str(prompt_yaml),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
            "--print-first-prompt",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert not output_dir.exists()
    output = capsys.readouterr().out
    assert "===== FIRST RENDERED PROMPT: video-1 =====" in output
    assert "Question: What happened?" in output
    assert '"B. Second"' in output
    assert "Gold: B" in output
    assert "AV: av_alignment_captions evidence" in output
    assert "Visual: video_consistent_captions evidence" in output
    assert "Audio: audio_revised_captions evidence" in output
    assert "{question}" not in output
