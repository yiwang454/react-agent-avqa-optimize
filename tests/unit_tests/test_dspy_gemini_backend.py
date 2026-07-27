import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "DSPy" / "dspy_avqa"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _install_package_stubs() -> None:
    package = types.ModuleType("dspy_avqa")
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules["dspy_avqa"] = package

    legacy = types.ModuleType("dspy_avqa.gemini_api")
    legacy.call_gemini_messages = lambda *args, **kwargs: ("legacy", {})
    sys.modules["dspy_avqa.gemini_api"] = legacy

    prompt_config = types.ModuleType("dspy_avqa.prompt_config")
    prompt_config.active_prompt_yaml_path = lambda: None
    prompt_config.load_prompt_config = lambda *args, **kwargs: {}
    prompt_config.prompt_config = lambda: {}
    prompt_config.prompt_value = lambda *args: "default system prompt"
    prompt_config.render_prompt = lambda *args, **kwargs: kwargs.get("perceptual_question", "")
    sys.modules["dspy_avqa.prompt_config"] = prompt_config


def _load_tools():
    _install_package_stubs()
    return _load_module("dspy_avqa.tools", PACKAGE_DIR / "tools.py")


def _load_runner():
    _install_package_stubs()
    deepseek = types.ModuleType("dspy_avqa.deepseek_dspy_lm")
    deepseek.consume_planner_call_trace = lambda: []
    sys.modules["dspy_avqa.deepseek_dspy_lm"] = deepseek

    context = types.ModuleType("dspy_avqa.context")
    context.AVQARuntimeContext = object
    context.CAPTION_PLACEMENT_CHOICES = ("conversation_state", "task")
    context.normalize_caption_placement = lambda: "conversation_state"
    context.resolve_allowed_tools = lambda value=None: ()
    sys.modules["dspy_avqa.context"] = context

    data = types.ModuleType("dspy_avqa.data")
    for name in ("build_input_state", "build_result_row", "maybe_dump_question_data", "read_jsonl", "write_results_jsonl"):
        setattr(data, name, lambda *args, **kwargs: {})
    sys.modules["dspy_avqa.data"] = data

    program = types.ModuleType("dspy_avqa.program")
    program.AVQADSPyReActProgram = object
    program.normalize_option_letter = lambda value: value
    sys.modules["dspy_avqa.program"] = program

    signatures = types.ModuleType("dspy_avqa.signatures")
    signatures.apply_prompt_config_to_signatures = lambda *args, **kwargs: None
    sys.modules["dspy_avqa.signatures"] = signatures
    return _load_module("dspy_avqa.runner", PACKAGE_DIR / "runner.py")


def test_gemini_api_new_maps_paths_and_rejects_outside_root(tmp_path):
    _install_package_stubs()
    _load_module("dspy_avqa.response_quality", PACKAGE_DIR / "response_quality.py")
    module = _load_module("dspy_avqa.gemini_api_new_test", PACKAGE_DIR / "gemini_api_new.py")
    root = tmp_path / "dataset"
    media = root / "Videos" / "sample.mp4"
    media.parent.mkdir(parents=True)
    media.touch()

    assert module.local_path_to_gcs_uri(str(media), local_data_root=str(root), gcs_data_root="gs://bucket/data") == "gs://bucket/data/Videos/sample.mp4"
    assert module.local_path_to_gcs_uri("gs://other/video.mp4", local_data_root=str(root), gcs_data_root="gs://bucket/data") == "gs://other/video.mp4"
    with pytest.raises(ValueError, match="outside Gemini local data root"):
        module.local_path_to_gcs_uri(str(tmp_path / "other.mp4"), local_data_root=str(root), gcs_data_root="gs://bucket/data")


def test_dspy_gemini_uses_gcs_messages_and_omits_audio_for_video_only(monkeypatch):
    tools = _load_tools()
    captured = {}

    new_api = types.ModuleType("dspy_avqa.gemini_api_new")
    new_api.VERTEX_PROJECT = "default-project"
    new_api.VERTEX_LOCATION = "global"

    def build(prompt, **kwargs):
        captured["build"] = (prompt, kwargs)
        return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    def call(messages, **kwargs):
        captured["call"] = (messages, kwargs)
        return "dspy response", {"total_tokens": 3}, ""

    new_api.build_gemini_messages = build
    new_api.call_gemini_messages = call
    sys.modules["dspy_avqa.gemini_api_new"] = new_api
    monkeypatch.setenv("GEMINI_API_BACKEND", "dspy")
    monkeypatch.setenv("GEMINI_LOCAL_DATA_ROOT", "/data")
    monkeypatch.setenv("GEMINI_GCS_DATA_ROOT", "gs://bucket/data")
    monkeypatch.setenv("GEMINI_VIDEO_ONLY", "true")
    monkeypatch.setenv("GEMINI_TOP_K", "64")
    monkeypatch.setenv("VERTEXAI_PROJECT", "project")

    assert tools.call_gemini_perception("/data/video.mp4", "/data/audio.wav", "question", system_prompt="system") == "dspy response"
    assert captured["build"][1]["audio_path"] is None
    assert captured["build"][1]["video_path"] == "/data/video.mp4"
    assert captured["call"][1]["vertex_project"] == "project"
    assert captured["call"][1]["top_k"] == 64
    assert tools.consume_last_perception_metadata()["api_backend"] == "dspy"


def test_legacy_gemini_keeps_native_contents(monkeypatch):
    tools = _load_tools()
    captured = {}
    monkeypatch.setattr(tools, "to_gemini_inline_data", lambda path: {"inline": path})

    def call(contents, **kwargs):
        captured["contents"] = contents
        captured["kwargs"] = kwargs
        return "legacy response", {}, "thinking"

    monkeypatch.setattr(tools, "call_gemini_messages", call)
    monkeypatch.setenv("GEMINI_API_BACKEND", "legacy")
    monkeypatch.setenv("GEMINI_VIDEO_ONLY", "false")

    assert tools.call_gemini_perception("video.mp4", "audio.wav", "question", system_prompt="system") == "legacy response"
    assert captured["contents"][0]["parts"] == [{"inline": "audio.wav"}, {"inline": "video.mp4"}, {"text": "question"}]
    assert "retry_delay_s" in captured["kwargs"]
    assert tools.consume_last_perception_metadata()["api_backend"] == "legacy"


def test_qwen_split_audio_keeps_standalone_audio_and_retry_fallback(monkeypatch):
    tools = _load_tools()
    captured = {}
    qwen_api = types.ModuleType("dspy_avqa.qwen3omni_api")
    qwen_api.to_data_url = lambda path: f"data:{path}"

    def call(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return "qwen response", {}, ""

    qwen_api.call_qwen_messages = call
    sys.modules["dspy_avqa.qwen3omni_api"] = qwen_api
    monkeypatch.setenv("QWEN_USE_AUDIO_IN_VIDEO", "false")
    monkeypatch.setenv("QWEN_VIDEO_ONLY", "false")

    tools.call_qwen_perception(
        "video.mp4",
        "audio.wav",
        "question",
        system_prompt="",
    )

    content = captured["messages"][0]["content"]
    assert [item["type"] for item in content] == [
        "audio_url",
        "video_url",
        "text",
    ]
    assert captured["kwargs"]["use_audio_in_video"] is False
    assert captured["kwargs"]["empty_response_retry_video_first"] is True


def test_qwen_audio_in_video_omits_duplicate_standalone_audio(monkeypatch):
    tools = _load_tools()
    captured = {}
    qwen_api = types.ModuleType("dspy_avqa.qwen3omni_api")
    qwen_api.to_data_url = lambda path: f"data:{path}"

    def call(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return "qwen response", {"total_tokens": 3}, ""

    qwen_api.call_qwen_messages = call
    sys.modules["dspy_avqa.qwen3omni_api"] = qwen_api
    monkeypatch.setenv("QWEN_USE_AUDIO_IN_VIDEO", "true")
    monkeypatch.setenv("QWEN_VIDEO_ONLY", "false")

    assert (
        tools.call_qwen_perception(
            "video.mp4",
            "audio.wav",
            "question",
            system_prompt="system",
        )
        == "qwen response"
    )
    content = captured["messages"][1]["content"]
    assert [item["type"] for item in content] == ["video_url", "text"]
    assert captured["kwargs"]["use_audio_in_video"] is True
    metadata = tools.consume_last_perception_metadata()
    assert metadata["use_audio_in_video"] is True


def test_qwen_video_only_takes_precedence_over_audio_in_video(monkeypatch):
    tools = _load_tools()
    captured = {}
    qwen_api = types.ModuleType("dspy_avqa.qwen3omni_api")
    qwen_api.to_data_url = lambda path: f"data:{path}"

    def call(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return "qwen response", {}, ""

    qwen_api.call_qwen_messages = call
    sys.modules["dspy_avqa.qwen3omni_api"] = qwen_api
    monkeypatch.setenv("QWEN_USE_AUDIO_IN_VIDEO", "true")
    monkeypatch.setenv("QWEN_VIDEO_ONLY", "true")

    tools.call_qwen_perception(
        "video.mp4",
        "audio.wav",
        "question",
        system_prompt="",
    )
    content = captured["messages"][0]["content"]
    assert [item["type"] for item in content] == ["video_url", "text"]
    assert captured["kwargs"]["use_audio_in_video"] is False


def test_runner_invalidates_cached_empty_qwen_observations(tmp_path):
    runner = _load_runner()
    cut = {"id": "sample-1"}
    cache_path = tmp_path / "sample-1.json"
    question_data = {
        "response": "A. answer",
        "turn_trace": [
            {
                "planner_action": "tool",
                "perception_backend": "qwen",
                "tool_observation": "",
            }
        ],
    }
    cache_path.write_text(json.dumps(question_data), encoding="utf-8")
    assert runner.load_cached_row_from_question_json(tmp_path, cut) is None

    question_data["turn_trace"][0]["tool_observation"] = "visible evidence"
    cache_path.write_text(json.dumps(question_data), encoding="utf-8")
    assert runner.load_cached_row_from_question_json(tmp_path, cut) is not None

    question_data["response"] = "[ERROR] Qwen API call failed"
    cache_path.write_text(json.dumps(question_data), encoding="utf-8")
    assert runner.load_cached_row_from_question_json(tmp_path, cut) is None


def test_runner_dspy_validation_requires_roots_and_validates_captioner_top_k(monkeypatch):
    runner = _load_runner()
    monkeypatch.setenv("PERCEPTION_MODEL", "qwen")
    monkeypatch.setenv("CAPTIONER_MODEL", "gemini")
    args = argparse.Namespace(gemini_api_backend="dspy", vertex_project=None, vertex_location=None, gemini_local_data_root=None, gemini_gcs_data_root=None)
    with pytest.raises(ValueError, match="local-to-GCS"):
        runner.configure_gemini_api_backend(args)

    args.gemini_local_data_root = "/data"
    args.gemini_gcs_data_root = "gs://bucket/data"
    monkeypatch.setenv("CAPTIONER_GEMINI_TOP_K", "0")
    with pytest.raises(ValueError, match="expected 1..64"):
        runner.configure_gemini_api_backend(args)

    monkeypatch.setenv("CAPTIONER_GEMINI_TOP_K", "64")
    assert runner.configure_gemini_api_backend(args) is True
    assert os.environ["GEMINI_API_BACKEND"] == "dspy"


def test_precomputed_audio_caption_dir_is_ignored_when_not_rendered(tmp_path):
    runner = _load_runner()
    caption_dir = tmp_path / "captions"
    runner.prompt_config = lambda: {
        "planner": {"task_prompt_template": "Question: {question}"}
    }

    resolved, reason = runner.resolve_preloaded_audio_caption_dir(
        caption_dir,
        caption_placement="task",
    )
    assert resolved is None
    assert reason == "planner.task_prompt_template has no {video_description}"

    runner.prompt_config = lambda: {
        "planner": {"task_prompt_template": "Description: {video_description}"}
    }
    resolved, reason = runner.resolve_preloaded_audio_caption_dir(
        caption_dir,
        caption_placement="conversation_state",
    )
    assert resolved is None
    assert reason == "caption placement is conversation_state"

    resolved, reason = runner.resolve_preloaded_audio_caption_dir(
        caption_dir,
        caption_placement="task",
        ignore_audio_caption_dir=True,
    )
    assert resolved is None
    assert reason == "explicitly ignored"


def test_response_error_sensitive_trips_only_for_gemini(monkeypatch):
    tools = _load_tools()
    monkeypatch.setattr(tools, "to_gemini_inline_data", lambda path: {"inline": path})
    monkeypatch.setenv("GEMINI_API_BACKEND", "legacy")
    monkeypatch.setenv("GEMINI_RESPONSE_ERROR_SENSITIVE", "true")

    monkeypatch.setattr(tools, "call_gemini_messages", lambda *args, **kwargs: ("[ERROR] LiteLLM request failed", {}))
    with pytest.raises(tools.GeminiResponseCircuitBreak, match="explicit \[ERROR\]"):
        tools.call_gemini_perception("video.mp4", None, "question", system_prompt="system")

    def fail_after_retries(*args, **kwargs):
        raise RuntimeError("LiteLLM APIConnectionError after retries")

    monkeypatch.setattr(tools, "call_gemini_messages", fail_after_retries)
    with pytest.raises(tools.GeminiResponseCircuitBreak, match="LiteLLM APIConnectionError"):
        tools.call_gemini_perception("video.mp4", None, "question", system_prompt="system")

    monkeypatch.setenv("GEMINI_RESPONSE_ERROR_SENSITIVE", "false")
    monkeypatch.setattr(tools, "call_gemini_messages", lambda *args, **kwargs: ("", {}))
    assert tools.call_gemini_perception("video.mp4", None, "question", system_prompt="system") == ""
