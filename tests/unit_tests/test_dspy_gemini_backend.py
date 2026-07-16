import argparse
import importlib.util
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
