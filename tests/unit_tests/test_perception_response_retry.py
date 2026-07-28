from __future__ import annotations

import importlib.util
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


def _load_api(package_name: str, filename: str):
    package = types.ModuleType(package_name)
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules[package_name] = package
    if filename == "qwen3omni_api.py":
        openai = types.ModuleType("openai")
        openai.OpenAI = object
        sys.modules["openai"] = openai
    _load_module(f"{package_name}.response_quality", PACKAGE_DIR / "response_quality.py")
    return _load_module(f"{package_name}.{filename[:-3]}", PACKAGE_DIR / filename)


def _qwen_completion(content: str, completion_tokens: int):
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=types.SimpleNamespace(completion_tokens=completion_tokens),
    )


def test_qwen_retries_empty_response_and_raises_after_exhaustion(monkeypatch):
    qwen = _load_api("perception_retry_qwen", "qwen3omni_api.py")
    responses = [
        _qwen_completion("\n\n", 4096),
        _qwen_completion("A detailed visual description.", 24),
    ]
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    monkeypatch.setattr(qwen, "_client", lambda **kwargs: client)
    monkeypatch.setattr(qwen.time, "sleep", lambda _: None)

    text, usage = qwen.call_qwen_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": "audio"}},
                    {"type": "video_url", "video_url": {"url": "video"}},
                    {"type": "text", "text": "question"},
                ],
            }
        ],
        stream=False,
        max_tokens=4096,
        max_retries=2,
        retry_delay_s=0,
        retry_degenerate_response=True,
        empty_response_retry_min_tokens=32,
        empty_response_retry_temperature=0.1,
        empty_response_retry_seed_step=1,
        empty_response_retry_video_first=True,
        qwen_seed=1234,
        raise_on_empty_response=True,
    )

    assert text == "A detailed visual description."
    assert usage["completion_tokens"] == 24
    assert usage["qwen_max_tokens"] == 4096
    assert usage["qwen_finish_reason"] == "stop"
    assert usage["qwen_response_attempts"][0]["degenerate_reason"] == "response is empty after strip"
    assert usage["qwen_response_attempts"][1]["degenerate_reason"] is None
    assert len(calls) == 2
    assert calls[0]["seed"] == 1234
    assert calls[0]["max_tokens"] == 4096
    assert "min_tokens" not in calls[0]["extra_body"]
    assert calls[1]["seed"] == 1235
    assert calls[1]["temperature"] == 0.1
    assert calls[1]["max_tokens"] == 4096
    assert calls[1]["extra_body"]["min_tokens"] == 32
    first_types = [item["type"] for item in calls[0]["messages"][0]["content"]]
    retry_types = [item["type"] for item in calls[1]["messages"][0]["content"]]
    assert first_types == ["audio_url", "video_url", "text"]
    assert retry_types == ["video_url", "audio_url", "text"]
    assert usage["qwen_response_attempts"][0]["video_first_retry"] is False
    assert usage["qwen_response_attempts"][1]["video_first_retry"] is True

    assert qwen.degenerate_response_reason(
        "ably",
        {"completion_tokens": 4096},
        max_tokens=4096,
    ) is None
    assert (
        qwen.degenerate_response_reason(" \n", None, max_tokens=0)
        == "response is empty after strip"
    )

    responses[:] = [_qwen_completion(" \n", 4096), _qwen_completion("\t", 4096)]
    calls.clear()
    with pytest.raises(qwen.EmptyQwenResponseError) as exc_info:
        qwen.call_qwen_messages(
            [{"role": "user", "content": "question"}],
            stream=False,
            max_tokens=4096,
            max_retries=2,
            retry_delay_s=0,
            retry_degenerate_response=True,
            raise_on_empty_response=True,
        )
    assert len(exc_info.value.response_attempts) == 2
    assert len(calls) == 2


def test_legacy_gemini_retries_degenerate_response(monkeypatch):
    gemini = _load_api("perception_retry_gemini", "gemini_api.py")
    payloads = [
        {
            "candidates": [{"content": {"parts": [{"text": "\n\n"}]}}],
            "usageMetadata": {"candidatesTokenCount": 4096},
        },
        {
            "candidates": [{"content": {"parts": [{"text": "useful evidence"}]}}],
            "usageMetadata": {"candidatesTokenCount": 20},
        },
    ]
    calls = []

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(ok=True, json=lambda: payloads.pop(0))

    monkeypatch.setattr(gemini.requests, "post", post)
    monkeypatch.setattr(gemini.time, "sleep", lambda _: None)
    text, usage = gemini.call_gemini_messages(
        [{"role": "user", "parts": [{"text": "question"}]}],
        max_tokens=4096,
        max_retries=2,
        retry_delay_s=0,
        retry_degenerate_response=True,
    )

    assert text == "useful evidence"
    assert usage["candidatesTokenCount"] == 20
    assert len(calls) == 2


def test_dspy_gemini_retries_degenerate_response(monkeypatch):
    gemini = _load_api("perception_retry_vertex", "gemini_api_new.py")
    responses = [("\n\n", 4096), ("useful evidence", 20)]
    state = {"calls": 0}

    class FakeLM:
        def __init__(self, *args, **kwargs):
            response_text, completion_tokens = responses[state["calls"]]
            state["calls"] += 1
            self.response_text = response_text
            self.history = [{"usage": {"completion_tokens": completion_tokens}}]

        def __call__(self, **kwargs):
            return [self.response_text]

    monkeypatch.setitem(sys.modules, "dspy", types.SimpleNamespace(LM=FakeLM))
    monkeypatch.setattr(gemini.time, "sleep", lambda _: None)
    text, usage = gemini.call_gemini_messages(
        [{"role": "user", "content": "question"}],
        max_tokens=4096,
        max_retries=2,
        retry_delay_s=0,
        retry_degenerate_response=True,
    )

    assert text == "useful evidence"
    assert usage["completion_tokens"] == 20
    assert state["calls"] == 2
