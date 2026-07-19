from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


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
    _load_module(f"{package_name}.response_quality", PACKAGE_DIR / "response_quality.py")
    return _load_module(f"{package_name}.{filename[:-3]}", PACKAGE_DIR / filename)


def _qwen_completion(content: str, completion_tokens: int):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))],
        usage=types.SimpleNamespace(completion_tokens=completion_tokens),
    )


def test_qwen_retries_empty_response_and_keeps_final_attempt(monkeypatch):
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
        [{"role": "user", "content": "question"}],
        stream=False,
        max_tokens=4096,
        max_retries=2,
        retry_delay_s=0,
        retry_degenerate_response=True,
    )

    assert text == "A detailed visual description."
    assert usage["completion_tokens"] == 24
    assert len(calls) == 2

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
    text, usage = qwen.call_qwen_messages(
        [{"role": "user", "content": "question"}],
        stream=False,
        max_tokens=4096,
        max_retries=2,
        retry_delay_s=0,
        retry_degenerate_response=True,
    )
    assert text == "\t"
    assert usage["completion_tokens"] == 4096
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
