import importlib.util
import os
import sys
import types
from pathlib import Path


DSPY_PACKAGE_DIR = Path(__file__).resolve().parents[2] / "DSPy" / "dspy_avqa"


def _install_context_import_stubs() -> None:
    package = types.ModuleType("dspy_avqa")
    package.__path__ = [str(DSPY_PACKAGE_DIR)]
    sys.modules["dspy_avqa"] = package

    dspy = types.ModuleType("dspy")

    class LM:
        kwargs = {}

    class BaseLM:
        kwargs = {}

    dspy.LM = LM
    dspy.BaseLM = BaseLM
    dspy.configure = lambda **kwargs: None
    sys.modules["dspy"] = dspy

    deepseek_api = types.ModuleType("dspy_avqa.deepseek_api")

    class DeepSeekPlannerClient:
        pass

    class DeepSeekPlannerConfig:
        pass

    deepseek_api.DeepSeekPlannerClient = DeepSeekPlannerClient
    deepseek_api.DeepSeekPlannerConfig = DeepSeekPlannerConfig
    sys.modules["dspy_avqa.deepseek_api"] = deepseek_api


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_context_modules():
    _install_context_import_stubs()
    deepseek_lm = _load_module(
        "dspy_avqa.deepseek_dspy_lm", DSPY_PACKAGE_DIR / "deepseek_dspy_lm.py"
    )
    context = _load_module("dspy_avqa.context", DSPY_PACKAGE_DIR / "context.py")
    return context, deepseek_lm


def test_signature_in_system_prompt_does_not_inject_fixed_system_prompt(monkeypatch):
    context, deepseek_lm = _load_context_modules()
    monkeypatch.setenv("DSPY_AVQA_SIGNATURE_IN_SYSTEM_PROMPT", "true")
    source_messages = [
        {"role": "system", "content": "SIGNATURE SCHEMA"},
        {"role": "user", "content": "[[ ## task ## ]]\nTASK BODY"},
    ]

    lm = context.SerialNOpenAICompatibleLM.__new__(context.SerialNOpenAICompatibleLM)

    assert lm._normalize_planner_messages(None, source_messages) == source_messages
    assert deepseek_lm.FIXED_PLANNER_SYSTEM_PROMPT not in [
        message["content"] for message in source_messages
    ]


def test_default_signature_placement_uses_fixed_system_and_moves_schema_to_user(monkeypatch):
    context, deepseek_lm = _load_context_modules()
    monkeypatch.delenv("DSPY_AVQA_SIGNATURE_IN_SYSTEM_PROMPT", raising=False)
    source_messages = [
        {"role": "system", "content": "SIGNATURE SCHEMA"},
        {"role": "user", "content": "[[ ## task ## ]]\nTASK BODY"},
    ]

    lm = context.SerialNOpenAICompatibleLM.__new__(context.SerialNOpenAICompatibleLM)
    normalized = lm._normalize_planner_messages(None, source_messages)

    assert normalized[0] == {
        "role": "system",
        "content": deepseek_lm.FIXED_PLANNER_SYSTEM_PROMPT,
    }
    assert normalized[1]["role"] == "user"
    assert "DSPy response format instruction:" in normalized[1]["content"]
    assert "SIGNATURE SCHEMA" in normalized[1]["content"]
    assert "[[ ## task ## ]]" in normalized[1]["content"]
