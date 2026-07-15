import importlib.util
import sys
import types
from pathlib import Path

import pytest


DSPY_PACKAGE_DIR = Path(__file__).resolve().parents[2] / "DSPy" / "dspy_avqa"


def _install_context_import_stubs():
    package = types.ModuleType("dspy_avqa")
    package.__path__ = [str(DSPY_PACKAGE_DIR)]
    sys.modules["dspy_avqa"] = package

    state = {}
    dspy = types.ModuleType("dspy")

    class LM:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = dict(kwargs)

    class BaseLM:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    def configure(**kwargs):
        state["configured_lm"] = kwargs["lm"]

    dspy.LM = LM
    dspy.BaseLM = BaseLM
    dspy.configure = configure
    sys.modules["dspy"] = dspy

    deepseek_api = types.ModuleType("dspy_avqa.deepseek_api")

    class DeepSeekPlannerClient:
        pass

    class DeepSeekPlannerConfig:
        pass

    deepseek_api.DeepSeekPlannerClient = DeepSeekPlannerClient
    deepseek_api.DeepSeekPlannerConfig = DeepSeekPlannerConfig
    sys.modules["dspy_avqa.deepseek_api"] = deepseek_api
    return state


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_context():
    state = _install_context_import_stubs()
    _load_module("dspy_avqa.deepseek_dspy_lm", DSPY_PACKAGE_DIR / "deepseek_dspy_lm.py")
    return _load_module("dspy_avqa.context", DSPY_PACKAGE_DIR / "context.py"), state


def _runtime_context(context):
    return context.AVQARuntimeContext(system_prompt="", allowed_tools=("ask_caption",))


def test_deepseek_provider_keeps_default_api_base(monkeypatch):
    context, state = _load_context()
    monkeypatch.setenv("PLANNER_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("PLANNER_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_BASE", raising=False)
    monkeypatch.delenv("PLANNER_API_BASE", raising=False)
    monkeypatch.delenv("DSPY_PLANNER_LM_BACKEND", raising=False)

    lm = context.configure_deepseek_lm(_runtime_context(context))

    assert lm.model == "openai/deepseek-v4-pro"
    assert lm.kwargs["api_base"] == "https://api.deepseek.com"
    assert state["configured_lm"] is lm


def test_elm_gpt_uses_planner_model_and_omits_api_base(monkeypatch):
    context, state = _load_context()
    monkeypatch.setenv("PLANNER_PROVIDER", "elm_gpt")
    monkeypatch.setenv("PLANNER_MODEL", "gpt-5-mini")
    monkeypatch.setenv("PLANNER_API_KEY", "elm-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_API_BASE", "https://unused.example/v1")
    monkeypatch.setenv("PLANNER_API_BASE", "https://also-unused.example/v1")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_TOKEN", raising=False)
    monkeypatch.delenv("DSPY_PLANNER_LM_BACKEND", raising=False)

    runtime = _runtime_context(context)
    lm = context.configure_deepseek_lm(runtime)

    assert runtime.planner_model == "gpt-5-mini"
    assert lm.model == "openai/gpt-5-mini"
    assert lm.kwargs["api_key"] == "elm-key"
    assert "api_base" not in lm.kwargs
    assert state["configured_lm"] is lm


def test_api_key_mapping_keeps_existing_deepseek_precedence(monkeypatch):
    context, _ = _load_context()
    monkeypatch.setenv("PLANNER_PROVIDER", "elm_gpt")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_TOKEN", "legacy-key")
    monkeypatch.setenv("PLANNER_API_KEY", "planner-key")

    assert _runtime_context(context).planner_api_key == "deepseek-key"

    monkeypatch.delenv("DEEPSEEK_API_KEY")
    assert _runtime_context(context).planner_api_key == "legacy-key"

    monkeypatch.delenv("DEEPSEEK_TOKEN")
    assert _runtime_context(context).planner_api_key == "planner-key"


def test_elm_gpt_rejects_legacy_custom_backend(monkeypatch):
    context, _ = _load_context()
    monkeypatch.setenv("PLANNER_PROVIDER", "elm_gpt")
    monkeypatch.setenv("DSPY_PLANNER_LM_BACKEND", "custom")

    with pytest.raises(ValueError, match="only supported with PLANNER_PROVIDER=deepseek"):
        context.configure_deepseek_lm(_runtime_context(context))
