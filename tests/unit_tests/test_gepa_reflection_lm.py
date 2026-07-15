import importlib.util
import os
import sys
import types
from pathlib import Path


OPTIMIZE_PATH = Path(__file__).resolve().parents[2] / "DSPy" / "dspy_avqa" / "optimize.py"


def _load_optimize_module():
    package = types.ModuleType("dspy_avqa")
    package.__path__ = [str(OPTIMIZE_PATH.parent)]
    sys.modules["dspy_avqa"] = package

    dspy = types.ModuleType("dspy")
    dspy.Module = type("Module", (), {})
    dspy.Prediction = type("Prediction", (dict,), {})
    dspy.Example = type("Example", (), {})
    dspy.Signature = type("Signature", (), {})
    dspy.InputField = lambda **kwargs: kwargs
    dspy.OutputField = lambda **kwargs: kwargs
    sys.modules["dspy"] = dspy

    context = types.ModuleType("dspy_avqa.context")
    context.AVQARuntimeContext = object
    context.CAPTION_PLACEMENT_CHOICES = ()
    context.normalize_caption_placement = lambda value=None: value
    context.resolve_allowed_tools = lambda value=None: ()
    sys.modules["dspy_avqa.context"] = context

    deepseek_lm = types.ModuleType("dspy_avqa.deepseek_dspy_lm")
    deepseek_lm.consume_planner_call_trace = lambda: []
    sys.modules["dspy_avqa.deepseek_dspy_lm"] = deepseek_lm

    data = types.ModuleType("dspy_avqa.data")
    for name in ("build_input_state", "build_result_row", "maybe_dump_question_data", "read_jsonl", "write_results_jsonl"):
        setattr(data, name, lambda *args, **kwargs: None)
    sys.modules["dspy_avqa.data"] = data

    program = types.ModuleType("dspy_avqa.program")
    program.AVQADSPyReActProgram = object
    program.normalize_option_letter = lambda value: value
    sys.modules["dspy_avqa.program"] = program

    prompt_config = types.ModuleType("dspy_avqa.prompt_config")
    prompt_config.active_prompt_yaml_path = lambda: None
    prompt_config.load_prompt_config = lambda *args, **kwargs: {}
    prompt_config.prompt_config = lambda: {}
    prompt_config.prompt_overrides = lambda *args, **kwargs: None
    prompt_config.prompt_value = lambda *args, **kwargs: ""
    sys.modules["dspy_avqa.prompt_config"] = prompt_config

    runner = types.ModuleType("dspy_avqa.runner")
    for name in (
        "add_gemini_backend_args",
        "configure_gemini_api_backend",
        "extract_error_info",
        "gemini_backend_log_lines",
        "load_captioner_config_yaml",
        "load_perception_config_yaml",
    ):
        setattr(runner, name, lambda *args, **kwargs: None)
    sys.modules["dspy_avqa.runner"] = runner

    signatures = types.ModuleType("dspy_avqa.signatures")
    signatures.apply_prompt_config_to_signatures = lambda: None
    sys.modules["dspy_avqa.signatures"] = signatures

    spec = importlib.util.spec_from_file_location("dspy_avqa.optimize", OPTIMIZE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dspy_avqa.optimize"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _PlannerLM:
    def __init__(self, model, kwargs):
        self.model = model
        self.kwargs = dict(kwargs)

    def copy(self, **overrides):
        return _PlannerLM(self.model, {**self.kwargs, **overrides})


def test_gepa_reflection_keeps_planner_connection_and_only_changes_temperature(monkeypatch):
    optimize = _load_optimize_module()
    planner_lm = _PlannerLM(
        "openai/gpt-5-mini",
        {"api_key": "elm-key", "max_tokens": 2048, "temperature": 0.0},
    )
    monkeypatch.setenv("GEPA_REFLECTION_TEMPERATURE", "0.7")

    reflection_lm = optimize.build_gepa_reflection_lm(planner_lm)

    assert reflection_lm is not planner_lm
    assert reflection_lm.model == planner_lm.model
    assert reflection_lm.kwargs == {
        "api_key": "elm-key",
        "max_tokens": 2048,
        "temperature": 0.7,
    }
    assert "api_base" not in reflection_lm.kwargs


def test_gepa_reflection_returns_planner_lm_when_no_temperature_is_set(monkeypatch):
    optimize = _load_optimize_module()
    planner_lm = _PlannerLM("openai/gpt-5-mini", {"api_key": "elm-key"})
    monkeypatch.delenv("GEPA_REFLECTION_TEMPERATURE", raising=False)

    assert optimize.build_gepa_reflection_lm(planner_lm) is planner_lm
