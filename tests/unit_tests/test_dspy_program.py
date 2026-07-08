from pathlib import Path
import importlib.util
import sys
import types

DSPY_PACKAGE_DIR = Path(__file__).resolve().parents[2] / "DSPy" / "dspy_avqa"


def _install_program_import_stubs() -> None:
    package = types.ModuleType("dspy_avqa")
    package.__path__ = [str(DSPY_PACKAGE_DIR)]
    sys.modules["dspy_avqa"] = package

    dspy = types.ModuleType("dspy")

    class Module:
        pass

    class Prediction(dict):
        pass

    dspy.Module = Module
    dspy.Prediction = Prediction
    dspy.Predict = lambda *args, **kwargs: None
    sys.modules["dspy"] = dspy

    deepseek = types.ModuleType("dspy_avqa.deepseek_dspy_lm")
    deepseek.clear_planner_call_trace = lambda: None
    deepseek.consume_planner_call_trace = lambda: []
    sys.modules["dspy_avqa.deepseek_dspy_lm"] = deepseek

    context = types.ModuleType("dspy_avqa.context")

    class AVQARuntimeContext:
        pass

    context.AVQARuntimeContext = AVQARuntimeContext
    context.configure_deepseek_lm = lambda *args, **kwargs: None
    context.resolve_allowed_tools = lambda value=None: ("ask_perception",)
    sys.modules["dspy_avqa.context"] = context

    signatures = types.ModuleType("dspy_avqa.signatures")
    signatures.PlanNextAction = object
    sys.modules["dspy_avqa.signatures"] = signatures

    tools = types.ModuleType("dspy_avqa.tools")
    tools.ask_caption = lambda *args, **kwargs: ""
    tools.ask_perception = lambda *args, **kwargs: ""
    tools.consume_last_perception_metadata = lambda: {}
    tools.selected_captioner_model = lambda: "gemini"
    tools.selected_perception_model = lambda: "gemini"
    tools.temporal_ground_video = lambda *args, **kwargs: ""
    sys.modules["dspy_avqa.tools"] = tools


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_install_program_import_stubs()
prompt_config = _load_module("dspy_avqa.prompt_config", DSPY_PACKAGE_DIR / "prompt_config.py")
program = _load_module("dspy_avqa.program", DSPY_PACKAGE_DIR / "program.py")


def test_perceptual_question_prefers_planner_payload(tmp_path):
    prompt_yaml = tmp_path / "prompt.yaml"
    prompt_yaml.write_text("""
perception:
  default_perceptual_question: |
    {perceptual_question}
""")
    prompt_config.load_prompt_config(prompt_yaml)

    payload = {"arguments": {"perceptual_question": "planner-generated question"}}

    assert program._perceptual_question(payload, source_question="original AVQA question") == "planner-generated question"


def test_default_perceptual_question_renders_explicit_placeholder(tmp_path):
    prompt_yaml = tmp_path / "prompt.yaml"
    prompt_yaml.write_text("""
perception:
  default_perceptual_question: |
    {perceptual_question}
""")
    prompt_config.load_prompt_config(prompt_yaml)

    assert program._perceptual_question({}, source_question="original AVQA question") == "original AVQA question"


def test_default_perceptual_question_keeps_hardcoded_text(tmp_path):
    prompt_yaml = tmp_path / "prompt.yaml"
    prompt_yaml.write_text("""
perception:
  default_perceptual_question: |
    What key visible and audible events help answer the question?
""")
    prompt_config.load_prompt_config(prompt_yaml)

    assert (
        program._perceptual_question({}, source_question="original AVQA question")
        == "What key visible and audible events help answer the question?"
    )
