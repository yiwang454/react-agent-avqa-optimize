import importlib.util
import json
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


def test_gepa_reflection_can_override_model_and_effort_while_omitting_temperature(monkeypatch):
    optimize = _load_optimize_module()
    planner_lm = _PlannerLM(
        "openai/gpt-5.4",
        {"api_key": "elm-key", "seed": 1234, "temperature": 0.0, "reasoning_effort": "none"},
    )
    monkeypatch.setenv("GEPA_REFLECTION_MODEL", "gpt-5.4")
    monkeypatch.setenv("GEPA_REFLECTION_REASONING_EFFORT", "medium")
    monkeypatch.delenv("GEPA_REFLECTION_TEMPERATURE", raising=False)

    reflection_lm = optimize.build_gepa_reflection_lm(planner_lm)

    assert reflection_lm.model == "openai/gpt-5.4"
    assert reflection_lm.kwargs == {
        "api_key": "elm-key",
        "seed": 1234,
        "reasoning_effort": "medium",
    }


class _ModelDumpWrapper:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return self.payload


def test_json_safe_serializes_nested_model_dump_wrapper():
    optimize = _load_optimize_module()

    normalized = optimize._json_safe({"usage": [_ModelDumpWrapper({"reasoning_tokens": 17})]})

    assert json.loads(json.dumps(normalized)) == {"usage": [{"reasoning_tokens": 17}]}


def test_make_reflective_dataset_keeps_only_reflection_relevant_trace_fields():
    optimize = _load_optimize_module()
    full_reasoning = "reasoning " * 400
    full_caption = "caption " * 400
    full_planner_response = "planner response " * 200
    full_planner_reasoning = "planner reasoning " * 200
    full_observation = "observation " * 300
    adapter = optimize.PromptTargetGEPAAdapter.__new__(optimize.PromptTargetGEPAAdapter)
    adapter.component_name = "prompt_component"
    adapter.target_path = "planner.workflow_prompt"
    adapter.failure_score = 0.0
    adapter.metric_fn = lambda *args: types.SimpleNamespace(feedback="feedback")
    example = types.SimpleNamespace(
        question="question",
        options_json='["A. first", "B. second"]',
        answer="B",
        video_id="video",
        video_description=full_caption,
    )
    prediction = types.SimpleNamespace(
        answer="A",
        reasoning_summary=full_reasoning,
        turn_trace=[
            {
                "turn_id": 1,
                "planner_action": "tool",
                "tool_name": "ask_perception",
                "tool_args": {
                    "video_path": "/private/video.mp4",
                    "perceptual_question": "What sound is heard?",
                },
                "tool_observation": full_observation,
                "planner_raw": "full raw action",
                "planner_calls": {
                    "action_decision": [{
                        "messages": [{"role": "system", "content": "large system prompt"}],
                        "response_text": full_planner_response,
                        "reasoning_content": full_planner_reasoning,
                        "usage": _ModelDumpWrapper({"reasoning_tokens": 17}),
                    }],
                },
                "perception_system_prompt": "large perception prompt",
                "perception_token_usage": _ModelDumpWrapper({"completion_tokens": 50}),
                "planner_parse_error": None,
            },
            {
                "turn_id": 2,
                "planner_action": "final",
                "final_answer": "A",
                "planner_calls": {"final": [{
                    "messages": ["full history"],
                    "response_text": "complete final response",
                    "reasoning_content": "complete final reasoning",
                }]},
                "planner_parse_error": "invalid final payload",
            },
        ],
    )
    eval_batch = types.SimpleNamespace(trajectories=[{
        "example": example,
        "prediction": prediction,
        "score": 1.0,
        "trace": None,
    }])

    dataset = adapter.make_reflective_dataset(
        {"prompt_component": "current prompt"},
        eval_batch,
        ["prompt_component"],
    )

    item = dataset["prompt_component"][0]
    assert item["Inputs"] == {
        "Question": "question",
        "Options": ["A. first", "B. second"],
        "Gold answer": "B",
        "Captioner response": full_caption.strip(),
    }
    assert item["Generated Outputs"] == {
        "Predicted answer": "A",
        "Score": 1.0,
        "Reasoning": full_reasoning.strip(),
        "Status": {
            "state": "error",
            "errors": ["turn 2: invalid final payload"],
        },
        "Reflection trace": [
            {
                "turn_id": 1,
                "planner_action": "tool",
                "tool_name": "ask_perception",
                "planner_tool_question": "What sound is heard?",
                "perception_observation": full_observation,
                "planner_response": full_planner_response.strip(),
                "planner_reasoning": full_planner_reasoning.strip(),
            },
            {
                "turn_id": 2,
                "planner_action": "final",
                "final_answer": "A",
                "planner_response": "complete final response",
                "planner_reasoning": "complete final reasoning",
                "parse_error": "invalid final payload",
            },
        ],
    }
    serialized = json.dumps(item)
    for removed_text in (
        "large system prompt",
        "large perception prompt",
        "reasoning_tokens",
        "completion_tokens",
        "full history",
        "/private/video.mp4",
        "current prompt",
    ):
        assert removed_text not in serialized


def test_compact_reflection_trace_records_failed_prediction_status():
    optimize = _load_optimize_module()
    failed_prediction = types.SimpleNamespace(
        completion_text="unparseable model completion",
    )

    turns, status = optimize._compact_reflection_trace(failed_prediction)

    assert turns == []
    assert status == {
        "state": "error",
        "errors": ["prediction parse failure: unparseable model completion"],
    }


def test_gepa_feedback_does_not_repeat_question_options_or_reasoning():
    optimize = _load_optimize_module()
    example = types.SimpleNamespace(
        answer="B",
        question="duplicate question",
        options_json='["duplicate options"]',
    )
    prediction = types.SimpleNamespace(answer="A", reasoning_summary="duplicate reasoning")

    result = optimize.avqa_gepa_feedback_metric(example, prediction)

    feedback = result["feedback"]
    assert "Incorrect (score=0)" in feedback
    assert "gold=B, predicted=A" in feedback
    assert "duplicate question" not in feedback
    assert "duplicate options" not in feedback
    assert "duplicate reasoning" not in feedback


def _reflective_dataset_event(iteration):
    return {
        "iteration": iteration,
        "candidate_idx": 0,
        "components": ["prompt_component"],
        "dataset": {"prompt_component": [{"Feedback": f"feedback-{iteration}"}]},
    }


def test_reflective_dataset_snapshots_follow_cadence_and_resume(tmp_path):
    optimize = _load_optimize_module()
    callback = optimize.ReflectiveDatasetSnapshotCallback(tmp_path, save_interval=50)
    for iteration in range(1, 50):
        callback.on_reflective_dataset_built(_reflective_dataset_event(iteration))

    resumed_callback = optimize.ReflectiveDatasetSnapshotCallback(tmp_path, save_interval=50)
    assert resumed_callback.datasets_seen == 49
    for iteration in range(50, 101):
        resumed_callback.on_reflective_dataset_built(_reflective_dataset_event(iteration))

    snapshot_dir = tmp_path / "reflective_datasets"
    snapshots = sorted(snapshot_dir.glob("reflective_dataset_*.json"))
    assert [json.loads(path.read_text())["reflective_dataset_index"] for path in snapshots] == [1, 50, 100]
    assert json.loads((snapshot_dir / "capture_state.json").read_text()) == {"datasets_seen": 100}


def test_reflective_dataset_snapshots_can_be_disabled(tmp_path):
    optimize = _load_optimize_module()
    callback = optimize.ReflectiveDatasetSnapshotCallback(tmp_path, save_interval=0)

    callback.on_reflective_dataset_built(_reflective_dataset_event(1))

    assert not (tmp_path / "reflective_datasets").exists()


def test_reflective_dataset_snapshot_interval_must_be_nonnegative(tmp_path):
    optimize = _load_optimize_module()

    try:
        optimize.ReflectiveDatasetSnapshotCallback(tmp_path, save_interval=-1)
    except ValueError as exc:
        assert "must be >= 0" in str(exc)
    else:
        raise AssertionError("negative save interval should fail")
