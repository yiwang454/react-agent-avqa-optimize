import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest


OPTIMIZE_PATH = Path(__file__).resolve().parents[2] / "DSPy" / "dspy_avqa" / "optimize.py"


def _load_optimize_module():
    package = types.ModuleType("dspy_avqa")
    package.__path__ = [str(OPTIMIZE_PATH.parent)]
    sys.modules["dspy_avqa"] = package

    dspy = types.ModuleType("dspy")
    dspy.Module = type("Module", (), {})
    dspy.Prediction = type("Prediction", (dict,), {})

    class Example:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self._input_keys = ()

        def with_inputs(self, *keys):
            self._input_keys = keys
            return self

        def inputs(self):
            return {key: getattr(self, key) for key in self._input_keys}

        def labels(self):
            return {
                key: value
                for key, value in self.__dict__.items()
                if key not in self._input_keys and key != "_input_keys"
            }

    dspy.Example = Example
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

    tools = types.ModuleType("dspy_avqa.tools")
    tools.build_caption_prompt = lambda *args, **kwargs: ""
    sys.modules["dspy_avqa.tools"] = tools

    runner = types.ModuleType("dspy_avqa.runner")
    for name in (
        "add_gemini_backend_args",
        "configure_gemini_api_backend",
        "cut_id",
        "extract_error_info",
        "gemini_backend_log_lines",
        "load_captioner_config_yaml",
        "load_cached_row_from_question_json",
        "load_perception_config_yaml",
        "resolve_preloaded_audio_caption_dir",
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


def test_gepa_reflection_model_override_does_not_inherit_deepseek_connection(monkeypatch):
    optimize = _load_optimize_module()
    planner_lm = _PlannerLM(
        "openai/deepseek-v4-pro",
        {
            "api_key": "deepseek-key",
            "api_base": "https://api.deepseek.com",
            "seed": 7,
            "temperature": 0.0,
        },
    )
    monkeypatch.setenv("GEPA_REFLECTION_MODEL", "gpt-5.4")
    monkeypatch.setenv("GEPA_REFLECTION_REASONING_EFFORT", "medium")
    monkeypatch.delenv("GEPA_REFLECTION_TEMPERATURE", raising=False)

    reflection_lm = optimize.build_gepa_reflection_lm(planner_lm)

    assert reflection_lm.model == "openai/gpt-5.4"
    assert reflection_lm.kwargs == {
        "seed": 7,
        "reasoning_effort": "medium",
    }


def _reflection_template_yaml_path():
    return OPTIMIZE_PATH.parent / "yamls" / "DSPy" / "reflection_template.yaml"


def test_gepa_reflection_template_auto_uses_original_without_densified_labels():
    optimize = _load_optimize_module()
    args = types.SimpleNamespace(
        gepa_reflection_template_yaml=_reflection_template_yaml_path(),
        gepa_reflection_template_version="auto",
    )

    info = optimize.configure_gepa_reflection_prompt_templates(
        args,
        densified_enabled=False,
    )

    assert info["resolved_version"] == "original"
    planner_template = optimize.GEPA_PLANNER_REFLECTION_PROMPT_TEMPLATE
    captioner_template = optimize.GEPA_CAPTIONER_REFLECTION_PROMPT_TEMPLATE
    assert "Below are optimization examples containing runtime trajectories and evaluation feedback" in planner_template
    assert "For each example with densified supervision" not in planner_template
    assert "matching Key evidence" not in planner_template
    assert "any privileged dataset evidence if enabled" in captioner_template
    assert "For each example with densified supervision" not in captioner_template


def test_gepa_reflection_template_auto_uses_key_evidence_with_densified_labels():
    optimize = _load_optimize_module()
    args = types.SimpleNamespace(
        gepa_reflection_template_yaml=_reflection_template_yaml_path(),
        gepa_reflection_template_version="auto",
    )

    info = optimize.configure_gepa_reflection_prompt_templates(
        args,
        densified_enabled=True,
    )

    assert info["resolved_version"] == "densified_key_evidence"
    planner_template = optimize.GEPA_PLANNER_REFLECTION_PROMPT_TEMPLATE
    captioner_template = optimize.GEPA_CAPTIONER_REFLECTION_PROMPT_TEMPLATE
    assert "For each example with densified supervision" in planner_template
    assert "matching Key evidence" in planner_template
    assert "Ideal perception target" not in planner_template
    assert "For each example with densified supervision" in captioner_template
    assert "matching Key evidence" in captioner_template
    assert "Ideal perception target" not in captioner_template


class _ModelDumpWrapper:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return self.payload


def test_json_safe_serializes_nested_model_dump_wrapper():
    optimize = _load_optimize_module()

    normalized = optimize._json_safe({"usage": [_ModelDumpWrapper({"reasoning_tokens": 17})]})

    assert json.loads(json.dumps(normalized)) == {"usage": [{"reasoning_tokens": 17}]}


def test_gepa_captioner_max_tokens_are_aggregated_per_batch(capsys):
    optimize = _load_optimize_module()
    adapter = optimize.PromptTargetGEPAAdapter.__new__(optimize.PromptTargetGEPAAdapter)
    predictions = [
        types.SimpleNamespace(
            turn_trace=[
                {
                    "tool_name": "ask_caption",
                    "perception_backend": "qwen",
                    "perception_token_usage": {
                        "qwen_max_tokens": 256,
                        "completion_tokens": 256,
                        "qwen_finish_reason": "length",
                    },
                },
                {
                    "tool_name": "ask_perception",
                    "perception_backend": "qwen",
                    "perception_token_usage": {
                        "qwen_max_tokens": 256,
                        "completion_tokens": 256,
                        "qwen_finish_reason": "length",
                    },
                },
            ]
        ),
        types.SimpleNamespace(
            turn_trace=[
                {
                    "tool_name": "ask_caption",
                    "perception_backend": "qwen",
                    "perception_token_usage": {
                        "qwen_max_tokens": 256,
                        "completion_tokens": 255,
                        "qwen_finish_reason": "stop",
                    },
                },
            ]
        ),
        types.SimpleNamespace(turn_trace=[]),
    ]

    stats = optimize._qwen_captioner_max_token_stats(predictions)

    assert stats == {
        "captioner_samples": 2,
        "hit_samples": 1,
        "captioner_calls": 2,
        "hit_calls": 1,
        "max_tokens_values": (256,),
    }
    adapter._report_captioner_max_token_hits(predictions, capture_traces=False)
    report = capsys.readouterr().out
    assert report.count("\n") == 1
    assert "batch=1 phase=evaluation batch_samples=3" in report
    assert "hit_max_tokens_samples=1" in report
    assert "hit_max_tokens_calls=1" in report
    assert "observed_max_tokens=256" in report


def test_gepa_adapter_retries_empty_qwen_predictions_after_batch(monkeypatch):
    optimize = _load_optimize_module()
    reliable_qwen = sys.modules["dspy_avqa.reliable_qwen"]

    class Example:
        def inputs(self):
            return {"sample": "one"}

    profiles = []

    class Program:
        def __call__(self, **kwargs):
            profile = reliable_qwen.active_qwen_request_profile()
            profiles.append(profile.name if profile is not None else None)
            return types.SimpleNamespace(
                turn_trace=[
                    {
                        "planner_action": "tool",
                        "perception_backend": "qwen",
                        "tool_observation": "recovered evidence",
                    }
                ]
            )

    adapter = optimize.PromptTargetGEPAAdapter.__new__(optimize.PromptTargetGEPAAdapter)
    adapter.qwen_executor = reliable_qwen.ReliableQwenExecutor(max_batch_retries=2)
    adapter.metric_fn = lambda example, prediction: 1.0
    predictions = [
        types.SimpleNamespace(
            turn_trace=[
                {
                    "planner_action": "tool",
                    "perception_backend": "qwen",
                    "tool_observation": "\n",
                }
            ]
        )
    ]
    raw_scores = [0.0]

    adapter._retry_empty_qwen_predictions(
        program=Program(),
        batch=[Example()],
        predictions=predictions,
        raw_scores=raw_scores,
    )

    assert profiles == ["retry_separate_video_first"]
    assert predictions[0].turn_trace[0]["tool_observation"] == "recovered evidence"
    assert raw_scores == [1.0]


def test_gepa_adapter_marks_exhausted_empty_qwen_rollout_for_reflection():
    optimize = _load_optimize_module()
    reliable_qwen = sys.modules["dspy_avqa.reliable_qwen"]

    class Example:
        def inputs(self):
            return {}

    class Program:
        def __call__(self, **kwargs):
            return types.SimpleNamespace(
                turn_trace=[
                    {
                        "planner_action": "tool",
                        "perception_backend": "qwen",
                        "tool_observation": "",
                    }
                ]
            )

    adapter = optimize.PromptTargetGEPAAdapter.__new__(optimize.PromptTargetGEPAAdapter)
    adapter.qwen_executor = reliable_qwen.ReliableQwenExecutor(max_batch_retries=2)
    adapter.metric_fn = lambda example, prediction: 0.0
    adapter.failure_score = 0.0
    predictions = [Program()()]

    invalid_indices, attempted_profiles = adapter._retry_empty_qwen_predictions(
        program=Program(),
        batch=[Example()],
        predictions=predictions,
        raw_scores=[0.0],
    )

    assert invalid_indices == [0]
    assert attempted_profiles == [
        "primary_separate_audio_first",
        "retry_separate_video_first",
        "retry_embedded_audio",
    ]
    scores = adapter._apply_exhausted_qwen_outcome(
        predictions=predictions,
        raw_scores=[1.0],
        invalid_indices=invalid_indices,
        attempted_profiles=attempted_profiles,
        is_validation_batch=False,
    )
    assert scores == [1.0]
    assert predictions[0].turn_trace[0]["reliable_qwen_error"]["error_type"] == "EmptyQwenResponse"


def test_exhausted_validation_qwen_scores_are_excluded_from_accuracy_without_shifting_ids():
    optimize = _load_optimize_module()

    adapter = optimize.PromptTargetGEPAAdapter.__new__(optimize.PromptTargetGEPAAdapter)
    adapter.failure_score = 0.0
    predictions = [
        types.SimpleNamespace(turn_trace=[]),
        types.SimpleNamespace(
            turn_trace=[
                {
                    "planner_action": "tool",
                    "perception_backend": "qwen",
                    "tool_observation": "",
                }
            ]
        ),
        types.SimpleNamespace(turn_trace=[]),
    ]

    scores = adapter._apply_exhausted_qwen_outcome(
        predictions=predictions,
        raw_scores=[1.0, 0.0, 0.0],
        invalid_indices=[1],
        attempted_profiles=["primary_separate_audio_first", "retry_embedded_audio"],
        is_validation_batch=True,
    )

    assert len(scores) == 3
    assert scores[1].included is False
    # GEPA's validation state computes sum(scores) / len(scores).  The custom
    # sum object deliberately ignores the exhausted sample's denominator.
    assert sum(scores) / len(scores) == 0.5
    assert predictions[1].turn_trace[0]["reliable_qwen_error"]["attempted_profiles"] == [
        "primary_separate_audio_first",
        "retry_embedded_audio",
    ]


def test_make_reflective_dataset_keeps_only_reflection_relevant_trace_fields():
    optimize = _load_optimize_module()
    full_reasoning = "reasoning " * 400
    full_caption = "caption " * 400
    full_planner_response = "planner response " * 200
    full_planner_reasoning = "planner reasoning " * 200
    full_observation = "observation " * 300
    adapter = optimize.PromptTargetGEPAAdapter.__new__(optimize.PromptTargetGEPAAdapter)
    adapter.component_names = ("prompt_component",)
    adapter.student = types.SimpleNamespace(target_paths=("planner.workflow_prompt",))
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
        "Optimization target": "planner.workflow_prompt",
        "Component role": optimize._target_component_role("planner.workflow_prompt"),
        "Output contract": optimize._target_output_contract("planner.workflow_prompt"),
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


def test_gepa_caption_supervision_loads_full_files_caches_and_warns_for_missing(tmp_path, caplog):
    optimize = _load_optimize_module()
    video_dir = tmp_path / "Videos" / "video-1"
    video_dir.mkdir(parents=True)
    av_alignment = "audio one -- visual one\r\nsecond alignment\r\n"
    visual_caption = "visual identity\nscene continuity\n"
    (video_dir / "av_alignment_captions.txt").write_text(av_alignment, encoding="utf-8")
    (video_dir / "video_consistent_captions.txt").write_text(visual_caption, encoding="utf-8")
    cache = {}
    stats = optimize._new_gepa_caption_supervision_stats()

    first = optimize._load_gepa_caption_supervision(
        daily_omni_root=tmp_path,
        video_id="video-1",
        cache=cache,
        stats=stats,
    )
    second = optimize._load_gepa_caption_supervision(
        daily_omni_root=tmp_path,
        video_id="video-1",
        cache=cache,
        stats=stats,
    )

    assert second is first
    assert first["gepa_privileged_av_alignment_captions"] == av_alignment
    assert first["gepa_privileged_video_consistent_captions"] == visual_caption
    assert first["gepa_privileged_audio_revised_captions"] == (
        "[MISSING PRIVILEGED SUPERVISION FILE: audio_revised_captions.txt]"
    )
    assert stats == {
        "videos_loaded": 1,
        "cache_hits": 1,
        "files_loaded": 2,
        "missing_files": 1,
        "unreadable_files": 0,
    }
    assert "audio_revised_captions.txt" in caplog.text


def test_gepa_caption_supervision_mode_none_disables_privileged_loading():
    optimize = _load_optimize_module()
    args = types.SimpleNamespace(
        algorithm="gepa",
        gepa_caption_supervision="none",
        daily_omni_root=None,
    )

    mode, root = optimize._resolve_gepa_caption_supervision(
        args,
        optimizes_captioner=True,
    )

    assert mode == "none"
    assert root is None


def test_gepa_densified_supervision_loads_key_evidence_from_current_and_legacy_schemas(tmp_path):
    optimize = _load_optimize_module()
    label_path = tmp_path / "video-1-2.json"
    label_path.write_text(
        json.dumps({
            "key_evidence": "  decisive visible action  ",
            "ideal_perception_target": "  legacy field is ignored  ",
        }),
        encoding="utf-8",
    )

    loaded = optimize._load_gepa_densified_supervision(
        label_dir=tmp_path,
        cut_id="video-1-2",
    )

    assert loaded == {
        "gepa_privileged_key_evidence": "decisive visible action",
    }

    label_path.write_text(
        json.dumps({"key_evidence": "  key-only evidence  "}),
        encoding="utf-8",
    )
    assert optimize._load_gepa_densified_supervision(
        label_dir=tmp_path,
        cut_id="video-1-2",
    ) == {
        "gepa_privileged_key_evidence": "key-only evidence",
    }

    label_path.write_text(
        json.dumps({"ideal_perception_target": "missing required key"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="required fields"):
        optimize._load_gepa_densified_supervision(
            label_dir=tmp_path,
            cut_id="video-1-2",
        )

    label_path.write_text(
        json.dumps({"key_evidence": "evidence", "metadata": "unsupported"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only supported fields"):
        optimize._load_gepa_densified_supervision(
            label_dir=tmp_path,
            cut_id="video-1-2",
        )

    label_path.write_text(
        json.dumps({"key_evidence": "  "}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be a non-empty string"):
        optimize._load_gepa_densified_supervision(
            label_dir=tmp_path,
            cut_id="video-1-2",
        )


def test_make_trainset_from_cuts_attaches_densified_labels_by_question_id(tmp_path, monkeypatch):
    optimize = _load_optimize_module()
    (tmp_path / "question-2.json").write_text(
        json.dumps({
            "key_evidence": "matching evidence",
            "ideal_perception_target": "matching perception target",
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(optimize, "build_input_state", lambda *args: {
        "question": "question",
        "options": ["A. first", "B. second"],
        "video_path": "/video.mp4",
        "audio_path": "/audio.wav",
        "video_id": "shared-video",
        "video_description": "runtime caption",
    })
    cuts = [{"id": "question-2", "supervisions": [{"custom": {"answer": "B"}}]}]

    examples, skipped = optimize.make_trainset_from_cuts(
        cuts,
        None,
        gepa_densified_label_dir=tmp_path,
    )

    assert skipped == []
    assert examples[0].inputs()["question_id"] == "question-2"
    assert examples[0].labels()["gepa_privileged_key_evidence"] == "matching evidence"
    assert "gepa_privileged_ideal_perception_target" not in examples[0].labels()


def test_gepa_caption_supervision_mode_auto_uses_root_when_available(tmp_path):
    optimize = _load_optimize_module()
    (tmp_path / "Videos").mkdir()
    args = types.SimpleNamespace(
        algorithm="gepa",
        gepa_caption_supervision="auto",
        daily_omni_root=tmp_path,
    )

    mode, root = optimize._resolve_gepa_caption_supervision(
        args,
        optimizes_captioner=True,
    )

    assert mode == "privileged"
    assert root == tmp_path


def test_gepa_caption_supervision_is_a_label_not_a_runtime_input():
    optimize = _load_optimize_module()
    raw_item = {
        "question": "question",
        "options": ["A. first", "B. second"],
        "video_path": "/video.mp4",
        "video_id": "video",
        "answer": "A",
        "gepa_privileged_av_alignment_captions": "alignment gold",
        "gepa_privileged_video_consistent_captions": "visual gold",
        "gepa_privileged_audio_revised_captions": "audio gold",
        "gepa_privileged_key_evidence": "decisive evidence",
        "gepa_privileged_ideal_perception_target": "ideal target",
    }

    example = optimize.make_trainset([raw_item])[0]

    assert example.inputs()["question_id"] == "video"
    assert "gepa_privileged_av_alignment_captions" not in example.inputs()
    assert example.labels()["gepa_privileged_av_alignment_captions"] == "alignment gold"
    assert example.labels()["gepa_privileged_video_consistent_captions"] == "visual gold"
    assert example.labels()["gepa_privileged_audio_revised_captions"] == "audio gold"
    assert "gepa_privileged_key_evidence" not in example.inputs()
    assert "gepa_privileged_ideal_perception_target" not in example.inputs()
    assert example.labels()["gepa_privileged_key_evidence"] == "decisive evidence"
    assert "gepa_privileged_ideal_perception_target" not in example.labels()


def test_reflective_dataset_and_proposal_prompts_are_target_specific():
    optimize = _load_optimize_module()
    optimize.configure_gepa_reflection_prompt_templates(
        types.SimpleNamespace(
            gepa_reflection_template_yaml=_reflection_template_yaml_path(),
            gepa_reflection_template_version="densified_key_evidence",
        ),
        densified_enabled=True,
    )
    adapter = optimize.PromptTargetGEPAAdapter.__new__(optimize.PromptTargetGEPAAdapter)
    adapter.component_names = ("planner_component", "captioner_component")
    adapter.student = types.SimpleNamespace(
        target_paths=("planner.workflow_prompt", "captioner.default_caption_instruction")
    )
    adapter.failure_score = 0.0
    adapter.metric_fn = lambda *args: types.SimpleNamespace(feedback="feedback")
    example = types.SimpleNamespace(
        question="question",
        options_json='["A. first", "B. second"]',
        answer="B",
        video_description="runtime caption",
        gepa_privileged_av_alignment_captions="full alignment\nwith newline\n",
        gepa_privileged_video_consistent_captions="full visual\n",
        gepa_privileged_audio_revised_captions="full audio\n",
        gepa_privileged_key_evidence="question-matched key evidence",
        gepa_privileged_ideal_perception_target="question-matched perception target",
    )
    prediction = types.SimpleNamespace(answer="A", reasoning_summary="reasoning", turn_trace=[])
    eval_batch = types.SimpleNamespace(trajectories=[{
        "example": example,
        "prediction": prediction,
        "score": 0.0,
        "trace": None,
    }])

    dataset = adapter.make_reflective_dataset(
        {"planner_component": "planner prompt", "captioner_component": "caption prompt"},
        eval_batch,
        ["planner_component", "captioner_component"],
    )

    planner_inputs = dataset["planner_component"][0]["Inputs"]
    captioner_inputs = dataset["captioner_component"][0]["Inputs"]
    assert planner_inputs["Optimization target"] == "planner.workflow_prompt"
    assert "AVQA ReAct planner" in planner_inputs["Component role"]
    assert "Privileged dataset evidence (optimization only)" not in planner_inputs
    assert captioner_inputs["Optimization target"] == "captioner.default_caption_instruction"
    assert "Captioner tool instruction" in captioner_inputs["Component role"]
    assert "Never output an answer letter" in captioner_inputs["Output contract"]
    assert "Caption quality rubric" in captioner_inputs
    expected_densified = {
        "Key evidence": "question-matched key evidence",
    }
    assert (
        planner_inputs["Privileged densified supervision (optimization only)"]
        == expected_densified
    )
    assert (
        captioner_inputs["Privileged densified supervision (optimization only)"]
        == expected_densified
    )
    evidence = captioner_inputs["Privileged dataset evidence (optimization only)"]
    assert evidence == (
        "=== PRIVILEGED DATASET EVIDENCE FOR OPTIMIZATION ONLY ===\n\n"
        "[PRIMARY: AV ALIGNMENT]\nfull alignment\nwith newline\n\n"
        "[AUXILIARY: VISUAL CAPTION]\nfull visual\n\n"
        "[AUXILIARY: AUDIO CAPTION]\nfull audio\n\n\n"
        "=== END PRIVILEGED EVIDENCE ==="
    )

    candidate = {"planner_component": "planner prompt", "captioner_component": "caption prompt"}
    planner_proposal = adapter._proposal_input(
        candidate=candidate,
        reflective_dataset=dataset,
        component_name="planner_component",
    )
    captioner_proposal = adapter._proposal_input(
        candidate=candidate,
        reflective_dataset=dataset,
        component_name="captioner_component",
    )
    planner_template = planner_proposal["prompt_template"]
    assert "planner workflow instruction" in planner_template
    assert "Infer the task format and the behavior required to solve it." in planner_template
    assert "planning and tool-use policy" in planner_template
    assert "not to learn facts from the example videos" in planner_template
    assert "Turn example-specific feedback into general rules" in planner_template
    assert "how to formulate one targeted perceptual question" in planner_template
    assert "matching Key evidence" in planner_template
    assert "Ideal perception target" not in planner_template
    assert "privileged supervision is unavailable at runtime" in planner_template
    assert "niche and domain specific factual information" not in planner_template
    prompt_template = captioner_proposal["prompt_template"]
    assert "ask_caption tool to obtain" in prompt_template
    assert "must not solve the multiple-choice question" in prompt_template
    assert "generalize to unseen videos" in prompt_template
    assert "Do not copy or encode any example-specific answer" in prompt_template
    assert "matching Key evidence" in prompt_template
    assert "Ideal perception target" not in prompt_template
    assert "All privileged supervision is unavailable at runtime" in prompt_template
    assert "ask_caption / ask_perception workflow" not in prompt_template


def test_captioner_reflective_dataset_omits_privileged_evidence_when_not_loaded():
    optimize = _load_optimize_module()
    adapter = optimize.PromptTargetGEPAAdapter.__new__(optimize.PromptTargetGEPAAdapter)
    adapter.component_names = ("captioner_component",)
    adapter.student = types.SimpleNamespace(
        target_paths=("captioner.default_caption_instruction",)
    )
    adapter.failure_score = 0.0
    adapter.metric_fn = lambda *args: types.SimpleNamespace(feedback="feedback")
    example = types.SimpleNamespace(
        question="question",
        options_json='["A. first", "B. second"]',
        answer="B",
        video_description="runtime caption",
    )
    prediction = types.SimpleNamespace(answer="A", reasoning_summary="reasoning", turn_trace=[])
    eval_batch = types.SimpleNamespace(trajectories=[{
        "example": example,
        "prediction": prediction,
        "score": 0.0,
        "trace": None,
    }])

    dataset = adapter.make_reflective_dataset(
        {"captioner_component": "caption prompt"},
        eval_batch,
        ["captioner_component"],
    )

    inputs = dataset["captioner_component"][0]["Inputs"]
    assert inputs["Optimization target"] == "captioner.default_caption_instruction"
    assert "Caption quality rubric" in inputs
    assert "Privileged dataset evidence (optimization only)" not in inputs
    assert "Privileged densified supervision (optimization only)" not in inputs


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


def test_gepa_feedback_is_target_aware_for_captioner_and_planner():
    optimize = _load_optimize_module()
    example = types.SimpleNamespace(answer="B")
    prediction = types.SimpleNamespace(answer="A")

    planner_feedback = optimize.avqa_gepa_feedback_metric(
        example,
        prediction,
        pred_name="planner.workflow_prompt",
    )["feedback"]
    captioner_feedback = optimize.avqa_gepa_feedback_metric(
        example,
        prediction,
        pred_name="captioner.default_caption_instruction",
    )["feedback"]

    assert "Revise the planner instruction" in planner_feedback
    assert "returns exactly one option letter" in planner_feedback
    assert "Revise only the captioner instruction" in captioner_feedback
    assert "timestamped audio-visual caption" in captioner_feedback
    assert "Revise the planner instruction" not in captioner_feedback
    assert "output an option letter" in captioner_feedback


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


def _install_final_eval_test_io(optimize):
    optimize.cut_id = lambda cut: str(cut.get("id") or "")

    def load_cached(output_dir, cut):
        sample_id = optimize.cut_id(cut)
        path = output_dir / f"{sample_id}.json"
        if not path.exists():
            return None
        try:
            question_data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(question_data, dict) or not str(question_data.get("response") or "").strip():
            return None
        return {
            "video_id": sample_id,
            "metadata": {"video_id": sample_id},
            "question_data": question_data,
        }

    def dump_sample(output_dir, row):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{row['video_id']}.json").write_text(
            json.dumps(row["question_data"]),
            encoding="utf-8",
        )

    def write_rows(rows, output_jsonl):
        output_jsonl.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    optimize.load_cached_row_from_question_json = load_cached
    optimize.maybe_dump_question_data = dump_sample
    optimize.write_results_jsonl = write_rows


def _final_eval_row(sample_id):
    return {
        "video_id": sample_id,
        "metadata": {"video_id": sample_id},
        "question_data": {"response": f"answer-{sample_id}"},
    }


def test_final_eval_resumes_after_interruption_and_preserves_input_order(tmp_path):
    optimize = _load_optimize_module()
    _install_final_eval_test_io(optimize)
    cuts = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    first_calls = []

    def interrupted_run(program, cut, audio_caption_dir, max_turns):
        sample_id = cut["id"]
        first_calls.append(sample_id)
        if sample_id == "c":
            raise KeyboardInterrupt
        return _final_eval_row(sample_id)

    optimize._run_program_on_cut = interrupted_run
    try:
        optimize.write_batch_style_program_outputs(
            object(), cuts, tmp_path, tmp_path / "output.jsonl", output_dir=tmp_path, max_turns=3
        )
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("the simulated final eval should be interrupted")

    assert first_calls == ["a", "b", "c"]
    assert (tmp_path / "a.json").exists()
    assert (tmp_path / "b.json").exists()
    assert not (tmp_path / "c.json").exists()

    resumed_calls = []

    def resumed_run(program, cut, audio_caption_dir, max_turns):
        resumed_calls.append(cut["id"])
        return _final_eval_row(cut["id"])

    optimize._run_program_on_cut = resumed_run
    result = optimize.write_batch_style_program_outputs(
        object(), cuts, tmp_path, tmp_path / "output.jsonl", output_dir=tmp_path, max_turns=3
    )

    assert resumed_calls == ["c"]
    assert result["final_eval_cached_samples"] == 2
    assert result["final_eval_new_samples"] == 1
    assert result["final_eval_complete"] is True
    output_rows = [json.loads(line) for line in (tmp_path / "output.jsonl").read_text().splitlines()]
    assert [row["video_id"] for row in output_rows] == ["a", "b", "c"]


def test_final_eval_regenerates_unreadable_or_empty_cache(tmp_path):
    optimize = _load_optimize_module()
    _install_final_eval_test_io(optimize)
    (tmp_path / "a.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"response": ""}), encoding="utf-8")
    (tmp_path / "c.json").write_text(json.dumps({"response": "cached-c"}), encoding="utf-8")
    calls = []
    optimize._run_program_on_cut = lambda program, cut, audio_caption_dir, max_turns: (
        calls.append(cut["id"]) or _final_eval_row(cut["id"])
    )

    result = optimize.write_batch_style_program_outputs(
        object(),
        [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        tmp_path,
        tmp_path / "output.jsonl",
        output_dir=tmp_path,
        max_turns=3,
    )

    assert calls == ["a", "b"]
    assert result["final_eval_cached_samples"] == 1
    assert result["final_eval_new_samples"] == 2


def test_final_eval_parallel_primary_uses_reliable_qwen_retry(tmp_path):
    optimize = _load_optimize_module()
    _install_final_eval_test_io(optimize)
    reliable_qwen = sys.modules["dspy_avqa.reliable_qwen"]
    profiles = []

    def run(program, cut, audio_caption_dir, max_turns):
        profile = reliable_qwen.active_qwen_request_profile()
        assert profile is not None
        profiles.append((cut["id"], profile.name))
        empty = cut["id"] == "retry" and profile.name == "primary_separate_audio_first"
        return {
            "video_id": cut["id"],
            "metadata": {"video_id": cut["id"]},
            "question_data": {
                "response": "A. answer",
                "turn_trace": [
                    {
                        "planner_action": "tool",
                        "perception_backend": "qwen",
                        "tool_observation": "" if empty else "visible evidence",
                    }
                ],
            },
        }

    optimize._run_program_on_cut = run
    result = optimize.write_batch_style_program_outputs(
        object(),
        [{"id": "valid"}, {"id": "retry"}],
        tmp_path,
        tmp_path / "output.jsonl",
        output_dir=tmp_path,
        max_turns=3,
        num_threads=2,
        batch_size=2,
    )

    assert result["final_eval_qwen_exhausted_samples"] == 0
    assert profiles.count(("valid", "primary_separate_audio_first")) == 1
    assert profiles.count(("retry", "primary_separate_audio_first")) == 1
    assert profiles.count(("retry", "retry_separate_video_first")) == 1


def test_final_eval_lock_rejects_a_second_writer(tmp_path):
    optimize = _load_optimize_module()

    with optimize._final_eval_directory_lock(tmp_path):
        try:
            with optimize._final_eval_directory_lock(tmp_path):
                raise AssertionError("second writer unexpectedly acquired the lock")
        except RuntimeError as exc:
            assert "already running" in str(exc)


def test_load_optimized_program_rejects_missing_or_corrupt_output(tmp_path):
    optimize = _load_optimize_module()

    try:
        optimize._load_optimized_program(object(), tmp_path / "missing.json")
    except RuntimeError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("missing compiled program should fail")

    corrupt_path = tmp_path / "compiled.json"
    corrupt_path.write_text("not a program", encoding="utf-8")

    class BrokenProgram:
        def load(self, path):
            raise ValueError("corrupt")

    try:
        optimize._load_optimized_program(BrokenProgram(), corrupt_path)
    except RuntimeError as exc:
        assert "could not be loaded" in str(exc)
    else:
        raise AssertionError("corrupt compiled program should fail")


def test_inference_only_skips_optimization_dataset_resolution(tmp_path):
    optimize = _load_optimize_module()
    output_program = tmp_path / "compiled.json"
    output_program.write_text("placeholder", encoding="utf-8")
    args = types.SimpleNamespace(
        inference_only=True,
        output_program=output_program,
        perception_model="qwen",
        signature_in_system_prompt=False,
        caption_placement="conversation_state",
        perception_config_yaml=None,
        captioner_config_yaml=None,
        prompt_yaml=None,
        optimize_targets=("planner.workflow_prompt",),
        allowed_tools="ask_perception",
        audio_caption_dir=None,
        ignore_audio_caption_dir=False,
        max_turns=3,
    )
    optimize.parse_optimize_args = lambda: args
    optimize.load_perception_config_yaml = lambda path: None
    optimize.load_captioner_config_yaml = lambda path: None
    optimize.configure_gemini_api_backend = lambda parsed_args: None
    optimize.load_prompt_config = lambda path: None
    optimize.resolve_preloaded_audio_caption_dir = lambda *args, **kwargs: (None, "not provided")
    optimize.validate_optimize_targets = lambda targets: None
    optimize.apply_prompt_config_to_signatures = lambda **kwargs: None
    optimize.resolve_allowed_tools = lambda tools: ("ask_perception",)
    optimize.AVQARuntimeContext = lambda **kwargs: types.SimpleNamespace(**kwargs)
    optimize.AVQADSPyReActProgram = lambda context: object()
    optimize.PromptTargetProgram = lambda base, targets: object()
    optimize.resolve_optimization_datasets = lambda parsed_args: (_ for _ in ()).throw(
        AssertionError("optimization datasets must not be resolved")
    )
    inference_calls = []
    optimize._run_inference_only = lambda parsed_args, program, context, started: inference_calls.append(
        (parsed_args, program, context)
    )

    optimize.run_optimization()

    assert len(inference_calls) == 1


def test_inference_only_updates_existing_metadata(tmp_path):
    optimize = _load_optimize_module()
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"keep": "value"}), encoding="utf-8")
    args = types.SimpleNamespace(algorithm="gepa", output_program=tmp_path / "compiled.json")
    final_eval = {"final_eval_rows": 3, "final_eval_cached_samples": 2}

    optimize._update_inference_only_metadata(
        metadata_path,
        args=args,
        final_eval=final_eval,
        elapsed_seconds=1.25,
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["keep"] == "value"
    assert payload["algorithm"] == "gepa"
    assert payload["output_program"] == str(args.output_program)
    assert payload["final_eval"] == final_eval
    assert payload["inference_only_elapsed_seconds"] == 1.25
