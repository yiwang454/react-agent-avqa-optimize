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
    tools.build_caption_prompt = lambda instruction=None: str(instruction or "").strip()
    tools.captioner_system_prompt = lambda: ""
    tools._metadata = {}
    tools.record_perception_metadata = lambda **kwargs: tools._metadata.update(kwargs)
    def consume_metadata():
        value = dict(tools._metadata)
        tools._metadata.clear()
        return value
    tools.consume_last_perception_metadata = consume_metadata
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

def test_caption_observation_is_not_truncated_in_conversation_state(tmp_path):
    prompt_yaml = tmp_path / "prompt.yaml"
    prompt_yaml.write_text("""
planner:
  conversation_empty: |
    <empty>
  assistant_turn_template: |
    [assistant]
    {assistant_text}
  tool_turn_template: |
    [tool:{tool_name}]
    question: {question}
    observation: {observation}
  truncated_marker: |
    [truncated]
""")
    prompt_config.load_prompt_config(prompt_yaml)

    long_caption = "caption-detail-" * 300
    state = program._conversation_state([
        {
            "tool_name": "ask_caption",
            "tool_args": {},
            "tool_observation": long_caption,
        }
    ])

    assert long_caption in state
    assert "[truncated]" not in state


def test_non_caption_observation_is_still_truncated_in_conversation_state(tmp_path):
    prompt_yaml = tmp_path / "prompt.yaml"
    prompt_yaml.write_text("""
planner:
  conversation_empty: |
    <empty>
  assistant_turn_template: |
    [assistant]
    {assistant_text}
  tool_turn_template: |
    [tool:{tool_name}]
    question: {question}
    observation: {observation}
  truncated_marker: |
    [truncated]
""")
    prompt_config.load_prompt_config(prompt_yaml)

    long_observation = "perception-detail-" * 300
    state = program._conversation_state([
        {
            "tool_name": "ask_perception",
            "tool_args": {},
            "tool_observation": long_observation,
        }
    ])

    assert long_observation not in state
    assert "[truncated]" in state


def _write_planner_prompt_yaml(path: Path, *, include_video_description: bool = True) -> None:
    coarse_description_block = (
        "    Coarse video/audio description: {video_description}\n\n"
        if include_video_description
        else ""
    )
    path.write_text(f"""
planner:
  workflow_prompt: |
    workflow instruction
  action_schema: |
    action schema
  task_prompt_template: |
    {{workflow_prompt}}

    {{planner_action_schema}}

    Video ID: {{video_id}}
    Video path: {{video_path}}
{coarse_description_block}    Question: {{question}}
    Options:
    {{formatted_options}}
  conversation_empty: |
    <empty>
  assistant_turn_template: |
    [assistant]
    {{assistant_text}}
  tool_turn_template: |
    [tool:{{tool_name}}]
    question: {{question}}
    observation: {{observation}}
  truncated_marker: |
    [truncated]
  final_fallback_instruction: |
    final only
captioner:
  default_caption_instruction: |
    default caption instruction
perception:
  default_perceptual_question: |
    {{perceptual_question}}
""")


class _FakeRuntimeContext:
    max_turns = 3
    allowed_tools = ("ask_caption", "ask_perception")
    caption_placement = "conversation_state"
    system_prompt = "workflow instruction"
    caption_cache_dir = None


def _run_caption_then_perception(tmp_path, caption_placement: str):
    prompt_yaml = tmp_path / "prompt.yaml"
    _write_planner_prompt_yaml(prompt_yaml, include_video_description=caption_placement == "task")
    prompt_config.load_prompt_config(prompt_yaml)

    context = _FakeRuntimeContext()
    context.caption_placement = caption_placement
    avqa_program = program.AVQADSPyReActProgram(context=context)
    planner_inputs = []
    long_caption = ("full caption detail " * 80).strip()

    def fake_plan_next_action(*, task, conversation_state, turn_index, max_turns):
        planner_inputs.append({"task": task, "conversation_state": conversation_state, "turn_index": turn_index})
        if turn_index == "1":
            return '{"action":"tool","tool_name":"ask_caption","arguments":{}}', [], None
        if turn_index == "2":
            return '{"action":"tool","tool_name":"ask_perception","arguments":{"perceptual_question":"what key detail distinguishes the options?"}}', [], None
        return '{"action":"final","answer":"A"}', [], None

    def fake_call_tool(*, tool_name, video_path, audio_path, tool_query, question_id=None):
        if tool_name == "ask_caption":
            return long_caption
        return "perception observation"

    avqa_program._plan_next_action = fake_plan_next_action
    avqa_program._call_tool = fake_call_tool
    result = avqa_program.forward(
        question="Which option is correct?",
        options_json='["first", "second"]',
        video_path="/tmp/video.mp4",
        audio_path=None,
        video_id="sample",
        video_description="seed description" if caption_placement == "task" else None,
        max_turns=3,
    )
    return result, planner_inputs, long_caption


def test_cached_caption_preserves_v8_tool_flow_without_live_captioner(tmp_path, monkeypatch):
    prompt_yaml = tmp_path / "prompt.yaml"
    _write_planner_prompt_yaml(prompt_yaml, include_video_description=True)
    prompt_config.load_prompt_config(prompt_yaml)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "sample-1.json").write_text(
        """{
  "question_id": "sample-1",
  "response": "fixed cached caption",
  "caption_prompt": "default caption instruction",
  "source_results_file": "repeat1/output_test.jsonl",
  "source_rank": 1,
  "source_backend": "gemini",
  "source_model": "gemini-2.5-flash",
  "source_token_usage": {"totalTokenCount": 10}
}
""",
        encoding="utf-8",
    )

    context = _FakeRuntimeContext()
    context.caption_placement = "task"
    context.caption_cache_dir = cache_dir
    avqa_program = program.AVQADSPyReActProgram(context=context)
    planner_inputs = []

    def fake_plan_next_action(*, task, conversation_state, turn_index, max_turns):
        planner_inputs.append((task, conversation_state))
        if turn_index == "1":
            return '{"action":"tool","tool_name":"ask_caption","arguments":{}}', [], None
        if turn_index == "2":
            return '{"action":"tool","tool_name":"ask_perception","arguments":{"perceptual_question":"detail?"}}', [], None
        return '{"action":"final","answer":"A"}', [], None

    monkeypatch.setattr(
        program,
        "ask_caption",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("live captioner must not be called")
        ),
    )
    monkeypatch.setattr(program, "ask_perception", lambda *args, **kwargs: "perception")
    avqa_program._plan_next_action = fake_plan_next_action

    result = avqa_program.forward(
        question_id="sample-1",
        question="question",
        options_json='["first", "second"]',
        video_path="/tmp/video.mp4",
        audio_path="/tmp/audio.wav",
        video_id="video",
        video_description=None,
        max_turns=3,
    )

    assert [turn["tool_name"] for turn in result["turn_trace"][:2]] == [
        "ask_caption",
        "ask_perception",
    ]
    caption_turn = result["turn_trace"][0]
    assert caption_turn["tool_observation"] == "fixed cached caption"
    assert caption_turn["perception_backend"] == "caption_cache"
    assert caption_turn["perception_model"] == "gemini-2.5-flash"
    assert caption_turn["perception_token_usage"] is None
    assert caption_turn["caption_cache_hit"] is True
    assert caption_turn["caption_cache_source_rank"] == 1
    assert "fixed cached caption" not in planner_inputs[0][0]
    assert "fixed cached caption" in planner_inputs[1][0]
    assert program.CAPTION_MOVED_TO_TASK_MARKER in planner_inputs[1][1]


def test_default_caption_placement_keeps_caption_in_conversation_state(tmp_path):
    _, planner_inputs, long_caption = _run_caption_then_perception(tmp_path, "conversation_state")

    assert long_caption not in planner_inputs[0]["task"]
    assert long_caption not in planner_inputs[1]["task"]
    assert long_caption in planner_inputs[1]["conversation_state"]
    assert program.CAPTION_MOVED_TO_TASK_MARKER not in planner_inputs[1]["conversation_state"]


def test_task_caption_placement_moves_caption_into_later_task(tmp_path):
    _, planner_inputs, long_caption = _run_caption_then_perception(tmp_path, "task")

    assert long_caption not in planner_inputs[0]["task"]
    assert "Coarse video/audio description: seed description" in planner_inputs[0]["task"]
    assert long_caption in planner_inputs[1]["task"]
    assert long_caption in planner_inputs[2]["task"]
    assert long_caption not in planner_inputs[1]["conversation_state"]
    assert program.CAPTION_MOVED_TO_TASK_MARKER in planner_inputs[1]["conversation_state"]
    assert "default caption instruction" not in planner_inputs[1]["conversation_state"]
    assert "perception observation" in planner_inputs[2]["conversation_state"]


def test_caption_instruction_still_falls_back_to_yaml_default(tmp_path):
    prompt_yaml = tmp_path / "prompt.yaml"
    _write_planner_prompt_yaml(prompt_yaml, include_video_description=False)
    prompt_config.load_prompt_config(prompt_yaml)

    assert program._caption_instruction({"arguments": {}}) == "default caption instruction"



def test_conversation_state_caption_placement_allows_unrendered_video_description(tmp_path):
    prompt_yaml = tmp_path / "prompt.yaml"
    _write_planner_prompt_yaml(prompt_yaml, include_video_description=False)
    prompt_config.load_prompt_config(prompt_yaml)

    program._validate_caption_placement_inputs(
        caption_placement="conversation_state",
        video_description="precomputed caption that template does not render",
    )


def test_conversation_state_caption_placement_allows_empty_video_description_template(tmp_path):
    prompt_yaml = tmp_path / "prompt.yaml"
    _write_planner_prompt_yaml(prompt_yaml, include_video_description=True)
    prompt_config.load_prompt_config(prompt_yaml)

    program._validate_caption_placement_inputs(
        caption_placement="conversation_state",
        video_description=None,
    )


def test_conversation_state_caption_placement_rejects_rendered_video_description(tmp_path):
    prompt_yaml = tmp_path / "prompt.yaml"
    _write_planner_prompt_yaml(prompt_yaml, include_video_description=True)
    prompt_config.load_prompt_config(prompt_yaml)

    try:
        program._validate_caption_placement_inputs(
            caption_placement="conversation_state",
            video_description="precomputed caption",
        )
    except ValueError as exc:
        message = str(exc)
        assert "planner.task_prompt_template" in message
        assert "{video_description}" in message
    else:
        raise AssertionError("Expected ValueError when video_description would render into task")

def test_task_caption_placement_rejects_template_without_video_description(tmp_path):
    prompt_yaml = tmp_path / "prompt.yaml"
    _write_planner_prompt_yaml(prompt_yaml, include_video_description=False)
    prompt_config.load_prompt_config(prompt_yaml)

    try:
        program._validate_caption_placement_inputs(
            caption_placement="task",
            video_description=None,
        )
    except ValueError as exc:
        message = str(exc)
        assert "caption_placement='task'" in message
        assert "{video_description}" in message
    else:
        raise AssertionError("Expected ValueError when task placement cannot render caption")
