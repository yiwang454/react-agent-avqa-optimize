"""DSPy signatures for explicit AVQA ReAct planning."""

from __future__ import annotations

import dspy


class PlanNextAction(dspy.Signature):
    """Choose the next AVQA ReAct action with a compact JSON interface."""

    task = dspy.InputField(desc="Question, options, and video context.")
    conversation_state = dspy.InputField(desc="Prior planner actions and tool observations.")
    turn_index = dspy.InputField()
    max_turns = dspy.InputField()
    action_json = dspy.OutputField(
        desc=(
            "Exactly one JSON object. Use "
            '{"action":"tool","tool_name":"ask_perception",'
            '"arguments":{"perceptual_question":"..."}} or '
            '{"action":"tool","tool_name":"temporal_ground_video",'
            '"arguments":{"perceptual_question":"..."}} or '
            '{"action":"final","answer":"<answer>A</answer>"}'
        )
    )


class DraftPerceptualQuestion(dspy.Signature):
    """Draft one discriminative perceptual question for the next tool call."""

    question = dspy.InputField()
    options_json = dspy.InputField()
    prior_evidence = dspy.InputField()
    prior_failures = dspy.InputField()
    remaining_uncertainty = dspy.InputField()
    next_question = dspy.OutputField()
    tool_name = dspy.OutputField(
        desc="ask_perception or temporal_ground_video; backend is selected by --perception-model/PERCEPTION_MODEL"
    )
    rationale = dspy.OutputField()


class UpdateBeliefFromObservation(dspy.Signature):
    """Update evidence and option beliefs after one tool observation."""

    question = dspy.InputField()
    options_json = dspy.InputField()
    prior_evidence = dspy.InputField()
    latest_question = dspy.InputField()
    latest_observation = dspy.InputField()
    updated_evidence = dspy.OutputField()
    supported_options = dspy.OutputField(desc="JSON list of option letters")
    weakened_options = dspy.OutputField(desc="JSON list of option letters")
    remaining_uncertainty = dspy.OutputField()
    updated_hypothesis = dspy.OutputField()


class DecideFinalAction(dspy.Signature):
    """Decide whether to continue tool use or produce final answer now."""

    question = dspy.InputField()
    options_json = dspy.InputField()
    accumulated_evidence = dspy.InputField()
    remaining_uncertainty = dspy.InputField()
    turn_index = dspy.InputField()
    max_turns = dspy.InputField()
    action = dspy.OutputField(desc="continue or final")
    answer = dspy.OutputField(desc="Option letter if action=final, else empty")
    reasoning_summary = dspy.OutputField()


class FinalAnswerFallback(dspy.Signature):
    """Produce final option when max turns are reached."""

    question = dspy.InputField()
    options_json = dspy.InputField()
    accumulated_evidence = dspy.InputField()
    answer = dspy.OutputField()
    reasoning_summary = dspy.OutputField()

