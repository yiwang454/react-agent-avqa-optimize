"""DSPy signatures for explicit AVQA ReAct planning."""

from __future__ import annotations

from typing import Any

import dspy

from .prompt_config import prompt_config, prompt_value


def _signature_instruction(name: str) -> str:
    return prompt_value("signatures", name, "instructions").strip()


def _field_desc(signature_name: str, field_name: str) -> str:
    return prompt_value("signatures", signature_name, "fields", field_name).strip()


class PlanNextAction(dspy.Signature):
    __doc__ = _signature_instruction("PlanNextAction")

    task = dspy.InputField(desc=_field_desc("PlanNextAction", "task"))
    conversation_state = dspy.InputField(desc=_field_desc("PlanNextAction", "conversation_state"))
    turn_index = dspy.InputField()
    max_turns = dspy.InputField()
    action_json = dspy.OutputField(desc=_field_desc("PlanNextAction", "action_json"))


class DraftPerceptualQuestion(dspy.Signature):
    __doc__ = _signature_instruction("DraftPerceptualQuestion")

    question = dspy.InputField()
    options_json = dspy.InputField()
    prior_evidence = dspy.InputField()
    prior_failures = dspy.InputField()
    remaining_uncertainty = dspy.InputField()
    next_question = dspy.OutputField()
    tool_name = dspy.OutputField(desc=_field_desc("DraftPerceptualQuestion", "tool_name"))
    rationale = dspy.OutputField()


class UpdateBeliefFromObservation(dspy.Signature):
    __doc__ = _signature_instruction("UpdateBeliefFromObservation")

    question = dspy.InputField()
    options_json = dspy.InputField()
    prior_evidence = dspy.InputField()
    latest_question = dspy.InputField()
    latest_observation = dspy.InputField()
    updated_evidence = dspy.OutputField()
    supported_options = dspy.OutputField(desc=_field_desc("UpdateBeliefFromObservation", "supported_options"))
    weakened_options = dspy.OutputField(desc=_field_desc("UpdateBeliefFromObservation", "weakened_options"))
    remaining_uncertainty = dspy.OutputField()
    updated_hypothesis = dspy.OutputField()


class DecideFinalAction(dspy.Signature):
    __doc__ = _signature_instruction("DecideFinalAction")

    question = dspy.InputField()
    options_json = dspy.InputField()
    accumulated_evidence = dspy.InputField()
    remaining_uncertainty = dspy.InputField()
    turn_index = dspy.InputField()
    max_turns = dspy.InputField()
    action = dspy.OutputField(desc=_field_desc("DecideFinalAction", "action"))
    answer = dspy.OutputField(desc=_field_desc("DecideFinalAction", "answer"))
    reasoning_summary = dspy.OutputField()


class FinalAnswerFallback(dspy.Signature):
    __doc__ = _signature_instruction("FinalAnswerFallback")

    question = dspy.InputField()
    options_json = dspy.InputField()
    accumulated_evidence = dspy.InputField()
    answer = dspy.OutputField()
    reasoning_summary = dspy.OutputField()


def _set_field_desc(field: Any, desc: str) -> None:
    if hasattr(field, "json_schema_extra"):
        extra = dict(getattr(field, "json_schema_extra") or {})
        extra["desc"] = desc
        field.json_schema_extra = extra
    if hasattr(field, "description"):
        try:
            field.description = desc
        except Exception:
            pass


def _iter_signature_fields(signature_cls: type[dspy.Signature]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for attr in ("input_fields", "output_fields", "fields", "model_fields"):
        value = getattr(signature_cls, attr, None)
        if isinstance(value, dict):
            fields.update(value)
    return fields


def apply_prompt_config_to_signatures() -> None:
    """Refresh DSPy signature prompt text after a custom prompt YAML is loaded."""
    signatures = {
        "PlanNextAction": PlanNextAction,
        "DraftPerceptualQuestion": DraftPerceptualQuestion,
        "UpdateBeliefFromObservation": UpdateBeliefFromObservation,
        "DecideFinalAction": DecideFinalAction,
        "FinalAnswerFallback": FinalAnswerFallback,
    }
    config = prompt_config().get("signatures") or {}
    if not isinstance(config, dict):
        raise ValueError("Prompt config signatures section must be a mapping")

    for signature_name, signature_cls in signatures.items():
        section = config.get(signature_name) or {}
        if not isinstance(section, dict):
            raise ValueError(f"Prompt config signatures.{signature_name} must be a mapping")

        instructions = section.get("instructions")
        if isinstance(instructions, str):
            instructions = instructions.strip()
            signature_cls.__doc__ = instructions
            try:
                setattr(signature_cls, "instructions", instructions)
            except Exception:
                pass

        field_descs = section.get("fields") or {}
        if not isinstance(field_descs, dict):
            raise ValueError(f"Prompt config signatures.{signature_name}.fields must be a mapping")
        fields = _iter_signature_fields(signature_cls)
        for field_name, desc in field_descs.items():
            if isinstance(desc, str) and field_name in fields:
                _set_field_desc(fields[field_name], desc.strip())


apply_prompt_config_to_signatures()
