"""DSPy signatures for explicit AVQA ReAct planning."""

from __future__ import annotations

from typing import Any

import dspy

from .prompt_config import prompt_config, prompt_value


FIXED_PLAN_NEXT_ACTION_INSTRUCTIONS = "Choose the next AVQA ReAct action with a compact JSON interface."


def _field_desc(signature_name: str, field_name: str) -> str:
    return prompt_value("signatures", signature_name, "fields", field_name).strip()


class PlanNextAction(dspy.Signature):
    __doc__ = FIXED_PLAN_NEXT_ACTION_INSTRUCTIONS

    task = dspy.InputField(desc=_field_desc("PlanNextAction", "task"))
    conversation_state = dspy.InputField(desc=_field_desc("PlanNextAction", "conversation_state"))
    turn_index = dspy.InputField()
    max_turns = dspy.InputField()
    action_json = dspy.OutputField(desc=_field_desc("PlanNextAction", "action_json"))


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


def _set_signature_instructions(signature_cls: type[dspy.Signature], instructions: str) -> None:
    signature_cls.__doc__ = instructions
    try:
        setattr(signature_cls, "instructions", instructions)
    except Exception:
        pass


def apply_prompt_config_to_signatures(*, apply_instructions: bool = False) -> None:
    """Refresh PlanNextAction field descriptions while keeping its instruction fixed."""
    signature_name = "PlanNextAction"
    signature_cls = PlanNextAction
    section = (prompt_config().get("signatures") or {}).get(signature_name) or {}
    if not isinstance(section, dict):
        raise ValueError(f"Prompt config signatures.{signature_name} must be a mapping")

    _set_signature_instructions(signature_cls, FIXED_PLAN_NEXT_ACTION_INSTRUCTIONS)

    field_descs = section.get("fields") or {}
    if not isinstance(field_descs, dict):
        raise ValueError(f"Prompt config signatures.{signature_name}.fields must be a mapping")
    fields = _iter_signature_fields(signature_cls)
    for field_name, desc in field_descs.items():
        if isinstance(desc, str) and field_name in fields:
            _set_field_desc(fields[field_name], desc.strip())


apply_prompt_config_to_signatures(apply_instructions=False)
