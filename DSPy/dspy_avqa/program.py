"""DSPy AVQA ReAct program."""

from __future__ import annotations

import json
import re
from typing import Any

import dspy

from .deepseek_dspy_lm import clear_planner_call_trace, consume_planner_call_trace
from .context import AVQARuntimeContext, configure_deepseek_lm, resolve_allowed_tools
from .prompt_config import prompt_value, render_prompt
from .signatures import PlanNextAction
from .tools import ask_perception, consume_last_perception_metadata, selected_perception_model, temporal_ground_video


OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_OBSERVATION_CHARS = 2400


def _strip_fences(text: str) -> str:
    """Remove optional markdown code fences from planner output."""
    value = text.strip()
    if not value.startswith("```"):
        return value

    lines = value.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _first_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object embedded in text, if present."""
    value = _strip_fences(text)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(value):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _coerce_action_payload(raw: Any) -> dict[str, Any]:
    """Turn DSPy or raw planner output into the compact action payload."""
    if isinstance(raw, dict):
        payload = raw
    else:
        text = str(raw or "").strip()
        payload = _first_json_object(text) or {}
        if not payload:
            answer = normalize_option_letter(text)
            if answer:
                return {"action": "final", "answer": text}
            return {"action": "tool", "tool_name": "ask_perception", "arguments": {"perceptual_question": text}}

    nested = payload.get("action_json")
    if isinstance(nested, dict):
        return _coerce_action_payload(nested)
    if isinstance(nested, str) and nested.strip():
        nested_payload = _first_json_object(nested)
        if nested_payload is not None:
            return _coerce_action_payload(nested_payload)
        answer = normalize_option_letter(nested)
        if answer:
            return {"action": "final", "answer": nested}

    return payload


def normalize_option_letter(answer: str) -> str:
    """Normalize model output to option letter form."""
    value = str(answer or "").strip()
    if not value:
        return ""

    tagged = re.search(r"<answer>\s*([A-Za-z])\s*</answer>", value, flags=re.IGNORECASE)
    if tagged:
        return tagged.group(1).upper()

    payload = _first_json_object(value)
    if payload is not None and "answer" in payload:
        nested = normalize_option_letter(str(payload.get("answer") or ""))
        if nested:
            return nested

    upper = value.upper()
    direct = re.match(r"^\s*([A-F])(?:\s*[\.)\]:-]|\s|$)", upper)
    if direct:
        return direct.group(1)

    for pattern in (
        r"\b(?:FINAL|ANSWER|OPTION|CHOICE|CHOOSE|SELECT|LABEL)\s*(?:IS|:|=)?\s*([A-F])\b",
        r"\b([A-F])\s+IS\s+(?:THE\s+)?(?:CORRECT|BEST|ANSWER)",
    ):
        match = re.search(pattern, upper)
        if match:
            return match.group(1)

    isolated = re.findall(r"\b([A-F])\b", upper)
    return isolated[0] if len(isolated) == 1 else ""


def _format_options(options_json: str) -> str:
    """Format options JSON as stable labeled lines."""
    try:
        parsed = json.loads(options_json)
    except json.JSONDecodeError:
        return options_json
    if not isinstance(parsed, list):
        return options_json

    lines: list[str] = []
    for idx, option in enumerate(parsed):
        label = OPTION_LABELS[idx] if idx < len(OPTION_LABELS) else f"O{idx + 1}"
        option_text = str(option).strip()
        if re.match(r"^[A-Z][\.)]\s", option_text):
            lines.append(option_text)
        else:
            lines.append(f"{label}. {option_text}")
    return "\n".join(lines)


def _compact_text(text: Any, limit: int = MAX_OBSERVATION_CHARS) -> str:
    """Keep planner context bounded without changing the tool observation record."""
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "\n" + prompt_value("planner", "truncated_marker").strip()


def _build_task_text(
    *,
    system_prompt: str,
    question: str,
    options_json: str,
    video_path: str,
    video_id: str | None,
    video_description: str | None,
) -> str:
    """Build the compact planner task used inside DSPy."""
    return render_prompt(
        "planner",
        "task_prompt_template",
        system_prompt=system_prompt,
        planner_action_schema=prompt_value("planner", "action_schema").strip(),
        video_id=video_id or "unknown",
        video_path=video_path,
        video_description=video_description or "unknown",
        question=question,
        formatted_options=_format_options(options_json),
    )


def _conversation_state(turn_trace: list[dict[str, Any]]) -> str:
    """Serialize prior planner/tool turns like the LangGraph implementation."""
    if not turn_trace:
        return prompt_value("planner", "conversation_empty").strip()

    lines: list[str] = []
    for turn in turn_trace:
        raw = str(turn.get("planner_raw") or "").strip()
        if raw:
            lines.append(
                render_prompt(
                    "planner",
                    "assistant_turn_template",
                    assistant_text=_compact_text(raw, 1200),
                )
            )

        observation = turn.get("tool_observation")
        if observation:
            tool_name = turn.get("tool_name") or "unknown"
            question = (turn.get("tool_args") or {}).get("perceptual_question") or ""
            lines.append(
                render_prompt(
                    "planner",
                    "tool_turn_template",
                    tool_name=tool_name,
                    question=_compact_text(question, 400),
                    observation=_compact_text(observation),
                )
            )

    return "\n\n".join(lines) if lines else prompt_value("planner", "conversation_empty").strip()


def _latest_response_text(planner_calls: list[dict[str, Any]]) -> str:
    """Return the latest raw LM text captured by the DSPy LM wrapper."""
    for call in reversed(planner_calls):
        response_text = call.get("response_text")
        if isinstance(response_text, str) and response_text.strip():
            return response_text.strip()
    return ""


def _exception_info(exc: Exception) -> dict[str, Any]:
    """Capture useful planner parse details without importing DSPy internals."""
    info: dict[str, Any] = {
        "error_type": exc.__class__.__name__,
        "message": str(exc),
    }
    for attr, key in (
        ("adapter_name", "adapter_name"),
        ("lm_response", "planner_lm_response"),
        ("parsed_result", "parsed_result"),
    ):
        value = getattr(exc, attr, None)
        if value is not None:
            info[key] = value
    signature = getattr(exc, "signature", None)
    output_fields = getattr(signature, "output_fields", None)
    if isinstance(output_fields, dict):
        info["expected_output_fields"] = list(output_fields.keys())
    return info


def _canonical_tool_name(raw_tool_name: str) -> str:
    value = raw_tool_name.strip().lower()
    if value == "temporal_ground_video":
        return "temporal_ground_video"
    if value in {"ask_qwen_perception", "ask_gemini_perception", "ask_perception"}:
        return "ask_perception"
    return "ask_perception"


def _normalize_tool_name(raw_tool_name: str, allowed_tools: tuple[str, ...]) -> str:
    """Map planner tool names onto enabled DSPy tool functions."""
    value = _canonical_tool_name(raw_tool_name)
    if value in allowed_tools:
        return value
    if "ask_perception" in allowed_tools:
        return "ask_perception"
    return allowed_tools[0]


def _perceptual_question(payload: dict[str, Any]) -> str:
    """Extract the perceptual question from the compact action payload."""
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    for candidate in (
        arguments.get("perceptual_question"),
        arguments.get("target_question"),
        payload.get("perceptual_question"),
        payload.get("question"),
    ):
        value = str(candidate or "").strip()
        if value:
            return value
    return prompt_value("perception", "default_perceptual_question").strip()


class AVQADSPyReActProgram(dspy.Module):
    """DSPy AVQA ReAct program with a compact planner-tool loop."""

    def __init__(
        self,
        context: AVQARuntimeContext | None = None,
        allowed_tools: str | list[str] | tuple[str, ...] | None = None,
    ):
        super().__init__()
        self.context = context or AVQARuntimeContext()
        if allowed_tools is not None:
            self.context.allowed_tools = resolve_allowed_tools(allowed_tools)
        configure_deepseek_lm(self.context)
        self.action_planner = dspy.Predict(PlanNextAction)

    def _plan_next_action(
        self,
        *,
        task: str,
        conversation_state: str,
        turn_index: str,
        max_turns: str,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
        """Run the DSPy planner and recover raw JSON if the adapter is too strict."""
        try:
            prediction = self.action_planner(
                task=task,
                conversation_state=conversation_state,
                turn_index=turn_index,
                max_turns=max_turns,
            )
        except Exception as exc:
            planner_calls = consume_planner_call_trace()
            raw = str(getattr(exc, "lm_response", "") or "").strip() or _latest_response_text(planner_calls)
            if raw:
                return raw, planner_calls, _exception_info(exc)
            raise

        planner_calls = consume_planner_call_trace()
        raw_action = getattr(prediction, "action_json", "")
        if isinstance(raw_action, dict):
            raw = json.dumps(raw_action, ensure_ascii=False)
        else:
            raw = str(raw_action or "").strip()
        if not raw:
            raw = _latest_response_text(planner_calls)
        return raw, planner_calls, None

    def _call_tool(
        self,
        *,
        tool_name: str,
        video_path: str,
        audio_path: str | None,
        perceptual_question: str,
    ) -> str:
        """Dispatch one perceptual tool call."""
        if tool_name == "temporal_ground_video":
            return temporal_ground_video(
                video_path=video_path,
                audio_path=audio_path,
                perceptual_question=perceptual_question,
            )
        return ask_perception(
            video_path=video_path,
            audio_path=audio_path,
            perceptual_question=perceptual_question,
        )

    def forward(
        self,
        question: str,
        options_json: str,
        video_path: str,
        audio_path: str | None = None,
        video_id: str | None = None,
        video_description: str | None = None,
        max_turns: int | None = None,
    ) -> dspy.Prediction:
        max_iters = max_turns or self.context.max_turns
        turn_trace: list[dict[str, Any]] = []
        clear_planner_call_trace()

        task = _build_task_text(
            system_prompt=self.context.system_prompt,
            question=question,
            options_json=options_json,
            video_path=video_path,
            video_id=video_id,
            video_description=video_description,
        )

        for turn_idx in range(1, max_iters + 1):
            raw_action, planner_calls, planner_error = self._plan_next_action(
                task=task,
                conversation_state=_conversation_state(turn_trace),
                turn_index=str(turn_idx),
                max_turns=str(max_iters),
            )
            payload = _coerce_action_payload(raw_action)
            action_name = str(payload.get("action") or "tool").strip().lower()

            if action_name == "final":
                raw_answer = str(payload.get("answer") or raw_action).strip()
                final_answer = normalize_option_letter(raw_answer)
                turn_trace.append(
                    {
                        "turn_id": turn_idx,
                        "planner_action": "final",
                        "tool_name": None,
                        "raw_tool_name": None,
                        "tool_args": None,
                        "tool_observation": None,
                        "planner_raw": raw_action,
                        "planner_calls": {"action_decision": planner_calls},
                        "planner_parse_error": planner_error,
                        "final_answer": final_answer,
                        "video_id": video_id,
                    }
                )
                evidence_summary = _conversation_state(turn_trace)
                return dspy.Prediction(
                    answer=final_answer,
                    reasoning_summary=raw_answer,
                    evidence_summary=evidence_summary,
                    turn_trace=turn_trace,
                    accumulated_evidence=evidence_summary,
                )

            raw_tool_name = str(payload.get("tool_name") or "ask_perception")
            tool_name = _normalize_tool_name(raw_tool_name, self.context.allowed_tools)
            perceptual_question = _perceptual_question(payload)
            perception_backend = selected_perception_model()
            tool_observation = self._call_tool(
                tool_name=tool_name,
                video_path=video_path,
                audio_path=audio_path,
                perceptual_question=perceptual_question,
            )
            perception_metadata = consume_last_perception_metadata()

            turn_trace.append(
                {
                    "turn_id": turn_idx,
                    "planner_action": "tool",
                    "tool_name": tool_name,
                    "raw_tool_name": raw_tool_name,
                    "perception_backend": perception_backend,
                    "tool_args": {
                        "video_path": video_path,
                        "audio_path": audio_path,
                        "perceptual_question": perceptual_question,
                    },
                    "tool_observation": tool_observation,
                    "perception_thinking": perception_metadata.get("thinking_text", ""),
                    "perception_token_usage": perception_metadata.get("token_usage"),
                    "planner_raw": raw_action,
                    "planner_calls": {"action_decision": planner_calls},
                    "planner_parse_error": planner_error,
                    "final_answer": None,
                    "video_id": video_id,
                }
            )

        fallback_state = (
            f"{_conversation_state(turn_trace)}\n\n"
            + prompt_value("planner", "final_fallback_instruction").strip()
        )
        raw_action, planner_calls, planner_error = self._plan_next_action(
            task=task,
            conversation_state=fallback_state,
            turn_index="final",
            max_turns=str(max_iters),
        )
        payload = _coerce_action_payload(raw_action)
        raw_answer = str(payload.get("answer") or raw_action).strip()
        final_answer = normalize_option_letter(raw_answer)
        turn_trace.append(
            {
                "turn_id": len(turn_trace) + 1,
                "planner_action": "final",
                "tool_name": None,
                "raw_tool_name": None,
                "tool_args": None,
                "tool_observation": None,
                "planner_raw": raw_action,
                "planner_calls": {"final_fallback": planner_calls},
                "planner_parse_error": planner_error,
                "final_answer": final_answer,
                "video_id": video_id,
            }
        )

        evidence_summary = _conversation_state(turn_trace)
        return dspy.Prediction(
            answer=final_answer,
            reasoning_summary=raw_answer,
            evidence_summary=evidence_summary,
            turn_trace=turn_trace,
            accumulated_evidence=evidence_summary,
        )
