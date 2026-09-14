"""DSPy AVQA ReAct program."""

from __future__ import annotations

import json
import re
from typing import Any

import dspy

from .caption_cache import load_cached_caption
from .context import AVQARuntimeContext, configure_deepseek_lm, resolve_allowed_tools
from .deepseek_dspy_lm import clear_planner_call_trace, consume_planner_call_trace
from .prompt_config import prompt_config, prompt_value, render_prompt
from .signatures import PlanNextAction
from .tools import (
    ask_caption,
    ask_perception,
    build_caption_prompt,
    captioner_system_prompt,
    consume_last_perception_metadata,
    record_perception_metadata,
    selected_captioner_model,
    selected_perception_model,
    temporal_ground_video,
)


OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_OBSERVATION_CHARS = 2400
CAPTION_MOVED_TO_TASK_MARKER = "<caption moved into task>"


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


def _is_caption_tool(tool_name: str | None) -> bool:
    return str(tool_name or "").strip().lower() in {"ask_caption", "ask_captioner", "caption_video", "captioner"}


def _format_observation_for_planner(
    tool_name: str,
    observation: Any,
    *,
    caption_placement: str = "conversation_state",
) -> str:
    """Format tool observations for planner context."""
    value = str(observation or "").strip()
    if _is_caption_tool(tool_name):
        if caption_placement == "task":
            return CAPTION_MOVED_TO_TASK_MARKER
        return value
    return _compact_text(value)


def _task_prompt_template_references_video_description() -> bool:
    template = prompt_value("planner", "task_prompt_template")
    return "{video_description}" in template


def _validate_caption_placement_inputs(*, caption_placement: str, video_description: str | None) -> None:
    template_references_video_description = _task_prompt_template_references_video_description()
    if caption_placement == "task":
        if not template_references_video_description:
            raise ValueError(
                "caption_placement='task' requires planner.task_prompt_template to reference "
                "{video_description}; otherwise the caption is removed from conversation_state "
                "but never rendered into task."
            )
        return
    if caption_placement != "conversation_state":
        return
    if str(video_description or "").strip() and template_references_video_description:
        raise ValueError(
            "caption_placement='conversation_state' requires video_description not to be rendered "
            "into planner.task_prompt_template. Pass an empty video_description, use a task template "
            "without {video_description}, or use --caption-placement task."
        )


def _build_task_text(
    *,
    workflow_prompt: str,
    question: str,
    options_json: str,
    video_path: str,
    video_id: str | None,
    video_description: str | None,
    turn_index: str = "unknown",
    max_turns: str = "unknown",
) -> str:
    """Build the compact planner task used inside DSPy."""
    return render_prompt(
        "planner",
        "task_prompt_template",
        system_prompt=workflow_prompt,
        workflow_prompt=workflow_prompt,
        planner_action_schema=prompt_value("planner", "action_schema").strip(),
        video_id=video_id or "unknown",
        video_path=video_path,
        video_description=video_description or "unknown",
        turn_index=turn_index,
        max_turns=max_turns,
        question=question,
        formatted_options=_format_options(options_json),
    )


def _conversation_state(
    turn_trace: list[dict[str, Any]],
    *,
    caption_placement: str = "conversation_state",
) -> str:
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
            tool_args = turn.get("tool_args") or {}
            if caption_placement == "task" and _is_caption_tool(tool_name):
                question = ""
            else:
                question = tool_args.get("perceptual_question") or tool_args.get("caption_instruction") or ""
            lines.append(
                render_prompt(
                    "planner",
                    "tool_turn_template",
                    tool_name=tool_name,
                    question=_compact_text(question, 400),
                    observation=_format_observation_for_planner(
                        tool_name,
                        observation,
                        caption_placement=caption_placement,
                    ),
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
    if value in {"ask_caption", "ask_captioner", "caption_video", "captioner"}:
        return "ask_caption"
    if value == "temporal_ground_video":
        return "temporal_ground_video"
    if value in {"ask_qwen_perception", "ask_gemini_perception", "ask_perception"}:
        return "ask_perception"
    return "ask_perception"


def _required_first_tool() -> str | None:
    """Return an optional prompt-configured first tool requirement."""
    tools_config = prompt_config().get("tools") or {}
    if not isinstance(tools_config, dict):
        raise ValueError("Prompt config tools section must be a mapping")
    raw_tool_name = str(tools_config.get("required_first") or "").strip()
    if not raw_tool_name:
        return None
    normalized = raw_tool_name.lower()
    known_names = {
        "ask_caption",
        "ask_captioner",
        "caption_video",
        "captioner",
        "ask_perception",
        "ask_qwen_perception",
        "ask_gemini_perception",
        "temporal_ground_video",
    }
    if normalized not in known_names:
        raise ValueError(
            f"Unsupported prompt config tools.required_first={raw_tool_name!r}"
        )
    return _canonical_tool_name(normalized)


def _normalize_tool_name(raw_tool_name: str, allowed_tools: tuple[str, ...]) -> str:
    """Map planner tool names onto enabled DSPy tool functions."""
    value = _canonical_tool_name(raw_tool_name)
    if value in allowed_tools:
        return value
    if "ask_perception" in allowed_tools:
        return "ask_perception"
    return allowed_tools[0]


def _render_default_perceptual_question(source_question: str | None = None) -> str:
    """Render the fallback perceptual question only when the YAML asks for it."""
    default_question = prompt_value("perception", "default_perceptual_question").strip()
    if "{perceptual_question}" not in default_question:
        return default_question
    return default_question.replace("{perceptual_question}", str(source_question or "").strip())


def _perceptual_question(payload: dict[str, Any], source_question: str | None = None) -> str:
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
    return _render_default_perceptual_question(source_question)


def _default_caption_instruction() -> str:
    try:
        return prompt_value("captioner", "default_caption_instruction").strip()
    except KeyError:
        return "Produce a detailed timestamped audio-visual caption of the video."


def _captioner_flag(name: str, *, default: bool = False) -> bool:
    """Read a boolean behavior flag from the active captioner prompt config."""
    captioner_config = prompt_config().get("captioner") or {}
    if not isinstance(captioner_config, dict):
        raise ValueError("Prompt config captioner section must be a mapping")
    value = captioner_config.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def caption_cache_missing_as_observation() -> bool:
    """Return whether a missing cached caption should become a tool observation."""
    return _captioner_flag("cache_missing_as_observation")


def _caption_instruction(payload: dict[str, Any]) -> str:
    """Extract the captioner instruction from the compact action payload."""
    if _captioner_flag("force_default_instruction"):
        return _default_caption_instruction()
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    for candidate in (
        arguments.get("caption_instruction"),
        arguments.get("instruction"),
        arguments.get("prompt"),
        payload.get("caption_instruction"),
    ):
        value = str(candidate or "").strip()
        if value:
            return value
    return _default_caption_instruction()


def _tool_query(payload: dict[str, Any], tool_name: str, source_question: str | None = None) -> str:
    if tool_name == "ask_caption":
        return _caption_instruction(payload)
    return _perceptual_question(payload, source_question=source_question)


def _tool_args(tool_name: str, video_path: str, audio_path: str | None, query: str) -> dict[str, Any]:
    args: dict[str, Any] = {"video_path": video_path, "audio_path": audio_path}
    if tool_name == "ask_caption":
        args["caption_instruction"] = query
    else:
        args["perceptual_question"] = query
    return args


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

    def _workflow_prompt(self) -> str:
        try:
            return prompt_value("planner", "workflow_prompt").strip()
        except KeyError:
            return self.context.system_prompt

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
        tool_query: str,
        question_id: str | None = None,
    ) -> str:
        """Dispatch one planner tool call."""
        if tool_name == "ask_caption":
            caption_cache_dir = getattr(self.context, "caption_cache_dir", None)
            if caption_cache_dir is not None:
                prompt = build_caption_prompt(tool_query)
                try:
                    cached = load_cached_caption(
                        caption_cache_dir,
                        str(question_id or ""),
                        expected_prompt=prompt,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    if not caption_cache_missing_as_observation():
                        raise
                    error_observation = (
                        "[ERROR] No usable cached caption is available for this question sample. "
                        "Continue the ReAct workflow with ask_perception or answer from other "
                        f"available evidence. Details: {exc}"
                    )
                    record_perception_metadata(
                        backend="caption_cache",
                        system_prompt=captioner_system_prompt(),
                        prompt=prompt,
                        model=None,
                        env_prefix="CAPTION_CACHE",
                        caption_cache_hit=False,
                        caption_cache_error=str(exc),
                    )
                    return error_observation
                record_perception_metadata(
                    backend="caption_cache",
                    system_prompt=captioner_system_prompt(),
                    prompt=prompt,
                    model=cached.source_model,
                    env_prefix="CAPTION_CACHE",
                    caption_cache_hit=True,
                    caption_cache_path=str(cached.path),
                    caption_cache_source_results=cached.source_results_file,
                    caption_cache_source_rank=cached.source_rank,
                    caption_cache_source_backend=cached.source_backend,
                    caption_cache_source_token_usage=cached.source_token_usage,
                )
                return cached.response
            return ask_caption(
                video_path=video_path,
                audio_path=audio_path,
                caption_instruction=tool_query,
            )
        if tool_name == "temporal_ground_video":
            return temporal_ground_video(
                video_path=video_path,
                audio_path=audio_path,
                perceptual_question=tool_query,
            )
        return ask_perception(
            video_path=video_path,
            audio_path=audio_path,
            perceptual_question=tool_query,
        )

    def forward(
        self,
        question: str,
        options_json: str,
        video_path: str,
        audio_path: str | None = None,
        video_id: str | None = None,
        question_id: str | None = None,
        video_description: str | None = None,
        max_turns: int | None = None,
    ) -> dspy.Prediction:
        max_iters = max_turns or self.context.max_turns
        _validate_caption_placement_inputs(
            caption_placement=self.context.caption_placement,
            video_description=video_description,
        )
        turn_trace: list[dict[str, Any]] = []
        caption_task_description: str | None = None
        workflow_prompt = self._workflow_prompt()
        required_first_tool = _required_first_tool()
        if required_first_tool not in (None, *self.context.allowed_tools):
            raise ValueError(
                f"Prompt requires first tool {required_first_tool!r}, but it is not enabled in "
                f"allowed_tools={self.context.allowed_tools!r}"
            )
        clear_planner_call_trace()

        def current_task_text(turn_index: str) -> str:
            active_video_description = (
                caption_task_description
                if self.context.caption_placement == "task" and caption_task_description is not None
                else video_description
            )
            return _build_task_text(
                workflow_prompt=workflow_prompt,
                question=question,
                options_json=options_json,
                video_path=video_path,
                video_id=video_id,
                video_description=active_video_description,
                turn_index=turn_index,
                max_turns=str(max_iters),
            )

        for turn_idx in range(1, max_iters + 1):
            task = current_task_text(str(turn_idx))
            raw_action, planner_calls, planner_error = self._plan_next_action(
                task=task,
                conversation_state=_conversation_state(
                    turn_trace,
                    caption_placement=self.context.caption_placement,
                ),
                turn_index=str(turn_idx),
                max_turns=str(max_iters),
            )
            payload = _coerce_action_payload(raw_action)
            action_name = str(payload.get("action") or "tool").strip().lower()
            requested_action = action_name
            requested_tool_name = str(payload.get("tool_name") or "").strip() or None
            first_tool_enforced = False
            if turn_idx == 1 and required_first_tool is not None:
                canonical_requested_tool = (
                    _canonical_tool_name(requested_tool_name)
                    if requested_action == "tool" and requested_tool_name
                    else None
                )
                if canonical_requested_tool != required_first_tool:
                    payload = {
                        "action": "tool",
                        "tool_name": required_first_tool,
                        "arguments": {},
                    }
                    action_name = "tool"
                    first_tool_enforced = True

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
                        "required_first_tool": required_first_tool,
                        "first_tool_enforced": first_tool_enforced,
                        "planner_requested_action": requested_action,
                        "planner_requested_tool_name": requested_tool_name,
                        "final_answer": final_answer,
                        "video_id": video_id,
                    }
                )
                evidence_summary = _conversation_state(
                    turn_trace,
                    caption_placement=self.context.caption_placement,
                )
                return dspy.Prediction(
                    answer=final_answer,
                    reasoning_summary=raw_answer,
                    evidence_summary=evidence_summary,
                    turn_trace=turn_trace,
                    accumulated_evidence=evidence_summary,
                )

            raw_tool_name = str(payload.get("tool_name") or "ask_perception")
            tool_name = _normalize_tool_name(raw_tool_name, self.context.allowed_tools)
            tool_query = _tool_query(payload, tool_name, source_question=question)
            default_perception_backend = (
                selected_captioner_model()
                if tool_name == "ask_caption"
                else selected_perception_model()
            )
            tool_observation = self._call_tool(
                tool_name=tool_name,
                video_path=video_path,
                audio_path=audio_path,
                tool_query=tool_query,
                question_id=question_id,
            )
            perception_metadata = consume_last_perception_metadata()
            perception_backend = (
                perception_metadata.get("backend") or default_perception_backend
            )
            if tool_name == "ask_caption" and self.context.caption_placement == "task":
                caption_task_description = str(tool_observation or "").strip()

            turn_trace.append(
                {
                    "turn_id": turn_idx,
                    "planner_action": "tool",
                    "tool_name": tool_name,
                    "raw_tool_name": raw_tool_name,
                    "perception_backend": perception_backend,
                    "tool_args": _tool_args(tool_name, video_path, audio_path, tool_query),
                    "tool_observation": tool_observation,
                    "perception_system_prompt": perception_metadata.get("system_prompt", ""),
                    "perception_prompt": perception_metadata.get("prompt", ""),
                    "perception_model": perception_metadata.get("model"),
                    "perception_thinking": perception_metadata.get("thinking_text", ""),
                    "perception_token_usage": perception_metadata.get("token_usage"),
                    "caption_cache_hit": perception_metadata.get("caption_cache_hit"),
                    "caption_cache_error": perception_metadata.get("caption_cache_error"),
                    "caption_cache_path": perception_metadata.get("caption_cache_path"),
                    "caption_cache_source_results": perception_metadata.get(
                        "caption_cache_source_results"
                    ),
                    "caption_cache_source_rank": perception_metadata.get(
                        "caption_cache_source_rank"
                    ),
                    "caption_cache_source_backend": perception_metadata.get(
                        "caption_cache_source_backend"
                    ),
                    "caption_cache_source_token_usage": perception_metadata.get(
                        "caption_cache_source_token_usage"
                    ),
                    "reliable_qwen_profile": perception_metadata.get("reliable_qwen_profile"),
                    "qwen_response_empty": perception_metadata.get("qwen_response_empty"),
                    "planner_raw": raw_action,
                    "planner_calls": {"action_decision": planner_calls},
                    "planner_parse_error": planner_error,
                    "required_first_tool": required_first_tool,
                    "first_tool_enforced": first_tool_enforced,
                    "planner_requested_action": requested_action,
                    "planner_requested_tool_name": requested_tool_name,
                    "final_answer": None,
                    "video_id": video_id,
                }
            )

        task = current_task_text("final")
        fallback_state = (
            f"{_conversation_state(turn_trace, caption_placement=self.context.caption_placement)}\n\n"
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

        evidence_summary = _conversation_state(
            turn_trace,
            caption_placement=self.context.caption_placement,
        )
        return dspy.Prediction(
            answer=final_answer,
            reasoning_summary=raw_answer,
            evidence_summary=evidence_summary,
            turn_trace=turn_trace,
            accumulated_evidence=evidence_summary,
        )
