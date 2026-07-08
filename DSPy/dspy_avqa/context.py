"""Runtime context and LM configuration for DSPy AVQA."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import dspy

from .deepseek_api import DeepSeekPlannerClient, DeepSeekPlannerConfig
from .deepseek_dspy_lm import DeepSeekDSPyLM, append_planner_call_trace
from .prompt_config import prompt_config, prompt_value

SUPPORTED_TOOL_NAMES = ("ask_caption", "ask_perception", "temporal_ground_video")
_SUPPORTED_TOOL_SET = set(SUPPORTED_TOOL_NAMES)


def pick_env(*keys: str, default: str) -> str:
    """Return the first non-empty env value from keys."""
    for key in keys:
        value = os.environ.get(key)
        if value and value.strip() and value.strip().upper() != "EMPTY":
            return value
    return default


def pick_optional_int_env(*keys: str) -> int | None:
    """Return the first non-empty integer env value from keys."""
    for key in keys:
        value = os.environ.get(key)
        if value and value.strip() and value.strip().upper() != "EMPTY":
            return int(value)
    return None


def _split_tool_names(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.replace(";", ",").split(",")
    else:
        raw_items = [str(item) for item in value]
    return [item.strip() for item in raw_items if item.strip()]


def _prompt_config_allowed_tools() -> list[str]:
    tools_config = prompt_config().get("tools") or {}
    if not isinstance(tools_config, dict):
        raise ValueError("Prompt config tools section must be a mapping")
    raw_allowed = tools_config.get("allowed") or tools_config.get("allowed_tools")
    return _split_tool_names(raw_allowed)


def resolve_allowed_tools(value: str | Iterable[str] | None = None) -> tuple[str, ...]:
    """Resolve enabled planner tools from CLI/env, then prompt YAML, then defaults."""
    raw_tools = _split_tool_names(value)
    if not raw_tools:
        raw_tools = _split_tool_names(
            os.environ.get("DSPY_AVQA_ALLOWED_TOOLS") or os.environ.get("DSPY_ALLOWED_TOOLS")
        )
    if not raw_tools:
        raw_tools = _prompt_config_allowed_tools()
    if not raw_tools:
        raw_tools = list(SUPPORTED_TOOL_NAMES)

    normalized: list[str] = []
    for tool_name in raw_tools:
        normalized_name = tool_name.strip().lower()
        if normalized_name in {"ask_captioner", "caption_video", "captioner"}:
            normalized_name = "ask_caption"
        if normalized_name in {"ask_qwen_perception", "ask_gemini_perception"}:
            normalized_name = "ask_perception"
        if normalized_name not in _SUPPORTED_TOOL_SET:
            raise ValueError(
                f"Unsupported DSPy AVQA tool {tool_name!r}; "
                f"expected one of {list(SUPPORTED_TOOL_NAMES)}"
            )
        if normalized_name not in normalized:
            normalized.append(normalized_name)

    if not normalized:
        raise ValueError("At least one DSPy AVQA tool must be enabled")
    return tuple(normalized)


@dataclass(kw_only=True)
class AVQARuntimeContext:
    """Runtime context aligned with original react_agent configuration."""

    system_prompt: str = field(default_factory=lambda: prompt_value("planner", "system_prompt").strip())
    planner_model: str = field(
        default_factory=lambda: pick_env(
            "DEEPSEEK_MODEL",
            "PLANNER_MODEL",
            default="deepseek-v4-pro",
        )
    )
    planner_api_key: str = field(
        default_factory=lambda: pick_env(
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_TOKEN",
            "PLANNER_API_KEY",
            default="EMPTY",
        )
    )
    planner_api_base: str = field(
        default_factory=lambda: pick_env(
            "DEEPSEEK_BASE_URL",
            "DEEPSEEK_API_BASE",
            "PLANNER_API_BASE",
            default="https://api.deepseek.com",
        )
    )
    planner_ss_url: str = field(
        default_factory=lambda: pick_env(
            "DEEPSEEK_BASE_URL",
            "DEEPSEEK_API_BASE",
            "PLANNER_API_BASE",
            default="https://api.deepseek.com",
        )
    )
    planner_wsid: str = field(
        default_factory=lambda: pick_env("DEEPSEEK_WSID", "PLANNER_WSID", default="12317")
    )
    planner_timeout: int = field(
        default_factory=lambda: int(os.environ.get("PLANNER_TIMEOUT", "180"))
    )
    planner_max_retries: int = field(
        default_factory=lambda: int(os.environ.get("PLANNER_MAX_RETRIES", "3"))
    )
    planner_retry_delay_s: float = field(
        default_factory=lambda: float(os.environ.get("PLANNER_RETRY_DELAY_S", "1"))
    )
    planner_temperature: float = field(
        default_factory=lambda: float(os.environ.get("PLANNER_TEMPERATURE", "0.2"))
    )
    planner_top_p: float = field(
        default_factory=lambda: float(os.environ.get("PLANNER_TOP_P", "0.6"))
    )
    planner_top_k: int = field(
        default_factory=lambda: int(os.environ.get("PLANNER_TOP_K", "20"))
    )
    planner_repetition_penalty: float = field(
        default_factory=lambda: float(os.environ.get("PLANNER_REPETITION_PENALTY", "1.05"))
    )
    planner_max_tokens: int = field(
        default_factory=lambda: int(os.environ.get("PLANNER_OUTPUT_SEQ_LEN", "2048"))
    )
    planner_max_input_seq_len: int = field(
        default_factory=lambda: int(os.environ.get("PLANNER_MAX_INPUT_SEQ_LEN", "30000"))
    )
    planner_intent_plugin_id: str = field(
        default_factory=lambda: os.environ.get("PLANNER_INTENT_PLUGIN_ID", "Adaptive")
    )
    planner_decoupled: int = field(
        default_factory=lambda: int(os.environ.get("PLANNER_DECOUPLED", "1"))
    )
    planner_thinking_mode: str = field(
        default_factory=lambda: pick_env("PLANNER_THINKING_MODE", "DEEPSEEK_THINKING_MODE", default="")
    )
    planner_deepseek_random_seed: int | None = field(
        default_factory=lambda: pick_optional_int_env("DEEPSEEK_SEED")
    )
    max_turns: int = field(default_factory=lambda: int(os.environ.get("DEFAULT_MAX_TURNS", "4")))
    allowed_tools: tuple[str, ...] = field(default_factory=resolve_allowed_tools)


def _obj_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _copy_lm_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for message in messages:
        copied.append({
            "role": message.get("role"),
            "content": message.get("content"),
        })
    return copied


def _response_choices(response: Any) -> list[Any]:
    choices = _obj_get(response, "choices", [])
    return list(choices or [])


def _message_from_choice(choice: Any) -> Any:
    return _obj_get(choice, "message", {})


def _message_content(message: Any) -> str:
    return str(_obj_get(message, "content", "") or "")


def _message_reasoning_content(message: Any) -> str:
    return str(
        _obj_get(message, "reasoning_content", None)
        or _obj_get(message, "reasoning", None)
        or ""
    )


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = _obj_get(response, "usage", {}) or {}
    try:
        return dict(usage)
    except Exception:
        return {}


class SerialNOpenAICompatibleLM(dspy.LM):
    """DSPy LM that records reasoning_content and emulates n>1 with serial n=1 calls."""

    def _record_response(
        self,
        *,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
        response: Any,
    ) -> None:
        normalized_messages = messages or [{"role": "user", "content": prompt or ""}]
        choices: list[dict[str, Any]] = []
        for choice in _response_choices(response):
            message = _message_from_choice(choice)
            choices.append({
                "content": _message_content(message),
                "reasoning_content": _message_reasoning_content(message),
                "finish_reason": _obj_get(choice, "finish_reason"),
            })

        first_choice = choices[0] if choices else {}
        call_record = {
            "model": self.model,
            "messages": _copy_lm_messages(normalized_messages),
            "params": {
                "temperature": kwargs.get("temperature", self.kwargs.get("temperature")),
                "top_p": kwargs.get("top_p", self.kwargs.get("top_p")),
                "max_tokens": kwargs.get("max_tokens", self.kwargs.get("max_tokens")),
                "deepseek_random_seed": kwargs.get("seed", self.kwargs.get("seed")),
                "n": kwargs.get("n", self.kwargs.get("n", 1)),
                "extra_body": kwargs.get("extra_body", self.kwargs.get("extra_body")),
            },
            "response_text": first_choice.get("content", ""),
            "reasoning_content": first_choice.get("reasoning_content", ""),
            "choices": choices,
            "usage": _usage_dict(response),
        }
        append_planner_call_trace(call_record)

    def _forward_once(self, prompt=None, messages=None, **kwargs):
        request_kwargs = dict(kwargs)
        if not _env_is_set("PLANNER_TEMPERATURE"):
            request_kwargs.pop("temperature", None)
        if not _env_is_set("PLANNER_TOP_P"):
            request_kwargs.pop("top_p", None)
        response = dspy.LM.forward(self, prompt=prompt, messages=messages, **request_kwargs)
        self._record_response(prompt=prompt, messages=messages, kwargs=request_kwargs, response=response)
        return response

    def forward(self, prompt=None, messages=None, **kwargs):
        n = int(kwargs.get("n") or 1)
        if n <= 1:
            return self._forward_once(prompt=prompt, messages=messages, **kwargs)

        single_kwargs = dict(kwargs)
        single_kwargs["n"] = 1
        responses = [
            self._forward_once(prompt=prompt, messages=messages, **single_kwargs)
            for _ in range(n)
        ]
        first = responses[0]
        choices = []
        for response in responses:
            choices.extend(_response_choices(response))

        try:
            first.choices = choices
        except Exception:
            first["choices"] = choices
        return first


def _planner_thinking_extra_body(mode: str) -> dict[str, Any] | None:
    value = str(mode or "").strip().lower().replace("_", "-")
    if not value or value in {"auto", "default", "none", "unset"}:
        return None
    if value in {"enabled", "enable", "thinking", "think", "on", "true", "1"}:
        return {"thinking": {"type": "enabled"}}
    if value in {"disabled", "disable", "non-thinking", "nonthinking", "no-thinking", "off", "false", "0"}:
        return {"thinking": {"type": "disabled"}}
    raise ValueError(
        f"Unsupported PLANNER_THINKING_MODE={mode!r}; expected enabled, disabled, or auto"
    )


def _native_litellm_model_name(model: str) -> str:
    """Route bare model names through LiteLLM's OpenAI-compatible provider."""
    value = model.strip()
    if "/" in value:
        return value
    return f"openai/{value}"


def _env_is_set(key: str) -> bool:
    value = os.environ.get(key)
    return value is not None and value.strip() and value.strip().upper() != "EMPTY"


def _configure_native_litellm(context: AVQARuntimeContext) -> dspy.BaseLM:
    """Configure DSPy native LM for an OpenAI-compatible DeepSeek endpoint."""
    api_key = None if context.planner_api_key.upper() == "EMPTY" else context.planner_api_key
    lm_kwargs: dict[str, Any] = {
        "model_type": "chat",
        "api_key": api_key,
        "api_base": context.planner_api_base,
        "max_tokens": context.planner_max_tokens,
        "timeout": context.planner_timeout,
        "num_retries": context.planner_max_retries,
        "cache": False,
    }
    if _env_is_set("PLANNER_TEMPERATURE"):
        lm_kwargs["temperature"] = context.planner_temperature
    if _env_is_set("PLANNER_TOP_P"):
        lm_kwargs["top_p"] = context.planner_top_p
    if context.planner_deepseek_random_seed is not None:
        lm_kwargs["seed"] = context.planner_deepseek_random_seed
    extra_body = _planner_thinking_extra_body(context.planner_thinking_mode)
    if extra_body is not None:
        lm_kwargs["extra_body"] = extra_body

    lm = SerialNOpenAICompatibleLM(
        _native_litellm_model_name(context.planner_model),
        **lm_kwargs,
    )
    if not _env_is_set("PLANNER_TEMPERATURE"):
        lm.kwargs.pop("temperature", None)
    if not _env_is_set("PLANNER_TOP_P"):
        lm.kwargs.pop("top_p", None)
    dspy.configure(lm=lm)
    return lm


def _configure_custom_deepseek_lm(context: AVQARuntimeContext) -> dspy.BaseLM:
    """Configure the legacy direct DeepSeek HTTP planner backend."""
    planner_cfg = DeepSeekPlannerConfig(
        ss_url=context.planner_api_base,
        wsid=context.planner_wsid,
        token=context.planner_api_key,
        model=context.planner_model,
        timeout=context.planner_timeout,
        max_retries=context.planner_max_retries,
        retry_delay_s=context.planner_retry_delay_s,
        temperature=context.planner_temperature,
        top_p=context.planner_top_p,
        top_k=context.planner_top_k,
        repetition_penalty=context.planner_repetition_penalty,
        output_seq_len=context.planner_max_tokens,
        max_input_seq_len=context.planner_max_input_seq_len,
        intent_plugin_id=context.planner_intent_plugin_id,
        decoupled=context.planner_decoupled,
        thinking_mode=context.planner_thinking_mode,
        deepseek_random_seed=context.planner_deepseek_random_seed,
    )
    client = DeepSeekPlannerClient(planner_cfg)
    lm = DeepSeekDSPyLM(
        client=client,
        model=context.planner_model,
        temperature=context.planner_temperature,
        max_tokens=context.planner_max_tokens,
        top_p=context.planner_top_p,
        top_k=context.planner_top_k,
        repetition_penalty=context.planner_repetition_penalty,
        thinking_mode=context.planner_thinking_mode,
        deepseek_random_seed=context.planner_deepseek_random_seed,
    )
    dspy.configure(lm=lm)
    return lm


def configure_deepseek_lm(context: AVQARuntimeContext) -> dspy.BaseLM:
    """Configure DSPy's planner LM.

    The default uses DSPy's native LiteLLM-backed LM against the DeepSeek
    OpenAI-compatible endpoint so optimizers such as COPRO can request multiple
    completions with `n`. Set DSPY_PLANNER_LM_BACKEND=custom to use the legacy
    one-completion DeepSeekDSPyLM wrapper.
    """
    backend = os.environ.get("DSPY_PLANNER_LM_BACKEND", "native_litellm").strip().lower()
    if backend in {"custom", "deepseek_custom", "legacy"}:
        return _configure_custom_deepseek_lm(context)
    if backend not in {"native", "native_litellm", "litellm", "openai_compatible"}:
        raise ValueError(
            "Unsupported DSPY_PLANNER_LM_BACKEND "
            f"{backend!r}; expected native_litellm or custom"
        )
    return _configure_native_litellm(context)

