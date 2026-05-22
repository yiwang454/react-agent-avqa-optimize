"""Runtime context and LM configuration for DSPy AVQA."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field

import dspy

from .deepseek_api import DeepSeekPlannerClient, DeepSeekPlannerConfig
from .deepseek_dspy_lm import DeepSeekDSPyLM
from .prompt_config import prompt_config, prompt_value

SUPPORTED_TOOL_NAMES = ("ask_perception", "temporal_ground_video")
_SUPPORTED_TOOL_SET = set(SUPPORTED_TOOL_NAMES)


def pick_env(*keys: str, default: str) -> str:
    """Return the first non-empty env value from keys."""
    for key in keys:
        value = os.environ.get(key)
        if value and value.strip() and value.strip().upper() != "EMPTY":
            return value
    return default


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
    max_turns: int = field(default_factory=lambda: int(os.environ.get("DEFAULT_MAX_TURNS", "4")))
    allowed_tools: tuple[str, ...] = field(default_factory=resolve_allowed_tools)


def configure_deepseek_lm(context: AVQARuntimeContext) -> dspy.BaseLM:
    """Configure DSPy to use direct DeepSeek HTTP as planner backend."""
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
    )
    dspy.configure(lm=lm)
    return lm

