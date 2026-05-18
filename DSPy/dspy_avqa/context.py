"""Runtime context and LM configuration for DSPy AVQA."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import dspy

from .deepseek_api import DeepSeekPlannerClient, DeepSeekPlannerConfig
from .deepseek_dspy_lm import DeepSeekDSPyLM

SYSTEM_PROMPT = """
You are the planning/reasoning model in an audio-visual question answering ReAct system.

You CANNOT watch the video directly.
You can only reason over:
1. the user question and options,
2. coarse video/audio description,
3. observations returned by perceptual tools.

Your job is to decide what perceptual evidence is missing, ask targeted questions via tools,
and provide a final option when evidence is sufficient.
""".strip()


def pick_env(*keys: str, default: str) -> str:
    """Return the first non-empty env value from keys."""
    for key in keys:
        value = os.environ.get(key)
        if value and value.strip() and value.strip().upper() != "EMPTY":
            return value
    return default


@dataclass(kw_only=True)
class AVQARuntimeContext:
    """Runtime context aligned with original react_agent configuration."""

    system_prompt: str = field(default=SYSTEM_PROMPT)
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

