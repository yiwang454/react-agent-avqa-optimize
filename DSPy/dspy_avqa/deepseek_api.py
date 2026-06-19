"""DeepSeek planner API compatibility layer using deepseek_api_new."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .deepseek_api_new import call_deepseek


def _pick_env(*keys: str, default: str) -> str:
    """Pick the first non-empty env value from keys."""
    for key in keys:
        value = os.environ.get(key)
        if value and value.strip() and value.strip().upper() != "EMPTY":
            return value
    return default


def _pick_optional_int_env(*keys: str) -> int | None:
    for key in keys:
        value = os.environ.get(key)
        if value and value.strip() and value.strip().upper() != "EMPTY":
            return int(value)
    return None


@dataclass(kw_only=True)
class DeepSeekPlannerConfig:
    """Configuration kept for the existing DSPy AVQA planner adapter."""

    ss_url: str = _pick_env("DEEPSEEK_BASE_URL", "PLANNER_API_BASE", default="https://api.deepseek.com")
    wsid: str = _pick_env("DEEPSEEK_WSID", "PLANNER_WSID", default="12317")
    token: str = _pick_env("DEEPSEEK_API_KEY", "DEEPSEEK_TOKEN", "PLANNER_API_KEY", default="EMPTY")
    model: str = _pick_env("DEEPSEEK_MODEL", "PLANNER_MODEL", default="deepseek-v4-pro")
    timeout: int = int(os.environ.get("PLANNER_TIMEOUT", "180"))
    max_retries: int = int(os.environ.get("PLANNER_MAX_RETRIES", "3"))
    retry_delay_s: float = float(os.environ.get("PLANNER_RETRY_DELAY_S", "1"))
    temperature: float = float(os.environ.get("PLANNER_TEMPERATURE", "0.2"))
    top_p: float = float(os.environ.get("PLANNER_TOP_P", "0.6"))
    top_k: int = int(os.environ.get("PLANNER_TOP_K", "20"))
    repetition_penalty: float = float(os.environ.get("PLANNER_REPETITION_PENALTY", "1.05"))
    output_seq_len: int = int(os.environ.get("PLANNER_OUTPUT_SEQ_LEN", "2048"))
    max_input_seq_len: int = int(os.environ.get("PLANNER_MAX_INPUT_SEQ_LEN", "30000"))
    intent_plugin_id: str = os.environ.get("PLANNER_INTENT_PLUGIN_ID", "Adaptive")
    decoupled: int = int(os.environ.get("PLANNER_DECOUPLED", "1"))
    thinking_mode: str = _pick_env("PLANNER_THINKING_MODE", "DEEPSEEK_THINKING_MODE", default="")
    deepseek_random_seed: int | None = _pick_optional_int_env("DEEPSEEK_SEED")


def _messages_to_prompts(messages: list[dict[str, Any]]) -> tuple[str, str]:
    system_parts: list[str] = []
    prompt_parts: list[str] = []
    for item in messages:
        role = str(item.get("role", "user")).strip().lower() or "user"
        content = item.get("content", "")
        if isinstance(content, list):
            text = "\n".join(
                str(part.get("text") or part.get("content") or part)
                if isinstance(part, dict)
                else str(part)
                for part in content
            )
        else:
            text = str(content)

        if role == "system":
            system_parts.append(text)
        else:
            prompt_parts.append(f"{role}: {text}" if role != "user" else text)

    return "\n".join(system_parts), "\n".join(prompt_parts)


class DeepSeekPlannerClient:
    """Thin client wrapper for planner calls via deepseek_api_new."""

    def __init__(self, config: DeepSeekPlannerConfig):
        self.config = config
        if self.config.token and self.config.token.upper() != "EMPTY":
            os.environ.setdefault("DEEPSEEK_API_KEY", self.config.token)
        os.environ.setdefault("DEEPSEEK_BASE_URL", self.config.ss_url)
        os.environ.setdefault("DEEPSEEK_MODEL", self.config.model)

    def call(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        """Call planner once with system and user prompt."""
        content = call_deepseek(
            user_prompt,
            system_prompt,
            model=kwargs.get("model", self.config.model),
            timeout=kwargs.get("timeout", self.config.timeout),
            max_retries=kwargs.get("max_retries", self.config.max_retries),
            retry_delay_s=kwargs.get("retry_delay_s", self.config.retry_delay_s),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
            repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
            output_seq_len=kwargs.get("output_seq_len", self.config.output_seq_len),
            max_input_seq_len=kwargs.get("max_input_seq_len", self.config.max_input_seq_len),
            intent_plugin_id=kwargs.get("intent_plugin_id", self.config.intent_plugin_id),
            decoupled=kwargs.get("decoupled", self.config.decoupled),
            deepseek_random_seed=kwargs.get("deepseek_random_seed", self.config.deepseek_random_seed),
            thinking_mode=kwargs.get("thinking_mode", self.config.thinking_mode),
        )
        if content is None:
            raise RuntimeError("DeepSeek planner request failed")
        return content

    def call_messages(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Call planner once with an OpenAI-style messages list."""
        system_prompt, user_prompt = _messages_to_prompts(messages)
        return self.call(system_prompt, user_prompt, **kwargs)
