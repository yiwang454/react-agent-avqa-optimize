"""Runtime context and planner client helpers for AVQA ReAct."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from react_agent import prompts
from react_agent.deepseek_api import DeepSeekPlannerClient, DeepSeekPlannerConfig


@dataclass(kw_only=True)
class RuntimeContext:
    """Runtime configuration for DeepSeek planner and ReAct budget."""

    system_prompt: str = field(default=prompts.SYSTEM_PROMPT)
    # Backward compatibility with template configuration/tests.
    model: str | None = field(default=None)
    planner_model: str = field(
        default_factory=lambda: os.environ.get("PLANNER_MODEL", "DeepSeek-V3.2-A37B-tziwang-3")
    )
    planner_api_key: str = field(
        default_factory=lambda: os.environ.get("PLANNER_API_KEY", "EMPTY")
    )
    planner_wsid: str = field(
        default_factory=lambda: os.environ.get("PLANNER_WSID", "12317")
    )
    planner_ss_url: str = field(
        default_factory=lambda: os.environ.get(
            "PLANNER_BASE_URL",
            "http://stream-server-online-openapi.turbotke.production.polaris:81/openapi/chat/completions",
        )
    )
    max_turns: int = field(
        default_factory=lambda: int(os.environ.get("DEFAULT_MAX_TURNS", "4"))
    )

    def __post_init__(self) -> None:
        """Bridge legacy `MODEL`/`model` onto `planner_model`."""
        if self.model:
            self.planner_model = self.model
            return

        legacy_model = os.environ.get("MODEL")
        if legacy_model and os.environ.get("PLANNER_MODEL") is None:
            self.planner_model = legacy_model
            self.model = legacy_model


def get_planner_llm(context: RuntimeContext | None = None) -> DeepSeekPlannerClient:
    """Build planner client using direct DeepSeek HTTP API calls."""
    cfg = context or RuntimeContext()
    planner_cfg = DeepSeekPlannerConfig(
        ss_url=os.environ.get("DEEPSEEK_SS_URL", cfg.planner_ss_url),
        wsid=os.environ.get("DEEPSEEK_WSID", cfg.planner_wsid),
        token=os.environ.get("DEEPSEEK_TOKEN", cfg.planner_api_key),
        model=os.environ.get("DEEPSEEK_MODEL", cfg.planner_model),
    )
    return DeepSeekPlannerClient(planner_cfg)


# Backward compatibility for template imports/tests.
Context = RuntimeContext
