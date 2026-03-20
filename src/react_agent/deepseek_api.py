"""DeepSeek planner API client based on direct HTTP requests."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass

import requests


@dataclass(kw_only=True)
class DeepSeekPlannerConfig:
    """Configuration for the local DeepSeek planner endpoint."""

    ss_url: str = os.environ.get(
        "DEEPSEEK_SS_URL",
        os.environ.get("DEEPSEEK_SS_URL", "http://stream-server-online-openapi.turbotke.production.polaris:81/openapi/chat/completions"),
    )
    wsid: str = os.environ.get("DEEPSEEK_WSID", os.environ.get("PLANNER_WSID", "12317"))
    token: str = os.environ.get("DEEPSEEK_TOKEN", os.getenv("DEEPSEEK_TOKEN", "7auGXNATFSKl7dF"))
    model: str = os.environ.get("DEEPSEEK_MODEL", os.getenv("DEEPSEEK_MODEL", "DeepSeek-V3.2-A37B-tziwang-3"))
    timeout: int = int(os.environ.get("PLANNER_TIMEOUT", "180"))
    max_retries: int = int(os.environ.get("PLANNER_MAX_RETRIES", "3"))
    temperature: float = float(os.environ.get("PLANNER_TEMPERATURE", "0.2"))
    top_p: float = float(os.environ.get("PLANNER_TOP_P", "0.6"))
    top_k: int = int(os.environ.get("PLANNER_TOP_K", "20"))
    repetition_penalty: float = float(os.environ.get("PLANNER_REPETITION_PENALTY", "1.05"))
    output_seq_len: int = int(os.environ.get("PLANNER_OUTPUT_SEQ_LEN", "2048"))
    max_input_seq_len: int = int(os.environ.get("PLANNER_MAX_INPUT_SEQ_LEN", "30000"))
    intent_plugin_id: str = os.environ.get("PLANNER_INTENT_PLUGIN_ID", "Adaptive")
    decoupled: int = int(os.environ.get("PLANNER_DECOUPLED", "1"))


def _headers(config: DeepSeekPlannerConfig) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.token}",
        "Wsid": config.wsid,
    }


def _post_json(
    config: DeepSeekPlannerConfig,
    payload: dict[str, object],
) -> requests.Response:
    session = requests.Session()
    session.trust_env = False
    return session.post(
        config.ss_url,
        headers=_headers(config),
        json=payload,
        timeout=config.timeout,
        proxies={"http": None, "https": None},
    )


def call_deepseek_once(
    system_prompt: str,
    user_prompt: str,
    *,
    config: DeepSeekPlannerConfig,
    random_seed: int | None = None,
) -> tuple[str | None, str | None]:
    """Single planner call that returns response text or error."""
    if random_seed is None:
        random_seed = abs(hash(user_prompt)) % 1_000_000

    payload: dict[str, object] = {
        "model": config.model,
        "query_id": "avqa_react_" + str(uuid.uuid4()),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "decoupled": config.decoupled,
        "random_seed": random_seed,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "repetition_penalty": config.repetition_penalty,
        "output_seq_len": config.output_seq_len,
        "max_input_seq_len": config.max_input_seq_len,
        "intent_plugin_id": config.intent_plugin_id,
    }

    last_err: str | None = None
    for attempt in range(1, config.max_retries + 1):
        try:
            resp = _post_json(config, payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"], None
        except Exception as exc:
            last_err = str(exc)
            if attempt < config.max_retries:
                time.sleep(10)
            else:
                return None, last_err

    return None, last_err


class DeepSeekPlannerClient:
    """Thin client wrapper for planner calls."""

    def __init__(self, config: DeepSeekPlannerConfig):
        self.config = config

    def call(self, system_prompt: str, user_prompt: str) -> str:
        """Call planner once and return plain content."""
        content, error = call_deepseek_once(
            system_prompt,
            user_prompt,
            config=self.config,
        )
        if content is None:
            raise RuntimeError(error or "DeepSeek planner request failed")
        return content
