"""DeepSeek planner API client based on direct HTTP requests."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass

import requests


def _pick_env(*keys: str, default: str) -> str:
    """Pick the first non-empty env value from keys."""
    for key in keys:
        value = os.environ.get(key)
        if value and value.strip() and value.strip().upper() != "EMPTY":
            return value
    return default


@dataclass(kw_only=True)
class DeepSeekPlannerConfig:
    """Configuration for the local DeepSeek planner endpoint."""

    ss_url: str = _pick_env(
        "DEEPSEEK_SS_URL",
        "PLANNER_BASE_URL",
        default="http://stream-server-online-openapi.turbotke.production.polaris:81/openapi/chat/completions",
    )
    wsid: str = _pick_env("DEEPSEEK_WSID", "PLANNER_WSID", default="12317")
    token: str = _pick_env(
        "DEEPSEEK_TOKEN",
        "PLANNER_API_KEY",
        default="7auGXNATFSKl7dF",
    )
    model: str = _pick_env(
        "DEEPSEEK_MODEL",
        "PLANNER_MODEL",
        default="DeepSeek-V3.2-A37B-tziwang-3",
    )
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


def _raise_http_error(resp: requests.Response) -> None:
    """Raise detailed HTTP error with gateway diagnostics."""
    preview = resp.text.strip()[:800]
    err_code = resp.headers.get("x-tai-errcode")
    err_msg = resp.headers.get("x-tai-errmsg")
    raise RuntimeError(
        "DeepSeek request failed. "
        f"status={resp.status_code}, errcode={err_code!r}, errmsg={err_msg!r}, "
        f"content_type={resp.headers.get('Content-Type', '')!r}, body_preview={preview!r}"
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
            if not resp.ok:
                _raise_http_error(resp)
            raw_text = resp.text.strip()
            if not raw_text:
                raise ValueError(
                    "DeepSeek response is empty. "
                    f"status={resp.status_code}, content_type={resp.headers.get('Content-Type', '')!r}"
                )
            try:
                data = resp.json()
            except ValueError:
                preview = raw_text[:500]
                raise ValueError(
                    "DeepSeek returned non-JSON response. "
                    f"status={resp.status_code}, content_type={resp.headers.get('Content-Type', '')!r}, "
                    f"body_preview={preview!r}"
                )
            choices = data.get("choices")
            if (
                not isinstance(choices, list)
                or not choices
                or not isinstance(choices[0], dict)
                or not isinstance(choices[0].get("message"), dict)
                or not isinstance(choices[0]["message"].get("content"), str)
            ):
                raise ValueError(
                    "DeepSeek JSON schema is unexpected. "
                    f"body_preview={raw_text[:500]!r}"
                )
            return choices[0]["message"]["content"], None
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
