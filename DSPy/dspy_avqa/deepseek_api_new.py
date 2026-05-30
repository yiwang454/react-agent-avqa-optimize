from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

from openai import OpenAI


API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")


def _thinking_extra_body(thinking_mode: Optional[str]) -> dict[str, Any] | None:
    mode = str(thinking_mode or "").strip().lower().replace("_", "-")
    if not mode or mode in {"auto", "default", "none", "unset"}:
        return None
    if mode in {"enabled", "enable", "thinking", "think", "on", "true", "1"}:
        return {"thinking": {"type": "enabled"}}
    if mode in {"disabled", "disable", "non-thinking", "nonthinking", "no-thinking", "off", "false", "0"}:
        return {"thinking": {"type": "disabled"}}
    raise ValueError(
        f"Unsupported PLANNER_THINKING_MODE={thinking_mode!r}; "
        "expected enabled, disabled, or auto"
    )


def _client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY", API_KEY)
    base_url = os.getenv("DEEPSEEK_BASE_URL", BASE_URL)
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY is not set.")
    return OpenAI(api_key=api_key, base_url=base_url)


def _messages(system_prompt: Optional[str], user_prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def call_deepseek(
    prompt: str,
    system_prompt: Optional[str] = "",
    *,
    model: str = MODEL,
    timeout: int = 60,
    max_retries: int = 3,
    retry_delay_s: float = 1.0,
    temperature: Optional[float] = 0.1,
    top_p: Optional[float] = 0.6,
    top_k: int = 20,
    repetition_penalty: float = 1.05,
    output_seq_len: int = 4096,
    max_input_seq_len: int = 30000,
    intent_plugin_id: str = "Adaptive",
    decoupled: int = 1,
    random_seed: Optional[int] = None,
    thinking_mode: Optional[str] = None,
) -> Optional[str]:
    """Call DeepSeek through the OpenAI-compatible SDK path.

    The legacy gateway-only parameters stay in the signature so callers from
    the optimize scripts do not need to change. The official DeepSeek endpoint
    safely receives the compatible subset: temperature, top_p, and max_tokens.
    """
    del top_k, repetition_penalty, max_input_seq_len, intent_plugin_id, decoupled, random_seed

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": _messages(system_prompt, prompt),
                "stream": False,
                "max_tokens": output_seq_len,
                "timeout": timeout,
            }
            if temperature is not None:
                request_kwargs["temperature"] = temperature
            if top_p is not None:
                request_kwargs["top_p"] = top_p
            extra_body = _thinking_extra_body(thinking_mode)
            if extra_body is not None:
                request_kwargs["extra_body"] = extra_body

            resp = _client().chat.completions.create(**request_kwargs)
            content = resp.choices[0].message.content
            if content is None or not str(content).strip():
                raise ValueError("The LM returned an empty or null response.")
            return content
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                print(
                    f"[warn] DeepSeek API attempt {attempt}/{max_retries} failed: {exc}. "
                    f"Retrying in {retry_delay_s}s.",
                    flush=True,
                )
                time.sleep(retry_delay_s)

    if last_err is not None:
        print(f"[error] DeepSeek API failed after {max_retries} attempt(s): {last_err}", flush=True)
    return None


def call_deepseek_batch(
    system_prompt: Optional[str],
    user_prompts: List[str],
    *,
    model: str = MODEL,
    max_workers: int = 8,
    timeout: int = 60,
    max_retries: int = 3,
    retry_delay_s: float = 1.0,
    temperature: Optional[float] = 0.1,
    top_p: Optional[float] = 0.6,
    top_k: int = 20,
    repetition_penalty: float = 1.05,
    output_seq_len: int = 4096,
    max_input_seq_len: int = 30000,
    intent_plugin_id: str = "Adaptive",
    decoupled: int = 1,
    random_seed: Optional[int] = None,
    thinking_mode: Optional[str] = None,
    return_raw_if_parse_fail: bool = True,
) -> List[str]:
    del return_raw_if_parse_fail

    results: list[Optional[str]] = [None] * len(user_prompts)

    def _worker(prompt: str) -> str:
        raw = call_deepseek(
            prompt,
            system_prompt,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay_s=retry_delay_s,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            output_seq_len=output_seq_len,
            max_input_seq_len=max_input_seq_len,
            intent_plugin_id=intent_plugin_id,
            decoupled=decoupled,
            random_seed=random_seed,
            thinking_mode=thinking_mode,
        )
        if raw is None:
            return "# 调用失败"
        return raw

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, prompt): i for i, prompt in enumerate(user_prompts)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = ""

    return [result if result is not None else "" for result in results]
