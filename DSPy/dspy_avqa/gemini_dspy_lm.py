"""DSPy BaseLM adapter for Gemini's native ``generateContent`` endpoint."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import dspy

from .deepseek_dspy_lm import FIXED_PLANNER_SYSTEM_PROMPT, append_planner_call_trace
from .gemini_api import call_gemini_messages


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


class _CompletionResponse:
    """The small OpenAI-shaped response surface DSPy needs to parse a reply."""

    def __init__(self, *, text: str, model: str, usage: dict[str, Any]):
        self.choices = [_Choice(message=_Message(content=text))]
        self.usage = usage
        self.model = model


class GeminiDSPyLM(dspy.BaseLM):
    """Make a text-only Gemini planner usable by the existing ReAct program."""

    def __init__(
        self, *, model: str, api_key: str, base_url: str, provider: str,
        auth_mode: str, timeout: int, max_retries: int, retry_delay_s: float,
        temperature: float, top_p: float, max_tokens: int, seed: int | None,
        include_thoughts: bool,
    ) -> None:
        super().__init__(model=model, model_type="chat", temperature=temperature,
                         top_p=top_p, max_tokens=max_tokens)
        self.api_key, self.base_url = api_key, base_url
        self.provider, self.auth_mode = provider, auth_mode
        self.timeout, self.max_retries, self.retry_delay_s = timeout, max_retries, retry_delay_s
        self.seed, self.include_thoughts = seed, include_thoughts

    @staticmethod
    def _messages(prompt: str | None, messages: list[dict[str, Any]] | None) -> tuple[str, list[dict[str, Any]]]:
        source = messages if messages is not None else [{"role": "user", "content": prompt or ""}]
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        for message in source:
            role = str(message.get("role") or "user").strip().lower()
            content = str(message.get("content") or "")
            if role == "system":
                system_parts.append(content)
            else:
                contents.append({"role": "model" if role == "assistant" else "user",
                                 "parts": [{"text": content}]})
        if not contents:
            contents = [{"role": "user", "parts": [{"text": prompt or ""}]}]
        if os.environ.get("DSPY_AVQA_SIGNATURE_IN_SYSTEM_PROMPT", "").lower() not in {"1", "true", "yes", "on"}:
            system_parts.insert(0, FIXED_PLANNER_SYSTEM_PROMPT)
        return "\n\n".join(part for part in system_parts if part), contents

    def forward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None,
                **kwargs: Any) -> _CompletionResponse:
        system_prompt, contents = self._messages(prompt, messages)
        params = {**self.kwargs, **kwargs}
        call_record = {
            "model": self.model,
            "messages": messages or [{"role": "user", "content": prompt or ""}],
            "params": {"temperature": params.get("temperature"), "top_p": params.get("top_p"),
                       "max_tokens": params.get("max_tokens"), "seed": params.get("seed", self.seed),
                       "include_thoughts": self.include_thoughts},
        }
        try:
            text, usage, _thinking = call_gemini_messages(
                contents, system_prompt=system_prompt, model=self.model, api_key=self.api_key,
                base_url=self.base_url, provider=self.provider, auth_mode=self.auth_mode,
                timeout=self.timeout, max_retries=self.max_retries, retry_delay_s=self.retry_delay_s,
                include_thoughts=self.include_thoughts, return_thinking=True,
                temperature=float(params.get("temperature", 0.0)),
                gemini_seed=params.get("seed", self.seed), top_p=float(params.get("top_p", 1.0)),
                top_k=None, max_tokens=int(params.get("max_tokens", 32768)),
                retry_degenerate_response=False,
            )
        except Exception as exc:
            call_record["error"] = str(exc)
            append_planner_call_trace(call_record)
            raise
        call_record["response_text"], call_record["usage"] = text, usage
        append_planner_call_trace(call_record)
        return _CompletionResponse(text=text, model=self.model, usage=usage)
