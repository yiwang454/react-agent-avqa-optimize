"""Custom DSPy BaseLM wrapper for direct DeepSeek HTTP calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import dspy

from .deepseek_api import DeepSeekPlannerClient


_PLANNER_CALL_TRACE: list[dict[str, Any]] = []


def clear_planner_call_trace() -> None:
    """Clear captured DeepSeek planner call records for the current sample."""
    _PLANNER_CALL_TRACE.clear()


def consume_planner_call_trace() -> list[dict[str, Any]]:
    """Return and clear captured DeepSeek planner call records."""
    calls = list(_PLANNER_CALL_TRACE)
    _PLANNER_CALL_TRACE.clear()
    return calls


def append_planner_call_trace(call_record: dict[str, Any]) -> None:
    """Append one planner call record for trajectory/debug output."""
    _PLANNER_CALL_TRACE.append(call_record)


def _copy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for message in messages:
        copied.append({
            "role": message.get("role"),
            "content": message.get("content"),
        })
    return copied


@dataclass
class _Message:
    """Minimal chat message object for DSPy response parsing."""

    content: str


@dataclass
class _Choice:
    """Minimal choice object carrying the generated message."""

    message: _Message


class _CompletionResponse:
    """Minimal OpenAI-compatible completion response object."""

    def __init__(self, *, text: str, model: str):
        self.choices = [_Choice(message=_Message(content=text))]
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.model = model


class DeepSeekDSPyLM(dspy.BaseLM):
    """DSPy LM backend that bypasses LiteLLM and calls DeepSeek directly."""

    def __init__(
        self,
        *,
        client: DeepSeekPlannerClient,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ):
        super().__init__(
            model=model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        self.client = client

    def _normalize_messages(
        self,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Convert DSPy inputs to an OpenAI-style messages array."""
        if messages is not None:
            return messages
        return [{"role": "user", "content": prompt or ""}]

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> _CompletionResponse:
        """Run one completion call through the direct DeepSeek client."""
        merged_kwargs = {**self.kwargs, **kwargs}
        normalized_messages = self._normalize_messages(prompt, messages)
        call_record: dict[str, Any] = {
            "model": self.model,
            "messages": _copy_messages(normalized_messages),
            "params": {
                "temperature": merged_kwargs.get("temperature"),
                "top_p": merged_kwargs.get("top_p"),
                "top_k": merged_kwargs.get("top_k"),
                "repetition_penalty": merged_kwargs.get("repetition_penalty"),
                "max_tokens": merged_kwargs.get("max_tokens"),
            },
        }
        try:
            response_text = self.client.call_messages(
                messages=normalized_messages,
                temperature=merged_kwargs.get("temperature"),
                top_p=merged_kwargs.get("top_p"),
                top_k=merged_kwargs.get("top_k"),
                repetition_penalty=merged_kwargs.get("repetition_penalty"),
                output_seq_len=merged_kwargs.get("max_tokens"),
            )
        except Exception as exc:
            call_record["error"] = str(exc)
            append_planner_call_trace(call_record)
            raise

        call_record["response_text"] = response_text
        append_planner_call_trace(call_record)
        return _CompletionResponse(text=response_text, model=self.model)
