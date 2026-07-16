"""Helpers for detecting malformed multimodal-model completions."""

from __future__ import annotations

import re
from typing import Any, Mapping


_OUTPUT_TOKEN_KEYS = (
    "completion_tokens",
    "output_tokens",
    "candidatesTokenCount",
    "candidates_token_count",
    "outputTokenCount",
)
_CJK_CHARACTER_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def completion_token_count(usage: Mapping[str, Any] | None) -> int | None:
    """Return a provider-neutral generated-token count when one is available."""
    if not usage:
        return None
    for key in _OUTPUT_TOKEN_KEYS:
        value = usage.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _text_unit_count(text: str) -> int:
    """Approximate words while treating contiguous CJK characters as text units."""
    cjk_count = len(_CJK_CHARACTER_RE.findall(text))
    non_cjk_text = _CJK_CHARACTER_RE.sub(" ", text)
    return cjk_count + len(re.findall(r"\w+", non_cjk_text, flags=re.UNICODE))


def degenerate_response_reason(
    response_text: str | None,
    token_usage: Mapping[str, Any] | None,
    *,
    max_tokens: int,
) -> str | None:
    """Return a reason when output is likely whitespace/repetition rather than text.

    A blank response is retried only after it has exhausted the requested output
    budget. A non-blank response is retried only when a substantial output
    budget was consumed yet there is fewer than one text unit per 50 generated
    tokens. These guards avoid retrying legitimate concise answers.
    """
    completion_tokens = completion_token_count(token_usage)
    if completion_tokens is None or max_tokens <= 0:
        return None

    stripped = str(response_text or "").strip()
    if not stripped and completion_tokens >= max_tokens:
        return (
            "response is empty after strip while completion_tokens="
            f"{completion_tokens} reached max_tokens={max_tokens}"
        )

    # Only scrutinize a response after it used enough of the budget for a
    # whitespace/repetition failure to be plausible.
    meaningful_output_floor = max(128, max_tokens // 4)
    text_units = _text_unit_count(stripped)
    if (
        completion_tokens >= meaningful_output_floor
        and text_units * 50 <= completion_tokens
    ):
        return (
            f"response has only {text_units} text units for "
            f"completion_tokens={completion_tokens}"
        )
    return None
