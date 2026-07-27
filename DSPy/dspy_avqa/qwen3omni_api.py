from __future__ import annotations

import base64
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .response_quality import degenerate_response_reason

API_KEY = os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY", os.getenv("QWEN_TOKEN", "")))
BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("QWEN_MODEL", "qwen3-omni-flash")


class EmptyQwenResponseError(RuntimeError):
    """Raised when every Qwen attempt returns empty visible content."""

    def __init__(self, message: str, response_attempts: list[dict[str, Any]]) -> None:
        """Initialize the error with structured response-attempt diagnostics."""
        super().__init__(message)
        self.response_attempts = response_attempts


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    resolved_api_key = api_key or API_KEY
    if not resolved_api_key:
        raise EnvironmentError("QWEN_API_KEY, DASHSCOPE_API_KEY, or QWEN_TOKEN is not set.")
    return OpenAI(api_key=resolved_api_key, base_url=base_url or BASE_URL)


def to_data_url(path: str, mime_type: Optional[str] = None) -> str:
    media_url_mode = os.getenv("QWEN_MEDIA_URL_MODE", "data_url").strip().lower()
    if media_url_mode in {"file", "file_url", "file://"}:
        return f"file://{Path(path).resolve()}"

    guessed_type = mime_type or mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{guessed_type};base64,{data}"


def _extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text", item.get("content"))
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)
    return str(content)


def _extract_choice_letter(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    patterns = [
        r"(?<![A-Z])([ABCD])(?![A-Z])",
        r"answer\s*(?:is|:)\s*([ABCD])\b",
        r"选择\s*[:：]?\s*([ABCD])\b",
        r"答案\s*[:：]?\s*([ABCD])\b",
        r"option\s*([ABCD])\b",
        r"^\s*([ABCD])\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.I)
        if matches:
            return matches[-1].upper()
    return ""


def parse_qwen_content(content: Any) -> Tuple[str, str]:
    text = _extract_text_content(content)
    thinking = ""
    remaining = text.strip()

    match = re.search(r"<think>(.*?)(</think>|$)", text, flags=re.S | re.I)
    if match:
        thinking = match.group(1).strip()
        remaining = text[match.end() :].strip()

    prediction = _extract_choice_letter(remaining) or _extract_choice_letter(text)
    return thinking, prediction


def parse_qwen_response(resp_json: Dict[str, Any]) -> Tuple[str, str]:
    return parse_qwen_content(resp_json["choices"][0]["message"]["content"])


def _extract_usage(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        data = usage.model_dump()
    elif isinstance(usage, dict):
        data = usage
    else:
        data = {
            key: getattr(usage, key)
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_tokens",
                "output_tokens",
            )
            if getattr(usage, key, None) is not None
        }
    return {key: value for key, value in data.items() if value is not None}


def _extract_stream_delta(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return ""
    return _extract_text_content(getattr(delta, "content", None))


def _extract_stream_reasoning(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return ""

    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning is not None:
        return _extract_text_content(reasoning)

    if hasattr(delta, "model_dump"):
        delta_data = delta.model_dump(exclude_none=True)
    elif isinstance(delta, dict):
        delta_data = delta
    else:
        delta_data = {}
    return _extract_text_content(delta_data.get("reasoning_content"))


def _extract_message_reasoning(message: Any) -> str:
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is not None:
        return _extract_text_content(reasoning)

    if hasattr(message, "model_dump"):
        message_data = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        message_data = message
    else:
        message_data = {}
    return _extract_text_content(message_data.get("reasoning_content"))


def _extract_finish_reason(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    finish_reason = getattr(choices[0], "finish_reason", None)
    return str(finish_reason) if finish_reason is not None else None


def _messages_with_video_first(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Place video items before audio items without moving non-media content."""
    reordered_messages: List[Dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            reordered_messages.append(message)
            continue

        media_positions = [
            index
            for index, item in enumerate(content)
            if isinstance(item, dict)
            and item.get("type") in {"audio_url", "video_url"}
        ]
        media_items = [content[index] for index in media_positions]
        media_items.sort(key=lambda item: item.get("type") != "video_url")
        reordered_content = list(content)
        for index, item in zip(media_positions, media_items):
            reordered_content[index] = item
        reordered_messages.append({**message, "content": reordered_content})
    return reordered_messages


def call_qwen_messages(
    messages: List[Dict[str, Any]],
    *,
    model: str = MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 180,
    max_retries: int = 3,
    retry_delay_s: float = 1.0,
    temperature: float = 0.6,
    qwen_seed: Optional[int] = None,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 1024,
    fps: float = 2.0,
    max_frames: int = 128,
    use_audio_in_video: Optional[bool] = None,
    repetition_penalty: Optional[float] = None,
    enable_thinking: Optional[bool] = None,
    stream: Optional[bool] = None,
    return_thinking: bool = False,
    retry_degenerate_response: bool = False,
    empty_response_retry_min_tokens: int = 0,
    empty_response_retry_temperature: Optional[float] = None,
    empty_response_retry_seed_step: int = 0,
    empty_response_retry_video_first: bool = False,
    raise_on_empty_response: bool = False,
) -> Tuple[str, Dict[str, Any]] | Tuple[str, Dict[str, Any], str]:
    last_err: Exception | None = None
    resolved_stream = _env_flag("QWEN_STREAM", True) if stream is None else stream
    response_attempts: list[dict[str, Any]] = []
    empty_response_retry_count = 0
    for attempt in range(1, max_retries + 1):
        try:
            is_empty_response_retry = (
                retry_degenerate_response and empty_response_retry_count > 0
            )
            effective_temperature = temperature
            effective_seed = qwen_seed
            effective_min_tokens = 0
            if is_empty_response_retry:
                if empty_response_retry_temperature is not None:
                    effective_temperature = empty_response_retry_temperature
                if effective_seed is not None:
                    effective_seed += empty_response_retry_seed_step * empty_response_retry_count
                effective_min_tokens = min(
                    max(0, empty_response_retry_min_tokens),
                    max_tokens,
                )

            use_video_first_retry = (
                is_empty_response_retry and empty_response_retry_video_first
            )
            effective_messages = (
                _messages_with_video_first(messages)
                if use_video_first_retry
                else messages
            )

            extra_body: Dict[str, Any] = {
                "top_k": top_k,
            }
            if effective_min_tokens:
                extra_body["min_tokens"] = effective_min_tokens
            if repetition_penalty is not None:
                extra_body["repetition_penalty"] = repetition_penalty
            mm_processor_kwargs: Dict[str, Any] = {
                "fps": fps,
                "max_frames": max_frames,
            }
            if use_audio_in_video is not None:
                mm_processor_kwargs["use_audio_in_video"] = use_audio_in_video
            extra_body["mm_processor_kwargs"] = mm_processor_kwargs
            if enable_thinking is not None and _env_flag("QWEN_INCLUDE_ENABLE_THINKING", True):
                extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = enable_thinking

            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": effective_messages,
                "stream": resolved_stream,
                "temperature": effective_temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "timeout": timeout,
                "extra_body": extra_body,
            }
            if effective_seed is not None:
                request_kwargs["seed"] = effective_seed
            if resolved_stream:
                request_kwargs["stream_options"] = {"include_usage": True}

            completion = _client(api_key=api_key, base_url=base_url).chat.completions.create(
                **request_kwargs
            )

            if resolved_stream:
                content_chunks: list[str] = []
                reasoning_chunks: list[str] = []
                token_usage: Dict[str, Any] = {}
                finish_reason: str | None = None
                for chunk in completion:
                    content = _extract_stream_delta(chunk)
                    if content:
                        content_chunks.append(content)
                    reasoning = _extract_stream_reasoning(chunk)
                    if reasoning:
                        reasoning_chunks.append(reasoning)
                    usage = _extract_usage(chunk)
                    if usage:
                        token_usage = usage
                    finish_reason = _extract_finish_reason(chunk) or finish_reason
                response_text = "".join(content_chunks)
                reasoning_text = "".join(reasoning_chunks).strip()
                if not response_text.strip() and not reasoning_text:
                    retry_kwargs = dict(request_kwargs)
                    retry_kwargs["stream"] = False
                    retry_kwargs.pop("stream_options", None)
                    retry_completion = _client(api_key=api_key, base_url=base_url).chat.completions.create(
                        **retry_kwargs
                    )
                    message = retry_completion.choices[0].message
                    response_text = _extract_text_content(getattr(message, "content", None))
                    reasoning_text = _extract_message_reasoning(message).strip()
                    token_usage = _extract_usage(retry_completion) or token_usage
                    finish_reason = _extract_finish_reason(retry_completion) or finish_reason
            else:
                message = completion.choices[0].message
                response_text = _extract_text_content(getattr(message, "content", None))
                reasoning_text = _extract_message_reasoning(message).strip()
                token_usage = _extract_usage(completion)
                finish_reason = _extract_finish_reason(completion)

            degenerate_reason = (
                degenerate_response_reason(
                    response_text,
                    token_usage,
                    max_tokens=max_tokens,
                )
                if retry_degenerate_response
                else None
            )
            response_attempts.append(
                {
                    "attempt": attempt,
                    "seed": effective_seed,
                    "temperature": effective_temperature,
                    "max_tokens": max_tokens,
                    "min_tokens": effective_min_tokens,
                    "video_first_retry": use_video_first_retry,
                    "completion_tokens": token_usage.get("completion_tokens"),
                    "content_chars": len(response_text),
                    "reasoning_chars": len(reasoning_text),
                    "finish_reason": finish_reason,
                    "degenerate_reason": degenerate_reason,
                }
            )
            if degenerate_reason and attempt < max_retries:
                print(
                    f"[warn] Qwen API attempt {attempt}/{max_retries} returned a degenerate "
                    f"perception response: {degenerate_reason}. Retrying in {retry_delay_s}s.",
                    flush=True,
                )
                empty_response_retry_count += 1
                time.sleep(retry_delay_s)
                continue
            if degenerate_reason:
                message = (
                    f"Qwen API exhausted {max_retries} attempts after a degenerate "
                    f"perception response: {degenerate_reason}"
                )
                print(f"[warn] {message}.", flush=True)
                if raise_on_empty_response:
                    raise EmptyQwenResponseError(message, response_attempts)

            if len(response_attempts) > 1:
                token_usage = dict(token_usage)
                token_usage["qwen_response_attempts"] = response_attempts

            if return_thinking:
                return response_text, token_usage, reasoning_text
            return response_text, token_usage
        except EmptyQwenResponseError:
            raise
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(retry_delay_s)

    raise RuntimeError(f"Qwen API call failed: {last_err}") from last_err


def call_qwen(
    video_data_url: str,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    *,
    model: str = MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 600,
    max_retries: int = 3,
    retry_delay_s: float = 1.0,
    temperature: float = 0.6,
    qwen_seed: Optional[int] = None,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 1024,
    fps: float = 2.0,
    max_frames: int = 128,
    repetition_penalty: Optional[float] = None,
    enable_thinking: Optional[bool] = None,
) -> Tuple[str, Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if video_data_url:
        content.append({"type": "video_url", "video_url": {"url": video_data_url}})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": content})

    return call_qwen_messages(
        messages,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay_s=retry_delay_s,
        temperature=temperature,
        qwen_seed=qwen_seed,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        fps=fps,
        max_frames=max_frames,
        repetition_penalty=repetition_penalty,
        enable_thinking=enable_thinking,
    )


def call_qwen_serial(
    video_data_url: str,
    system_prompt: str,
    batches: list[str],
    delay_s: float = 0.0,
    **kwargs: Any,
) -> list[Tuple[str, Dict[str, Any]]]:
    outputs: list[Tuple[str, Dict[str, Any]]] = []
    for prompt in batches:
        outputs.append(call_qwen(video_data_url, prompt, system_prompt, **kwargs))
        if delay_s:
            time.sleep(delay_s)
    return outputs


def call_qwen3omni(
    video_path: str,
    audio_path: str,
    prompt_text: str,
    max_retries: int = 10,
    *,
    system_prompt: str = "",
    model: str = MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 180,
    retry_delay_s: float = 1.0,
    temperature: float = 0.6,
    qwen_seed: Optional[int] = None,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 1024,
    fps: float = 2.0,
    max_frames: int = 128,
    repetition_penalty: Optional[float] = None,
    video_first: bool = False,
    enable_thinking: bool = True,
) -> Tuple[str, str, Dict[str, Any]]:
    """Qwen3-Omni call used by optimize_dailyomni scripts.

    Request construction mirrors the old in-script call_qwen3omni payload, while
    transport follows deepseek_api_old/qwen_api_test.py's OpenAI-compatible SDK
    pattern.
    """
    video_data_url = to_data_url(video_path)
    audio_data_url = to_data_url(audio_path)
    audio_content = {"type": "audio_url", "audio_url": {"url": audio_data_url}}
    video_content = {"type": "video_url", "video_url": {"url": video_data_url}}
    content = [video_content, audio_content] if video_first else [audio_content, video_content]
    content.append({"type": "text", "text": prompt_text})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    response_text, token_usage, reasoning_text = call_qwen_messages(
        messages,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay_s=retry_delay_s,
        temperature=temperature,
        qwen_seed=qwen_seed,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        fps=fps,
        max_frames=max_frames,
        repetition_penalty=repetition_penalty,
        enable_thinking=enable_thinking,
        return_thinking=True,
    )
    parsed_thinking, prediction = parse_qwen_content(response_text)
    thinking = reasoning_text or parsed_thinking
    return thinking, prediction, token_usage
