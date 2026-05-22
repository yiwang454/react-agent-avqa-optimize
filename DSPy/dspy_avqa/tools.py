"""Perception and grounding tools backed by selectable AV models."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests

from .gemini_api import call_gemini_messages
from .prompt_config import prompt_value, render_prompt


SUPPORTED_PERCEPTION_MODELS = {"qwen", "gemini"}
_LAST_PERCEPTION_METADATA: dict[str, Any] = {}


def _env_flag(name: str, default: str = "false") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _set_last_perception_metadata(**metadata: Any) -> None:
    global _LAST_PERCEPTION_METADATA
    _LAST_PERCEPTION_METADATA = {key: value for key, value in metadata.items() if value is not None}


def consume_last_perception_metadata() -> dict[str, Any]:
    """Return and clear metadata from the most recent perception call."""
    global _LAST_PERCEPTION_METADATA
    metadata = dict(_LAST_PERCEPTION_METADATA)
    _LAST_PERCEPTION_METADATA = {}
    return metadata


def selected_perception_model() -> str:
    """Return the configured perceptual backend name."""
    backend = os.environ.get("PERCEPTION_MODEL", "qwen").strip().lower()
    if not backend:
        return "qwen"
    if backend not in SUPPORTED_PERCEPTION_MODELS:
        raise ValueError(
            f"Unsupported PERCEPTION_MODEL={backend!r}; "
            f"expected one of {sorted(SUPPORTED_PERCEPTION_MODELS)}"
        )
    return backend


def build_qwen_url() -> str:
    """Build Qwen OpenAI-compatible chat endpoint URL."""
    base_url = os.environ.get("QWEN_BASE_URL", "http://29.232.225.71:8000").rstrip("/")
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def qwen_headers() -> dict[str, str]:
    """Build HTTP headers for Qwen endpoint."""
    api_key = os.environ.get("QWEN_API_KEY", "").strip()
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.upper() != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def mime_type_for_path(path: str) -> str:
    """Guess a stable MIME type for local audio/video payloads."""
    suffix = Path(path).suffix.lower()
    mime_overrides = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    return mime_overrides.get(suffix) or mimetypes.guess_type(path)[0] or "application/octet-stream"


def b64_file(path: str) -> str:
    """Return base64-encoded file contents."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def to_data_url(path: str) -> str:
    """Convert a local file to data-url payload."""
    return f"data:{mime_type_for_path(path)};base64,{b64_file(path)}"


def to_gemini_inline_data(path: str) -> dict[str, Any]:
    """Convert a local file to Gemini inlineData payload."""
    return {
        "inlineData": {
            "mimeType": mime_type_for_path(path),
            "data": b64_file(path),
        }
    }


def call_qwen_perception(video_path: str, audio_path: str | None, prompt: str) -> str:
    """Call Qwen3-omni for AV perception/grounding."""
    if not audio_path:
        raise ValueError("audio_path is required for Qwen3-omni calls")

    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": prompt_value("perception", "system_prompt").strip()},
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": to_data_url(audio_path)}},
                    {"type": "video_url", "video_url": {"url": to_data_url(video_path)}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    }

    qwen_model = os.environ.get("QWEN_MODEL", "").strip()
    if qwen_model and qwen_model.upper() != "EMPTY":
        payload["model"] = qwen_model

    response = requests.post(
        build_qwen_url(),
        headers=qwen_headers(),
        json=payload,
        timeout=int(os.environ.get("QWEN_TIMEOUT", "300")),
    )
    if not response.ok:
        raise RuntimeError(
            "Qwen request failed. "
            f"status={response.status_code}, body_preview={response.text[:1200]!r}"
        )
    data = response.json()
    _set_last_perception_metadata(backend="qwen", token_usage=data.get("usage"))
    return str(data["choices"][0]["message"]["content"]).strip()


def gemini_video_only() -> bool:
    """Return whether Gemini should receive only video input."""
    value = os.environ.get("GEMINI_VIDEO_ONLY", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def call_gemini_perception(video_path: str, audio_path: str | None, prompt: str) -> str:
    """Call Gemini for AV perception/grounding."""
    parts: list[dict[str, Any]] = []
    if audio_path and not gemini_video_only():
        parts.append(to_gemini_inline_data(audio_path))
    parts.extend([to_gemini_inline_data(video_path), {"text": prompt}])
    contents = [{"role": "user", "parts": parts}]
    result = call_gemini_messages(
        contents,
        system_prompt=prompt_value("perception", "system_prompt").strip(),
        timeout=int(os.environ.get("GEMINI_TIMEOUT", "180")),
        max_retries=int(os.environ.get("GEMINI_MAX_RETRIES", "3")),
        retry_delay_s=float(os.environ.get("GEMINI_RETRY_DELAY_S", "5")),
        include_thoughts=_env_flag("GEMINI_INCLUDE_THOUGHTS", "true"),
        return_thinking=_env_flag("GEMINI_RETURN_THINKING", "true"),
        temperature=float(os.environ.get("GEMINI_TEMPERATURE", "0.6")),
        top_p=float(os.environ.get("GEMINI_TOP_P", "0.95")),
        top_k=int(os.environ.get("GEMINI_TOP_K", "20")),
        max_tokens=int(os.environ.get("GEMINI_MAX_TOKENS", "1024")),
    )
    if len(result) == 3:
        response_text, usage, thinking_text = result
    else:
        response_text, usage = result
        thinking_text = ""
    _set_last_perception_metadata(
        backend="gemini",
        token_usage=usage,
        thinking_text=thinking_text,
    )
    return response_text.strip()


def call_perception(video_path: str, audio_path: str | None, prompt: str) -> str:
    """Call the selected perceptual backend."""
    backend = selected_perception_model()
    if backend == "gemini":
        return call_gemini_perception(video_path=video_path, audio_path=audio_path, prompt=prompt)
    return call_qwen_perception(video_path=video_path, audio_path=audio_path, prompt=prompt)


def build_perception_prompt(
    perceptual_question: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> str:
    """Build the shared AVQA perception prompt."""
    clip_note = ""
    if start_time is not None and end_time is not None:
        clip_note = f"Focus only on {start_time:.2f}s to {end_time:.2f}s."

    return render_prompt(
        "perception",
        "evidence_prompt_template",
        clip_note=clip_note,
        perceptual_question=perceptual_question,
    )


def ask_qwen_perception(
    video_path: str,
    perceptual_question: str,
    audio_path: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> str:
    """Tool: ask Qwen3-omni for perceptual evidence from audio-video input."""
    prompt = build_perception_prompt(perceptual_question, start_time, end_time)
    return call_qwen_perception(video_path=video_path, audio_path=audio_path, prompt=prompt)


def ask_gemini_perception(
    video_path: str,
    perceptual_question: str,
    audio_path: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> str:
    """Tool: ask Gemini for perceptual evidence from audio-video input."""
    prompt = build_perception_prompt(perceptual_question, start_time, end_time)
    return call_gemini_perception(video_path=video_path, audio_path=audio_path, prompt=prompt)


def ask_perception(
    video_path: str,
    perceptual_question: str,
    audio_path: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> str:
    """Tool: ask the selected backend for perceptual evidence from audio-video input."""
    prompt = build_perception_prompt(perceptual_question, start_time, end_time)
    return call_perception(video_path=video_path, audio_path=audio_path, prompt=prompt)


def temporal_ground_video(
    video_path: str,
    perceptual_question: str,
    audio_path: str | None = None,
) -> str:
    """Tool: find relevant temporal spans for a perceptual question."""
    prompt = render_prompt(
        "perception",
        "temporal_grounding_prompt_template",
        perceptual_question=perceptual_question,
    )
    return call_perception(video_path=video_path, audio_path=audio_path, prompt=prompt)
