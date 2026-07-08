"""Perception and grounding tools backed by selectable AV models."""

from __future__ import annotations

import base64
import mimetypes
import os
import threading
from pathlib import Path
from typing import Any

from .gemini_api import call_gemini_messages
from .prompt_config import prompt_value, render_prompt


SUPPORTED_PERCEPTION_MODELS = {"qwen", "gemini"}
_PERCEPTION_METADATA_STATE = threading.local()

def _env_flag(name: str, default: str = "false") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}

def _env_optional_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _env_optional_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()

def _env_optional_int(name: str) -> int | None:
    value = _env_optional_value(name)
    if value is None:
        return None
    return int(value)

def _env_optional_float(name: str) -> float | None:
    value = _env_optional_value(name)
    if value is None:
        return None
    return float(value)

def _set_last_perception_metadata(**metadata: Any) -> None:
    _PERCEPTION_METADATA_STATE.last = {
        key: value for key, value in metadata.items() if value is not None
    }

def consume_last_perception_metadata() -> dict[str, Any]:
    """Return and clear metadata from the most recent perception call in this thread."""
    metadata = dict(getattr(_PERCEPTION_METADATA_STATE, "last", {}) or {})
    _PERCEPTION_METADATA_STATE.last = {}
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

def selected_captioner_model() -> str:
    """Return the configured captioner backend name."""
    backend = (
        os.environ.get("CAPTIONER_MODEL")
        or os.environ.get("CAPTION_MODEL")
        or os.environ.get("PERCEPTION_MODEL", "qwen")
    ).strip().lower()
    if not backend:
        return selected_perception_model()
    if backend not in SUPPORTED_PERCEPTION_MODELS:
        raise ValueError(
            f"Unsupported CAPTIONER_MODEL={backend!r}; "
            f"expected one of {sorted(SUPPORTED_PERCEPTION_MODELS)}"
        )
    return backend

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

def to_gemini_inline_data(path: str) -> dict[str, Any]:
    """Convert a local file to Gemini inlineData payload."""
    return {
        "inlineData": {
            "mimeType": mime_type_for_path(path),
            "data": b64_file(path),
        }
    }

def call_qwen_perception(
    video_path: str,
    audio_path: str | None,
    prompt: str,
    system_prompt: str | None = None,
) -> str:
    """Call Qwen3-omni for AV perception/grounding."""
    from .qwen3omni_api import call_qwen_messages, to_data_url

    qwen_video_only = _env_flag("QWEN_VIDEO_ONLY", "false")
    if not audio_path and not qwen_video_only:
        raise ValueError("audio_path is required for Qwen3-omni calls")

    content: list[dict[str, Any]] = []
    video_content = {"type": "video_url", "video_url": {"url": to_data_url(video_path)}}
    audio_content = (
        {"type": "audio_url", "audio_url": {"url": to_data_url(audio_path)}}
        if audio_path and not qwen_video_only
        else None
    )
    if _env_flag("QWEN_VIDEO_FIRST", "false"):
        content.append(video_content)
        if audio_content is not None:
            content.append(audio_content)
    else:
        if audio_content is not None:
            content.append(audio_content)
        content.append(video_content)
    content.append({"type": "text", "text": prompt})
    system_prompt = system_prompt if system_prompt is not None else prompt_value("perception", "system_prompt").strip()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    response_text, token_usage, thinking_text = call_qwen_messages(
        messages,
        model=os.environ.get("QWEN_MODEL", "qwen3-omni-flash").strip() or "qwen3-omni-flash",
        api_key=_env_optional_value("QWEN_API_KEY"),
        base_url=_env_optional_value("QWEN_BASE_URL"),
        timeout=int(os.environ.get("QWEN_TIMEOUT", "180")),
        max_retries=int(os.environ.get("QWEN_MAX_RETRIES", "3")),
        retry_delay_s=float(os.environ.get("QWEN_DELAY_S", "1.0")),
        temperature=float(os.environ.get("QWEN_TEMPERATURE", "0.6")),
        qwen_seed=_env_optional_int("QWEN_SEED"),
        top_p=float(os.environ.get("QWEN_TOP_P", "0.95")),
        top_k=int(os.environ.get("QWEN_TOP_K", "20")),
        max_tokens=int(os.environ.get("QWEN_MAX_TOKENS", "1024")),
        fps=float(os.environ.get("QWEN_FPS", "2.0")),
        max_frames=int(os.environ.get("QWEN_MAX_FRAMES", "128")),
        repetition_penalty=_env_optional_float("QWEN_REPETITION_PENALTY"),
        enable_thinking=_env_optional_bool("QWEN_ENABLE_THINKING"),
        return_thinking=True,
    )
    _set_last_perception_metadata(
        backend="qwen",
        system_prompt=system_prompt,
        prompt=prompt,
        model=os.environ.get("QWEN_MODEL", "qwen3-omni").strip() or "qwen3-omni",
        # base_url=_env_optional_value("QWEN_BASE_URL"),
        video_first=_env_flag("QWEN_VIDEO_FIRST", "false"),
        token_usage=token_usage,
        thinking_text=thinking_text,
    )
    return response_text.strip()

def gemini_video_only() -> bool:
    """Return whether Gemini should receive only video input."""
    value = os.environ.get("GEMINI_VIDEO_ONLY", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}

def call_gemini_perception(
    video_path: str,
    audio_path: str | None,
    prompt: str,
    system_prompt: str | None = None,
) -> str:
    """Call Gemini for AV perception/grounding."""
    parts: list[dict[str, Any]] = []
    if audio_path and not gemini_video_only():
        parts.append(to_gemini_inline_data(audio_path))
    parts.extend([to_gemini_inline_data(video_path), {"text": prompt}])
    contents = [{"role": "user", "parts": parts}]
    system_prompt = system_prompt if system_prompt is not None else prompt_value("perception", "system_prompt").strip()
    result = call_gemini_messages(
        contents,
        system_prompt=system_prompt,
        timeout=int(os.environ.get("GEMINI_TIMEOUT", "180")),
        max_retries=int(os.environ.get("GEMINI_MAX_RETRIES", "3")),
        retry_delay_s=float(os.environ.get("GEMINI_RETRY_DELAY_S", "5")),
        include_thoughts=_env_flag("GEMINI_INCLUDE_THOUGHTS", "true"),
        return_thinking=_env_flag("GEMINI_RETURN_THINKING", "true"),
        temperature=float(os.environ.get("GEMINI_TEMPERATURE", "0.6")),
        gemini_seed=_env_optional_int("GEMINI_SEED"),
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
        system_prompt=system_prompt,
        prompt=prompt,
        model=os.environ.get("GEMINI_MODEL", ""),
        # base_url=_env_optional_value("GEMINI_BASE_URL"),
        token_usage=usage,
        thinking_text=thinking_text,
    )
    return response_text.strip()

def call_perception(
    video_path: str,
    audio_path: str | None,
    prompt: str,
    backend: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Call the selected perceptual backend."""
    backend = backend or selected_perception_model()
    if backend == "gemini":
        return call_gemini_perception(
            video_path=video_path,
            audio_path=audio_path,
            prompt=prompt,
            system_prompt=system_prompt,
        )
    return call_qwen_perception(
        video_path=video_path,
        audio_path=audio_path,
        prompt=prompt,
        system_prompt=system_prompt,
    )

def _caption_prompt_value(*keys: str, default: str = "") -> str:
    try:
        return prompt_value(*keys).strip()
    except KeyError:
        return default


def build_caption_prompt(caption_instruction: str | None = None) -> str:
    """Build the captioner prompt from YAML, with a stable fallback."""
    instruction = (caption_instruction or "").strip()
    if not instruction:
        instruction = _caption_prompt_value(
            "captioner",
            "default_caption_instruction",
            default=(
                "Given a video, produce a detailed timestamped shot list describing "
                "all salient visible and audible events."
            ),
        )
    try:
        return render_prompt(
            "captioner",
            "caption_prompt_template",
            caption_instruction=instruction,
        )
    except KeyError:
        return instruction


def captioner_system_prompt() -> str:
    """Return the captioner system prompt, falling back to a factual tool role."""
    return _caption_prompt_value(
        "captioner",
        "system_prompt",
        default="You are an audio-visual captioning system. Be factual and concise.",
    )


def ask_caption(
    video_path: str,
    caption_instruction: str | None = None,
    audio_path: str | None = None,
) -> str:
    """Tool: ask the selected captioner backend for a factual AV caption."""
    return call_perception(
        video_path=video_path,
        audio_path=audio_path,
        prompt=build_caption_prompt(caption_instruction),
        backend=selected_captioner_model(),
        system_prompt=captioner_system_prompt(),
    )


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
