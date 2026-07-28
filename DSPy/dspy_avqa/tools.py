"""Perception and grounding tools backed by selectable AV models."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
from pathlib import Path
from typing import Any

from .gemini_api import call_gemini_messages
from .prompt_config import prompt_value, render_prompt
from .reliable_qwen import active_qwen_request_profile, qwen_request_slot


SUPPORTED_PERCEPTION_MODELS = {"qwen", "gemini"}
_PERCEPTION_METADATA_STATE = threading.local()
_DSPY_GEMINI_PROMPT_LOCK = threading.Lock()
_PRINTED_DSPY_GEMINI_PROMPT = False


class GeminiResponseCircuitBreak(BaseException):
    """Fatal Gemini-only circuit breaker that bypasses normal per-sample error saves."""


def _gemini_response_error_sensitive() -> bool:
    return _env_flag("GEMINI_RESPONSE_ERROR_SENSITIVE", "false")


def _gemini_response_error_reason(response_text: Any) -> str | None:
    text = str(response_text or "").strip()
    if not text:
        return "Gemini returned an empty response"
    lowered = text.lower()
    if lowered.startswith("[error]"):
        return "Gemini returned an explicit [ERROR] response"
    if "litellm" in lowered and any(token in lowered for token in ("error", "exception", "failed")):
        return "Gemini returned a LiteLLM error response"
    return None


def _trip_gemini_response_circuit(reason: str, *, backend: str, env_prefix: str) -> None:
    if _gemini_response_error_sensitive():
        raise GeminiResponseCircuitBreak(
            f"Gemini response circuit breaker tripped ({backend}, {env_prefix}): {reason}"
        )


def _call_gemini_with_circuit_breaker(
    call: Any,
    *args: Any,
    backend: str,
    env_prefix: str,
    **kwargs: Any,
) -> Any:
    try:
        return call(*args, **kwargs)
    except Exception as exc:
        _trip_gemini_response_circuit(str(exc), backend=backend, env_prefix=env_prefix)
        raise

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

def _env_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"

def _env_flag_prefixed(prefix: str, suffix: str, default: str = "false") -> bool:
    value = _env_optional_value_prefixed(prefix, suffix)
    if value is None:
        value = default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _env_fallback_name(prefix: str, suffix: str) -> str | None:
    if prefix == "CAPTIONER_QWEN":
        return _env_name("QWEN", suffix)
    return None

def _env_optional_value_prefixed(prefix: str, suffix: str) -> str | None:
    value = _env_optional_value(_env_name(prefix, suffix))
    if value is not None:
        return value
    fallback = _env_fallback_name(prefix, suffix)
    return _env_optional_value(fallback) if fallback else None

def _env_optional_int_prefixed(prefix: str, suffix: str) -> int | None:
    value = _env_optional_value_prefixed(prefix, suffix)
    return int(value) if value is not None else None

def _env_optional_float_prefixed(prefix: str, suffix: str) -> float | None:
    value = _env_optional_value_prefixed(prefix, suffix)
    return float(value) if value is not None else None

def _env_optional_bool_prefixed(prefix: str, suffix: str) -> bool | None:
    value = _env_optional_value_prefixed(prefix, suffix)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _env_value_prefixed(prefix: str, suffix: str, default: str) -> str:
    value = _env_optional_value_prefixed(prefix, suffix)
    return value if value is not None else default

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

def to_gemini_media_data(path: str) -> dict[str, Any]:
    """Convert local media to inlineData, or a GCS URI to Vertex fileData."""
    if path.startswith("gs://"):
        return {
            "fileData": {
                "mimeType": mime_type_for_path(path),
                "fileUri": path,
            }
        }
    return {
        "inlineData": {
            "mimeType": mime_type_for_path(path),
            "data": b64_file(path),
        }
    }


def to_gemini_inline_data(path: str) -> dict[str, Any]:
    """Backward-compatible alias for Gemini media payload conversion."""
    return to_gemini_media_data(path)

def call_qwen_perception(
    video_path: str,
    audio_path: str | None,
    prompt: str,
    system_prompt: str | None = None,
    env_prefix: str = "QWEN",
) -> str:
    """Call Qwen3-omni for AV perception/grounding."""
    from .qwen3omni_api import call_qwen_messages, to_data_url

    request_profile = active_qwen_request_profile()
    if request_profile is None:
        qwen_video_only = _env_flag_prefixed(env_prefix, "VIDEO_ONLY", "false")
        qwen_use_audio_in_video = (
            not qwen_video_only
            and _env_flag_prefixed(env_prefix, "USE_AUDIO_IN_VIDEO", "false")
        )
        qwen_video_first = _env_flag_prefixed(env_prefix, "VIDEO_FIRST", "false")
    else:
        # The reliable policy is intentionally request-local.  In particular,
        # it never mutates QWEN_VIDEO_FIRST while DSPy worker threads are live.
        qwen_video_only = False
        qwen_use_audio_in_video = request_profile.use_audio_in_video
        qwen_video_first = request_profile.video_first
    defer_empty_response_retries = bool(
        request_profile is not None
        and request_profile.defer_empty_response_retries
    )
    if not audio_path and not qwen_video_only and not qwen_use_audio_in_video:
        raise ValueError("audio_path is required for Qwen3-omni calls")

    content: list[dict[str, Any]] = []
    video_content = {"type": "video_url", "video_url": {"url": to_data_url(video_path)}}
    audio_content = (
        {"type": "audio_url", "audio_url": {"url": to_data_url(audio_path)}}
        if audio_path and not qwen_video_only and not qwen_use_audio_in_video
        else None
    )
    if qwen_video_first:
        content.append(video_content)
        if audio_content is not None:
            content.append(audio_content)
    else:
        if audio_content is not None:
            content.append(audio_content)
        content.append(video_content)
    content.append({"type": "text", "text": prompt})
    system_prompt = system_prompt if system_prompt is not None else prompt_value("perception", "system_prompt").strip()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    with qwen_request_slot():
        response_text, token_usage, thinking_text = call_qwen_messages(
            messages,
            model=_env_value_prefixed(env_prefix, "MODEL", "qwen3-omni-flash") or "qwen3-omni-flash",
            api_key=_env_optional_value_prefixed(env_prefix, "API_KEY"),
            base_url=_env_optional_value_prefixed(env_prefix, "BASE_URL"),
            timeout=int(_env_value_prefixed(env_prefix, "TIMEOUT", "180")),
            max_retries=int(_env_value_prefixed(env_prefix, "MAX_RETRIES", "3")),
            retry_delay_s=float(_env_value_prefixed(env_prefix, "DELAY_S", "1.0")),
            temperature=float(_env_value_prefixed(env_prefix, "TEMPERATURE", "0.6")),
            qwen_seed=_env_optional_int_prefixed(env_prefix, "SEED"),
            top_p=float(_env_value_prefixed(env_prefix, "TOP_P", "0.95")),
            top_k=int(_env_value_prefixed(env_prefix, "TOP_K", "20")),
            max_tokens=int(_env_value_prefixed(env_prefix, "MAX_TOKENS", "1024")),
            fps=float(_env_value_prefixed(env_prefix, "FPS", "2.0")),
            max_frames=int(_env_value_prefixed(env_prefix, "MAX_FRAMES", "128")),
            use_audio_in_video=qwen_use_audio_in_video,
            repetition_penalty=_env_optional_float_prefixed(env_prefix, "REPETITION_PENALTY"),
            enable_thinking=_env_optional_bool_prefixed(env_prefix, "ENABLE_THINKING"),
            return_thinking=True,
            # During a reliable batch, preserve transport retries but return an
            # empty visible completion to the adapter.  The adapter retries only
            # after all concurrent workers have completed.
            retry_degenerate_response=not defer_empty_response_retries,
            empty_response_retry_min_tokens=int(
                _env_value_prefixed(env_prefix, "EMPTY_RESPONSE_RETRY_MIN_TOKENS", "0")
            ),
            empty_response_retry_temperature=_env_optional_float_prefixed(
                env_prefix, "EMPTY_RESPONSE_RETRY_TEMPERATURE"
            ),
            empty_response_retry_seed_step=int(
                _env_value_prefixed(env_prefix, "EMPTY_RESPONSE_RETRY_SEED_STEP", "0")
            ),
            empty_response_retry_video_first=_env_flag_prefixed(
                env_prefix, "EMPTY_RESPONSE_RETRY_VIDEO_FIRST", "true"
            ),
            raise_on_empty_response=(
                not defer_empty_response_retries
                and _env_flag_prefixed(env_prefix, "RAISE_ON_EMPTY_RESPONSE", "true")
            ),
            stream_empty_fallback=not defer_empty_response_retries,
        )
    _set_last_perception_metadata(
        backend="qwen",
        system_prompt=system_prompt,
        prompt=prompt,
        model=_env_value_prefixed(env_prefix, "MODEL", "qwen3-omni") or "qwen3-omni",
        base_url=_env_optional_value_prefixed(env_prefix, "BASE_URL"),
        video_first=qwen_video_first,
        use_audio_in_video=qwen_use_audio_in_video,
        reliable_qwen_profile=request_profile.name if request_profile is not None else None,
        qwen_response_empty=not response_text.strip(),
        env_prefix=env_prefix,
        token_usage=token_usage,
        thinking_text=thinking_text,
    )
    return response_text.strip()

def _gemini_env_optional_value(prefix: str, suffix: str) -> str | None:
    value = _env_optional_value(_env_name(prefix, suffix))
    if value is not None:
        return value
    if prefix != "GEMINI":
        return _env_optional_value(_env_name("GEMINI", suffix))
    return None

def _gemini_env_value(prefix: str, suffix: str, default: str) -> str:
    value = _gemini_env_optional_value(prefix, suffix)
    return value if value is not None else default

def _gemini_env_flag(prefix: str, suffix: str, default: str = "false") -> bool:
    return _gemini_env_value(prefix, suffix, default).strip().lower() in {"1", "true", "yes", "on"}

def _gemini_env_optional_int(prefix: str, suffix: str) -> int | None:
    value = _gemini_env_optional_value(prefix, suffix)
    return int(value) if value is not None else None

def gemini_video_only(env_prefix: str = "GEMINI") -> bool:
    """Return whether Gemini should receive only video input."""
    return _gemini_env_value(env_prefix, "VIDEO_ONLY", "true").strip().lower() not in {"0", "false", "no", "off"}

def gemini_api_backend() -> str:
    """Return the configured Gemini transport, validating direct tool use too."""
    backend = os.environ.get("GEMINI_API_BACKEND", "legacy").strip().lower() or "legacy"
    if backend not in {"legacy", "dspy"}:
        raise ValueError("GEMINI_API_BACKEND must be legacy or dspy, got {!r}.".format(backend))
    return backend


def _dspy_gemini_top_k(env_prefix: str) -> int:
    top_k = int(_gemini_env_value(env_prefix, "TOP_K", "20"))
    if not 1 <= top_k <= 64:
        raise ValueError(
            f"{env_prefix}_TOP_K={top_k} is invalid for DSPy Vertex Gemini; expected 1..64. "
            "Update the Gemini YAML sampling_params.top_k (for example, to 64)."
        )
    return top_k


def _maybe_print_dspy_gemini_messages(messages: list[dict[str, Any]], env_prefix: str) -> None:
    global _PRINTED_DSPY_GEMINI_PROMPT
    if _PRINTED_DSPY_GEMINI_PROMPT or not _gemini_env_flag(env_prefix, "PRINT_FIRST_PROMPT"):
        return
    with _DSPY_GEMINI_PROMPT_LOCK:
        if _PRINTED_DSPY_GEMINI_PROMPT:
            return
        print("[Gemini DSPy first prompt]", flush=True)
        print(json.dumps(messages, ensure_ascii=False, indent=2), flush=True)
        _PRINTED_DSPY_GEMINI_PROMPT = True


def call_gemini_perception(
    video_path: str,
    audio_path: str | None,
    prompt: str,
    system_prompt: str | None = None,
    env_prefix: str = "GEMINI",
) -> str:
    """Call Gemini for AV perception through the selected legacy or DSPy backend."""
    backend = gemini_api_backend()
    system_prompt = system_prompt if system_prompt is not None else prompt_value("perception", "system_prompt").strip()
    include_audio = bool(audio_path and not gemini_video_only(env_prefix))

    if backend == "dspy":
        from .gemini_api_new import (
            VERTEX_LOCATION as default_vertex_location,
            VERTEX_PROJECT as default_vertex_project,
            build_gemini_messages,
            call_gemini_messages as call_dspy_gemini_messages,
        )

        local_data_root = _env_optional_value("GEMINI_LOCAL_DATA_ROOT")
        gcs_data_root = _env_optional_value("GEMINI_GCS_DATA_ROOT")
        if not local_data_root or not gcs_data_root:
            raise ValueError(
                "DSPy Vertex Gemini requires GEMINI_LOCAL_DATA_ROOT and GEMINI_GCS_DATA_ROOT. "
                "Pass --gemini-local-data-root and --gemini-gcs-data-root to the runner."
            )
        messages = build_gemini_messages(
            prompt,
            system_prompt=system_prompt,
            video_path=video_path,
            audio_path=audio_path if include_audio else None,
            local_data_root=local_data_root,
            gcs_data_root=gcs_data_root,
        )
        _maybe_print_dspy_gemini_messages(messages, env_prefix)
        vertex_project = _env_optional_value("VERTEXAI_PROJECT") or default_vertex_project
        vertex_location = _env_optional_value("VERTEXAI_LOCATION") or default_vertex_location
        result = _call_gemini_with_circuit_breaker(
            call_dspy_gemini_messages,
            messages,
            backend="dspy",
            env_prefix=env_prefix,
            model=_gemini_env_value(env_prefix, "MODEL", "gemini-2.5-flash"),
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            timeout=int(_gemini_env_value(env_prefix, "TIMEOUT", "180")),
            max_retries=int(_gemini_env_value(env_prefix, "MAX_RETRIES", "3")),
            return_thinking=True,
            temperature=float(_gemini_env_value(env_prefix, "TEMPERATURE", "0.6")),
            gemini_seed=_gemini_env_optional_int(env_prefix, "SEED"),
            top_p=float(_gemini_env_value(env_prefix, "TOP_P", "0.95")),
            top_k=_dspy_gemini_top_k(env_prefix),
            max_tokens=int(_gemini_env_value(env_prefix, "MAX_TOKENS", "1024")),
            retry_delay_s=float(_gemini_env_value(env_prefix, "RETRY_DELAY_S", "5")),
            retry_degenerate_response=True,
        )
        api_metadata: dict[str, Any] = {
            "api_backend": "dspy",
            "vertex_project": vertex_project,
            "vertex_location": vertex_location,
            "local_data_root": local_data_root,
            "gcs_data_root": gcs_data_root,
        }
    else:
        parts: list[dict[str, Any]] = []
        if include_audio:
            parts.append(to_gemini_inline_data(audio_path))
        parts.extend([to_gemini_inline_data(video_path), {"text": prompt}])
        contents = [{"role": "user", "parts": parts}]
        result = _call_gemini_with_circuit_breaker(
            call_gemini_messages,
            contents,
            backend="legacy",
            env_prefix=env_prefix,
            system_prompt=system_prompt,
            model=_gemini_env_optional_value(env_prefix, "MODEL"),
            api_key=_gemini_env_optional_value(env_prefix, "API_KEY"),
            base_url=_gemini_env_optional_value(env_prefix, "BASE_URL"),
            provider=_gemini_env_optional_value(env_prefix, "PROVIDER"),
            auth_mode=_gemini_env_optional_value(env_prefix, "AUTH_MODE"),
            timeout=int(_gemini_env_value(env_prefix, "TIMEOUT", "180")),
            max_retries=int(_gemini_env_value(env_prefix, "MAX_RETRIES", "3")),
            retry_delay_s=float(_gemini_env_value(env_prefix, "RETRY_DELAY_S", "5")),
            include_thoughts=_gemini_env_flag(env_prefix, "INCLUDE_THOUGHTS", "true"),
            return_thinking=_gemini_env_flag(env_prefix, "RETURN_THINKING", "true"),
            temperature=float(_gemini_env_value(env_prefix, "TEMPERATURE", "0.6")),
            gemini_seed=_gemini_env_optional_int(env_prefix, "SEED"),
            top_p=float(_gemini_env_value(env_prefix, "TOP_P", "0.95")),
            top_k=int(_gemini_env_value(env_prefix, "TOP_K", "20")),
            max_tokens=int(_gemini_env_value(env_prefix, "MAX_TOKENS", "1024")),
            retry_degenerate_response=True,
        )
        api_metadata = {
            "api_backend": "legacy",
            "base_url": _gemini_env_optional_value(env_prefix, "BASE_URL"),
        }

    if len(result) == 3:
        response_text, usage, thinking_text = result
    else:
        response_text, usage = result
        thinking_text = ""
    response_error_reason = _gemini_response_error_reason(response_text)
    if response_error_reason:
        _trip_gemini_response_circuit(response_error_reason, backend=backend, env_prefix=env_prefix)
    _set_last_perception_metadata(
        backend="gemini",
        system_prompt=system_prompt,
        prompt=prompt,
        model=_gemini_env_value(env_prefix, "MODEL", ""),
        env_prefix=env_prefix,
        token_usage=usage,
        thinking_text=thinking_text,
        **api_metadata,
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
    backend = selected_captioner_model()
    if backend == "gemini":
        return call_gemini_perception(
            video_path=video_path,
            audio_path=audio_path,
            prompt=build_caption_prompt(caption_instruction),
            system_prompt=captioner_system_prompt(),
            env_prefix="CAPTIONER_GEMINI",
        )
    return call_qwen_perception(
        video_path=video_path,
        audio_path=audio_path,
        prompt=build_caption_prompt(caption_instruction),
        system_prompt=captioner_system_prompt(),
        env_prefix="CAPTIONER_QWEN",
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
