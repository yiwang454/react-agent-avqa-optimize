from __future__ import annotations

import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .latency import finish_model_attempt, install_litellm_attempt_observer

from .response_quality import degenerate_response_reason

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
VERTEX_PROJECT = os.getenv("VERTEXAI_PROJECT", "decoupled-avqa-501420")
VERTEX_LOCATION = os.getenv("VERTEXAI_LOCATION", "global")


def _mime_type(path: str) -> str:
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
    return mime_overrides.get(Path(path).suffix.lower()) or mimetypes.guess_type(path)[0] or "application/octet-stream"


def local_path_to_gcs_uri(
    path: str,
    *,
    local_data_root: str,
    gcs_data_root: str,
) -> str:
    """Map a dataset-local media path to the corresponding GCS object URI."""
    if path.startswith("gs://"):
        return path
    if not gcs_data_root.startswith("gs://"):
        raise ValueError(f"gcs_data_root must start with gs://, got {gcs_data_root!r}")

    local_path = Path(path).expanduser().resolve()
    root_path = Path(local_data_root).expanduser().resolve()
    try:
        relative_path = local_path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(
            f"Media path {path!r} is outside Gemini local data root {local_data_root!r}. "
            "Pass --gemini-local-data-root/--gemini-gcs-data-root to configure the mapping."
        ) from exc
    return f"{gcs_data_root.rstrip('/')}/{relative_path.as_posix()}"


def _file_content(
    path: str,
    *,
    local_data_root: str,
    gcs_data_root: str,
) -> Dict[str, Any]:
    uri = local_path_to_gcs_uri(
        path,
        local_data_root=local_data_root,
        gcs_data_root=gcs_data_root,
    )
    return {
        "type": "file",
        "file": {
            "file_data": uri,
            "format": _mime_type(uri),
        },
    }


def build_gemini_messages(
    prompt: str,
    *,
    system_prompt: str = "",
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    local_data_root: str,
    gcs_data_root: str,
) -> List[Dict[str, Any]]:
    """Build LiteLLM-style messages for DSPy's Vertex Gemini backend."""
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content: List[Dict[str, Any]] = []
    if audio_path:
        content.append(
            _file_content(
                audio_path,
                local_data_root=local_data_root,
                gcs_data_root=gcs_data_root,
            )
        )
    if video_path:
        content.append(
            _file_content(
                video_path,
                local_data_root=local_data_root,
                gcs_data_root=gcs_data_root,
            )
        )
    content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})
    return messages


def _response_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        content = output.get("content")
        if isinstance(content, str):
            return content
    content = getattr(output, "content", None)
    if isinstance(content, str):
        return content
    raise RuntimeError(f"Unexpected DSPy output: {output!r}")


def _json_safe(value: Any) -> Any:
    """Recursively convert LiteLLM usage metadata to JSON-native values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    return str(value)


def _usage_dict(usage: Any) -> Dict[str, Any]:
    normalized = _json_safe(usage)
    return normalized if isinstance(normalized, dict) else {}


def call_gemini_messages(
    messages: List[Dict[str, Any]],
    *,
    model: str = MODEL,
    vertex_project: str = VERTEX_PROJECT,
    vertex_location: str = VERTEX_LOCATION,
    timeout: int = 180,
    max_retries: int = 3,
    return_thinking: bool = False,
    temperature: float = 0.6,
    gemini_seed: Optional[int] = None,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 1024,
    retry_delay_s: float = 5.0,
    retry_degenerate_response: bool = False,
) -> Tuple[str, Dict[str, Any]] | Tuple[str, Dict[str, Any], str]:
    """Call Vertex Gemini through DSPy's LiteLLM-backed LM."""
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            "The DSPy Gemini backend requires dspy. Run with the Python environment "
            "used by scripts/smoke_test_dspy_gemini_video.py."
        ) from exc

    install_litellm_attempt_observer()

    lm_kwargs: Dict[str, Any] = {
        "vertex_project": vertex_project,
        "vertex_location": vertex_location,
        "cache": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "num_retries": max_retries,
        "top_p": top_p,
        "top_k": top_k,
    }
    if gemini_seed is not None:
        lm_kwargs["seed"] = gemini_seed

    for attempt in range(1, max_retries + 1):
        lm = dspy.LM(f"vertex_ai/{model}", **lm_kwargs)
        outputs = lm(messages=messages)
        if not outputs:
            raise RuntimeError("DSPy returned no Gemini output.")

        response_text = _response_text(outputs[0])
        usage: Dict[str, Any] = {}
        if lm.history:
            usage = _usage_dict(lm.history[-1].get("usage"))
        degenerate_reason = (
            degenerate_response_reason(response_text, usage, max_tokens=max_tokens)
            if retry_degenerate_response
            else None
        )
        if degenerate_reason and attempt < max_retries:
            finish_model_attempt(success=False)
            print(
                f"[warn] DSPy Gemini attempt {attempt}/{max_retries} returned a degenerate "
                f"perception response: {degenerate_reason}. Retrying in {retry_delay_s}s.",
                flush=True,
            )
            time.sleep(retry_delay_s)
            continue
        if degenerate_reason:
            print(
                f"[warn] DSPy Gemini exhausted {max_retries} attempts after a degenerate "
                f"perception response: {degenerate_reason}. Keeping the final response.",
                flush=True,
            )
        # A final degenerate response is still an accepted call when the
        # existing policy keeps it.  Only responses that trigger a retry are
        # excluded from retry-adjusted latency above.
        finish_model_attempt(success=True)
        if return_thinking:
            return response_text, usage, ""
        return response_text, usage

    raise RuntimeError("Gemini API call failed without attempting a request.")
