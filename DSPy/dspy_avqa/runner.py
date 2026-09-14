"""Batch runner for DSPy AVQA program."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .caption_cache import validate_caption_cache_coverage
from .context import (
    AVQARuntimeContext,
    CAPTION_CACHE_SCOPE_CHOICES,
    CAPTION_PLACEMENT_CHOICES,
    normalize_caption_placement,
    resolve_allowed_tools,
)
from .deepseek_dspy_lm import consume_planner_call_trace
from .experiment_config import (
    load_reasoner_config_yaml,
    planner_effective_config,
    save_resolved_experiment_config,
)
from .data import (
    build_input_state,
    build_result_row,
    maybe_dump_question_data,
    read_jsonl,
    write_results_jsonl,
)
from .program import (
    AVQADSPyReActProgram,
    caption_cache_missing_as_observation,
    normalize_option_letter,
)
from .prompt_config import active_prompt_yaml_path, load_prompt_config, prompt_config
from .reliable_qwen import (
    ReliableQwenExecutor,
    mark_empty_qwen_turn_trace,
    turn_trace_has_empty_qwen_observation,
)
from .signatures import apply_prompt_config_to_signatures
from .tools import build_caption_prompt


GEMINI_API_BACKENDS = ("legacy", "dspy")


def planner_task_template_uses_video_description() -> bool:
    """Return whether the active task template renders a seeded video description."""
    planner_config = prompt_config().get("planner") or {}
    if not isinstance(planner_config, dict):
        return False
    template = planner_config.get("task_prompt_template")
    return isinstance(template, str) and "{video_description}" in template


def resolve_preloaded_audio_caption_dir(
    audio_caption_dir: Path | None,
    *,
    caption_placement: str,
    ignore_audio_caption_dir: bool = False,
) -> tuple[Path | None, str | None]:
    """Determine whether a precomputed caption directory is relevant to this run.

    The precomputed caption is only rendered into the initial task when caption
    placement is ``task`` and the active task template references
    ``{video_description}``.  In all other configurations, passing the directory
    must not trigger caption loading or validation.  Callers may explicitly ignore
    it when the initial description is merely a placeholder and ``ask_caption``
    supplies the real caption in a later turn.
    """
    if audio_caption_dir is None:
        return None, "not provided"
    if ignore_audio_caption_dir:
        return None, "explicitly ignored"
    if caption_placement != "task":
        return None, f"caption placement is {caption_placement}"
    if not planner_task_template_uses_video_description():
        return None, "planner.task_prompt_template has no {video_description}"
    return audio_caption_dir, None


def add_gemini_backend_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared Gemini legacy/Vertex backend arguments to a CLI parser."""
    parser.add_argument(
        "--gemini-api-backend",
        choices=GEMINI_API_BACKENDS,
        default=os.environ.get("GEMINI_API_BACKEND", "legacy").strip().lower() or "legacy",
        help="Gemini transport: legacy generateContent API (default) or DSPy Vertex AI.",
    )
    parser.add_argument(
        "--vertex-project",
        default=os.environ.get("VERTEXAI_PROJECT"),
        help="Override VERTEXAI_PROJECT for the DSPy Vertex Gemini backend.",
    )
    parser.add_argument(
        "--vertex-location",
        default=os.environ.get("VERTEXAI_LOCATION"),
        help="Override VERTEXAI_LOCATION for the DSPy Vertex Gemini backend.",
    )
    parser.add_argument(
        "--gemini-local-data-root",
        default=os.environ.get("GEMINI_LOCAL_DATA_ROOT"),
        help="Required in DSPy Gemini mode: local media root mapped to --gemini-gcs-data-root.",
    )
    parser.add_argument(
        "--gemini-gcs-data-root",
        default=os.environ.get("GEMINI_GCS_DATA_ROOT"),
        help="Required in DSPy Gemini mode: gs:// root corresponding to local media files.",
    )
    parser.add_argument(
        "--response-error-sensitive",
        action=argparse.BooleanOptionalAction,
        default=_env_flag_value("GEMINI_RESPONSE_ERROR_SENSITIVE"),
        help=(
            "Trip the run on an empty or explicit Gemini error response, including "
            "Gemini API/LiteLLM failures after retries. Gemini only."
        ),
    )


def _env_flag_value(name: str, default: str = "false") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Batch run DSPy AVQA ReAct over cut JSONL.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="Run only these cut/question IDs. Comma-separated values are also accepted.",
    )
    parser.add_argument("--audio-caption-dir", type=Path, default=None)
    parser.add_argument(
        "--caption-cache-dir",
        type=Path,
        default=None,
        help=(
            "Question-scoped cache used only when the planner calls ask_caption. "
            "A cache-routed call never falls back to live captioning when its "
            "entry is missing or its prompt does not match."
        ),
    )
    parser.add_argument(
        "--caption-cache-scope",
        choices=CAPTION_CACHE_SCOPE_CHOICES,
        default="all",
        help=(
            "Use the caption cache for all ask_caption calls (default), or only "
            "for the first call and dispatch later calls to the live captioner."
        ),
    )
    parser.add_argument(
        "--ignore-audio-caption-dir",
        action="store_true",
        help=(
            "Do not load precomputed captions from --audio-caption-dir. "
            "Use when video_description is only an initial placeholder and "
            "ask_caption provides the actual caption."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--debug", action="store_true", help="Run only the first debug-limit samples.")
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=int(os.environ.get("DEBUG_LIMIT", "4")),
        help="Number of leading samples to run when --debug is enabled.",
    )
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument(
        "--inference-num-threads",
        type=int,
        default=int(os.environ.get("DSPY_AVQA_INFERENCE_NUM_THREADS", "4")),
        help=(
            "Concurrent ReAct rollouts in the primary inference pass (default: 4). "
            "Qwen-only recovery retries remain serial."
        ),
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=None,
        help=(
            "Primary rollouts per reliable-Qwen batch. Defaults to "
            "--inference-num-threads; empty-Qwen retries start after each batch."
        ),
    )
    parser.add_argument(
        "--planner-config-yaml",
        type=Path,
        default=None,
        help="Optional YAML file with explicit planner model and sampling parameters.",
    )
    parser.add_argument(
        "--perception-config-yaml",
        type=Path,
        default=None,
        help="Optional YAML file with perception backend runtime/sampling parameters.",
    )
    parser.add_argument(
        "--captioner-config-yaml",
        type=Path,
        default=None,
        help="Optional YAML file with captioner backend runtime/sampling parameters.",
    )
    parser.add_argument(
        "--prompt-yaml",
        type=Path,
        default=None,
        help="Optional YAML file with DSPy AVQA planner/perception/signature prompts.",
    )
    parser.add_argument(
        "--allowed-tools",
        default=os.environ.get("DSPY_AVQA_ALLOWED_TOOLS") or os.environ.get("DSPY_ALLOWED_TOOLS"),
        help=(
            "Comma-separated DSPy AVQA tools to expose to the planner. "
            "Defaults to DSPY_AVQA_ALLOWED_TOOLS/DSPY_ALLOWED_TOOLS, "
            "then prompt YAML tools.allowed, then all tools."
        ),
    )
    parser.add_argument(
        "--perception-model",
        choices=("qwen", "gemini"),
        default=os.environ.get("PERCEPTION_MODEL", "qwen").strip().lower() or "qwen",
        help="Perceptual backend used by DSPy tools.",
    )
    add_gemini_backend_args(parser)
    parser.add_argument(
        "--signature-in-system-prompt",
        action="store_true",
        default=_env_flag_value("DSPY_AVQA_SIGNATURE_IN_SYSTEM_PROMPT"),
        help=(
            "Keep the DSPy-rendered PlanNextAction signature/schema prompt as the "
            "chat-level system message instead of moving it into the user prompt; "
            "when enabled, the fixed short planner system prompt is not injected."
        ),
    )
    parser.add_argument(
        "--caption-placement",
        choices=CAPTION_PLACEMENT_CHOICES,
        default=normalize_caption_placement(),
        help=(
            "Where to place caption tool observations in later planner turns. "
            "Use 'conversation_state' for the current behavior or 'task' to move "
            "the caption into Coarse video/audio description."
        ),
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the same redacted resolved experiment config saved in the output directory.",
    )
    return parser.parse_args()

def _env_bool(value: Any) -> str:
    if isinstance(value, str):
        return "false" if value.strip().lower() in {"0", "false", "no", "off"} else "true"
    return "true" if bool(value) else "false"

def _set_env_from_mapping(mapping: dict[str, Any], key_map: dict[str, str]) -> None:
    for source_key, env_key in key_map.items():
        value = mapping.get(source_key)
        if value is not None:
            os.environ[env_key] = str(value)

def _selected_captioner_model() -> str:
    return (
        os.environ.get("CAPTIONER_MODEL")
        or os.environ.get("CAPTION_MODEL")
        or os.environ.get("PERCEPTION_MODEL", "qwen")
    ).strip().lower() or "qwen"


def _validate_dspy_gemini_top_k(prefix: str) -> None:
    raw_value = os.environ.get(f"{prefix}_TOP_K")
    if raw_value is None and prefix == "CAPTIONER_GEMINI":
        raw_value = os.environ.get("GEMINI_TOP_K")
    if raw_value is None or not raw_value.strip():
        return
    try:
        top_k = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{prefix}_TOP_K must be an integer for DSPy Vertex Gemini, got {raw_value!r}."
        ) from exc
    if not 1 <= top_k <= 64:
        raise ValueError(
            f"{prefix}_TOP_K={top_k} is invalid for DSPy Vertex Gemini; expected 1..64. "
            "Update the Gemini YAML sampling_params.top_k (for example, to 64)."
        )


def configure_gemini_api_backend(args: argparse.Namespace) -> bool:
    """Apply CLI Gemini backend settings and validate an active Vertex configuration."""
    backend = str(getattr(args, "gemini_api_backend", "legacy") or "legacy").strip().lower()
    if backend not in GEMINI_API_BACKENDS:
        raise ValueError(f"Unsupported Gemini API backend {backend!r}; expected one of {GEMINI_API_BACKENDS}.")
    os.environ["GEMINI_API_BACKEND"] = backend
    os.environ["GEMINI_RESPONSE_ERROR_SENSITIVE"] = _env_bool(
        getattr(args, "response_error_sensitive", _env_flag_value("GEMINI_RESPONSE_ERROR_SENSITIVE"))
    )

    for arg_name, env_name in (
        ("vertex_project", "VERTEXAI_PROJECT"),
        ("vertex_location", "VERTEXAI_LOCATION"),
        ("gemini_local_data_root", "GEMINI_LOCAL_DATA_ROOT"),
        ("gemini_gcs_data_root", "GEMINI_GCS_DATA_ROOT"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None and str(value).strip():
            os.environ[env_name] = str(value).strip()

    perception_is_gemini = os.environ.get("PERCEPTION_MODEL", "qwen").strip().lower() == "gemini"
    caption_cache_scope = getattr(args, "caption_cache_scope", "all")
    captioner_is_gemini = _selected_captioner_model() == "gemini" and (
        not bool(getattr(args, "caption_cache_dir", None))
        or caption_cache_scope == "first_call_only"
    )
    gemini_is_active = perception_is_gemini or captioner_is_gemini
    if backend != "dspy" or not gemini_is_active:
        return gemini_is_active

    missing = [
        name
        for name in ("GEMINI_LOCAL_DATA_ROOT", "GEMINI_GCS_DATA_ROOT")
        if not os.environ.get(name, "").strip()
    ]
    if missing:
        flags = " and ".join(
            "--gemini-local-data-root" if name == "GEMINI_LOCAL_DATA_ROOT" else "--gemini-gcs-data-root"
            for name in missing
        )
        raise ValueError(
            "DSPy Vertex Gemini requires an explicit local-to-GCS media mapping. "
            f"Set {flags} (or the corresponding environment variable)."
        )
    if perception_is_gemini:
        _validate_dspy_gemini_top_k("GEMINI")
    if captioner_is_gemini:
        _validate_dspy_gemini_top_k("CAPTIONER_GEMINI")
    return gemini_is_active


def gemini_backend_log_lines() -> list[str]:
    """Return non-secret Gemini transport settings suitable for startup logs."""
    backend = os.environ.get("GEMINI_API_BACKEND", "legacy")
    lines = [f"Gemini API backend: {backend}"]
    lines.append(
        f"Gemini response-error-sensitive: {os.environ.get('GEMINI_RESPONSE_ERROR_SENSITIVE', 'false')}"
    )
    if backend == "dspy":
        lines.extend(
            [
                f"Vertex project: {os.environ.get('VERTEXAI_PROJECT', '<gemini_api_new default>')}",
                f"Vertex location: {os.environ.get('VERTEXAI_LOCATION', '<gemini_api_new default>')}",
                f"Gemini local data root: {os.environ.get('GEMINI_LOCAL_DATA_ROOT', '')}",
                f"Gemini GCS data root: {os.environ.get('GEMINI_GCS_DATA_ROOT', '')}",
            ]
        )
    return lines


def _load_qwen_config_yaml(
    path: Path | None,
    *,
    prefix: str,
) -> dict[str, Any]:
    """Load Qwen runtime parameters into env vars under the requested prefix."""
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Qwen config must be a YAML mapping: {path}")

    task = config.get("task") or {}
    model = config.get("model") or {}
    sampling = config.get("sampling_params") or {}
    if not isinstance(task, dict):
        raise ValueError(f"Qwen config task section must be a mapping: {path}")
    if not isinstance(model, dict):
        raise ValueError(f"Qwen config model section must be a mapping: {path}")
    if not isinstance(sampling, dict):
        raise ValueError(f"Qwen config sampling_params section must be a mapping: {path}")

    if task.get("video_only") is not None:
        os.environ[f"{prefix}_VIDEO_ONLY"] = _env_bool(task["video_only"])
    if task.get("enable_thinking") is not None:
        os.environ[f"{prefix}_ENABLE_THINKING"] = _env_bool(task["enable_thinking"])
    if task.get("video_first") is not None:
        os.environ[f"{prefix}_VIDEO_FIRST"] = _env_bool(task["video_first"])
    if task.get("use_audio_in_video") is not None:
        os.environ[f"{prefix}_USE_AUDIO_IN_VIDEO"] = _env_bool(task["use_audio_in_video"])

    _set_env_from_mapping(
        task,
        {
            "timeout": f"{prefix}_TIMEOUT",
            "max_retries": f"{prefix}_MAX_RETRIES",
            "qwen_delay_s": f"{prefix}_DELAY_S",
            "empty_response_retry_min_tokens": f"{prefix}_EMPTY_RESPONSE_RETRY_MIN_TOKENS",
            "empty_response_retry_temperature": f"{prefix}_EMPTY_RESPONSE_RETRY_TEMPERATURE",
            "empty_response_retry_seed_step": f"{prefix}_EMPTY_RESPONSE_RETRY_SEED_STEP",
            "empty_response_retry_video_first": f"{prefix}_EMPTY_RESPONSE_RETRY_VIDEO_FIRST",
        },
    )
    if task.get("raise_on_empty_response") is not None:
        os.environ[f"{prefix}_RAISE_ON_EMPTY_RESPONSE"] = _env_bool(
            task["raise_on_empty_response"]
        )
    _set_env_from_mapping(
        model,
        {
            "qwen_model": f"{prefix}_MODEL",
            "qwen_base_url": f"{prefix}_BASE_URL",
            "qwen_api_key": f"{prefix}_API_KEY",
        },
    )
    _set_env_from_mapping(
        sampling,
        {
            "temperature": f"{prefix}_TEMPERATURE",
            "top_p": f"{prefix}_TOP_P",
            "top_k": f"{prefix}_TOP_K",
            "max_tokens": f"{prefix}_MAX_TOKENS",
            "fps": f"{prefix}_FPS",
            "max_frames": f"{prefix}_MAX_FRAMES",
            "seed": f"{prefix}_SEED",
            "repetition_penalty": f"{prefix}_REPETITION_PENALTY",
        },
    )
    return config


def load_captioner_config_yaml(path: Path | None) -> dict[str, Any]:
    """Load captioner-specific runtime parameters under captioner env vars."""
    backend = (
        os.environ.get("CAPTIONER_MODEL")
        or os.environ.get("CAPTION_MODEL")
        or os.environ.get("PERCEPTION_MODEL", "qwen")
    ).strip().lower()
    if backend == "gemini":
        return load_gemini_captioner_config_yaml(path)
    return _load_qwen_config_yaml(
        path,
        prefix="CAPTIONER_QWEN",
    )


def load_gemini_captioner_config_yaml(path: Path | None) -> dict[str, Any]:
    """Load Gemini captioner runtime parameters under CAPTIONER_GEMINI_* env vars."""
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Gemini captioner config must be a YAML mapping: {path}")

    task = config.get("task") or {}
    model = config.get("model") or {}
    sampling = config.get("sampling_params") or {}
    if not isinstance(task, dict):
        raise ValueError(f"Gemini captioner config task section must be a mapping: {path}")
    if not isinstance(model, dict):
        raise ValueError(f"Gemini captioner config model section must be a mapping: {path}")
    if not isinstance(sampling, dict):
        raise ValueError(f"Gemini captioner config sampling_params section must be a mapping: {path}")

    if task.get("video_only") is not None:
        os.environ["CAPTIONER_GEMINI_VIDEO_ONLY"] = _env_bool(task["video_only"])
    if task.get("print_first_prompt") is not None:
        os.environ["CAPTIONER_GEMINI_PRINT_FIRST_PROMPT"] = _env_bool(task["print_first_prompt"])
        os.environ["GEMINI_PRINT_FIRST_PROMPT"] = _env_bool(task["print_first_prompt"])

    _set_env_from_mapping(
        task,
        {
            "batch_size": "CAPTIONER_GEMINI_BATCH_SIZE",
            "timeout": "CAPTIONER_GEMINI_TIMEOUT",
            "max_retries": "CAPTIONER_GEMINI_MAX_RETRIES",
            "retry_delay_s": "CAPTIONER_GEMINI_RETRY_DELAY_S",
        },
    )
    _set_env_from_mapping(
        model,
        {
            "gemini_model": "CAPTIONER_GEMINI_MODEL",
            "vertex_project": "VERTEXAI_PROJECT",
            "vertex_location": "VERTEXAI_LOCATION",
            "gemini_base_url": "CAPTIONER_GEMINI_BASE_URL",
            "gemini_api_key": "CAPTIONER_GEMINI_API_KEY",
            "gemini_provider": "CAPTIONER_GEMINI_PROVIDER",
            "gemini_auth_mode": "CAPTIONER_GEMINI_AUTH_MODE",
        },
    )
    _set_env_from_mapping(
        sampling,
        {
            "temperature": "CAPTIONER_GEMINI_TEMPERATURE",
            "top_p": "CAPTIONER_GEMINI_TOP_P",
            "top_k": "CAPTIONER_GEMINI_TOP_K",
            "max_tokens": "CAPTIONER_GEMINI_MAX_TOKENS",
            "gemini_seed": "CAPTIONER_GEMINI_SEED",
            "seed": "CAPTIONER_GEMINI_SEED",
        },
    )
    return config


def load_perception_config_yaml(path: Path | None) -> dict[str, Any]:
    """Load perception backend parameters from a baseline-style YAML config."""
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Perception config must be a YAML mapping: {path}")

    task = config.get("task") or {}
    model = config.get("model") or {}
    sampling = config.get("sampling_params") or {}
    if not isinstance(task, dict):
        raise ValueError(f"Perception config task section must be a mapping: {path}")
    if not isinstance(model, dict):
        raise ValueError(f"Perception config model section must be a mapping: {path}")
    if not isinstance(sampling, dict):
        raise ValueError(f"Perception config sampling_params section must be a mapping: {path}")

    if task.get("video_only") is not None:
        video_only = _env_bool(task["video_only"])
        os.environ["GEMINI_VIDEO_ONLY"] = video_only
        os.environ["QWEN_VIDEO_ONLY"] = video_only
    if task.get("enable_thinking") is not None:
        os.environ["QWEN_ENABLE_THINKING"] = _env_bool(task["enable_thinking"])
    if task.get("video_first") is not None:
        os.environ["QWEN_VIDEO_FIRST"] = _env_bool(task["video_first"])
    if task.get("use_audio_in_video") is not None:
        os.environ["QWEN_USE_AUDIO_IN_VIDEO"] = _env_bool(
            task["use_audio_in_video"]
        )

    _set_env_from_mapping(
        task,
        {
            "batch_size": "GEMINI_BATCH_SIZE",
            "timeout": "GEMINI_TIMEOUT",
            "max_retries": "GEMINI_MAX_RETRIES",
            "print_first_prompt": "GEMINI_PRINT_FIRST_PROMPT",
            "retry_delay_s": "GEMINI_RETRY_DELAY_S",
        },
    )
    _set_env_from_mapping(
        task,
        {
            "batch_size": "QWEN_BATCH_SIZE",
            "timeout": "QWEN_TIMEOUT",
            "max_retries": "QWEN_MAX_RETRIES",
            "qwen_delay_s": "QWEN_DELAY_S",
            "empty_response_retry_min_tokens": "QWEN_EMPTY_RESPONSE_RETRY_MIN_TOKENS",
            "empty_response_retry_temperature": "QWEN_EMPTY_RESPONSE_RETRY_TEMPERATURE",
            "empty_response_retry_seed_step": "QWEN_EMPTY_RESPONSE_RETRY_SEED_STEP",
            "empty_response_retry_video_first": "QWEN_EMPTY_RESPONSE_RETRY_VIDEO_FIRST",
        },
    )
    # Keep an explicit shell override available for long-running GEPA jobs while
    # retaining the YAML value as the normal default.
    if (
        task.get("reliable_max_batch_retries") is not None
        and not os.environ.get("QWEN_RELIABLE_MAX_BATCH_RETRIES", "").strip()
    ):
        os.environ["QWEN_RELIABLE_MAX_BATCH_RETRIES"] = str(
            task["reliable_max_batch_retries"]
        )
    if task.get("raise_on_empty_response") is not None:
        os.environ["QWEN_RAISE_ON_EMPTY_RESPONSE"] = _env_bool(
            task["raise_on_empty_response"]
        )
    if task.get("print_first_prompt") is not None:
        os.environ["GEMINI_PRINT_FIRST_PROMPT"] = _env_bool(task["print_first_prompt"])

    _set_env_from_mapping(
        model,
        {
            "gemini_model": "GEMINI_MODEL",
            "vertex_project": "VERTEXAI_PROJECT",
            "vertex_location": "VERTEXAI_LOCATION",
            "gemini_base_url": "GEMINI_BASE_URL",
            "gemini_api_key": "GEMINI_API_KEY",
            "qwen_model": "QWEN_MODEL",
            "qwen_base_url": "QWEN_BASE_URL",
            "qwen_api_key": "QWEN_API_KEY",
        },
    )
    qwen_base_url_override = os.environ.get("QWEN_BASE_URL_OVERRIDE", "").strip()
    if qwen_base_url_override:
        os.environ["QWEN_BASE_URL"] = qwen_base_url_override
    _set_env_from_mapping(
        sampling,
        {
            "temperature": "GEMINI_TEMPERATURE",
            "top_p": "GEMINI_TOP_P",
            "top_k": "GEMINI_TOP_K",
            "max_tokens": "GEMINI_MAX_TOKENS",
            "gemini_seed": "GEMINI_SEED",
        },
    )
    _set_env_from_mapping(
        sampling,
        {
            "temperature": "QWEN_TEMPERATURE",
            "top_p": "QWEN_TOP_P",
            "top_k": "QWEN_TOP_K",
            "max_tokens": "QWEN_MAX_TOKENS",
            "fps": "QWEN_FPS",
            "max_frames": "QWEN_MAX_FRAMES",
            "seed": "QWEN_SEED",
            "repetition_penalty": "QWEN_REPETITION_PENALTY",
        },
    )
    return config

def extract_error_info(exc: Exception) -> dict[str, Any]:
    """Extract structured details from DSPy adapter/planner exceptions."""
    info: dict[str, Any] = {
        "error_type": exc.__class__.__name__,
        "message": str(exc),
    }

    adapter_name = getattr(exc, "adapter_name", None)
    if adapter_name is not None:
        info["adapter_name"] = adapter_name

    lm_response = getattr(exc, "lm_response", None)
    if lm_response is not None:
        info["planner_lm_response"] = lm_response

    parsed_result = getattr(exc, "parsed_result", None)
    if isinstance(parsed_result, dict):
        info["parsed_output_fields"] = list(parsed_result.keys())
        info["parsed_result"] = parsed_result

    response_attempts = getattr(exc, "response_attempts", None)
    if isinstance(response_attempts, list):
        info["qwen_response_attempts"] = response_attempts

    signature = getattr(exc, "signature", None)
    output_fields = getattr(signature, "output_fields", None)
    if isinstance(output_fields, dict):
        info["expected_output_fields"] = list(output_fields.keys())

    return info

def cut_id(cut: dict[str, Any]) -> str:
    """Return the per-question sample id used for cache files and output rows."""
    value = cut.get("id")
    return str(value) if value is not None else ""


def _has_empty_qwen_tool_observation(question_data: dict[str, Any]) -> bool:
    for turn in question_data.get("turn_trace") or []:
        if not isinstance(turn, dict):
            continue
        observation = turn.get("tool_observation")
        if (
            turn.get("planner_action") == "tool"
            and str(turn.get("perception_backend") or "").strip().lower() == "qwen"
            and isinstance(observation, str)
            and not observation.strip()
        ):
            return True
    return False


def _has_parseable_final_answer(question_data: dict[str, Any]) -> bool:
    """Return whether a cached trajectory ends with a usable A-F answer."""
    for turn in question_data.get("turn_trace") or []:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("planner_action") or "").strip().lower() != "final":
            continue
        if normalize_option_letter(str(turn.get("final_answer") or "")) in {
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        }:
            return True
    return False


def load_cached_row_from_question_json(output_dir: Path | None, cut: dict[str, Any]) -> dict[str, Any] | None:
    """Load one completed per-sample JSON cache and wrap it as a result row."""
    sample_id = cut_id(cut)
    if output_dir is None or not sample_id:
        return None

    path = output_dir / f"{sample_id}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            question_data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Cached DSPy sample at {path} is unreadable, regenerating: {exc}")
        return None

    if not isinstance(question_data, dict):
        print(f"Cached DSPy sample at {path} is not a JSON object, regenerating.")
        return None

    response = str(question_data.get("response") or "").strip()
    if not response:
        print(f"Cached DSPy sample at {path} missing response, regenerating.")
        return None
    if response.startswith("[ERROR]"):
        print(f"Cached DSPy sample at {path} contains an error response, regenerating.")
        return None
    if not _has_parseable_final_answer(question_data):
        print(f"Cached DSPy sample at {path} has no parseable final answer, regenerating.")
        return None
    if _has_empty_qwen_tool_observation(question_data):
        print(
            f"Cached DSPy sample at {path} contains an empty Qwen tool observation, "
            "regenerating."
        )
        return None

    row = build_result_row(cut, response)
    row["question_data"] = question_data
    return row

def load_cached_row(cut: dict[str, Any], output_dir: Path | None) -> dict[str, Any] | None:
    """Load a cached row for one cut from its per-sample JSON file."""
    return load_cached_row_from_question_json(output_dir, cut)

def run_one(
    program: AVQADSPyReActProgram,
    cut: dict[str, Any],
    audio_caption_dir: Path | None,
    max_turns: int,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], dict[str, Any], str | None, dict[str, Any] | None]:
    """Run one cut and return response text, trace, and optional error."""
    payload = build_input_state(cut, audio_caption_dir)
    try:
        pred = program(
            question=payload["question"],
            options_json=json.dumps(payload["options"], ensure_ascii=False),
            video_path=payload["video_path"],
            audio_path=payload["audio_path"],
            video_id=payload.get("video_id"),
            question_id=payload.get("question_id"),
            video_description=payload.get("video_description"),
            max_turns=max_turns,
        )
        answer = normalize_option_letter(str(pred.answer).strip())
        response_text = f"{answer}. {str(pred.reasoning_summary).strip()}"
        turn_trace = list(pred.turn_trace) if hasattr(pred, "turn_trace") else []
        return cut, response_text, turn_trace, payload, None, None
    except Exception as exc:
        error_info = extract_error_info(exc)
        planner_calls = consume_planner_call_trace()
        if planner_calls:
            error_info["planner_calls"] = planner_calls
        return cut, "", [], payload, str(exc), error_info

def run_batch() -> None:
    """Entrypoint for batch execution."""
    args = parse_args()
    if args.inference_num_threads < 1:
        raise ValueError("--inference-num-threads must be >= 1")
    if args.inference_batch_size is not None and args.inference_batch_size < 1:
        raise ValueError("--inference-batch-size must be >= 1")
    planner_source_config = load_reasoner_config_yaml(
        args.planner_config_yaml,
        role="planner",
    )
    os.environ["PERCEPTION_MODEL"] = args.perception_model
    os.environ["DSPY_AVQA_SIGNATURE_IN_SYSTEM_PROMPT"] = _env_bool(args.signature_in_system_prompt)
    os.environ["DSPY_AVQA_CAPTION_PLACEMENT"] = args.caption_placement
    caption_cache_scope = getattr(args, "caption_cache_scope", "all")
    if args.caption_cache_dir is not None:
        args.caption_cache_dir = args.caption_cache_dir.expanduser().resolve()
    perception_source_config = load_perception_config_yaml(args.perception_config_yaml)
    if args.caption_cache_dir is None or caption_cache_scope == "first_call_only":
        captioner_source_config = load_captioner_config_yaml(args.captioner_config_yaml)
    else:
        captioner_source_config = {}
    configure_gemini_api_backend(args)
    prompt_source_config = load_prompt_config(args.prompt_yaml)
    audio_caption_dir, audio_caption_skip_reason = resolve_preloaded_audio_caption_dir(
        args.audio_caption_dir,
        caption_placement=args.caption_placement,
        ignore_audio_caption_dir=args.ignore_audio_caption_dir,
    )
    if args.caption_cache_dir is not None and audio_caption_dir is not None:
        raise ValueError(
            "--caption-cache-dir and an active --audio-caption-dir are mutually exclusive; "
            "use --ignore-audio-caption-dir for V8 cache-backed ask_caption."
        )
    apply_prompt_config_to_signatures()
    allowed_tools = resolve_allowed_tools(args.allowed_tools)
    if args.caption_cache_dir is not None and "ask_caption" not in allowed_tools:
        raise ValueError("--caption-cache-dir requires ask_caption in --allowed-tools")
    if args.caption_cache_dir is None and caption_cache_scope != "all":
        raise ValueError("--caption-cache-scope first_call_only requires --caption-cache-dir")
    cuts = read_jsonl(args.input_jsonl)
    requested_sample_ids = {
        sample_id
        for value in (args.sample_ids or [])
        for sample_id in (part.strip() for part in value.split(","))
        if sample_id
    }
    selected = (
        [cut for cut in cuts if cut_id(cut) in requested_sample_ids]
        if requested_sample_ids
        else cuts
    )
    if requested_sample_ids:
        found_sample_ids = {cut_id(cut) for cut in selected}
        missing_sample_ids = sorted(requested_sample_ids - found_sample_ids)
        if missing_sample_ids:
            raise ValueError(
                "Requested sample IDs not found in input JSONL: "
                + ", ".join(missing_sample_ids)
            )
    if args.debug:
        selected = selected[: args.debug_limit]
    caption_cache_coverage = None
    if args.caption_cache_dir is not None:
        caption_cache_coverage = validate_caption_cache_coverage(
            args.caption_cache_dir,
            [cut_id(cut) for cut in selected],
            expected_prompt=build_caption_prompt(),
            allow_entry_errors=caption_cache_missing_as_observation(),
        )

    context = AVQARuntimeContext(
        max_turns=args.max_turns,
        allowed_tools=allowed_tools,
        caption_placement=args.caption_placement,
        caption_cache_dir=args.caption_cache_dir,
        caption_cache_scope=caption_cache_scope,
    )
    program = AVQADSPyReActProgram(context=context)

    resolved_config_path = save_resolved_experiment_config(
        args.output_dir or args.output_jsonl.parent,
        {
            "mode": "inference",
            "arguments": args,
            "source_configs": {
                "planner": {
                    "path": args.planner_config_yaml,
                    "config": planner_source_config,
                },
                "perception": {
                    "path": args.perception_config_yaml,
                    "config": perception_source_config,
                },
                "captioner": {
                    "path": args.captioner_config_yaml,
                    "config": captioner_source_config,
                },
                "prompt": {
                    "path": active_prompt_yaml_path(),
                    "config": prompt_source_config,
                },
            },
            "effective": {
                "planner": planner_effective_config(context),
                "allowed_tools": context.allowed_tools,
                "caption_placement": context.caption_placement,
                "caption_cache_scope": context.caption_cache_scope,
                "max_turns": context.max_turns,
                "perception_model": args.perception_model,
                "gemini_api_backend": args.gemini_api_backend,
            },
        },
        print_config=args.print_config,
    )
    print(f"Saved resolved experiment config to {resolved_config_path}")

    print(f"Loaded cuts: {len(cuts)}")
    if args.debug:
        print(f"Debug mode: first {args.debug_limit} sample(s)")
    print(f"Selected cuts: {len(selected)}")
    print(f"Planner model: {context.planner_model}")
    print(f"Perception model: {args.perception_model}")
    for line in gemini_backend_log_lines():
        print(line)
    print(f"Prompt yaml: {active_prompt_yaml_path()}")
    print(f"Allowed tools: {','.join(context.allowed_tools)}")
    print(f"Signature in system prompt: {args.signature_in_system_prompt}")
    print(f"Caption placement: {context.caption_placement}")
    print(f"Caption cache scope: {context.caption_cache_scope}")
    if caption_cache_coverage is not None:
        print(
            "Caption cache: "
            f"{caption_cache_coverage['cache_dir']} "
            f"(validated={caption_cache_coverage['validated']}/"
            f"{caption_cache_coverage['requested']}, "
            f"missing={caption_cache_coverage['missing']}, "
            f"invalid={caption_cache_coverage['invalid']}, "
            f"manifest_sha256={caption_cache_coverage['manifest_sha256']}, "
            f"content_sha256={caption_cache_coverage['content_sha256']})"
        )
    if audio_caption_dir:
        print(f"Audio caption dir: {audio_caption_dir}")
    elif args.audio_caption_dir:
        print(f"Audio caption dir: <ignored; {audio_caption_skip_reason}>")
    else:
        print("Audio caption dir: <none; use captioner tool if needed>")
    if args.perception_config_yaml:
        print(f"Perception config yaml: {args.perception_config_yaml}")
    if args.captioner_config_yaml and (
        args.caption_cache_dir is None or caption_cache_scope == "first_call_only"
    ):
        print(f"Captioner config yaml: {args.captioner_config_yaml}")
    print(
        "Qwen perception config: "
        f"model={os.environ.get('QWEN_MODEL', '')}, "
        f"base_url={os.environ.get('QWEN_BASE_URL', '')}, "
        f"temperature={os.environ.get('QWEN_TEMPERATURE', '')}, "
        f"top_p={os.environ.get('QWEN_TOP_P', '')}, "
        f"top_k={os.environ.get('QWEN_TOP_K', '')}, "
        f"max_tokens={os.environ.get('QWEN_MAX_TOKENS', '')}, "
        f"fps={os.environ.get('QWEN_FPS', '')}, "
        f"max_frames={os.environ.get('QWEN_MAX_FRAMES', '')}, "
        f"seed={os.environ.get('QWEN_SEED', '')}"
    )
    print(
        "Qwen captioner config: "
        f"model={os.environ.get('CAPTIONER_QWEN_MODEL', '')}, "
        f"base_url={os.environ.get('CAPTIONER_QWEN_BASE_URL', '')}, "
        f"temperature={os.environ.get('CAPTIONER_QWEN_TEMPERATURE', '')}, "
        f"top_p={os.environ.get('CAPTIONER_QWEN_TOP_P', '')}, "
        f"top_k={os.environ.get('CAPTIONER_QWEN_TOP_K', '')}, "
        f"max_tokens={os.environ.get('CAPTIONER_QWEN_MAX_TOKENS', '')}, "
        f"fps={os.environ.get('CAPTIONER_QWEN_FPS', '')}, "
        f"max_frames={os.environ.get('CAPTIONER_QWEN_MAX_FRAMES', '')}, "
        f"seed={os.environ.get('CAPTIONER_QWEN_SEED', '')}"
    )
    if (os.environ.get("CAPTIONER_MODEL") or os.environ.get("CAPTION_MODEL", "")).strip().lower() == "gemini":
        print(
            "Gemini captioner config: "
            f"model={os.environ.get('CAPTIONER_GEMINI_MODEL', '')}, "
            f"base_url={os.environ.get('CAPTIONER_GEMINI_BASE_URL', '')}, "
            f"api_key={'***set***' if os.environ.get('CAPTIONER_GEMINI_API_KEY') else ''}, "
            f"video_only={os.environ.get('CAPTIONER_GEMINI_VIDEO_ONLY', '')}, "
            f"timeout={os.environ.get('CAPTIONER_GEMINI_TIMEOUT', '')}, "
            f"max_retries={os.environ.get('CAPTIONER_GEMINI_MAX_RETRIES', '')}, "
            f"retry_delay_s={os.environ.get('CAPTIONER_GEMINI_RETRY_DELAY_S', '')}, "
            f"temperature={os.environ.get('CAPTIONER_GEMINI_TEMPERATURE', '')}, "
            f"top_p={os.environ.get('CAPTIONER_GEMINI_TOP_P', '')}, "
            f"top_k={os.environ.get('CAPTIONER_GEMINI_TOP_K', '')}, "
            f"max_tokens={os.environ.get('CAPTIONER_GEMINI_MAX_TOKENS', '')}, "
            f"seed={os.environ.get('CAPTIONER_GEMINI_SEED', '')}, "
            f"print_first_prompt={os.environ.get('CAPTIONER_GEMINI_PRINT_FIRST_PROMPT', '')}, "
            f"batch_size={os.environ.get('CAPTIONER_GEMINI_BATCH_SIZE', '')}"
        )
    if args.perception_model == "gemini":
        print(
            "Gemini config: "
            f"video_only={os.environ.get('GEMINI_VIDEO_ONLY', '')}, "
            f"timeout={os.environ.get('GEMINI_TIMEOUT', '')}, "
            f"max_retries={os.environ.get('GEMINI_MAX_RETRIES', '')}, "
            f"retry_delay_s={os.environ.get('GEMINI_RETRY_DELAY_S', '')}, "
            f"temperature={os.environ.get('GEMINI_TEMPERATURE', '')}, "
            f"top_p={os.environ.get('GEMINI_TOP_P', '')}, "
            f"top_k={os.environ.get('GEMINI_TOP_K', '')}, "
            f"max_tokens={os.environ.get('GEMINI_MAX_TOKENS', '')}, "
            f"seed={os.environ.get('GEMINI_SEED', '')}, "
            f"print_first_prompt={os.environ.get('GEMINI_PRINT_FIRST_PROMPT', '')}, "
            f"batch_size={os.environ.get('GEMINI_BATCH_SIZE', '')}"
        )

    if args.perception_model == "qwen":
        print(
            "Qwen config: "
            f"model={os.environ.get('QWEN_MODEL', '')}, "
            f"base_url={os.environ.get('QWEN_BASE_URL', '')}, "
            f"video_only={os.environ.get('QWEN_VIDEO_ONLY', '')}, "
            f"video_first={os.environ.get('QWEN_VIDEO_FIRST', '')}, "
            f"use_audio_in_video={os.environ.get('QWEN_USE_AUDIO_IN_VIDEO', '')}, "
            f"enable_thinking={os.environ.get('QWEN_ENABLE_THINKING', '')}, "
            f"timeout={os.environ.get('QWEN_TIMEOUT', '')}, "
            f"max_retries={os.environ.get('QWEN_MAX_RETRIES', '')}, "
            f"delay_s={os.environ.get('QWEN_DELAY_S', '')}, "
            f"empty_retry_min_tokens={os.environ.get('QWEN_EMPTY_RESPONSE_RETRY_MIN_TOKENS', '')}, "
            f"empty_retry_temperature={os.environ.get('QWEN_EMPTY_RESPONSE_RETRY_TEMPERATURE', '')}, "
            f"empty_retry_seed_step={os.environ.get('QWEN_EMPTY_RESPONSE_RETRY_SEED_STEP', '')}, "
            f"raise_on_empty={os.environ.get('QWEN_RAISE_ON_EMPTY_RESPONSE', '')}, "
            f"temperature={os.environ.get('QWEN_TEMPERATURE', '')}, "
            f"top_p={os.environ.get('QWEN_TOP_P', '')}, "
            f"top_k={os.environ.get('QWEN_TOP_K', '')}, "
            f"max_tokens={os.environ.get('QWEN_MAX_TOKENS', '')}, "
            f"seed={os.environ.get('QWEN_SEED', '')}, "
            f"fps={os.environ.get('QWEN_FPS', '')}, "
            f"max_frames={os.environ.get('QWEN_MAX_FRAMES', '')}, "
            f"repetition_penalty={os.environ.get('QWEN_REPETITION_PENALTY', '')}"
        )

    rows_by_sample_id: dict[str, dict[str, Any]] = {}
    remaining: list[dict[str, Any]] = []
    for cut in selected:
        cached_row = load_cached_row(cut, args.output_dir)
        if cached_row is None:
            remaining.append(cut)
            continue
        rows_by_sample_id[cut_id(cut)] = cached_row

    print(f"Cached samples: {len(selected) - len(remaining)}")
    print(f"Remaining samples: {len(remaining)}")
    reliable_qwen = ReliableQwenExecutor()
    inference_batch_size = args.inference_batch_size or args.inference_num_threads
    print(
        "Inference rollout config: "
        f"primary_threads={args.inference_num_threads}, "
        f"reliable_batch_size={inference_batch_size}, "
        f"qwen_batch_retries={reliable_qwen.max_batch_retries}"
    )

    for batch_start in range(0, len(remaining), inference_batch_size):
        rollout_batch = remaining[batch_start : batch_start + inference_batch_size]
        recovered = reliable_qwen.run_batch(
            rollout_batch,
            run_item=lambda cut: run_one(
                program=program,
                cut=cut,
                audio_caption_dir=audio_caption_dir,
                max_turns=args.max_turns,
            ),
            has_empty_qwen_response=lambda result: turn_trace_has_empty_qwen_observation(
                result[2]
            ),
            max_workers=args.inference_num_threads,
        )
        exhausted_indices = set(recovered.exhausted_indices)
        if exhausted_indices:
            print(
                "Qwen recovery exhausted for "
                f"{len(exhausted_indices)} inference sample(s) in batch "
                f"{batch_start // inference_batch_size + 1}; recording [ERROR] rows."
            )

        for batch_index, result in enumerate(recovered.results):
            cut_item, response_text, turn_trace, payload, error, error_info = result
            row = build_result_row(cut_item, response_text)
            if error is None:
                if batch_index in exhausted_indices:
                    mark_empty_qwen_turn_trace(
                        turn_trace,
                        attempted_profiles=recovered.attempted_profiles,
                    )
                    row["question_data"]["response"] = (
                        "[ERROR] Qwen returned no visible content after reliable recovery"
                    )
                row["question_data"]["turn_trace"] = turn_trace
            else:
                row["question_data"]["response"] = f"[ERROR] {error}"
                if error_info:
                    row["question_data"]["planner_error"] = error_info
                row["question_data"]["turn_trace"] = [
                    {
                        "turn_id": 1,
                        "planner_action": "error",
                        "tool_name": None,
                        "tool_args": {
                            "video_path": payload.get("video_path"),
                            "audio_path": payload.get("audio_path"),
                        },
                        "tool_observation": None,
                        "final_answer": None,
                        "tool_error": error,
                        "planner_lm_response": (error_info or {}).get("planner_lm_response"),
                        "planner_calls": (error_info or {}).get("planner_calls", []),
                        "planner_error": error_info,
                    }
                ]
            maybe_dump_question_data(args.output_dir, row)
            rows_by_sample_id[cut_id(cut_item)] = row

    rows = [rows_by_sample_id[cut_id(cut)] for cut in selected if cut_id(cut) in rows_by_sample_id]
    write_results_jsonl(rows, args.output_jsonl)
    print(f"Wrote {len(rows)} rows to {args.output_jsonl}")
