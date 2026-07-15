"""Batch runner for DSPy AVQA program."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .deepseek_dspy_lm import consume_planner_call_trace
from .context import AVQARuntimeContext, CAPTION_PLACEMENT_CHOICES, normalize_caption_placement, resolve_allowed_tools
from .data import (
    build_input_state,
    build_result_row,
    maybe_dump_question_data,
    read_jsonl,
    write_results_jsonl,
)
from .program import AVQADSPyReActProgram, normalize_option_letter
from .prompt_config import active_prompt_yaml_path, load_prompt_config
from .signatures import apply_prompt_config_to_signatures


GEMINI_API_BACKENDS = ("legacy", "dspy")


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
    parser.add_argument("--audio-caption-dir", type=Path, default=None)
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
    captioner_is_gemini = _selected_captioner_model() == "gemini"
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


def _load_qwen_config_yaml(path: Path | None, *, prefix: str, preserve_base_url_override: bool) -> dict[str, Any]:
    """Load Qwen runtime parameters into env vars, optionally under a prefix."""
    base_url_key = f"{prefix}_BASE_URL"
    qwen_base_url_override = os.environ.get(base_url_key) if preserve_base_url_override else None
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
        },
    )
    _set_env_from_mapping(
        model,
        {
            "qwen_model": f"{prefix}_MODEL",
            "qwen_base_url": f"{prefix}_BASE_URL",
            "qwen_api_key": f"{prefix}_API_KEY",
        },
    )
    if qwen_base_url_override:
        os.environ[base_url_key] = qwen_base_url_override

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
    return _load_qwen_config_yaml(path, prefix="CAPTIONER_QWEN", preserve_base_url_override=True)


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
    qwen_base_url_override = os.environ.get("QWEN_BASE_URL")
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
        },
    )
    if task.get("print_first_prompt") is not None:
        os.environ["GEMINI_PRINT_FIRST_PROMPT"] = _env_bool(task["print_first_prompt"])

    _set_env_from_mapping(
        model,
        {
            "gemini_model": "GEMINI_MODEL",
            "gemini_base_url": "GEMINI_BASE_URL",
            "gemini_api_key": "GEMINI_API_KEY",
            "qwen_model": "QWEN_MODEL",
            "qwen_base_url": "QWEN_BASE_URL",
            "qwen_api_key": "QWEN_API_KEY",
        },
    )
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

    signature = getattr(exc, "signature", None)
    output_fields = getattr(signature, "output_fields", None)
    if isinstance(output_fields, dict):
        info["expected_output_fields"] = list(output_fields.keys())

    return info

def cut_id(cut: dict[str, Any]) -> str:
    """Return the per-question sample id used for cache files and output rows."""
    value = cut.get("id")
    return str(value) if value is not None else ""

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

    if not isinstance(question_data, dict) or not str(question_data.get("response") or "").strip():
        print(f"Cached DSPy sample at {path} missing response, regenerating.")
        return None

    row = build_result_row(cut, str(question_data.get("response") or ""))
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
    os.environ["PERCEPTION_MODEL"] = args.perception_model
    os.environ["DSPY_AVQA_SIGNATURE_IN_SYSTEM_PROMPT"] = _env_bool(args.signature_in_system_prompt)
    os.environ["DSPY_AVQA_CAPTION_PLACEMENT"] = args.caption_placement
    load_perception_config_yaml(args.perception_config_yaml)
    load_captioner_config_yaml(args.captioner_config_yaml)
    configure_gemini_api_backend(args)
    load_prompt_config(args.prompt_yaml)
    apply_prompt_config_to_signatures()
    allowed_tools = resolve_allowed_tools(args.allowed_tools)
    cuts = read_jsonl(args.input_jsonl)
    selected = cuts[: args.debug_limit] if args.debug else cuts

    context = AVQARuntimeContext(
        max_turns=args.max_turns,
        allowed_tools=allowed_tools,
        caption_placement=args.caption_placement,
    )
    program = AVQADSPyReActProgram(context=context)

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
    if args.audio_caption_dir:
        print(f"Audio caption dir: {args.audio_caption_dir}")
    else:
        print("Audio caption dir: <none; use captioner tool if needed>")
    if args.perception_config_yaml:
        print(f"Perception config yaml: {args.perception_config_yaml}")
    if args.captioner_config_yaml:
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
            f"enable_thinking={os.environ.get('QWEN_ENABLE_THINKING', '')}, "
            f"timeout={os.environ.get('QWEN_TIMEOUT', '')}, "
            f"max_retries={os.environ.get('QWEN_MAX_RETRIES', '')}, "
            f"delay_s={os.environ.get('QWEN_DELAY_S', '')}, "
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

    for cut in remaining:
        cut_item, response_text, turn_trace, payload, error, error_info = run_one(
            program=program,
            cut=cut,
            audio_caption_dir=args.audio_caption_dir,
            max_turns=args.max_turns,
        )
        row = build_result_row(cut_item, response_text)
        if error is None:
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

