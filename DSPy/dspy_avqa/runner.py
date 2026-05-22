"""Batch runner for DSPy AVQA program."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .deepseek_dspy_lm import consume_planner_call_trace
from .context import AVQARuntimeContext, resolve_allowed_tools
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Batch run DSPy AVQA ReAct over cut JSONL.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audio-caption-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--debug", action="store_true", help="Run only the first debug-limit samples.")
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=int(os.environ.get("DEBUG_LIMIT", "4")),
        help="Number of leading samples to run when --debug is enabled.",
    )
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--perception-config-yaml",
        type=Path,
        default=None,
        help="Optional YAML file with Gemini perception runtime/sampling parameters.",
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


def load_perception_config_yaml(path: Path | None) -> dict[str, Any]:
    """Load Gemini perception parameters from a baseline-style YAML config."""
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Perception config must be a YAML mapping: {path}")

    task = config.get("task") or {}
    sampling = config.get("sampling_params") or {}
    if not isinstance(task, dict):
        raise ValueError(f"Perception config task section must be a mapping: {path}")
    if not isinstance(sampling, dict):
        raise ValueError(f"Perception config sampling_params section must be a mapping: {path}")

    if task.get("video_only") is not None:
        os.environ["GEMINI_VIDEO_ONLY"] = _env_bool(task["video_only"])
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
    if task.get("print_first_prompt") is not None:
        os.environ["GEMINI_PRINT_FIRST_PROMPT"] = _env_bool(task["print_first_prompt"])

    _set_env_from_mapping(
        sampling,
        {
            "temperature": "GEMINI_TEMPERATURE",
            "top_p": "GEMINI_TOP_P",
            "top_k": "GEMINI_TOP_K",
            "max_tokens": "GEMINI_MAX_TOKENS",
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
    audio_caption_dir: Path,
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
    load_perception_config_yaml(args.perception_config_yaml)
    load_prompt_config(args.prompt_yaml)
    apply_prompt_config_to_signatures()
    allowed_tools = resolve_allowed_tools(args.allowed_tools)
    cuts = read_jsonl(args.input_jsonl)
    selected = cuts[: args.debug_limit] if args.debug else cuts

    context = AVQARuntimeContext(max_turns=args.max_turns, allowed_tools=allowed_tools)
    program = AVQADSPyReActProgram(context=context)

    print(f"Loaded cuts: {len(cuts)}")
    if args.debug:
        print(f"Debug mode: first {args.debug_limit} sample(s)")
    print(f"Selected cuts: {len(selected)}")
    print(f"Planner model: {context.planner_model}")
    print(f"Perception model: {args.perception_model}")
    print(f"Prompt yaml: {active_prompt_yaml_path()}")
    print(f"Allowed tools: {','.join(context.allowed_tools)}")
    if args.perception_config_yaml:
        print(f"Perception config yaml: {args.perception_config_yaml}")
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
            f"print_first_prompt={os.environ.get('GEMINI_PRINT_FIRST_PROMPT', '')}, "
            f"batch_size={os.environ.get('GEMINI_BATCH_SIZE', '')}"
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

