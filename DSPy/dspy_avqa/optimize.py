"""Optimization hooks, dataset helpers, and CLI for DSPy."""

from __future__ import annotations

import argparse
import copy
import fcntl
import functools
import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import dspy
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
    lm_effective_config,
    load_reasoner_config_yaml,
    planner_effective_config,
    save_resolved_experiment_config,
)
from .data import build_input_state, build_result_row, maybe_dump_question_data, read_jsonl, write_results_jsonl
from .program import AVQADSPyReActProgram, normalize_option_letter
from .prompt_config import active_prompt_yaml_path, load_prompt_config, prompt_config, prompt_overrides, prompt_value
from .reliable_qwen import (
    ReliableQwenExecutor,
    QwenValidationScore,
    mark_empty_qwen_prediction,
    mark_empty_qwen_turn_trace,
    prediction_has_empty_qwen_observation,
    turn_trace_has_empty_qwen_observation,
)
from .runner import (
    add_gemini_backend_args,
    configure_gemini_api_backend,
    cut_id,
    extract_error_info,
    gemini_backend_log_lines,
    load_cached_row_from_question_json,
    load_captioner_config_yaml,
    load_perception_config_yaml,
    resolve_preloaded_audio_caption_dir,
)
from .signatures import apply_prompt_config_to_signatures
from .tools import build_caption_prompt

logger = logging.getLogger(__name__)

GEPA_OPTIMIZER_LOG_FILENAME = "gepa_optimizer.log"
GEPA_REFLECTIVE_DATASET_DIRNAME = "reflective_datasets"
GEPA_REFLECTIVE_DATASET_STATE_FILENAME = "capture_state.json"
DEFAULT_GEPA_REFLECTIVE_DATASET_SAVE_INTERVAL = 50
MAX_GEPA_REFLECTION_FEEDBACK_CHARS = 2000
MAX_GEPA_REFLECTION_ERROR_CHARS = 2000

GEPA_PRIVILEGED_CAPTION_FILES = (
    ("gepa_privileged_av_alignment_captions", "av_alignment_captions.txt"),
    ("gepa_privileged_video_consistent_captions", "video_consistent_captions.txt"),
    ("gepa_privileged_audio_revised_captions", "audio_revised_captions.txt"),
)
GEPA_DENSIFIED_LABEL_FIELDS = (
    ("gepa_privileged_key_evidence", "key_evidence"),
)
GEPA_DENSIFIED_OPTIONAL_SOURCE_KEYS = frozenset({"ideal_perception_target"})
GEPA_CAPTIONER_TARGET_PREFIX = "captioner."
GEPA_PLANNER_TARGET_PREFIX = "planner."
GEPA_CAPTION_SUPERVISION_CHOICES = ("auto", "none", "privileged")


def _caption_cache_allows_missing() -> bool:
    """Read the missing-caption behavior from the active prompt config."""
    captioner_config = prompt_config().get("captioner") or {}
    if not isinstance(captioner_config, dict):
        raise ValueError("Prompt config captioner section must be a mapping")
    value = captioner_config.get("cache_missing_as_observation", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


GEPA_CAPTIONER_REFLECTION_PROMPT_TEMPLATE = """I provided an assistant with the following captioning instruction:
```
<curr_instructions>
```

Below are optimization examples containing runtime trajectories, evaluation feedback, and any optimization-only privileged supervision if enabled:
```
<inputs_outputs_feedback>
```

This instruction is used by the ask_caption tool to obtain a factual audio-visual caption. It must not solve the multiple-choice question, choose an option, output a final answer letter, plan tool calls, or instruct another model.

Write an improved captioning instruction. For each example with densified supervision, compare the caption step and its response against the matching Key evidence. Use Key evidence to judge whether the caption step captured the decisive observable facts, then infer reusable improvements that will generalize to unseen videos.

All privileged supervision is unavailable at runtime. Do not copy or encode any example-specific answer, option, timestamp, person, object, scene, event, wording, caption sentence, or other dataset fact into the new instruction. Do not mention privileged supervision or assume access to it.

Provide only the new captioner instruction within ``` blocks."""

GEPA_PLANNER_REFLECTION_PROMPT_TEMPLATE = """I provided an assistant with the following planner workflow instruction:
```
<curr_instructions>
```

Below are optimization examples containing runtime trajectories, evaluation feedback, and optimization-only densified supervision when available:
```
<inputs_outputs_feedback>
```

Your task is to write a new instruction for the assistant.

Infer the task format and the behavior required to solve it.

Read the assistant responses and feedback. For each example with densified supervision, compare the rollout against its matching Key evidence. Use Key evidence to judge whether caption/perception steps obtained and preserved the decisive facts. Identify recurring mistakes and general strategies that could improve performance on unseen examples.

The goal is to improve the assistant's planning and tool-use policy, not to learn facts from the example videos. The privileged supervision is unavailable at runtime.

Turn example-specific feedback into general rules. Do not copy or include any specific answer, option, timestamp, person, object, scene, event, caption text, tool observation, or other example-specific detail.

Preserve useful task-level information, such as:
- how to identify the most important missing evidence,
- how to formulate one targeted perceptual question,
- how to combine the caption and perceptual observation before answering,

Only include strategies that are likely to help across different unseen inputs. Keep the new instruction concise, clear, and actionable.

Provide only the new instruction within a single ``` block."""

GEPA_REFLECTION_TEMPERATURE_ENV = "GEPA_REFLECTION_TEMPERATURE"
GEPA_REFLECTION_MODEL_ENV = "GEPA_REFLECTION_MODEL"
GEPA_REFLECTION_REASONING_EFFORT_ENV = "GEPA_REFLECTION_REASONING_EFFORT"
GEPA_REFLECTION_MAX_TOKENS_ENV = "GEPA_REFLECTION_MAX_TOKENS"
GEPA_REFLECTION_SEED_ENV = "GEPA_REFLECTION_SEED"
GEPA_REFLECTION_API_KEY_ENV = "GEPA_REFLECTION_API_KEY"
GEPA_REFLECTION_API_BASE_ENV = "GEPA_REFLECTION_API_BASE"
GEPA_REFLECTION_THINKING_MODE_ENV = "GEPA_REFLECTION_THINKING_MODE"
GEPA_REFLECTION_TEMPLATE_YAML_ENV = "GEPA_REFLECTION_TEMPLATE_YAML"
GEPA_REFLECTION_TEMPLATE_VERSION_ENV = "GEPA_REFLECTION_TEMPLATE_VERSION"
GEPA_REFLECTION_TEMPLATE_VERSION_AUTO = "auto"
GEPA_REFLECTION_TEMPLATE_VERSION_ORIGINAL = "original"
GEPA_REFLECTION_TEMPLATE_VERSION_DENSIFIED_KEY_EVIDENCE = "densified_key_evidence"
GEPA_REFLECTION_TEMPLATE_VERSION_ALIASES = {
    "g0": GEPA_REFLECTION_TEMPLATE_VERSION_ORIGINAL,
    "no_densified": GEPA_REFLECTION_TEMPLATE_VERSION_ORIGINAL,
    "none": GEPA_REFLECTION_TEMPLATE_VERSION_ORIGINAL,
    "original": GEPA_REFLECTION_TEMPLATE_VERSION_ORIGINAL,
    "g3": GEPA_REFLECTION_TEMPLATE_VERSION_DENSIFIED_KEY_EVIDENCE,
    "key_evidence": GEPA_REFLECTION_TEMPLATE_VERSION_DENSIFIED_KEY_EVIDENCE,
    "densified": GEPA_REFLECTION_TEMPLATE_VERSION_DENSIFIED_KEY_EVIDENCE,
    "densified_key_evidence": GEPA_REFLECTION_TEMPLATE_VERSION_DENSIFIED_KEY_EVIDENCE,
}
GEPA_REFLECTION_TEMPLATE_YAML_KEYS = (
    "GEPA_CAPTIONER_REFLECTION_PROMPT_TEMPLATE",
    "GEPA_PLANNER_REFLECTION_PROMPT_TEMPLATE",
)
GEPA_REFLECTION_TEMPLATE_DEFAULT_PATH = (
    Path(__file__).resolve().parent / "yamls" / "DSPy" / "reflection_template.yaml"
)
GEPA_REFLECTION_TEMPLATE_LEGACY_PATH = (
    Path(__file__).resolve().parent / "yamls" / "reflection_template.yaml"
)

_current_gepa_reflection_template_info: dict[str, Any] = {
    "yaml": None,
    "requested_version": "hardcoded",
    "resolved_version": "hardcoded_densified_key_evidence",
}


def _log_gepa_stage_exceptions(stage: str):
    """Log a traceback before GEPA converts a reflection failure into no proposal."""
    def decorator(method: Any) -> Any:
        @functools.wraps(method)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return method(*args, **kwargs)
            except Exception as exc:
                logger.exception("GEPA %s failed", stage)
                if "missing credentials" in str(exc).lower():
                    print(
                        "FATAL: GEPA reflection LM is missing credentials; stopping optimization. "
                        f"Original error: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    # GEPA catches ordinary Exception instances and turns them into
                    # skipped proposals.  SystemExit deliberately bypasses that path.
                    raise SystemExit(1) from exc
                print(
                    f"WARNING: GEPA {stage} failed; skipping this proposal. "
                    f"Original error: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

        return wrapped

    return decorator


def _start_gepa_file_logging(log_dir: str | None) -> tuple[logging.FileHandler | None, int]:
    """Route this module and GEPA's LoggerAdapter messages to the run directory."""
    previous_level = logger.level
    if log_dir is None:
        return None, previous_level

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(
        log_path / GEPA_OPTIMIZER_LOG_FILENAME,
        mode="a",
        encoding="utf-8",
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("GEPA optimizer logging started")
    return handler, previous_level


def _stop_gepa_file_logging(handler: logging.FileHandler | None, previous_level: int) -> None:
    if handler is None:
        return
    logger.info("GEPA optimizer logging finished")
    handler.flush()
    logger.removeHandler(handler)
    handler.close()
    logger.setLevel(previous_level)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
    os.replace(temporary_path, path)


class ReflectiveDatasetSnapshotCallback:
    """Persist a sparse, resume-aware sample of GEPA reflective datasets."""

    def __init__(
        self,
        log_dir: str | Path,
        save_interval: int = DEFAULT_GEPA_REFLECTIVE_DATASET_SAVE_INTERVAL,
    ):
        if save_interval < 0:
            raise ValueError("reflective_dataset_save_interval must be >= 0")
        self.save_interval = save_interval
        self.output_dir = Path(log_dir) / GEPA_REFLECTIVE_DATASET_DIRNAME
        self.state_path = self.output_dir / GEPA_REFLECTIVE_DATASET_STATE_FILENAME
        self.datasets_seen = 0
        if self.save_interval > 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.datasets_seen = self._load_datasets_seen()

    def _load_datasets_seen(self) -> int:
        if not self.state_path.exists():
            return 0
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            datasets_seen = int(payload.get("datasets_seen", 0))
            if datasets_seen < 0:
                raise ValueError("datasets_seen must be >= 0")
            return datasets_seen
        except Exception:
            logger.exception("Failed to load reflective dataset capture state: %s", self.state_path)
            return 0

    def _should_save(self, dataset_index: int) -> bool:
        return dataset_index == 1 or dataset_index % self.save_interval == 0

    def on_reflective_dataset_built(self, event: Any) -> None:
        if self.save_interval == 0:
            return
        try:
            dataset_index = self.datasets_seen + 1
            iteration = int(event["iteration"])
            candidate_idx = int(event["candidate_idx"])
            if self._should_save(dataset_index):
                snapshot_path = self.output_dir / (
                    f"reflective_dataset_{dataset_index:06d}_"
                    f"iteration_{iteration:06d}_candidate_{candidate_idx:06d}.json"
                )
                _write_json_atomic(snapshot_path, {
                    "reflective_dataset_index": dataset_index,
                    "gepa_iteration": iteration,
                    "candidate_index": candidate_idx,
                    "components": event.get("components", []),
                    "dataset": event.get("dataset", {}),
                })
                logger.info("Saved GEPA reflective dataset snapshot: %s", snapshot_path)

            self.datasets_seen = dataset_index
            _write_json_atomic(self.state_path, {"datasets_seen": self.datasets_seen})
        except Exception:
            logger.exception("Failed to persist GEPA reflective dataset snapshot")


def _env_flag_value(name: str, default: str = "false") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _env_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _optional_env_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip() or value.strip().upper() == "EMPTY":
        return None
    return value.strip()


def _normalize_openai_model_name(model: str) -> str:
    return model if "/" in model else f"openai/{model}"


def _validate_reasoning_effort(value: str, *, env_name: str) -> str:
    effort = value.lower()
    valid_efforts = {"none", "minimal", "low", "medium", "high", "xhigh"}
    if effort not in valid_efforts:
        raise ValueError(
            f"{env_name}={value!r} is invalid; expected one of {sorted(valid_efforts)}"
        )
    return effort


def _deepseek_thinking_extra_body(value: str, *, env_name: str) -> dict[str, Any] | None:
    mode = value.strip().lower().replace("_", "-")
    if mode in {"auto", "default", "none", "unset"}:
        return None
    if mode in {"enabled", "enable", "thinking", "think", "on", "true", "1"}:
        return {"thinking": {"type": "enabled"}}
    if mode in {"disabled", "disable", "non-thinking", "nonthinking", "no-thinking", "off", "false", "0"}:
        return {"thinking": {"type": "disabled"}}
    raise ValueError(
        f"{env_name}={value!r} is invalid; expected enabled, disabled, or auto"
    )


def build_gepa_reflection_lm(planner_lm: Any) -> Any:
    """Return a GEPA reflection LM with optional model, effort, and temperature overrides."""
    overrides: dict[str, Any] = {}
    raw_model = _optional_env_value(GEPA_REFLECTION_MODEL_ENV)
    raw_effort = _optional_env_value(GEPA_REFLECTION_REASONING_EFFORT_ENV)
    raw_api_key = _optional_env_value(GEPA_REFLECTION_API_KEY_ENV)
    raw_api_base = _optional_env_value(GEPA_REFLECTION_API_BASE_ENV)
    raw_thinking_mode = _optional_env_value(GEPA_REFLECTION_THINKING_MODE_ENV)
    raw_max_tokens = _optional_env_value(GEPA_REFLECTION_MAX_TOKENS_ENV)
    raw_seed = _optional_env_value(GEPA_REFLECTION_SEED_ENV)
    raw_temperature = os.environ.get(GEPA_REFLECTION_TEMPERATURE_ENV)
    if raw_model is not None:
        overrides["model"] = _normalize_openai_model_name(raw_model)
    if raw_effort is not None:
        overrides["reasoning_effort"] = _validate_reasoning_effort(
            raw_effort,
            env_name=GEPA_REFLECTION_REASONING_EFFORT_ENV,
        )
    if raw_api_key is not None:
        overrides["api_key"] = raw_api_key
    if raw_api_base is not None:
        overrides["api_base"] = raw_api_base
    if raw_thinking_mode is not None:
        reflection_model = str(overrides.get("model", getattr(planner_lm, "model", ""))).lower()
        if "deepseek" not in reflection_model:
            raise ValueError(
                f"{GEPA_REFLECTION_THINKING_MODE_ENV} is only supported for DeepSeek reflection models; "
                f"got {reflection_model!r}"
            )
        thinking_extra_body = _deepseek_thinking_extra_body(
            raw_thinking_mode,
            env_name=GEPA_REFLECTION_THINKING_MODE_ENV,
        )
        if thinking_extra_body is not None:
            overrides["extra_body"] = thinking_extra_body
    for raw_value, key, env_name in (
        (raw_max_tokens, "max_tokens", GEPA_REFLECTION_MAX_TOKENS_ENV),
        (raw_seed, "seed", GEPA_REFLECTION_SEED_ENV),
    ):
        if raw_value is not None:
            try:
                overrides[key] = int(raw_value)
            except ValueError as exc:
                raise ValueError(f"{env_name} must be an integer, got {raw_value!r}") from exc

    planner_kwargs = getattr(planner_lm, "kwargs", None)
    inherited_api_base = isinstance(planner_kwargs, dict) and bool(planner_kwargs.get("api_base"))
    if raw_model is not None and inherited_api_base:
        if raw_api_base is None:
            overrides["api_base"] = None
        if raw_api_key is None:
            overrides["api_key"] = None

    if raw_temperature is not None and raw_temperature.strip():
        try:
            overrides["temperature"] = float(raw_temperature)
        except ValueError as exc:
            raise ValueError(
                f"{GEPA_REFLECTION_TEMPERATURE_ENV} must be a float, got {raw_temperature!r}"
            ) from exc
    elif overrides:
        # A distinct reflection LM should omit temperature unless explicitly set,
        # rather than inheriting the planner's sampling temperature.
        overrides["temperature"] = None

    if not overrides:
        return planner_lm

    reflection_lm = planner_lm.copy(**overrides)
    if raw_model is not None:
        reflection_lm.model = overrides["model"]
    reflection_kwargs = getattr(reflection_lm, "kwargs", None)
    if isinstance(reflection_kwargs, dict):
        reflection_kwargs.pop("model", None)
        if raw_model is not None and overrides["model"].lower().startswith("openai/gpt-"):
            reflection_kwargs.pop("extra_body", None)
        for key in ("api_base", "api_key", "temperature"):
            if key in overrides and overrides.get(key) is None:
                reflection_kwargs.pop(key, None)
    return reflection_lm


def avqa_metric(example: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> float:
    """Exact-match metric on option letter."""
    gold = normalize_option_letter(str(example.answer))
    got = normalize_option_letter(str(pred.answer))
    return 1.0 if gold == got else 0.0


def avqa_gepa_feedback_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> dspy.Prediction:
    """Exact-match metric with target-aware, non-duplicative GEPA feedback."""
    gold = normalize_option_letter(str(example.answer))
    got = normalize_option_letter(str(getattr(pred, "answer", "")))
    score = 1.0 if gold == got else 0.0
    target_path = str(pred_name or "")
    result_label = "Correct" if score == 1.0 else "Incorrect"
    if _is_captioner_target(target_path):
        feedback_parts = [
            f"Downstream QA {result_label.lower()} (score={score:g}); "
            f"gold={gold or '<empty>'}, predicted={got or '<empty>'}.",
        ]
        if score < 1.0:
            feedback_parts.append(
                "Revise only the captioner instruction so ask_caption returns a factual, "
                "timestamped audio-visual caption with the observable evidence needed by the "
                "planner. Do not make the captioner solve the multiple-choice question, "
                "plan tool calls, or output an option letter."
            )
        else:
            feedback_parts.append(
                "This trajectory is correct; preserve the captioning behavior that supplied "
                "useful factual evidence, while keeping the captioner as a caption-only tool."
            )
    else:
        feedback_parts = [
            f"{result_label} (score={score:g}); "
            f"gold={gold or '<empty>'}, predicted={got or '<empty>'}.",
        ]
        if score < 1.0:
            feedback_parts.append(
                "Revise the planner instruction so it gathers the right audio/video evidence, "
                "uses tools only when helpful, and returns exactly one option letter."
            )
        else:
            feedback_parts.append(
                "This trajectory is correct; preserve the behavior that led to this answer."
            )
    return dspy.Prediction(score=score, feedback="\n".join(feedback_parts))


def _bounded_reflection_text(value: Any, *, limit: int) -> str:
    """Normalize diagnostic text and bound only low-value reasoning/error payloads."""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[truncated for GEPA reflection]"


def _reflection_options(value: Any) -> Any:
    """Expose options as a compact structure when the source is JSON."""
    if not isinstance(value, str):
        return _json_safe(value)
    try:
        return _json_safe(json.loads(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return value.strip()


def _planner_tool_question(turn: dict[str, Any]) -> str:
    args = turn.get("tool_args")
    if not isinstance(args, dict):
        return ""
    for key in ("perceptual_question", "caption_instruction", "question", "query", "prompt"):
        value = args.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _planner_call_texts(turn: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Extract complete planner outputs while discarding request/history metadata."""
    responses: list[str] = []
    reasonings: list[str] = []
    planner_calls = turn.get("planner_calls")
    if not isinstance(planner_calls, dict):
        return responses, reasonings
    for calls in planner_calls.values():
        if not isinstance(calls, (list, tuple)):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            response_text = str(call.get("response_text") or "").strip()
            reasoning_content = str(call.get("reasoning_content") or "").strip()
            if response_text:
                responses.append(response_text)
            if reasoning_content:
                reasonings.append(reasoning_content)
    return responses, reasonings


def _compact_reflection_trace(prediction: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep reflection-relevant execution evidence without LM messages or token metadata."""
    raw_trace = getattr(prediction, "turn_trace", None) or []
    turns: list[dict[str, Any]] = []
    errors: list[str] = []

    for turn_index, raw_turn in enumerate(raw_trace, start=1):
        if not isinstance(raw_turn, dict):
            errors.append(
                f"turn {turn_index}: invalid trace record "
                f"{_bounded_reflection_text(raw_turn, limit=MAX_GEPA_REFLECTION_ERROR_CHARS)}"
            )
            continue

        turn_id = raw_turn.get("turn_id", turn_index)
        action = str(raw_turn.get("planner_action") or "unknown").strip()
        compact_turn: dict[str, Any] = {
            "turn_id": turn_id,
            "planner_action": action,
        }
        tool_name = str(raw_turn.get("tool_name") or "").strip()
        if tool_name:
            compact_turn["tool_name"] = tool_name
        tool_question = _planner_tool_question(raw_turn)
        if tool_question:
            compact_turn["planner_tool_question"] = tool_question
        observation = raw_turn.get("tool_observation")
        if observation is not None and str(observation).strip():
            compact_turn["perception_observation"] = _json_safe(observation)
        final_answer = str(raw_turn.get("final_answer") or "").strip()
        if final_answer:
            compact_turn["final_answer"] = final_answer
        planner_responses, planner_reasonings = _planner_call_texts(raw_turn)
        if planner_responses:
            compact_turn["planner_response"] = "\n\n".join(planner_responses)
        if planner_reasonings:
            compact_turn["planner_reasoning"] = "\n\n".join(planner_reasonings)

        parse_error = raw_turn.get("planner_parse_error")
        if parse_error is not None and str(parse_error).strip():
            error_text = _bounded_reflection_text(
                parse_error,
                limit=MAX_GEPA_REFLECTION_ERROR_CHARS,
            )
            compact_turn["parse_error"] = error_text
            errors.append(f"turn {turn_id}: {error_text}")
        qwen_error = raw_turn.get("reliable_qwen_error")
        if isinstance(qwen_error, dict):
            error_text = _bounded_reflection_text(
                qwen_error.get("message") or "Qwen returned an empty response.",
                limit=MAX_GEPA_REFLECTION_ERROR_CHARS,
            )
            compact_turn["qwen_error"] = _json_safe(qwen_error)
            errors.append(f"turn {turn_id}: {error_text}")
        turns.append(compact_turn)

    completion_text = getattr(prediction, "completion_text", None)
    if completion_text is not None:
        errors.append(
            "prediction parse failure: "
            + _bounded_reflection_text(
                completion_text,
                limit=MAX_GEPA_REFLECTION_ERROR_CHARS,
            )
        )

    if errors:
        status: dict[str, Any] = {"state": "error", "errors": errors}
    elif turns:
        status = {"state": "ok"}
    else:
        status = {"state": "no_turn_trace"}
    return turns, status


def _new_gepa_caption_supervision_stats() -> dict[str, int]:
    return {
        "videos_loaded": 0,
        "cache_hits": 0,
        "files_loaded": 0,
        "missing_files": 0,
        "unreadable_files": 0,
    }


def _is_captioner_target(target_path: str) -> bool:
    return str(target_path).startswith(GEPA_CAPTIONER_TARGET_PREFIX)


def _is_planner_target(target_path: str) -> bool:
    return str(target_path).startswith(GEPA_PLANNER_TARGET_PREFIX)


def _target_component_role(target_path: str) -> str:
    if _is_captioner_target(target_path):
        return (
            "Captioner tool instruction for ask_caption. Produce factual timestamped "
            "audio-visual captions only; do not answer the multiple-choice question."
        )
    if _is_planner_target(target_path):
        return (
            "AVQA ReAct planner workflow instruction. Decide when to call tools and "
            "return exactly one final option letter."
        )
    return "Prompt component under optimization. Preserve its existing runtime interface."


def _target_output_contract(target_path: str) -> str:
    if _is_captioner_target(target_path):
        return (
            "Output a timestamped shot list/caption with observable visual events, "
            "audible events, speech when clear, identities/continuity/actions, sound "
            "sources, simultaneity, and audio-visual correspondence. Never output an "
            "answer letter or final QA decision."
        )
    if _is_planner_target(target_path):
        return (
            "Use question/options and tool observations to gather evidence, then return "
            "one final answer label through the planner JSON interface."
        )
    return "Keep the component's original output format and runtime contract."


def _caption_quality_rubric() -> str:
    return (
        "Evaluate the caption instruction by whether ask_caption supplies factual, "
        "timestamped, non-decision-biased evidence that helps the planner answer later: "
        "visual events, audio events, speech, identity/continuity, source attribution, "
        "timing, simultaneity, and audio-visual alignment."
    )


def _reflection_prompt_template_for_target(target_path: str) -> str | None:
    if _is_captioner_target(target_path):
        return GEPA_CAPTIONER_REFLECTION_PROMPT_TEMPLATE
    if _is_planner_target(target_path):
        return GEPA_PLANNER_REFLECTION_PROMPT_TEMPLATE
    return None


def _missing_gepa_caption_marker(filename: str) -> str:
    return f"[MISSING PRIVILEGED SUPERVISION FILE: {filename}]"


def _load_gepa_caption_supervision(
    *,
    daily_omni_root: Path,
    video_id: str,
    cache: dict[str, dict[str, str]],
    stats: dict[str, int],
) -> dict[str, str]:
    """Load full optimization-only captions once per DailyOmni video."""
    normalized_video_id = str(video_id or "").strip()
    if not normalized_video_id:
        raise ValueError("Cannot load GEPA caption supervision without video_id")
    if normalized_video_id in cache:
        stats["cache_hits"] += 1
        return cache[normalized_video_id]

    video_dir = daily_omni_root / "Videos" / normalized_video_id
    captions: dict[str, str] = {}
    for label_name, filename in GEPA_PRIVILEGED_CAPTION_FILES:
        caption_path = video_dir / filename
        if not caption_path.is_file():
            logger.warning(
                "Missing GEPA optimization supervision for video_id=%s: %s",
                normalized_video_id,
                caption_path,
            )
            captions[label_name] = _missing_gepa_caption_marker(filename)
            stats["missing_files"] += 1
            continue
        try:
            with caption_path.open("r", encoding="utf-8", newline="") as caption_file:
                captions[label_name] = caption_file.read()
            stats["files_loaded"] += 1
        except (OSError, UnicodeError) as exc:
            logger.warning(
                "Unreadable GEPA optimization supervision for video_id=%s: %s (%s)",
                normalized_video_id,
                caption_path,
                exc,
            )
            captions[label_name] = _missing_gepa_caption_marker(filename)
            stats["unreadable_files"] += 1

    cache[normalized_video_id] = captions
    stats["videos_loaded"] += 1
    return captions


def _has_gepa_privileged_caption_evidence(example: dspy.Example) -> bool:
    return any(
        bool(str(getattr(example, label_name, "") or "").strip())
        for label_name, _ in GEPA_PRIVILEGED_CAPTION_FILES
    )


def _format_gepa_privileged_caption_evidence(example: dspy.Example) -> str:
    av_alignment = str(getattr(example, "gepa_privileged_av_alignment_captions", "") or "")
    visual_caption = str(getattr(example, "gepa_privileged_video_consistent_captions", "") or "")
    audio_caption = str(getattr(example, "gepa_privileged_audio_revised_captions", "") or "")
    return (
        "=== PRIVILEGED DATASET EVIDENCE FOR OPTIMIZATION ONLY ===\n\n"
        f"[PRIMARY: AV ALIGNMENT]\n{av_alignment}\n"
        f"[AUXILIARY: VISUAL CAPTION]\n{visual_caption}\n"
        f"[AUXILIARY: AUDIO CAPTION]\n{audio_caption}\n\n"
        "=== END PRIVILEGED EVIDENCE ==="
    )


def _load_gepa_densified_supervision(
    *,
    label_dir: Path,
    cut_id: str,
) -> dict[str, str]:
    """Load one strict, optimization-only densified label by question ID."""
    normalized_cut_id = str(cut_id or "").strip()
    if not normalized_cut_id:
        raise ValueError("Cannot load GEPA densified supervision without cut_id")
    if Path(normalized_cut_id).name != normalized_cut_id:
        raise ValueError(f"Invalid cut_id for GEPA densified supervision: {cut_id!r}")

    label_path = label_dir / f"{normalized_cut_id}.json"
    if not label_path.is_file():
        raise ValueError(
            f"Missing GEPA densified supervision for cut_id={normalized_cut_id}: "
            f"{label_path}"
        )
    try:
        payload = json.loads(label_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid GEPA densified supervision for cut_id={normalized_cut_id}: "
            f"{label_path} ({exc})"
        ) from exc

    required_keys = {source_key for _, source_key in GEPA_DENSIFIED_LABEL_FIELDS}
    allowed_keys = required_keys | GEPA_DENSIFIED_OPTIONAL_SOURCE_KEYS
    payload_keys = set(payload) if isinstance(payload, dict) else set()
    if (
        not isinstance(payload, dict)
        or not required_keys.issubset(payload_keys)
        or not payload_keys.issubset(allowed_keys)
    ):
        raise ValueError(
            f"GEPA densified label must contain required fields {sorted(required_keys)} "
            f"and only supported fields {sorted(allowed_keys)} "
            f"for cut_id={normalized_cut_id}: {label_path}"
        )

    supervision: dict[str, str] = {}
    for label_name, source_key in GEPA_DENSIFIED_LABEL_FIELDS:
        value = payload[source_key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"GEPA densified label field {source_key!r} must be a non-empty string "
                f"for cut_id={normalized_cut_id}: {label_path}"
            )
        supervision[label_name] = value.strip()
    return supervision


def _format_gepa_densified_supervision(example: dspy.Example) -> dict[str, str]:
    return {
        "Key evidence": str(getattr(example, "gepa_privileged_key_evidence", "") or "").strip(),
    }


def make_trainset(raw_items: list[dict[str, Any]]) -> list[dspy.Example]:
    """Convert raw rows to DSPy Example list."""
    trainset: list[dspy.Example] = []
    for item in raw_items:
        example_fields = {
            "question_id": (
                item.get("question_id") or item.get("cut_id") or item.get("video_id")
            ),
            "question": item["question"],
            "options_json": json.dumps(item["options"], ensure_ascii=False),
            "video_path": item["video_path"],
            "audio_path": item.get("audio_path"),
            "video_id": item.get("video_id"),
            "video_description": item.get("video_description", ""),
            "max_turns": item.get("max_turns"),
            "answer": item["answer"],
        }
        for label_name, _ in GEPA_PRIVILEGED_CAPTION_FILES:
            if label_name in item:
                example_fields[label_name] = item[label_name]
        for label_name, _ in GEPA_DENSIFIED_LABEL_FIELDS:
            if label_name in item:
                example_fields[label_name] = item[label_name]
        ex = dspy.Example(**example_fields).with_inputs(
            "question_id",
            "question",
            "options_json",
            "video_path",
            "audio_path",
            "video_id",
            "video_description",
            "max_turns",
        )
        trainset.append(ex)
    return trainset


def make_trainset_from_cuts(
    cuts: list[dict[str, Any]],
    audio_caption_dir: Path | None,
    *,
    max_turns: int | None = None,
    skip_bad_examples: bool = False,
    gepa_caption_root: Path | None = None,
    gepa_caption_cache: dict[str, dict[str, str]] | None = None,
    gepa_caption_stats: dict[str, int] | None = None,
    gepa_densified_label_dir: Path | None = None,
) -> tuple[list[dspy.Example], list[dict[str, Any]]]:
    """Convert Daily Omni cut rows into DSPy Examples."""
    raw_items: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    caption_cache = gepa_caption_cache if gepa_caption_cache is not None else {}
    caption_stats = (
        gepa_caption_stats
        if gepa_caption_stats is not None
        else _new_gepa_caption_supervision_stats()
    )
    for cut in cuts:
        cut_id = str(cut.get("id") or "")
        try:
            payload = build_input_state(cut, audio_caption_dir)
            supervisions = cut.get("supervisions") or []
            custom = (supervisions[0].get("custom") if supervisions else {}) or {}
            answer = custom.get("answer") or custom.get("Answer")
            if answer is None:
                raise ValueError(f"Missing answer in cut_id={cut_id}")
            video_id = payload.get("video_id") or cut_id.rsplit("-", 1)[0]
            raw_item = {
                "question_id": cut_id,
                "question": payload["question"],
                "options": payload["options"],
                "video_path": payload["video_path"],
                "audio_path": payload.get("audio_path"),
                "video_id": video_id,
                "video_description": payload.get("video_description", ""),
                "max_turns": max_turns,
                "answer": answer,
                "cut_id": cut_id,
            }
            if gepa_caption_root is not None:
                raw_item.update(
                    _load_gepa_caption_supervision(
                        daily_omni_root=gepa_caption_root,
                        video_id=str(video_id or ""),
                        cache=caption_cache,
                        stats=caption_stats,
                    )
                )
            if gepa_densified_label_dir is not None:
                raw_item.update(
                    _load_gepa_densified_supervision(
                        label_dir=gepa_densified_label_dir,
                        cut_id=cut_id,
                    )
                )
            raw_items.append(raw_item)
        except Exception as exc:
            if not skip_bad_examples:
                raise
            skipped.append({"cut_id": cut_id, "error": str(exc)})
    return make_trainset(raw_items), skipped


def _filtered_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only keyword arguments accepted by this installed DSPy version."""
    import inspect

    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _load_optimizer_config(path: Path | None) -> dict[str, Any]:
    """Load an optimizer config from JSON or YAML."""
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(f) or {}
        else:
            payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Optimizer config must be a mapping: {path}")
    return payload


_GEPA_RUN_PATH_FIELDS = {
    "trainset_jsonl",
    "valset_jsonl",
    "daily_omni_root",
    "audio_caption_dir",
    "output_program",
    "initial_program",
    "metadata_json",
    "signature_search_json",
    "trajectory_jsonl",
    "optimizer_log_dir",
    "optimized_prompt_config_yaml",
    "final_eval_output_jsonl",
    "final_eval_output_dir",
    "gepa_reflection_template_yaml",
}
_GEPA_RUN_INT_FIELDS = {"train_limit", "gepa_seed"}
_GEPA_RUN_STRING_FIELDS = {
    "algorithm",
    "gepa_reflection_template_version",
    "gepa_log_dir",
}
_GEPA_RUN_FIELDS = (
    _GEPA_RUN_PATH_FIELDS
    | _GEPA_RUN_INT_FIELDS
    | _GEPA_RUN_STRING_FIELDS
    | {"optimize_targets"}
)


def _expand_gepa_run_value(value: str, *, key: str, config_path: Path) -> str:
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(
            f"Unresolved environment variable in GEPA run config {key}: "
            f"{value!r} ({config_path})"
        )
    return expanded


def _apply_gepa_run_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    """Apply the optional ``run`` section from --gepa-config to parsed arguments."""
    config = _load_optimizer_config(args.gepa_config)
    run_config = config.get("run") or {}
    if not isinstance(run_config, dict):
        parser.error(f"GEPA config run section must be a mapping: {args.gepa_config}")
    unknown = sorted(set(run_config) - _GEPA_RUN_FIELDS)
    if unknown:
        parser.error(
            "Unknown GEPA run config key(s): " + ", ".join(unknown)
        )

    for key, value in run_config.items():
        if key == "optimize_targets":
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                parser.error("GEPA run config optimize_targets must be a list of strings")
            args.optimize_targets = tuple(value)
            continue
        if key in _GEPA_RUN_PATH_FIELDS:
            if not isinstance(value, str) or not value.strip():
                parser.error(f"GEPA run config {key} must be a non-empty path string")
            setattr(
                args,
                key,
                Path(_expand_gepa_run_value(value, key=key, config_path=args.gepa_config)),
            )
            continue
        if key in _GEPA_RUN_INT_FIELDS:
            if isinstance(value, bool) or not isinstance(value, int):
                parser.error(f"GEPA run config {key} must be an integer")
            setattr(args, key, value)
            continue
        if not isinstance(value, str) or not value.strip():
            parser.error(f"GEPA run config {key} must be a non-empty string")
        if key == "algorithm" and value not in {"copro", "simba", "miprov2", "gepa"}:
            parser.error(f"GEPA run config algorithm is unsupported: {value!r}")
        setattr(
            args,
            key,
            _expand_gepa_run_value(value, key=key, config_path=args.gepa_config),
        )
    return run_config


def _config_value(config: dict[str, Any], key: str, fallback: Any) -> Any:
    return config[key] if key in config else fallback


def _normalize_gepa_caption_supervision_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower()
    if mode not in GEPA_CAPTION_SUPERVISION_CHOICES:
        choices = ", ".join(GEPA_CAPTION_SUPERVISION_CHOICES)
        raise ValueError(f"gepa_caption_supervision must be one of: {choices}; got {value!r}")
    return mode


def _validate_gepa_caption_root(root: Path | None, *, required: bool) -> Path | None:
    if root is None:
        if required:
            raise ValueError(
                "Captioner GEPA privileged supervision requires --daily-omni-root "
                "or DAILY_OMNI_ROOT"
            )
        return None
    if not root.is_dir() or not (root / "Videos").is_dir():
        raise ValueError(
            "DailyOmni root must contain a Videos directory: "
            f"{root}"
        )
    return root


def _resolve_gepa_caption_supervision(
    args: argparse.Namespace,
    *,
    optimizes_captioner: bool,
) -> tuple[str, Path | None]:
    """Return the caption-supervision mode and optional DailyOmni root for GEPA."""
    if args.algorithm != "gepa" or not optimizes_captioner:
        return "none", None

    requested_mode = _normalize_gepa_caption_supervision_mode(
        getattr(args, "gepa_caption_supervision", "auto")
    )
    if requested_mode == "none":
        return "none", None

    root = getattr(args, "daily_omni_root", None)
    if requested_mode == "privileged":
        return "privileged", _validate_gepa_caption_root(root, required=True)

    validated_root = _validate_gepa_caption_root(root, required=False)
    if validated_root is None:
        return "none", None
    return "privileged", validated_root


def _resolve_gepa_densified_label_dir(args: argparse.Namespace) -> Path | None:
    """Resolve opt-in densified supervision for GEPA training examples only."""
    if args.algorithm != "gepa":
        return None
    label_dir = getattr(args, "gepa_densified_label_dir", None)
    if label_dir is None:
        return None
    label_dir = Path(label_dir)
    if not label_dir.is_dir():
        raise ValueError(
            "GEPA densified label directory does not exist or is not a directory: "
            f"{label_dir}"
        )
    return label_dir


def _default_gepa_reflection_template_yaml_path() -> Path:
    if GEPA_REFLECTION_TEMPLATE_DEFAULT_PATH.is_file():
        return GEPA_REFLECTION_TEMPLATE_DEFAULT_PATH
    return GEPA_REFLECTION_TEMPLATE_LEGACY_PATH


def _normalize_gepa_reflection_template_version(
    value: Any,
    *,
    densified_enabled: bool,
) -> str:
    raw = str(value or GEPA_REFLECTION_TEMPLATE_VERSION_AUTO).strip().lower()
    if not raw or raw == GEPA_REFLECTION_TEMPLATE_VERSION_AUTO:
        if densified_enabled:
            return GEPA_REFLECTION_TEMPLATE_VERSION_DENSIFIED_KEY_EVIDENCE
        return GEPA_REFLECTION_TEMPLATE_VERSION_ORIGINAL
    return GEPA_REFLECTION_TEMPLATE_VERSION_ALIASES.get(raw, raw)


def _template_value(entry: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"GEPA reflection template entry must define one of {keys}")


def _load_gepa_reflection_template_versions(
    template_yaml: Path,
) -> dict[str, dict[str, str]]:
    if not template_yaml.is_file():
        raise ValueError(f"GEPA reflection template YAML does not exist: {template_yaml}")
    try:
        payload = yaml.safe_load(template_yaml.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid GEPA reflection template YAML: {template_yaml} ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"GEPA reflection template YAML must be a mapping: {template_yaml}")

    versions_payload = payload.get("versions")
    if versions_payload is None and all(key in payload for key in GEPA_REFLECTION_TEMPLATE_YAML_KEYS):
        versions_payload = {"default": payload}
    if not isinstance(versions_payload, dict) or not versions_payload:
        raise ValueError(
            "GEPA reflection template YAML must contain a non-empty 'versions' mapping: "
            f"{template_yaml}"
        )

    versions: dict[str, dict[str, str]] = {}
    for version_name, entry in versions_payload.items():
        normalized_name = str(version_name or "").strip()
        if not normalized_name:
            raise ValueError(f"GEPA reflection template version name is empty: {template_yaml}")
        if not isinstance(entry, dict):
            raise ValueError(
                f"GEPA reflection template version {normalized_name!r} must be a mapping: "
                f"{template_yaml}"
            )
        versions[normalized_name] = {
            "captioner": _template_value(
                entry,
                "GEPA_CAPTIONER_REFLECTION_PROMPT_TEMPLATE",
                "captioner",
                "captioner_template",
            ),
            "planner": _template_value(
                entry,
                "GEPA_PLANNER_REFLECTION_PROMPT_TEMPLATE",
                "planner",
                "planner_template",
            ),
        }
    return versions


def configure_gepa_reflection_prompt_templates(
    args: argparse.Namespace,
    *,
    densified_enabled: bool,
) -> dict[str, Any]:
    """Select the GEPA reflection prompt templates for this optimization run."""
    global GEPA_CAPTIONER_REFLECTION_PROMPT_TEMPLATE
    global GEPA_PLANNER_REFLECTION_PROMPT_TEMPLATE
    global _current_gepa_reflection_template_info

    raw_version = getattr(args, "gepa_reflection_template_version", None)
    resolved_version = _normalize_gepa_reflection_template_version(
        raw_version,
        densified_enabled=densified_enabled,
    )
    template_yaml = getattr(args, "gepa_reflection_template_yaml", None)
    template_yaml = Path(template_yaml) if template_yaml is not None else _default_gepa_reflection_template_yaml_path()
    versions = _load_gepa_reflection_template_versions(template_yaml)
    if resolved_version not in versions:
        available_versions = ", ".join(sorted(versions))
        raise ValueError(
            f"GEPA reflection template version {resolved_version!r} is not defined in "
            f"{template_yaml}; available versions: {available_versions}"
        )

    selected = versions[resolved_version]
    GEPA_CAPTIONER_REFLECTION_PROMPT_TEMPLATE = selected["captioner"]
    GEPA_PLANNER_REFLECTION_PROMPT_TEMPLATE = selected["planner"]
    _current_gepa_reflection_template_info = {
        "yaml": str(template_yaml),
        "requested_version": str(raw_version or GEPA_REFLECTION_TEMPLATE_VERSION_AUTO),
        "resolved_version": resolved_version,
        "densified_enabled": densified_enabled,
    }
    return dict(_current_gepa_reflection_template_info)


def current_gepa_reflection_template_info() -> dict[str, Any]:
    return dict(_current_gepa_reflection_template_info)


def _none_if_unset(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "unset"}:
        return None
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if result < 0:
        raise ValueError(f"{name} must be >= 0, got {result}")
    return result


def _resolve_miprov2_config(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_optimizer_config(args.miprov2_config)
    auto = _none_if_unset(_config_value(config, "auto", args.miprov2_auto))
    return {
        "auto": auto,
        "num_candidates": None if auto is not None else _config_value(config, "num_candidates", args.miprov2_num_candidates),
        "num_trials": None if auto is not None else _config_value(config, "num_trials", args.miprov2_num_trials),
        "max_bootstrapped_demos": _config_value(config, "max_bootstrapped_demos", args.miprov2_max_bootstrapped_demos),
        "max_labeled_demos": _config_value(config, "max_labeled_demos", args.miprov2_max_labeled_demos),
        "seed": _config_value(config, "seed", args.miprov2_seed),
        "init_temperature": _config_value(config, "init_temperature", args.miprov2_init_temperature),
        "num_threads": _none_if_unset(_config_value(config, "num_threads", args.miprov2_num_threads)),
        "max_errors": _none_if_unset(_config_value(config, "max_errors", args.miprov2_max_errors)),
        "minibatch": _config_value(config, "minibatch", not args.miprov2_no_minibatch),
        "minibatch_size": _config_value(config, "minibatch_size", args.miprov2_minibatch_size),
        "minibatch_full_eval_steps": _config_value(config, "minibatch_full_eval_steps", args.miprov2_minibatch_full_eval_steps),
        "view_data_batch_size": _config_value(config, "view_data_batch_size", args.miprov2_view_data_batch_size),
        "track_stats": _config_value(config, "track_stats", not args.miprov2_no_track_stats),
        "log_dir": _none_if_unset(_config_value(config, "log_dir", args.miprov2_log_dir)),
    }


def _resolve_gepa_config(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_optimizer_config(args.gepa_config)
    auto = _none_if_unset(_config_value(config, "auto", args.gepa_auto))
    max_metric_calls = _none_if_unset(_config_value(config, "max_metric_calls", args.gepa_max_metric_calls))
    max_full_evals = _none_if_unset(_config_value(config, "max_full_evals", args.gepa_max_full_evals))
    if auto is not None or max_metric_calls is not None:
        max_full_evals = None
    return {
        "auto": auto,
        "max_full_evals": max_full_evals,
        "max_metric_calls": max_metric_calls,
        "reflection_minibatch_size": _config_value(config, "reflection_minibatch_size", args.gepa_reflection_minibatch_size),
        "candidate_selection_strategy": _config_value(config, "candidate_selection_strategy", args.gepa_candidate_selection_strategy),
        "skip_perfect_score": _config_value(config, "skip_perfect_score", not args.gepa_dont_skip_perfect_score),
        "use_merge": _config_value(config, "use_merge", not args.gepa_no_merge),
        "max_merge_invocations": _none_if_unset(_config_value(config, "max_merge_invocations", args.gepa_max_merge_invocations)),
        "num_threads": _none_if_unset(_config_value(config, "num_threads", args.gepa_num_threads)),
        "seed": _none_if_unset(_config_value(config, "seed", args.gepa_seed)),
        "log_dir": _none_if_unset(_config_value(config, "log_dir", args.gepa_log_dir)),
        "reflective_dataset_save_interval": _nonnegative_int(
            _config_value(
                config,
                "reflective_dataset_save_interval",
                args.gepa_reflective_dataset_save_interval,
            ),
            name="reflective_dataset_save_interval",
        ),
        "track_stats": _config_value(config, "track_stats", args.gepa_track_stats),
        "track_best_outputs": _config_value(config, "track_best_outputs", args.gepa_track_best_outputs),
    }


PROMPT_OPTIMIZE_TARGETS = {
    "planner.workflow_prompt",
    "captioner.default_caption_instruction",
    "captioner.caption_prompt_template",
    "perception.default_perceptual_question",
    "perception.evidence_prompt_template",
}
PROMPT_OPTIMIZE_EXCLUDED_TARGETS = {
    "signatures.PlanNextAction.instructions",
    "captioner.system_prompt",
    "perception.system_prompt",
    "planner.action_schema",
    "planner.task_prompt_template",
    "planner.final_fallback_instruction",
    "planner.conversation_empty",
    "planner.assistant_turn_template",
    "planner.tool_turn_template",
    "planner.truncated_marker",
}


class PromptComponentSignature(dspy.Signature):
    """Prompt component text being optimized."""

    prompt_context = dspy.InputField(desc="Description of where this prompt component is used.")
    prompt_text = dspy.OutputField(desc="The prompt component text.")


def _target_keys(target_path: str) -> tuple[str, ...]:
    return tuple(part for part in target_path.split(".") if part)


def _get_nested(mapping: dict[str, Any], target_path: str) -> Any:
    value: Any = mapping
    for key in _target_keys(target_path):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(target_path)
        value = value[key]
    return value


def _set_nested(mapping: dict[str, Any], target_path: str, value: str) -> None:
    keys = _target_keys(target_path)
    current: dict[str, Any] = mapping
    for key in keys[:-1]:
        next_value = current.setdefault(key, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set {target_path}; {key} is not a mapping")
        current = next_value
    current[keys[-1]] = value


def validate_optimize_targets(target_paths: tuple[str, ...]) -> None:
    if not target_paths:
        raise ValueError("At least one --optimize-target must be provided")
    if len(set(target_paths)) != len(target_paths):
        raise ValueError(f"Duplicate prompt optimize targets are not allowed: {target_paths}")
    for target_path in target_paths:
        if target_path == "signature":
            raise ValueError("Signature optimization is disabled for v8; use --optimize-target planner.workflow_prompt or another explicit prompt dot path")
        if target_path in PROMPT_OPTIMIZE_EXCLUDED_TARGETS:
            raise ValueError(f"Prompt target {target_path!r} is intentionally fixed and excluded from optimization")
        if target_path not in PROMPT_OPTIMIZE_TARGETS:
            allowed = ", ".join(sorted(PROMPT_OPTIMIZE_TARGETS))
            raise ValueError(f"Unsupported prompt optimize target {target_path!r}; expected one of: {allowed}")
        value = _get_nested(prompt_config(), target_path)
        if not isinstance(value, str):
            raise ValueError(f"Prompt optimize target must resolve to a string: {target_path}")


def prompt_target_initial_text(target_path: str) -> str:
    return str(_get_nested(prompt_config(), target_path)).strip()


class PromptTargetProgram(dspy.Module):
    """Expose one or more prompt config strings as optimizer-visible components."""

    def __init__(self, base_program: AVQADSPyReActProgram, target_paths: tuple[str, ...]):
        super().__init__()
        object.__setattr__(self, "_base_program", base_program)
        self.target_paths = target_paths
        self.component_names = tuple(f"prompt_component_{idx}" for idx in range(len(target_paths)))
        for target_path, component_name in zip(self.target_paths, self.component_names):
            signature = PromptComponentSignature.with_instructions(prompt_target_initial_text(target_path))
            setattr(self, component_name, dspy.Predict(signature))

    def _component_name(self, target_path: str) -> str:
        return self.component_names[self.target_paths.index(target_path)]

    def _candidate_text(self, target_path: str) -> str:
        component = getattr(self, self._component_name(target_path))
        signature = getattr(component, "signature", None)
        return str(getattr(signature, "instructions", "") or "").strip()

    def forward(self, *args: Any, **kwargs: Any) -> dspy.Prediction:
        overrides = {
            _target_keys(target_path): self._candidate_text(target_path)
            for target_path in self.target_paths
        }
        with prompt_overrides(overrides):
            return self._base_program(*args, **kwargs)

    def named_predictors(self):
        return [(name, getattr(self, name)) for name in self.component_names]

    def predictors(self):
        return [getattr(self, name) for name in self.component_names]


def _prompt_component_text(program: dspy.Module, target_path: str) -> str:
    component_name = program._component_name(target_path)
    component = getattr(program, component_name, None)
    signature = getattr(component, "signature", None)
    return str(getattr(signature, "instructions", "") or "").strip()


_CAPTION_TOOL_NAMES = frozenset({
    "ask_caption",
    "ask_captioner",
    "caption_video",
    "captioner",
})
_QWEN_LENGTH_FINISH_REASONS = frozenset({
    "length",
    "max_token",
    "max_tokens",
    "max_output_tokens",
    "token_limit",
})
_QWEN_OUTPUT_TOKEN_USAGE_KEYS = ("completion_tokens", "output_tokens")


def _mapping_value(value: Any) -> dict[str, Any]:
    """Return a dict from API metadata without failing GEPA diagnostics."""
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            return {}
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _qwen_call_hit_max_tokens(token_usage: Any) -> bool:
    """Detect a Qwen generation that stopped because of its output cap."""
    usage = _mapping_value(token_usage)
    attempt_usages = [usage]
    attempts = usage.get("qwen_response_attempts")
    if isinstance(attempts, list):
        attempt_usages.extend(_mapping_value(attempt) for attempt in attempts)

    for attempt_usage in attempt_usages:
        finish_reason = str(
            attempt_usage.get("qwen_finish_reason")
            or attempt_usage.get("finish_reason")
            or ""
        ).strip().lower()
        if finish_reason in _QWEN_LENGTH_FINISH_REASONS:
            return True

        max_tokens = _positive_int(
            attempt_usage.get("qwen_max_tokens", attempt_usage.get("max_tokens"))
        )
        if max_tokens is None:
            continue
        for token_key in _QWEN_OUTPUT_TOKEN_USAGE_KEYS:
            completion_tokens = _positive_int(attempt_usage.get(token_key))
            if completion_tokens is not None and completion_tokens >= max_tokens:
                return True
    return False


def _qwen_captioner_max_token_stats(predictions: list[Any]) -> dict[str, Any]:
    """Aggregate Qwen captioner cap hits across a completed GEPA batch."""
    captioner_samples = 0
    hit_samples = 0
    captioner_calls = 0
    hit_calls = 0
    max_tokens_values: set[int] = set()

    for prediction in predictions:
        sample_has_captioner_call = False
        sample_hit_max_tokens = False
        for turn in getattr(prediction, "turn_trace", None) or []:
            if not isinstance(turn, dict):
                continue
            tool_name = str(turn.get("tool_name") or "").strip().lower()
            backend = str(turn.get("perception_backend") or "").strip().lower()
            if tool_name not in _CAPTION_TOOL_NAMES or backend != "qwen":
                continue
            sample_has_captioner_call = True
            captioner_calls += 1
            token_usage = _mapping_value(turn.get("perception_token_usage"))
            max_tokens = _positive_int(token_usage.get("qwen_max_tokens"))
            if max_tokens is not None:
                max_tokens_values.add(max_tokens)
            if _qwen_call_hit_max_tokens(token_usage):
                sample_hit_max_tokens = True
                hit_calls += 1
        if sample_has_captioner_call:
            captioner_samples += 1
        if sample_hit_max_tokens:
            hit_samples += 1

    return {
        "captioner_samples": captioner_samples,
        "hit_samples": hit_samples,
        "captioner_calls": captioner_calls,
        "hit_calls": hit_calls,
        "max_tokens_values": tuple(sorted(max_tokens_values)),
    }


class PromptTargetGEPAAdapter:
    """GEPA adapter for prompt components that are read but not called as predictors."""

    def __init__(
        self,
        *,
        student_module: PromptTargetProgram,
        metric_fn: Any,
        failure_score: float = 0.0,
        num_threads: int | None = None,
        rng: random.Random | None = None,
        reflection_lm: Any = None,
        reflection_minibatch_size: int | None = None,
        qwen_executor: ReliableQwenExecutor | None = None,
        validation_examples: list[dspy.Example] | None = None,
    ):
        self.student = student_module
        self.component_names = student_module.component_names
        self.metric_fn = metric_fn
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.rng = rng or random.Random(0)
        self.reflection_lm = reflection_lm
        self.reflection_minibatch_size = reflection_minibatch_size
        self.qwen_executor = qwen_executor or ReliableQwenExecutor()
        self.validation_example_ids = frozenset(
            id(example) for example in (validation_examples or [])
        )

    def stripped_lm_call(self, x: str) -> list[str]:
        raw_outputs = (self.reflection_lm or dspy.settings.lm)(x)
        outputs: list[str] = []
        for raw_output in raw_outputs:
            if isinstance(raw_output, str):
                outputs.append(raw_output)
            elif isinstance(raw_output, dict) and "text" in raw_output:
                outputs.append(str(raw_output["text"]))
            else:
                outputs.append(str(raw_output))
        return outputs

    def build_program(self, candidate: dict[str, str]) -> PromptTargetProgram:
        new_prog = self.student.deepcopy()
        for component_name in self.component_names:
            candidate_text = candidate.get(component_name)
            if candidate_text is not None:
                component = getattr(new_prog, component_name)
                component.signature = component.signature.with_instructions(candidate_text)
        return new_prog

    @staticmethod
    def _metric_score_value(score: Any) -> Any:
        return score["score"] if hasattr(score, "score") else score

    def _retry_empty_qwen_predictions(
        self,
        *,
        program: dspy.Module,
        batch: list[dspy.Example],
        predictions: list[Any],
        raw_scores: list[Any],
    ) -> tuple[list[int], list[str]]:
        """Replace only empty-Qwen rollouts after a concurrent evaluation batch.

        The primary pass is already complete when this runs.  Every fallback
        rollout is executed one at a time, and the request-level semaphore in
        ``ReliableQwenExecutor`` keeps Qwen max_inflight at one even if a caller
        later introduces another retry worker.
        """
        executor = getattr(self, "qwen_executor", None) or ReliableQwenExecutor()
        invalid_indices = [
            index
            for index, prediction in enumerate(predictions)
            if prediction_has_empty_qwen_observation(prediction)
        ]
        if not invalid_indices:
            return [], [executor.primary_profile.name]

        attempted_profiles = [executor.primary_profile.name]
        for retry_number in range(1, executor.max_batch_retries + 1):
            with executor.retry_batch(retry_number) as profile:
                attempted_profiles.append(profile.name)
                logger.warning(
                    "Retrying %d empty-Qwen rollout(s) serially after evaluation batch "
                    "with profile=%s (retry %d/%d).",
                    len(invalid_indices),
                    profile.name,
                    retry_number,
                    executor.max_batch_retries,
                )
                for index in invalid_indices:
                    prediction = program(**batch[index].inputs())
                    predictions[index] = prediction
                    raw_scores[index] = self.metric_fn(batch[index], prediction)

            invalid_indices = [
                index
                for index in invalid_indices
                if prediction_has_empty_qwen_observation(predictions[index])
            ]
            if not invalid_indices:
                return [], attempted_profiles

        # Keep the rollout so training/reflection can see its explicit error;
        # validation wraps its score below so the sample is excluded from mean
        # accuracy without changing GEPA's positional val-id mapping.
        return invalid_indices, attempted_profiles

    def _is_validation_batch(
        self,
        batch: list[dspy.Example],
        *,
        capture_traces: bool,
    ) -> bool:
        if capture_traces or not batch:
            return False
        validation_ids = getattr(self, "validation_example_ids", frozenset())
        return bool(validation_ids) and all(id(example) in validation_ids for example in batch)

    def _apply_exhausted_qwen_outcome(
        self,
        *,
        predictions: list[Any],
        raw_scores: list[Any],
        invalid_indices: list[int],
        attempted_profiles: list[str],
        is_validation_batch: bool,
    ) -> list[Any]:
        invalid_set = set(invalid_indices)
        for index in invalid_indices:
            mark_empty_qwen_prediction(
                predictions[index],
                attempted_profiles=attempted_profiles,
            )

        if not is_validation_batch:
            # Discovery/training keeps its normal score and sample membership.
            # The reflection trace above carries the reliable_qwen_error so the
            # teacher can see why a rollout was unreliable.
            return raw_scores

        # Keep one score object per original validation example: GEPA maps
        # scores positionally to validation IDs. QwenValidationScore excludes
        # failed rollouts from the aggregate while preserving that mapping.
        if invalid_indices:
            logger.warning(
                "Excluding %d exhausted empty-Qwen rollout(s) from validation accuracy; "
                "their traces remain attached to the corresponding examples.",
                len(invalid_indices),
            )
        return [
            QwenValidationScore(
                self._metric_score_value(score),
                included=index not in invalid_set,
            )
            for index, score in enumerate(raw_scores)
        ]

    def _report_captioner_max_token_hits(
        self,
        predictions: list[Any],
        *,
        capture_traces: bool,
    ) -> None:
        """Print one captioner token-cap summary after each GEPA batch."""
        stats = _qwen_captioner_max_token_stats(predictions)
        batch_index = getattr(self, "_gepa_evaluation_batch_index", 0) + 1
        self._gepa_evaluation_batch_index = batch_index
        phase = "reflection_trace" if capture_traces else "evaluation"
        observed_max_tokens = ",".join(map(str, stats["max_tokens_values"])) or "unknown"
        print(
            "[GEPA captioner max-tokens] "
            f"batch={batch_index} phase={phase} batch_samples={len(predictions)} "
            f"qwen_captioner_samples={stats['captioner_samples']} "
            f"hit_max_tokens_samples={stats['hit_samples']} "
            f"qwen_captioner_calls={stats['captioner_calls']} "
            f"hit_max_tokens_calls={stats['hit_calls']} "
            f"observed_max_tokens={observed_max_tokens}",
            flush=True,
        )

    def evaluate(self, batch: list[dspy.Example], candidate: dict[str, str], capture_traces: bool = False):
        from dspy.evaluate import Evaluate
        from gepa.core.adapter import EvaluationBatch

        program = self.build_program(candidate)
        callback_metadata = (
            {"metric_key": "eval_full"}
            if self.reflection_minibatch_size is None or len(batch) > self.reflection_minibatch_size
            else {"disable_logging": True}
        )
        qwen_executor = getattr(self, "qwen_executor", None) or ReliableQwenExecutor()
        if capture_traces:
            from dspy.teleprompt import bootstrap_trace as bootstrap_trace_module

            with qwen_executor.primary_batch():
                trajs = bootstrap_trace_module.bootstrap_trace_data(
                    program=program,
                    dataset=batch,
                    metric=self.metric_fn,
                    num_threads=self.num_threads,
                    raise_on_error=False,
                    capture_failed_parses=True,
                    failure_score=self.failure_score,
                    format_failure_score=self.failure_score,
                    callback_metadata=callback_metadata,
                )
            # bootstrap_trace_data keeps the original example index.  GEPA
            # normally returns every item here; retaining the explicit mapping
            # also avoids silently retrying a non-Qwen parse failure.
            trajectories_by_index = {
                int(item["example_ind"]): item
                for item in trajs
                if isinstance(item, dict) and "example_ind" in item
            }
            if len(trajectories_by_index) == len(batch):
                predictions = [trajectories_by_index[index]["prediction"] for index in range(len(batch))]
                raw_scores = [
                    trajectories_by_index[index].get("score", self.failure_score)
                    for index in range(len(batch))
                ]
                invalid_indices, attempted_profiles = self._retry_empty_qwen_predictions(
                    program=program,
                    batch=batch,
                    predictions=predictions,
                    raw_scores=raw_scores,
                )
                raw_scores = self._apply_exhausted_qwen_outcome(
                    predictions=predictions,
                    raw_scores=raw_scores,
                    invalid_indices=invalid_indices,
                    attempted_profiles=attempted_profiles,
                    is_validation_batch=False,
                )
                for index, item in trajectories_by_index.items():
                    item["prediction"] = predictions[index]
                    item["score"] = raw_scores[index]

            scores = []
            outputs = []
            for item in trajs:
                outputs.append(item["prediction"])
                scores.append(self._metric_score_value(item.get("score", self.failure_score)))
            self._report_captioner_max_token_hits(outputs, capture_traces=True)
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajs)

        evaluator = Evaluate(
            devset=batch,
            metric=self.metric_fn,
            num_threads=self.num_threads,
            return_all_scores=True,
            failure_score=self.failure_score,
            provide_traceback=True,
            max_errors=len(batch) * 100,
            callback_metadata=callback_metadata,
        )
        with qwen_executor.primary_batch():
            result = evaluator(program)
        outputs = [row[1] for row in result.results]
        raw_scores = [row[2] for row in result.results]
        invalid_indices, attempted_profiles = self._retry_empty_qwen_predictions(
            program=program,
            batch=batch,
            predictions=outputs,
            raw_scores=raw_scores,
        )
        raw_scores = self._apply_exhausted_qwen_outcome(
            predictions=outputs,
            raw_scores=raw_scores,
            invalid_indices=invalid_indices,
            attempted_profiles=attempted_profiles,
            is_validation_batch=self._is_validation_batch(
                batch,
                capture_traces=False,
            ),
        )
        scores = [
            score if isinstance(score, QwenValidationScore) else self._metric_score_value(score)
            for score in raw_scores
        ]
        self._report_captioner_max_token_hits(outputs, capture_traces=False)
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=None)

    def _target_path(self, component_name: str) -> str:
        try:
            component_index = self.component_names.index(component_name)
        except ValueError as exc:
            raise KeyError(f"Unknown prompt component: {component_name}") from exc
        return self.student.target_paths[component_index]

    def _proposal_input(
        self,
        *,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        component_name: str,
    ) -> dict[str, Any]:
        target_path = self._target_path(component_name)
        input_dict: dict[str, Any] = {
            "current_instruction_doc": candidate[component_name],
            "dataset_with_feedback": reflective_dataset[component_name],
        }
        prompt_template = _reflection_prompt_template_for_target(target_path)
        if prompt_template is not None:
            input_dict["prompt_template"] = prompt_template
        return input_dict

    def _make_reflective_item(
        self,
        data: dict[str, Any],
        *,
        component_name: str,
        target_path: str,
    ) -> dict[str, Any]:
        example = data.get("example")
        prediction = data.get("prediction")
        score = data.get("score", self.failure_score)
        if hasattr(score, "score"):
            score = score["score"]
        feedback = self.metric_fn(example, prediction, data.get("trace"), target_path, None)
        if hasattr(feedback, "feedback"):
            feedback_text = str(feedback.feedback)
        elif hasattr(feedback, "get"):
            feedback_text = str(feedback.get("feedback", ""))
        else:
            feedback_text = str(feedback)
        gold_answer = normalize_option_letter(str(getattr(example, "answer", "") or ""))
        predicted_answer = normalize_option_letter(str(getattr(prediction, "answer", "") or ""))
        reflection_trace, status = _compact_reflection_trace(prediction)
        inputs = {
            "Optimization target": target_path,
            "Component role": _target_component_role(target_path),
            "Output contract": _target_output_contract(target_path),
            "Question": str(getattr(example, "question", "") or ""),
            "Options": _reflection_options(getattr(example, "options_json", "")),
            "Gold answer": gold_answer,
        }
        captioner_response = str(getattr(example, "video_description", "") or "").strip()
        if captioner_response:
            inputs["Captioner response"] = captioner_response
        if _is_captioner_target(target_path):
            inputs["Caption quality rubric"] = _caption_quality_rubric()
            if _has_gepa_privileged_caption_evidence(example):
                inputs["Privileged dataset evidence (optimization only)"] = (
                    _format_gepa_privileged_caption_evidence(example)
                )
        densified_supervision = _format_gepa_densified_supervision(example)
        if all(densified_supervision.values()):
            inputs["Privileged densified supervision (optimization only)"] = (
                densified_supervision
            )
        return {
            "Inputs": inputs,
            "Generated Outputs": {
                "Predicted answer": predicted_answer,
                "Score": _json_safe(score),
                "Reasoning": str(getattr(prediction, "reasoning_summary", "") or "").strip(),
                "Status": status,
                "Reflection trace": reflection_trace,
            },
            "Feedback": _bounded_reflection_text(
                feedback_text,
                limit=MAX_GEPA_REFLECTION_FEEDBACK_CHARS,
            ),
        }

    @_log_gepa_stage_exceptions("reflective dataset construction")
    def make_reflective_dataset(self, candidate: dict[str, str], eval_batch: Any, components_to_update: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Convert trajectories to component-specific, reflection-focused datasets."""
        trajectories = eval_batch.trajectories or []
        if not trajectories:
            raise Exception("No valid reflective examples found for prompt components")

        datasets: dict[str, list[dict[str, Any]]] = {}
        for component_name in components_to_update:
            target_path = self._target_path(component_name)
            datasets[component_name] = [
                self._make_reflective_item(
                    data,
                    component_name=component_name,
                    target_path=target_path,
                )
                for data in trajectories
            ]
        return datasets

    @_log_gepa_stage_exceptions("instruction proposal")
    def propose_new_texts(self, candidate: dict[str, str], reflective_dataset: dict[str, list[dict[str, Any]]], components_to_update: list[str]) -> dict[str, str]:
        from dspy.teleprompt.gepa.gepa_utils import InstructionProposalSignature

        results: dict[str, str] = {}
        reflection_lm = self.reflection_lm or dspy.settings.lm
        with dspy.context(lm=reflection_lm):
            for name in components_to_update:
                results[name] = InstructionProposalSignature.run(
                    lm=(lambda x: self.stripped_lm_call(x)[0]),
                    input_dict=self._proposal_input(
                        candidate=candidate,
                        reflective_dataset=reflective_dataset,
                        component_name=name,
                    ),
                )["new_instruction"]
        return results


def optimize_prompt_target_with_gepa(
    program: PromptTargetProgram,
    trainset: list[dspy.Example],
    *,
    reflection_lm: Any = None,
    valset: list[dspy.Example] | None = None,
    auto: str | None = None,
    max_full_evals: int | None = 6,
    max_metric_calls: int | None = None,
    reflection_minibatch_size: int = 3,
    candidate_selection_strategy: str = "pareto",
    skip_perfect_score: bool = True,
    use_merge: bool = True,
    max_merge_invocations: int | None = 5,
    num_threads: int | None = None,
    seed: int | None = 0,
    log_dir: str | None = None,
    reflective_dataset_save_interval: int = DEFAULT_GEPA_REFLECTIVE_DATASET_SAVE_INTERVAL,
    track_stats: bool = False,
    track_best_outputs: bool = False,
) -> dspy.Module:
    """Compile one or more prompt components with GEPA without runtime LM calls."""
    from gepa import optimize
    from dspy.teleprompt import GEPA
    from dspy.teleprompt.gepa.gepa import AUTO_RUN_SETTINGS
    from dspy.teleprompt.gepa.gepa_utils import LoggerAdapter

    valset = valset or trainset
    reflection_lm = reflection_lm or dspy.settings.lm
    teleprompter = GEPA(
        metric=avqa_gepa_feedback_metric,
        auto=auto,
        max_full_evals=max_full_evals,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=reflection_minibatch_size,
        candidate_selection_strategy=candidate_selection_strategy,
        reflection_lm=reflection_lm,
        skip_perfect_score=skip_perfect_score,
        use_merge=use_merge,
        max_merge_invocations=max_merge_invocations,
        num_threads=num_threads,
        seed=seed,
        log_dir=log_dir,
        track_stats=track_stats,
        track_best_outputs=track_best_outputs,
    )
    if teleprompter.auto is not None:
        teleprompter.max_metric_calls = teleprompter.auto_budget(
            num_preds=len(program.component_names),
            num_candidates=AUTO_RUN_SETTINGS[teleprompter.auto]["n"],
            valset_size=len(valset),
        )
    elif teleprompter.max_full_evals is not None:
        teleprompter.max_metric_calls = teleprompter.max_full_evals * (len(trainset) + len(valset))
    rng = random.Random(seed)
    adapter = PromptTargetGEPAAdapter(
        student_module=program,
        metric_fn=avqa_gepa_feedback_metric,
        failure_score=teleprompter.failure_score,
        num_threads=num_threads,
        rng=rng,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=reflection_minibatch_size,
        validation_examples=valset,
    )
    seed_candidate = {name: program._candidate_text(path) for path, name in zip(program.target_paths, adapter.component_names)}
    reflective_dataset_save_interval = _nonnegative_int(
        reflective_dataset_save_interval,
        name="reflective_dataset_save_interval",
    )
    file_handler, previous_logger_level = _start_gepa_file_logging(log_dir)
    try:
        callbacks = None
        if log_dir is not None and reflective_dataset_save_interval > 0:
            callbacks = [
                ReflectiveDatasetSnapshotCallback(
                    log_dir,
                    save_interval=reflective_dataset_save_interval,
                )
            ]
        elif log_dir is None and reflective_dataset_save_interval > 0:
            logger.warning(
                "GEPA reflective dataset snapshots are disabled because log_dir is not set"
            )
        result = optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=(lambda x: adapter.stripped_lm_call(x)[0]),
            candidate_selection_strategy=candidate_selection_strategy,
            skip_perfect_score=skip_perfect_score,
            reflection_minibatch_size=reflection_minibatch_size,
            module_selector="round_robin",
            perfect_score=teleprompter.perfect_score,
            use_merge=use_merge,
            max_merge_invocations=max_merge_invocations,
            max_metric_calls=teleprompter.max_metric_calls,
            logger=LoggerAdapter(logger),
            run_dir=log_dir,
            callbacks=callbacks,
            track_best_outputs=track_best_outputs,
            display_progress_bar=True,
            raise_on_exception=True,
            seed=seed,
        )
    finally:
        _stop_gepa_file_logging(file_handler, previous_logger_level)
    new_prog = adapter.build_program(result.best_candidate)
    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(result.candidates):
        for target_path, component_name in zip(program.target_paths, adapter.component_names):
            rows.append(_drop_none_values({
                "candidate_index": idx,
                "target_path": target_path,
                "prompt_text": candidate.get(component_name),
                "full_valset_score": result.val_aggregate_scores[idx] if idx < len(result.val_aggregate_scores) else None,
                "parents": result.parents[idx] if idx < len(result.parents) else None,
                "discovery_eval_count": result.discovery_eval_counts[idx] if idx < len(result.discovery_eval_counts) else None,
                "whether_selected_as_best": idx == result.best_idx,
            }))
    new_prog.prompt_target_candidate_rows = rows
    new_prog.prompt_target_gepa_result = result
    return new_prog


def optimize_with_copro(
    program: AVQADSPyReActProgram,
    trainset: list[dspy.Example],
    *,
    valset: list[dspy.Example] | None = None,
    breadth: int | None = None,
    depth: int | None = None,
    init_temperature: float | None = None,
    track_stats: bool = False,
) -> dspy.Module:
    """Compile the program with COPRO."""
    from dspy.teleprompt import COPRO

    copro_kwargs = {
        "breadth": breadth,
        "depth": depth,
        "init_temperature": init_temperature,
        "track_stats": track_stats,
    }
    copro_kwargs = {key: value for key, value in copro_kwargs.items() if value is not None}
    teleprompter = COPRO(metric=avqa_metric, **_filtered_kwargs(COPRO.__init__, copro_kwargs))
    compile_kwargs = _filtered_kwargs(teleprompter.compile, {"eval_kwargs": {}, "valset": valset})
    return teleprompter.compile(student=program, trainset=trainset, **compile_kwargs)


def optimize_with_simba(
    program: AVQADSPyReActProgram,
    trainset: list[dspy.Example],
    *,
    valset: list[dspy.Example] | None = None,
    bsize: int = 32,
    num_candidates: int = 6,
    max_steps: int = 8,
    max_demos: int = 4,
    num_threads: int | None = None,
    seed: int = 0,
    temperature_for_sampling: float | None = None,
    temperature_for_candidates: float | None = None,
) -> dspy.Module:
    """Compile the program with SIMBA."""
    from dspy.teleprompt import SIMBA

    simba_kwargs = {
        "bsize": bsize,
        "num_candidates": num_candidates,
        "max_steps": max_steps,
        "max_demos": max_demos,
        "num_threads": num_threads,
        "temperature_for_sampling": temperature_for_sampling,
        "temperature_for_candidates": temperature_for_candidates,
    }
    simba_kwargs = {key: value for key, value in simba_kwargs.items() if value is not None}
    teleprompter = SIMBA(metric=avqa_metric, **_filtered_kwargs(SIMBA.__init__, simba_kwargs))
    compile_kwargs = _filtered_kwargs(teleprompter.compile, {"seed": seed, "valset": valset})
    return teleprompter.compile(student=program, trainset=trainset, **compile_kwargs)


def optimize_with_miprov2(
    program: AVQADSPyReActProgram,
    trainset: list[dspy.Example],
    *,
    valset: list[dspy.Example] | None = None,
    auto: str | None = None,
    num_candidates: int | None = None,
    num_trials: int | None = None,
    max_bootstrapped_demos: int = 2,
    max_labeled_demos: int = 2,
    seed: int = 9,
    init_temperature: float = 1.0,
    num_threads: int | None = None,
    max_errors: int | None = None,
    minibatch: bool = True,
    minibatch_size: int = 16,
    minibatch_full_eval_steps: int = 5,
    view_data_batch_size: int = 8,
    track_stats: bool = True,
    log_dir: str | None = None,
) -> dspy.Module:
    """Compile the program with MIPROv2."""
    from dspy.teleprompt import MIPROv2

    mipro_kwargs = {
        "auto": auto,
        "num_candidates": num_candidates,
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "max_labeled_demos": max_labeled_demos,
        "seed": seed,
        "init_temperature": init_temperature,
        "num_threads": num_threads,
        "max_errors": max_errors,
        "track_stats": track_stats,
        "log_dir": log_dir,
    }
    # MIPROv2 defaults to auto="light"; keep an explicit auto=None so DSPy
    # does not reject the explicit num_candidates/num_trials settings below.
    mipro_kwargs = {key: value for key, value in mipro_kwargs.items() if value is not None or key == "auto"}
    teleprompter = MIPROv2(metric=avqa_metric, **_filtered_kwargs(MIPROv2.__init__, mipro_kwargs))
    compile_kwargs = {
        "num_trials": num_trials,
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "max_labeled_demos": max_labeled_demos,
        "seed": seed,
        "minibatch": minibatch,
        "minibatch_size": minibatch_size,
        "minibatch_full_eval_steps": minibatch_full_eval_steps,
        "view_data_batch_size": view_data_batch_size,
        "valset": valset,
    }
    compile_kwargs = {key: value for key, value in compile_kwargs.items() if value is not None}
    return teleprompter.compile(
        student=program,
        trainset=trainset,
        **_filtered_kwargs(teleprompter.compile, compile_kwargs),
    )


def optimize_with_gepa(
    program: AVQADSPyReActProgram,
    trainset: list[dspy.Example],
    *,
    valset: list[dspy.Example] | None = None,
    auto: str | None = None,
    max_full_evals: int | None = 6,
    max_metric_calls: int | None = None,
    reflection_minibatch_size: int = 3,
    candidate_selection_strategy: str = "pareto",
    skip_perfect_score: bool = True,
    use_merge: bool = True,
    max_merge_invocations: int | None = 5,
    num_threads: int | None = None,
    seed: int | None = 0,
    log_dir: str | None = None,
    track_stats: bool = False,
    track_best_outputs: bool = False,
) -> dspy.Module:
    """Compile the program with GEPA."""
    from dspy.teleprompt import GEPA

    gepa_kwargs = {
        "auto": auto,
        "max_full_evals": max_full_evals,
        "max_metric_calls": max_metric_calls,
        "reflection_minibatch_size": reflection_minibatch_size,
        "candidate_selection_strategy": candidate_selection_strategy,
        "reflection_lm": dspy.settings.lm,
        "skip_perfect_score": skip_perfect_score,
        "use_merge": use_merge,
        "max_merge_invocations": max_merge_invocations,
        "num_threads": num_threads,
        "seed": seed,
        "log_dir": log_dir,
        "track_stats": track_stats,
        "track_best_outputs": track_best_outputs,
    }
    gepa_kwargs = {key: value for key, value in gepa_kwargs.items() if value is not None}
    teleprompter = GEPA(
        metric=avqa_gepa_feedback_metric,
        **_filtered_kwargs(GEPA.__init__, gepa_kwargs),
    )
    compile_kwargs = _filtered_kwargs(teleprompter.compile, {"valset": valset})
    return teleprompter.compile(student=program, trainset=trainset, **compile_kwargs)


def parse_optimize_args() -> argparse.Namespace:
    """Parse optimizer CLI arguments."""
    parser = argparse.ArgumentParser(description="Optimize DSPy AVQA ReAct with DSPy optimizers.")
    parser.add_argument("--algorithm", choices=("copro", "simba", "miprov2", "gepa"), default="copro")
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help=(
            "Skip optimization, load --output-program, and run/resume final test inference. "
            "Requires --final-eval-output-jsonl."
        ),
    )
    parser.add_argument(
        "--optimization-train-only",
        action="store_true",
        help=(
            "Run optimization and save its artifacts, then stop before final-test inference. "
            "Cannot be combined with --inference-only or --final-eval-output-jsonl."
        ),
    )
    parser.add_argument(
        "--optimize-target",
        action="append",
        dest="optimize_targets",
        default=None,
        help="Prompt-config dot path to optimize. Repeat to jointly optimize multiple targets.",
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--trainset-jsonl", type=Path, default=None)
    parser.add_argument("--valset-jsonl", type=Path, default=None)
    parser.add_argument("--data-seed", type=int, default=int(os.environ.get("DSPY_AVQA_DATA_SEED", "0")))
    parser.add_argument("--audio-caption-dir", type=Path, default=None)
    parser.add_argument(
        "--caption-cache-dir",
        type=Path,
        default=None,
        help=(
            "Question-scoped cache used by ask_caption. A cache-routed call never "
            "falls back to live captioning when its entry is missing or its prompt "
            "does not match."
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
        "--daily-omni-root",
        type=Path,
        default=(Path(os.environ["DAILY_OMNI_ROOT"]) if os.environ.get("DAILY_OMNI_ROOT") else None),
        help=(
            "DailyOmni dataset root containing Videos/<video_id>/ caption annotations. "
            "Used only when GEPA caption supervision mode resolves to privileged; "
            "defaults to DAILY_OMNI_ROOT."
        ),
    )
    parser.add_argument(
        "--gepa-caption-supervision",
        choices=GEPA_CAPTION_SUPERVISION_CHOICES,
        default=os.environ.get("GEPA_CAPTION_SUPERVISION", "auto"),
        help=(
            "Captioner GEPA supervision mode: none uses only downstream answer feedback; "
            "privileged loads DailyOmni caption supervision; auto uses privileged only "
            "when --daily-omni-root/DAILY_OMNI_ROOT is available."
        ),
    )
    parser.add_argument(
        "--gepa-densified-label-dir",
        type=Path,
        default=(
            Path(os.environ["GEPA_DENSIFIED_LABEL_DIR"])
            if os.environ.get("GEPA_DENSIFIED_LABEL_DIR")
            else None
        ),
        help=(
            "Optional directory containing per-cut key_evidence JSON labels. Legacy "
            "files may also contain ideal_perception_target, which is ignored. Labels "
            "are loaded only as GEPA reflection supervision for training examples and "
            "are never passed to runtime modules."
        ),
    )
    parser.add_argument(
        "--gepa-reflection-template-yaml",
        type=Path,
        default=(
            Path(os.environ[GEPA_REFLECTION_TEMPLATE_YAML_ENV])
            if os.environ.get(GEPA_REFLECTION_TEMPLATE_YAML_ENV)
            else _default_gepa_reflection_template_yaml_path()
        ),
        help=(
            "YAML file containing versioned GEPA reflection prompt templates. "
            "Defaults to DSPy/dspy_avqa/yamls/DSPy/reflection_template.yaml."
        ),
    )
    parser.add_argument(
        "--gepa-reflection-template-version",
        default=os.environ.get(
            GEPA_REFLECTION_TEMPLATE_VERSION_ENV,
            GEPA_REFLECTION_TEMPLATE_VERSION_AUTO,
        ),
        help=(
            "GEPA reflection template version to load from --gepa-reflection-template-yaml. "
            "Use 'auto' to select original when no densified labels are loaded and "
            "densified_key_evidence when --gepa-densified-label-dir is set."
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
    parser.add_argument("--output-program", type=Path, default=None)
    parser.add_argument(
        "--initial-program",
        type=Path,
        default=None,
        help="Optional JSON file saving the unoptimized starting program.",
    )
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument(
        "--signature-search-json",
        type=Path,
        default=None,
        help="Optional JSON file recording optimizer candidate signatures.",
    )
    parser.add_argument(
        "--trajectory-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file recording final optimized program trajectories on the selected trainset.",
    )
    parser.add_argument(
        "--optimizer-log-dir",
        type=Path,
        default=None,
        help="Directory for normalized optimizer candidate logs. Defaults beside output-program.",
    )
    parser.add_argument(
        "--optimized-prompt-config-yaml",
        type=Path,
        default=None,
        help="Optional YAML path for the optimized prompt config artifact.",
    )
    parser.add_argument(
        "--final-eval-output-jsonl",
        type=Path,
        default=None,
        help="Optional batch-runner-style JSONL for the optimized program on the full input JSONL.",
    )
    parser.add_argument(
        "--final-eval-output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for resumable per-sample JSON files from "
            "--final-eval-output-jsonl. Defaults to the --output-program directory."
        ),
    )
    parser.add_argument(
        "--final-eval-num-threads",
        type=int,
        default=int(os.environ.get("DSPY_AVQA_FINAL_EVAL_NUM_THREADS", "4")),
        help=(
            "Concurrent ReAct rollouts for final evaluation (default: 4). "
            "Reliable-Qwen retries are always serial."
        ),
    )
    parser.add_argument(
        "--final-eval-batch-size",
        type=int,
        default=None,
        help=(
            "Primary rollouts per final-evaluation reliable-Qwen batch. Defaults to "
            "--final-eval-num-threads."
        ),
    )
    parser.add_argument("--max-turns", type=int, default=int(os.environ.get("DEFAULT_MAX_TURNS", "4")))
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="Use only the first debug-limit samples.")
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=int(os.environ.get("DEBUG_LIMIT", "4")),
        help="Number of leading samples to optimize on when --debug is enabled.",
    )
    parser.add_argument(
        "--skip-bad-examples",
        action="store_true",
        help="Skip cuts with missing captions/audio/options instead of failing fast.",
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
        "--signature-in-system-prompt",
        action="store_true",
        default=_env_flag_value("DSPY_AVQA_SIGNATURE_IN_SYSTEM_PROMPT"),
        help=(
            "Evaluate optimizer rollouts with the DSPy-rendered PlanNextAction "
            "signature/schema prompt kept as the chat-level system message. This "
            "does not change --optimize-target."
        ),
    )
    parser.add_argument(
        "--caption-placement",
        choices=CAPTION_PLACEMENT_CHOICES,
        default=normalize_caption_placement(),
        help=(
            "Where to place caption tool observations in later planner turns. "
            "This changes rollout prompt topology but does not change --optimize-target."
        ),
    )
    parser.add_argument(
        "--allowed-tools",
        default=os.environ.get("DSPY_AVQA_ALLOWED_TOOLS") or os.environ.get("DSPY_ALLOWED_TOOLS"),
        help="Comma-separated DSPy AVQA tools to expose to the planner.",
    )
    parser.add_argument(
        "--perception-model",
        choices=("qwen", "gemini"),
        default=os.environ.get("PERCEPTION_MODEL", "qwen").strip().lower() or "qwen",
        help="Perceptual backend used by DSPy tools.",
    )
    add_gemini_backend_args(parser)
    parser.add_argument("--copro-breadth", type=int, default=None)
    parser.add_argument("--copro-depth", type=int, default=None)
    parser.add_argument("--copro-init-temperature", type=float, default=None)
    parser.add_argument("--copro-track-stats", action="store_true")
    parser.add_argument("--simba-bsize", type=int, default=32)
    parser.add_argument("--simba-num-candidates", type=int, default=6)
    parser.add_argument("--simba-max-steps", type=int, default=8)
    parser.add_argument("--simba-max-demos", type=int, default=4)
    parser.add_argument("--simba-num-threads", type=int, default=None)
    parser.add_argument("--simba-seed", type=int, default=0)
    parser.add_argument("--simba-temperature-for-sampling", type=float, default=None)
    parser.add_argument("--simba-temperature-for-candidates", type=float, default=None)
    parser.add_argument("--miprov2-config", type=Path, default=None, help="JSON/YAML file with MIPROv2 optimizer parameters.")
    parser.add_argument("--miprov2-auto", choices=("none", "light", "medium", "heavy"), default="none")
    parser.add_argument("--miprov2-num-candidates", type=int, default=6)
    parser.add_argument("--miprov2-num-trials", type=int, default=10)
    parser.add_argument("--miprov2-max-bootstrapped-demos", type=int, default=2)
    parser.add_argument("--miprov2-max-labeled-demos", type=int, default=2)
    parser.add_argument("--miprov2-seed", type=int, default=9)
    parser.add_argument("--miprov2-init-temperature", type=float, default=1.0)
    parser.add_argument("--miprov2-num-threads", type=int, default=None)
    parser.add_argument("--miprov2-max-errors", type=int, default=None)
    parser.add_argument("--miprov2-no-minibatch", action="store_true")
    parser.add_argument("--miprov2-minibatch-size", type=int, default=16)
    parser.add_argument("--miprov2-minibatch-full-eval-steps", type=int, default=5)
    parser.add_argument("--miprov2-view-data-batch-size", type=int, default=8)
    parser.add_argument("--miprov2-no-track-stats", action="store_true")
    parser.add_argument("--miprov2-log-dir", type=str, default=None)
    parser.add_argument("--gepa-config", type=Path, default=None, help="JSON/YAML file with GEPA optimizer parameters.")
    parser.add_argument(
        "--gepa-reflection-config-yaml",
        type=Path,
        default=None,
        help="Optional reasoner YAML whose model/sampling values override the GEPA reflection LM.",
    )
    parser.add_argument("--gepa-auto", choices=("none", "light", "medium", "heavy"), default="none")
    parser.add_argument("--gepa-max-full-evals", type=int, default=6)
    parser.add_argument("--gepa-max-metric-calls", type=int, default=None)
    parser.add_argument("--gepa-reflection-minibatch-size", type=int, default=3)
    parser.add_argument("--gepa-candidate-selection-strategy", choices=("pareto", "current_best"), default="pareto")
    parser.add_argument("--gepa-dont-skip-perfect-score", action="store_true")
    parser.add_argument("--gepa-no-merge", action="store_true")
    parser.add_argument("--gepa-max-merge-invocations", type=int, default=5)
    parser.add_argument("--gepa-num-threads", type=int, default=None)
    parser.add_argument("--gepa-seed", type=int, default=0)
    parser.add_argument("--gepa-log-dir", type=str, default=None)
    parser.add_argument(
        "--gepa-reflective-dataset-save-interval",
        type=int,
        default=DEFAULT_GEPA_REFLECTIVE_DATASET_SAVE_INTERVAL,
        help="Save the first GEPA reflective dataset and then every Nth one; 0 disables snapshots.",
    )
    parser.add_argument("--gepa-track-stats", action="store_true")
    parser.add_argument("--gepa-track-best-outputs", action="store_true")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the same redacted resolved experiment config saved beside output-program.",
    )
    args = parser.parse_args()
    _apply_gepa_run_config(args, parser)
    args.optimize_targets = tuple(args.optimize_targets or ("planner.workflow_prompt",))
    if args.output_program is None:
        parser.error("--output-program is required unless supplied by --gepa-config run.output_program")
    if args.inference_only and args.final_eval_output_jsonl is None:
        parser.error("--inference-only requires --final-eval-output-jsonl")
    if args.inference_only and args.optimization_train_only:
        parser.error("--inference-only and --optimization-train-only cannot be combined")
    if args.optimization_train_only and args.final_eval_output_jsonl is not None:
        parser.error("--optimization-train-only cannot be combined with --final-eval-output-jsonl")
    if args.final_eval_output_jsonl is not None and args.final_eval_output_dir is None:
        args.final_eval_output_dir = args.output_program.parent
    if args.final_eval_num_threads < 1:
        parser.error("--final-eval-num-threads must be >= 1")
    if args.final_eval_batch_size is not None and args.final_eval_batch_size < 1:
        parser.error("--final-eval-batch-size must be >= 1")
    return args


def _save_program(program: dspy.Module, output_program: Path) -> None:
    output_program.parent.mkdir(parents=True, exist_ok=True)
    save_method = getattr(program, "save", None)
    if save_method is None:
        raise AttributeError(f"{program.__class__.__name__} does not expose a save() method")
    save_method(str(output_program))


def _json_safe(value: Any) -> Any:
    """Convert metadata values into JSON-serializable objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    for converter_name in ("model_dump", "to_dict"):
        converter = getattr(value, converter_name, None)
        if callable(converter):
            try:
                return _json_safe(converter())
            except Exception:
                logger.debug(
                    "Failed to serialize %s via %s",
                    type(value).__name__,
                    converter_name,
                    exc_info=True,
                )
    return str(value)


def _field_summary(field: Any) -> dict[str, Any]:
    extra = getattr(field, "json_schema_extra", None) or {}
    return {
        "prefix": extra.get("prefix"),
        "description": extra.get("desc") or getattr(field, "description", None),
    }


def _signature_summary(signature: Any) -> dict[str, Any]:
    fields = getattr(signature, "fields", None) or getattr(signature, "model_fields", None) or {}
    return {
        "instructions": getattr(signature, "instructions", None),
        "fields": {name: _field_summary(field) for name, field in fields.items()},
    }


def _program_signature_summary(program: dspy.Module) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    try:
        named_predictors = list(program.named_predictors())
    except Exception:
        named_predictors = []
    if named_predictors:
        iterator = named_predictors
    else:
        iterator = [(f"predictor_{idx}", predictor) for idx, predictor in enumerate(program.predictors())]
    for name, predictor in iterator:
        signature = getattr(predictor, "signature", None)
        summaries.append({
            "name": name,
            "signature": _signature_summary(signature) if signature is not None else None,
        })
    return summaries


def write_signature_search_json(
    program: dspy.Module,
    output_path: Path,
    *,
    initial_program_signatures: list[dict[str, Any]] | None = None,
) -> None:
    """Write starting, optimized, and candidate signatures kept by the optimizer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(getattr(program, "candidate_programs", []) or []):
        candidate_program = candidate.get("program") if isinstance(candidate, dict) else None
        rows.append({
            "candidate_index": idx,
            "score": candidate.get("score") if isinstance(candidate, dict) else None,
            "depth": candidate.get("depth") if isinstance(candidate, dict) else None,
            "instruction": candidate.get("instruction") if isinstance(candidate, dict) else None,
            "prefix": candidate.get("prefix") if isinstance(candidate, dict) else None,
            "signatures": _program_signature_summary(candidate_program) if candidate_program is not None else [],
        })

    payload = {
        "total_evaluate_calls": getattr(program, "total_calls", None),
        "initial_program_signatures": initial_program_signatures or [],
        "best_program_signatures": _program_signature_summary(program),
        "candidate_programs": rows,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2, default=str)


def _example_summary(example: dspy.Example) -> dict[str, Any]:
    return {
        "question": getattr(example, "question", None),
        "options_json": getattr(example, "options_json", None),
        "video_id": getattr(example, "video_id", None),
        "video_path": getattr(example, "video_path", None),
        "audio_path": getattr(example, "audio_path", None),
        "answer": getattr(example, "answer", None),
    }


def write_final_trajectories_jsonl(program: dspy.Module, trainset: list[dspy.Example], output_path: Path) -> None:
    """Run the final optimized program on trainset and write per-example trajectories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, example in enumerate(trainset):
            row: dict[str, Any] = {
                "example_index": idx,
                "example": _example_summary(example),
                "gold_answer": normalize_option_letter(str(getattr(example, "answer", ""))),
            }
            try:
                pred = program(**example.inputs())
                pred_answer = normalize_option_letter(str(getattr(pred, "answer", "")))
                row.update({
                    "pred_answer": pred_answer,
                    "score": avqa_metric(example, pred),
                    "reasoning_summary": getattr(pred, "reasoning_summary", None),
                    "turn_trace": getattr(pred, "turn_trace", []),
                    "accumulated_evidence": getattr(pred, "accumulated_evidence", None),
                    "error": None,
                })
            except Exception as exc:
                row.update({
                    "pred_answer": "",
                    "score": 0.0,
                    "reasoning_summary": None,
                    "turn_trace": [],
                    "accumulated_evidence": None,
                    "error": {
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                })
            f.write(json.dumps(_json_safe(row), ensure_ascii=False, default=str) + "\n")




def _drop_none_values(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if value is not None}


def _safe_filename_part(value: Any) -> str:
    text = str(value or "unknown")
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return cleaned[:120] or "unknown"


def _write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False, default=str) + "\n")


def _candidate_predictor_rows(
    *,
    algorithm: str,
    candidate_index: int,
    candidate_program: dspy.Module | None,
    base_row: dict[str, Any],
    instructions_dir: Path,
) -> list[dict[str, Any]]:
    if candidate_program is None:
        return []

    rows: list[dict[str, Any]] = []
    for predictor_idx, predictor_summary in enumerate(_program_signature_summary(candidate_program)):
        predictor_name = predictor_summary.get("name") or f"predictor_{predictor_idx}"
        signature = predictor_summary.get("signature") or {}
        instruction_text = signature.get("instructions")
        row = {
            "algorithm": algorithm,
            "candidate_index": candidate_index,
            "predictor_name": predictor_name,
            **base_row,
        }
        if instruction_text is not None:
            instruction_path = instructions_dir / f"candidate_{candidate_index:04d}_{_safe_filename_part(predictor_name)}.txt"
            _write_text_file(instruction_path, str(instruction_text))
            row["instruction_text"] = instruction_text
            row["instruction_path"] = str(instruction_path)
        rows.append(_drop_none_values(row))
    return rows


def write_optimizer_candidate_logs(program: dspy.Module, algorithm: str, output_dir: Path) -> dict[str, str]:
    """Write normalized candidate logs for GEPA, MIPROv2, and COPRO when available."""
    output_dir.mkdir(parents=True, exist_ok=True)
    instructions_dir = output_dir / "instructions"
    candidate_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    details = getattr(program, "detailed_results", None)
    if algorithm == "gepa" and details is not None:
        candidates = list(getattr(details, "candidates", []) or [])
        best_idx = getattr(details, "best_idx", None)
        aggregate_scores = list(getattr(details, "val_aggregate_scores", []) or [])
        subscores = list(getattr(details, "val_subscores", []) or [])
        discovery_counts = list(getattr(details, "discovery_eval_counts", []) or [])
        parents = list(getattr(details, "parents", []) or [])
        for candidate_index, candidate_program in enumerate(candidates):
            per_example_scores = None
            if candidate_index < len(subscores):
                per_example_scores = {str(idx): score for idx, score in enumerate(subscores[candidate_index])}
            base_row = _drop_none_values({
                "iteration": discovery_counts[candidate_index] if candidate_index < len(discovery_counts) else None,
                "full_valset_score": aggregate_scores[candidate_index] if candidate_index < len(aggregate_scores) else None,
                "per_example_score_dict": per_example_scores,
                "whether_selected_as_best": candidate_index == best_idx if best_idx is not None else None,
                "parents": parents[candidate_index] if candidate_index < len(parents) else None,
                "discovery_eval_count": discovery_counts[candidate_index] if candidate_index < len(discovery_counts) else None,
            })
            score_rows.append(_drop_none_values({
                "algorithm": algorithm,
                "candidate_index": candidate_index,
                **base_row,
            }))
            candidate_rows.extend(_candidate_predictor_rows(
                algorithm=algorithm,
                candidate_index=candidate_index,
                candidate_program=candidate_program,
                base_row=base_row,
                instructions_dir=instructions_dir,
            ))
    else:
        candidate_index = 0
        sources = [
            ("candidate_programs", "full"),
            ("mb_candidate_programs", "subsample"),
        ]
        for attr_name, eval_scope in sources:
            for candidate in list(getattr(program, attr_name, []) or []):
                if not isinstance(candidate, dict):
                    continue
                candidate_program = candidate.get("program")
                score = candidate.get("score")
                base_row = _drop_none_values({
                    "iteration": candidate.get("depth") if "depth" in candidate else candidate.get("trial"),
                    "full_valset_score": score if eval_scope == "full" else None,
                    "subsample_score": score if eval_scope == "subsample" else None,
                    "whether_selected_as_best": candidate_index == 0 and eval_scope == "full",
                    "evaluation_scope": eval_scope,
                    "prefix": candidate.get("prefix"),
                })
                score_rows.append(_drop_none_values({
                    "algorithm": algorithm,
                    "candidate_index": candidate_index,
                    **base_row,
                }))
                candidate_rows.extend(_candidate_predictor_rows(
                    algorithm=algorithm,
                    candidate_index=candidate_index,
                    candidate_program=candidate_program,
                    base_row=base_row,
                    instructions_dir=instructions_dir,
                ))
                candidate_index += 1

        if not candidate_rows:
            base_row = {"whether_selected_as_best": True, "evaluation_scope": "final_program"}
            score_rows.append({"algorithm": algorithm, "candidate_index": 0, **base_row})
            candidate_rows.extend(_candidate_predictor_rows(
                algorithm=algorithm,
                candidate_index=0,
                candidate_program=program,
                base_row=base_row,
                instructions_dir=instructions_dir,
            ))

    candidates_path = output_dir / "optimizer_candidates.jsonl"
    scores_path = output_dir / "optimizer_candidate_scores.jsonl"
    summary_path = output_dir / "optimizer_summary.json"
    _write_jsonl_rows(candidates_path, candidate_rows)
    _write_jsonl_rows(scores_path, score_rows)

    summary = {
        "algorithm": algorithm,
        "candidate_rows": len(candidate_rows),
        "score_rows": len(score_rows),
        "candidates_jsonl": str(candidates_path),
        "candidate_scores_jsonl": str(scores_path),
        "instructions_dir": str(instructions_dir),
        "fields": [
            "candidate_index",
            "iteration",
            "predictor_name",
            "instruction_text",
            "instruction_path",
            "full_valset_score",
            "subsample_score",
            "per_example_score_dict",
            "whether_selected_as_best",
        ],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved optimizer candidate logs to {output_dir}")
    return {
        "optimizer_log_dir": str(output_dir),
        "optimizer_candidates_jsonl": str(candidates_path),
        "optimizer_candidate_scores_jsonl": str(scores_path),
        "optimizer_summary_json": str(summary_path),
    }


def _run_program_on_cut(
    program: dspy.Module,
    cut: dict[str, Any],
    audio_caption_dir: Path | None,
    max_turns: int,
) -> dict[str, Any]:
    payload = build_input_state(cut, audio_caption_dir)
    try:
        pred = program(
            question_id=payload.get("question_id"),
            question=payload["question"],
            options_json=json.dumps(payload["options"], ensure_ascii=False),
            video_path=payload["video_path"],
            audio_path=payload["audio_path"],
            video_id=payload.get("video_id"),
            video_description=payload.get("video_description"),
            max_turns=max_turns,
        )
        answer = normalize_option_letter(str(getattr(pred, "answer", "")).strip())
        reasoning_summary = str(getattr(pred, "reasoning_summary", "")).strip()
        response_text = f"{answer}. {reasoning_summary}" if reasoning_summary else answer
        row = build_result_row(cut, response_text)
        row["question_data"]["turn_trace"] = list(getattr(pred, "turn_trace", []))
        return row
    except Exception as exc:
        error_info = extract_error_info(exc)
        planner_calls = consume_planner_call_trace()
        if planner_calls:
            error_info["planner_calls"] = planner_calls
        row = build_result_row(cut, f"[ERROR] {exc}")
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
                "tool_error": str(exc),
                "planner_lm_response": error_info.get("planner_lm_response"),
                "planner_calls": error_info.get("planner_calls", []),
                "planner_error": error_info,
            }
        ]
        return row




def _select_fallback_train_cuts(
    cuts: list[dict[str, Any]],
    *,
    train_limit: int | None,
    debug: bool,
    debug_limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    pool = cuts[: debug_limit] if debug else list(cuts)
    if train_limit is None:
        return pool
    rng = random.Random(seed)
    if train_limit >= len(pool):
        shuffled = list(pool)
        rng.shuffle(shuffled)
        return shuffled
    return rng.sample(pool, train_limit)


def _build_examples_from_cuts(
    cuts: list[dict[str, Any]],
    audio_caption_dir: Path | None,
    *,
    max_turns: int,
    skip_bad_examples: bool,
    gepa_caption_root: Path | None = None,
    gepa_caption_cache: dict[str, dict[str, str]] | None = None,
    gepa_caption_stats: dict[str, int] | None = None,
    gepa_densified_label_dir: Path | None = None,
) -> tuple[list[dspy.Example], list[dict[str, Any]]]:
    return make_trainset_from_cuts(
        cuts,
        audio_caption_dir,
        max_turns=max_turns,
        skip_bad_examples=skip_bad_examples,
        gepa_caption_root=gepa_caption_root,
        gepa_caption_cache=gepa_caption_cache,
        gepa_caption_stats=gepa_caption_stats,
        gepa_densified_label_dir=gepa_densified_label_dir,
    )


def resolve_optimization_datasets(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve train/val examples while preserving legacy defaults."""
    input_cuts = read_jsonl(args.input_jsonl)
    optimizes_captioner = any(_is_captioner_target(target) for target in args.optimize_targets)
    gepa_caption_supervision_mode, gepa_caption_root = _resolve_gepa_caption_supervision(
        args,
        optimizes_captioner=optimizes_captioner,
    )
    use_gepa_caption_supervision = gepa_caption_supervision_mode == "privileged"
    gepa_densified_label_dir = _resolve_gepa_densified_label_dir(args)
    gepa_caption_cache: dict[str, dict[str, str]] = {}
    gepa_caption_stats = _new_gepa_caption_supervision_stats()

    if args.trainset_jsonl is not None:
        train_source_path = args.trainset_jsonl
        train_source_cuts = read_jsonl(args.trainset_jsonl)
        train_cuts = train_source_cuts[: args.debug_limit] if args.debug else train_source_cuts
        train_selection = "explicit_trainset_jsonl"
    else:
        train_source_path = args.input_jsonl
        train_source_cuts = input_cuts
        train_cuts = _select_fallback_train_cuts(
            input_cuts,
            train_limit=args.train_limit,
            debug=args.debug,
            debug_limit=args.debug_limit,
            seed=args.data_seed,
        )
        train_selection = "random_sample_from_input_jsonl" if args.train_limit is not None else "input_jsonl"

    trainset, skipped_train = _build_examples_from_cuts(
        train_cuts,
        args.audio_caption_dir,
        max_turns=args.max_turns,
        skip_bad_examples=args.skip_bad_examples,
        gepa_caption_root=gepa_caption_root,
        gepa_caption_cache=gepa_caption_cache,
        gepa_caption_stats=gepa_caption_stats,
        gepa_densified_label_dir=gepa_densified_label_dir,
    )
    if not trainset:
        raise ValueError("No train examples were built; check train/input JSONL and audio-caption-dir.")

    if args.valset_jsonl is not None:
        val_source_path = args.valset_jsonl
        val_source_cuts = read_jsonl(args.valset_jsonl)
        val_cuts = val_source_cuts[: args.debug_limit] if args.debug else val_source_cuts
        valset, skipped_val = _build_examples_from_cuts(
            val_cuts,
            args.audio_caption_dir,
            max_turns=args.max_turns,
            skip_bad_examples=args.skip_bad_examples,
            gepa_caption_root=gepa_caption_root,
            gepa_caption_cache=gepa_caption_cache,
            gepa_caption_stats=gepa_caption_stats,
        )
        if not valset:
            raise ValueError("No val examples were built; check valset JSONL and audio-caption-dir.")
        val_selection = "explicit_valset_jsonl"
        valset_is_trainset = False
    else:
        val_source_path = train_source_path
        val_source_cuts = train_source_cuts
        val_cuts = train_cuts
        valset = trainset
        skipped_val = []
        val_selection = "trainset"
        valset_is_trainset = True

    caption_cache_coverage = None
    if getattr(args, "caption_cache_dir", None) is not None:
        cache_question_ids = list(
            dict.fromkeys(
                cut_id(cut)
                for source_cuts in (input_cuts, train_source_cuts, val_source_cuts)
                for cut in source_cuts
            )
        )
        caption_cache_coverage = validate_caption_cache_coverage(
            args.caption_cache_dir,
            cache_question_ids,
            expected_prompt=build_caption_prompt(),
            allow_entry_errors=_caption_cache_allows_missing(),
        )

    return {
        "input_cuts": input_cuts,
        "caption_cache_coverage": caption_cache_coverage,
        "train_source_path": train_source_path,
        "train_source_cuts": train_source_cuts,
        "train_cuts": train_cuts,
        "trainset": trainset,
        "skipped_train": skipped_train,
        "train_selection": train_selection,
        "val_source_path": val_source_path,
        "val_source_cuts": val_source_cuts,
        "val_cuts": val_cuts,
        "valset": valset,
        "skipped_val": skipped_val,
        "val_selection": val_selection,
        "valset_is_trainset": valset_is_trainset,
        "gepa_caption_supervision": {
            "mode": gepa_caption_supervision_mode,
            "enabled": use_gepa_caption_supervision,
            "requested_mode": _normalize_gepa_caption_supervision_mode(
                getattr(args, "gepa_caption_supervision", "auto")
            ),
            "daily_omni_root": str(gepa_caption_root) if gepa_caption_root else None,
            "files": [filename for _, filename in GEPA_PRIVILEGED_CAPTION_FILES],
            **gepa_caption_stats,
        },
        "gepa_densified_supervision": {
            "enabled": gepa_densified_label_dir is not None,
            "label_dir": str(gepa_densified_label_dir) if gepa_densified_label_dir else None,
            "labels_loaded": len(trainset) if gepa_densified_label_dir else 0,
            "fields": [source_key for _, source_key in GEPA_DENSIFIED_LABEL_FIELDS],
        },
    }


@contextmanager
def _final_eval_directory_lock(output_dir: Path):
    """Prevent concurrent final-eval writers from sharing one cache directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".final_eval.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Final test inference is already running in {output_dir}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def write_batch_style_program_outputs(
    program: dspy.Module,
    cuts: list[dict[str, Any]],
    audio_caption_dir: Path | None,
    output_jsonl: Path,
    *,
    output_dir: Path | None = None,
    max_turns: int,
    skip_bad_examples: bool = False,
    num_threads: int = 1,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run final test inference, resuming from valid per-sample JSON files.

    Primary rollouts may run concurrently.  Empty Qwen observations are retried
    only after the primary batch completes, under the serial fallback layouts.
    """
    if num_threads < 1:
        raise ValueError("num_threads must be >= 1")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    cache_dir = output_dir or output_jsonl.parent
    rows_by_sample_id: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    cached_samples = 0
    new_samples = 0
    exhausted_qwen_samples = 0

    with _final_eval_directory_lock(cache_dir):
        remaining: list[tuple[int, dict[str, Any]]] = []
        for idx, cut in enumerate(cuts):
            cached_row = load_cached_row_from_question_json(cache_dir, cut)
            if cached_row is None:
                remaining.append((idx, cut))
                continue
            rows_by_sample_id[cut_id(cut)] = cached_row
            cached_samples += 1

        print(f"Cached final-test samples: {cached_samples}")
        print(f"Remaining final-test samples: {len(remaining)}")
        reliable_qwen = ReliableQwenExecutor()
        resolved_batch_size = batch_size or num_threads
        print(
            "Final-eval rollout config: "
            f"primary_threads={num_threads}, "
            f"reliable_batch_size={resolved_batch_size}, "
            f"qwen_batch_retries={reliable_qwen.max_batch_retries}"
        )

        def run_item(item: tuple[int, dict[str, Any]]) -> tuple[dict[str, Any] | None, Exception | None]:
            _, cut = item
            try:
                return _run_program_on_cut(program, cut, audio_caption_dir, max_turns), None
            except Exception as exc:
                return None, exc

        def has_empty_qwen(result: tuple[dict[str, Any] | None, Exception | None]) -> bool:
            row, error = result
            if error is not None or not isinstance(row, dict):
                return False
            question_data = row.get("question_data")
            return isinstance(question_data, dict) and turn_trace_has_empty_qwen_observation(
                question_data.get("turn_trace") or []
            )

        for batch_start in range(0, len(remaining), resolved_batch_size):
            rollout_batch = remaining[batch_start : batch_start + resolved_batch_size]
            recovered = reliable_qwen.run_batch(
                rollout_batch,
                run_item=run_item,
                has_empty_qwen_response=has_empty_qwen,
                max_workers=num_threads,
            )
            exhausted_indices = set(recovered.exhausted_indices)
            exhausted_qwen_samples += len(exhausted_indices)
            if exhausted_indices:
                print(
                    "Qwen recovery exhausted for "
                    f"{len(exhausted_indices)} final-eval sample(s) in batch "
                    f"{batch_start // resolved_batch_size + 1}; recording [ERROR] rows."
                )

            for batch_index, ((idx, cut), result) in enumerate(zip(rollout_batch, recovered.results)):
                row, error = result
                if error is not None:
                    if not skip_bad_examples:
                        raise error
                    skipped.append({
                        "example_index": idx,
                        "cut_id": str(cut.get("id") or ""),
                        "error": str(error),
                    })
                    continue
                assert row is not None
                if batch_index in exhausted_indices:
                    question_data = row.setdefault("question_data", {})
                    if isinstance(question_data, dict):
                        mark_empty_qwen_turn_trace(
                            question_data.get("turn_trace") or [],
                            attempted_profiles=recovered.attempted_profiles,
                        )
                        question_data["response"] = (
                            "[ERROR] Qwen returned no visible content after reliable recovery"
                        )
                maybe_dump_question_data(cache_dir, row)
                rows_by_sample_id[cut_id(cut)] = row
                new_samples += 1

        rows = [
            rows_by_sample_id[cut_id(cut)]
            for cut in cuts
            if cut_id(cut) in rows_by_sample_id
        ]
        write_results_jsonl(rows, output_jsonl)

    print(f"Wrote {len(rows)} optimized-program rows to {output_jsonl}")
    return {
        "final_eval_output_jsonl": str(output_jsonl),
        "final_eval_output_dir": str(cache_dir),
        "final_eval_total_samples": len(cuts),
        "final_eval_cached_samples": cached_samples,
        "final_eval_new_samples": new_samples,
        "final_eval_qwen_exhausted_samples": exhausted_qwen_samples,
        "final_eval_rows": len(rows),
        "final_eval_skipped_examples": skipped,
        "final_eval_complete": len(rows) + len(skipped) == len(cuts),
    }


def _optimized_prompt_config(program: dspy.Module, target_paths: tuple[str, ...]) -> dict[str, Any]:
    config = copy.deepcopy(prompt_config())
    for target_path in target_paths:
        _set_nested(config, target_path, _optimized_prompt_text_for_target(program, target_path))
    return config


def _optimized_prompt_text_for_target(program: dspy.Module, target_path: str) -> str:
    return _prompt_component_text(program, target_path)


def _candidate_prompt_text_for_target(program: dspy.Module | None, target_path: str) -> str:
    if program is None:
        return ""
    return _prompt_component_text(program, target_path)


def _candidate_prompt_rows_from_program(program: dspy.Module, target_path: str) -> list[dict[str, Any]]:
    rows = list(getattr(program, "prompt_target_candidate_rows", []) or [])
    if rows:
        return [row for row in rows if row.get("target_path", target_path) == target_path]
    out: list[dict[str, Any]] = []
    candidate_index = 0
    for attr_name, eval_scope in (("candidate_programs", "full"), ("mb_candidate_programs", "subsample")):
        for candidate in list(getattr(program, attr_name, []) or []):
            if not isinstance(candidate, dict):
                continue
            candidate_program = candidate.get("program")
            prompt_text = _candidate_prompt_text_for_target(candidate_program, target_path)
            if not prompt_text:
                continue
            out.append(_drop_none_values({
                "candidate_index": candidate_index,
                "prompt_text": prompt_text,
                "full_valset_score": candidate.get("score") if eval_scope == "full" else None,
                "subsample_score": candidate.get("score") if eval_scope == "subsample" else None,
                "iteration": candidate.get("depth") if "depth" in candidate else candidate.get("trial"),
                "whether_selected_as_best": candidate_index == 0 and eval_scope == "full",
                "evaluation_scope": eval_scope,
            }))
            candidate_index += 1
    if not out:
        out.append({
            "candidate_index": 0,
            "prompt_text": _optimized_prompt_text_for_target(program, target_path),
            "whether_selected_as_best": True,
        })
    return out

def write_prompt_target_artifacts(
    program: dspy.Module,
    target_paths: tuple[str, ...],
    output_program: Path,
    optimizer_log_dir: Path,
    optimized_prompt_config_yaml: Path | None = None,
) -> dict[str, str]:
    """Write optimized prompt component artifacts for explicit dot-path optimization."""
    first_target = target_paths[0]
    safe_target = _safe_filename_part("__".join(target_path.replace(".", "_") for target_path in target_paths))
    output_dir = output_program.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer_log_dir.mkdir(parents=True, exist_ok=True)
    instructions_dir = optimizer_log_dir / "instructions"
    instructions_dir.mkdir(parents=True, exist_ok=True)

    optimized_text = _optimized_prompt_text_for_target(program, first_target)
    optimized_txt = output_dir / f"optimized_{safe_target}.txt"
    optimized_txt.write_text(optimized_text, encoding="utf-8")

    optimized_yaml = optimized_prompt_config_yaml or (output_dir / f"optimized_prompt_config_{safe_target}.yaml")
    optimized_yaml.parent.mkdir(parents=True, exist_ok=True)
    with optimized_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_optimized_prompt_config(program, target_paths), f, allow_unicode=True, sort_keys=False)

    candidate_rows: list[dict[str, Any]] = []
    for row in _candidate_prompt_rows_from_program(program, first_target):
        candidate_index = int(row.get("candidate_index", len(candidate_rows)))
        prompt_text = str(row.get("prompt_text") or "")
        instruction_path = instructions_dir / f"candidate_{candidate_index:04d}_{safe_target}.txt"
        instruction_path.write_text(prompt_text, encoding="utf-8")
        candidate_rows.append(_drop_none_values({
            **row,
            "optimize_target": first_target,
            "instruction_text": prompt_text,
            "instruction_path": str(instruction_path),
        }))

    candidates_jsonl = output_dir / f"{safe_target}_candidates.jsonl"
    _write_jsonl_rows(candidates_jsonl, candidate_rows)
    return {
        "optimized_prompt_txt": str(optimized_txt),
        "optimized_prompt_config_yaml": str(optimized_yaml),
        "prompt_target_candidates_jsonl": str(candidates_jsonl),
    }


def _load_optimized_program(program: dspy.Module, output_program: Path) -> None:
    """Load a completed optimizer output into an equivalent program instance."""
    if not output_program.is_file():
        raise RuntimeError(
            f"Optimization is not complete: compiled program does not exist at {output_program}"
        )
    load_method = getattr(program, "load", None)
    if load_method is None:
        raise RuntimeError(f"{program.__class__.__name__} does not expose a load() method")
    try:
        load_method(str(output_program))
    except Exception as exc:
        raise RuntimeError(
            f"Optimization output at {output_program} could not be loaded; "
            "the optimization may be incomplete or the file may be corrupt"
        ) from exc


def _update_inference_only_metadata(
    metadata_path: Path | None,
    *,
    args: argparse.Namespace,
    final_eval: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    if metadata_path is None:
        return

    payload: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"Cannot update unreadable metadata file {metadata_path}") from exc
        if not isinstance(loaded, dict):
            raise RuntimeError(f"Metadata file must contain a JSON object: {metadata_path}")
        payload = loaded

    payload.setdefault("algorithm", args.algorithm)
    payload.setdefault("output_program", str(args.output_program))
    payload["final_eval"] = final_eval
    payload["inference_only_elapsed_seconds"] = elapsed_seconds
    _write_json_atomic(metadata_path, payload)
    print(f"Updated final-eval metadata in {metadata_path}")


def _run_inference_only(
    args: argparse.Namespace,
    program: dspy.Module,
    context: AVQARuntimeContext,
    *,
    started: float,
) -> None:
    _load_optimized_program(program, args.output_program)
    cuts = read_jsonl(args.input_jsonl)
    caption_cache_coverage = None
    if getattr(args, "caption_cache_dir", None) is not None:
        caption_cache_coverage = validate_caption_cache_coverage(
            args.caption_cache_dir,
            [cut_id(cut) for cut in cuts],
            expected_prompt=build_caption_prompt(),
            allow_entry_errors=_caption_cache_allows_missing(),
        )
    print(f"Inference-only mode: loaded optimized program from {args.output_program}")
    print(f"Loaded final-test cuts: {len(cuts)}")
    print(f"Final-test cache directory: {args.final_eval_output_dir}")
    print(f"Planner model: {context.planner_model}")
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
    if args.audio_caption_dir:
        print(f"Audio caption dir: {args.audio_caption_dir}")
    elif args.requested_audio_caption_dir:
        print(f"Audio caption dir: <ignored; {args.audio_caption_skip_reason}>")
    else:
        print("Audio caption dir: <none; use captioner tool if needed>")

    final_eval = write_batch_style_program_outputs(
        program,
        cuts,
        args.audio_caption_dir,
        args.final_eval_output_jsonl,
        output_dir=args.final_eval_output_dir,
        max_turns=args.max_turns,
        skip_bad_examples=args.skip_bad_examples,
        num_threads=args.final_eval_num_threads,
        batch_size=args.final_eval_batch_size,
    )
    elapsed = time.perf_counter() - started
    _update_inference_only_metadata(
        args.metadata_json,
        args=args,
        final_eval=final_eval,
        elapsed_seconds=elapsed,
    )
    print(f"Inference-only elapsed time: {elapsed:.2f}s")


def run_optimization() -> None:
    """Entrypoint for DSPy AVQA optimization."""
    args = parse_optimize_args()
    optimization_train_only = bool(getattr(args, "optimization_train_only", False))
    started = time.perf_counter()

    if args.inference_only and not args.output_program.is_file():
        raise RuntimeError(
            f"Optimization is not complete: compiled program does not exist at {args.output_program}"
        )

    planner_source_config = load_reasoner_config_yaml(
        getattr(args, "planner_config_yaml", None),
        role="planner",
    )
    reflection_source_config = (
        load_reasoner_config_yaml(
            getattr(args, "gepa_reflection_config_yaml", None),
            role="reflection",
        )
        if getattr(args, "algorithm", None) == "gepa"
        else {}
    )

    os.environ["PERCEPTION_MODEL"] = args.perception_model
    os.environ["DSPY_AVQA_SIGNATURE_IN_SYSTEM_PROMPT"] = _env_bool(args.signature_in_system_prompt)
    os.environ["DSPY_AVQA_CAPTION_PLACEMENT"] = args.caption_placement
    caption_cache_dir = getattr(args, "caption_cache_dir", None)
    caption_cache_scope = getattr(args, "caption_cache_scope", "all")
    if caption_cache_dir is not None:
        args.caption_cache_dir = caption_cache_dir.expanduser().resolve()
    perception_source_config = load_perception_config_yaml(args.perception_config_yaml)
    if caption_cache_dir is None or caption_cache_scope == "first_call_only":
        captioner_source_config = load_captioner_config_yaml(args.captioner_config_yaml)
    else:
        captioner_source_config = {}
    configure_gemini_api_backend(args)
    prompt_source_config = load_prompt_config(args.prompt_yaml)
    args.requested_audio_caption_dir = args.audio_caption_dir
    args.audio_caption_dir, args.audio_caption_skip_reason = resolve_preloaded_audio_caption_dir(
        args.audio_caption_dir,
        caption_placement=args.caption_placement,
        ignore_audio_caption_dir=args.ignore_audio_caption_dir,
    )
    if caption_cache_dir is not None and args.audio_caption_dir is not None:
        raise ValueError(
            "--caption-cache-dir and an active --audio-caption-dir are mutually exclusive; "
            "use --ignore-audio-caption-dir for V8 cache-backed ask_caption."
        )
    if caption_cache_dir is not None and any(
        _is_captioner_target(target) for target in args.optimize_targets
    ):
        raise ValueError(
            "--caption-cache-dir cannot be used while optimizing captioner.* targets"
        )
    validate_optimize_targets(args.optimize_targets)
    apply_prompt_config_to_signatures(apply_instructions=False)

    allowed_tools = resolve_allowed_tools(args.allowed_tools)
    if caption_cache_dir is not None and "ask_caption" not in allowed_tools:
        raise ValueError("--caption-cache-dir requires ask_caption in --allowed-tools")
    if caption_cache_dir is None and caption_cache_scope != "all":
        raise ValueError("--caption-cache-scope first_call_only requires --caption-cache-dir")
    context = AVQARuntimeContext(
        max_turns=args.max_turns,
        allowed_tools=allowed_tools,
        caption_placement=args.caption_placement,
        caption_cache_dir=getattr(args, "caption_cache_dir", None),
        caption_cache_scope=caption_cache_scope,
    )
    base_program = AVQADSPyReActProgram(context=context)
    program: dspy.Module = PromptTargetProgram(
        base_program,
        args.optimize_targets,
    )

    if args.inference_only:
        _run_inference_only(args, program, context, started=started)
        return

    dataset_info = resolve_optimization_datasets(args)
    cuts = dataset_info["input_cuts"]
    selected = dataset_info["train_cuts"]
    trainset = dataset_info["trainset"]
    valset = dataset_info["valset"]
    skipped = dataset_info["skipped_train"]
    skipped_val = dataset_info["skipped_val"]

    planner_lm = dspy.settings.lm
    if args.algorithm == "gepa":
        gepa_reflection_template = configure_gepa_reflection_prompt_templates(
            args,
            densified_enabled=dataset_info["gepa_densified_supervision"]["enabled"],
        )
    else:
        gepa_reflection_template = None
    gepa_reflection_lm = build_gepa_reflection_lm(planner_lm) if args.algorithm == "gepa" else planner_lm
    initial_program_signatures = _program_signature_summary(program)

    if args.initial_program is not None:
        _save_program(program, args.initial_program)
        print(f"Saved initial program to {args.initial_program}")

    print(f"Algorithm: {args.algorithm}")
    print(f"Optimize targets: {', '.join(args.optimize_targets)}")
    print(f"Signature in system prompt: {args.signature_in_system_prompt}")
    print(f"Caption placement: {context.caption_placement}")
    caption_cache_coverage = dataset_info["caption_cache_coverage"]
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
    print(f"Loaded input cuts: {len(cuts)}")
    print(f"Train source jsonl: {dataset_info['train_source_path']}")
    print(f"Val source jsonl: {dataset_info['val_source_path']}")
    if args.debug:
        print(f"Debug mode: first {args.debug_limit} sample(s) per selected source")
    if args.trainset_jsonl is None and args.train_limit is not None:
        print(f"Train limit: {args.train_limit} random sample(s), data_seed={args.data_seed}")
    print(f"Train selection: {dataset_info['train_selection']}")
    print(f"Val selection: {dataset_info['val_selection']}")
    print(f"Train cuts: {len(dataset_info['train_cuts'])}")
    print(f"Val cuts: {len(dataset_info['val_cuts'])}")
    print(f"Train examples: {len(trainset)}")
    print(f"Val examples: {len(valset)}")
    print(f"Skipped train examples: {len(skipped)}")
    print(f"Skipped val examples: {len(skipped_val)}")
    gepa_caption_supervision = dataset_info["gepa_caption_supervision"]
    print(
        "GEPA caption supervision: "
        f"mode={gepa_caption_supervision['mode']}, "
        f"requested={gepa_caption_supervision['requested_mode']}, "
        f"root={gepa_caption_supervision['daily_omni_root']}"
    )
    if gepa_caption_supervision["enabled"]:
        print(
            "GEPA caption supervision stats: "
            f"videos_loaded={gepa_caption_supervision['videos_loaded']}, "
            f"cache_hits={gepa_caption_supervision['cache_hits']}, "
            f"files_loaded={gepa_caption_supervision['files_loaded']}, "
            f"missing_files={gepa_caption_supervision['missing_files']}, "
            f"unreadable_files={gepa_caption_supervision['unreadable_files']}"
        )
    gepa_densified_supervision = dataset_info["gepa_densified_supervision"]
    print(
        "GEPA densified supervision: "
        f"enabled={gepa_densified_supervision['enabled']}, "
        f"label_dir={gepa_densified_supervision['label_dir']}, "
        f"labels_loaded={gepa_densified_supervision['labels_loaded']}"
    )
    print(f"Planner model: {context.planner_model}")
    print(f"Planner temperature: {planner_lm.kwargs.get('temperature')}")
    if args.algorithm == "gepa":
        print(
            f"GEPA reflection temperature: {gepa_reflection_lm.kwargs.get('temperature')}"
        )
        print(
            "GEPA reflection template: "
            f"version={gepa_reflection_template['resolved_version']}, "
            f"requested={gepa_reflection_template['requested_version']}, "
            f"yaml={gepa_reflection_template['yaml']}"
        )
    print(f"Perception model: {args.perception_model}")
    for line in gemini_backend_log_lines():
        print(line)
    print(f"Prompt yaml: {active_prompt_yaml_path()}")
    if args.audio_caption_dir:
        print(f"Audio caption dir: {args.audio_caption_dir}")
    elif args.requested_audio_caption_dir:
        print(f"Audio caption dir: <ignored; {args.audio_caption_skip_reason}>")
    else:
        print("Audio caption dir: <none; use captioner tool if needed>")
    print(f"Allowed tools: {','.join(context.allowed_tools)}")
    print(f"Max turns: {context.max_turns}")
    print(f"Caption cache scope: {context.caption_cache_scope}")
    if args.perception_config_yaml:
        print(f"Perception config yaml: {args.perception_config_yaml}")
    if args.captioner_config_yaml:
        print(f"Captioner config yaml: {args.captioner_config_yaml}")

    optimizer_log_dir = args.optimizer_log_dir or (args.output_program.parent / f"{args.algorithm}_optimizer_logs")
    miprov2_config = _resolve_miprov2_config(args)
    gepa_config = _resolve_gepa_config(args)
    if args.algorithm == "miprov2" and miprov2_config.get("log_dir") is None:
        miprov2_config["log_dir"] = str(optimizer_log_dir / "dspy_native")
    if args.algorithm == "gepa" and not gepa_config.get("track_stats"):
        print("GEPA track_stats was false; enabling it so candidate-level optimizer logs can be saved.")
        gepa_config["track_stats"] = True

    resolved_config_path = save_resolved_experiment_config(
        args.output_program.parent,
        {
            "mode": "optimization",
            "arguments": args,
            "source_configs": {
                "planner": {
                    "path": getattr(args, "planner_config_yaml", None),
                    "config": planner_source_config,
                },
                "gepa_reflection": {
                    "path": getattr(args, "gepa_reflection_config_yaml", None),
                    "config": reflection_source_config,
                },
                "gepa": {
                    "path": args.gepa_config,
                    "config": _load_optimizer_config(args.gepa_config),
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
                "reflection_lm": lm_effective_config(gepa_reflection_lm)
                if args.algorithm == "gepa"
                else None,
                "gepa": gepa_config if args.algorithm == "gepa" else None,
                "miprov2": miprov2_config if args.algorithm == "miprov2" else None,
                "reflection_template": gepa_reflection_template,
                "allowed_tools": context.allowed_tools,
                "caption_placement": context.caption_placement,
                "caption_cache_scope": context.caption_cache_scope,
                "max_turns": context.max_turns,
                "perception_model": args.perception_model,
                "gemini_api_backend": args.gemini_api_backend,
            },
        },
        print_config=getattr(args, "print_config", False),
    )
    print(f"Saved resolved experiment config to {resolved_config_path}")

    if args.algorithm == "copro":
        print(
            "COPRO config: "
            f"breadth={args.copro_breadth}, "
            f"depth={args.copro_depth}, "
            f"init_temperature={args.copro_init_temperature}, "
            f"track_stats={args.copro_track_stats}"
        )
        compiled = optimize_with_copro(
            program,
            trainset,
            valset=valset,
            breadth=args.copro_breadth,
            depth=args.copro_depth,
            init_temperature=args.copro_init_temperature,
            track_stats=args.copro_track_stats,
        )
    elif args.algorithm == "simba":
        print(
            "SIMBA config: "
            f"bsize={args.simba_bsize}, "
            f"num_candidates={args.simba_num_candidates}, "
            f"max_steps={args.simba_max_steps}, "
            f"max_demos={args.simba_max_demos}, "
            f"num_threads={args.simba_num_threads}, "
            f"seed={args.simba_seed}, "
            f"temperature_for_sampling={args.simba_temperature_for_sampling}, "
            f"temperature_for_candidates={args.simba_temperature_for_candidates}"
        )
        compiled = optimize_with_simba(
            program,
            trainset,
            valset=valset,
            bsize=args.simba_bsize,
            num_candidates=args.simba_num_candidates,
            max_steps=args.simba_max_steps,
            max_demos=args.simba_max_demos,
            num_threads=args.simba_num_threads,
            seed=args.simba_seed,
            temperature_for_sampling=args.simba_temperature_for_sampling,
            temperature_for_candidates=args.simba_temperature_for_candidates,
        )
    elif args.algorithm == "miprov2":
        print(
            "MIPROv2 config: "
            f"config_file={args.miprov2_config}, "
            f"auto={miprov2_config['auto']}, "
            f"num_candidates={miprov2_config['num_candidates']}, "
            f"num_trials={miprov2_config['num_trials']}, "
            f"max_bootstrapped_demos={miprov2_config['max_bootstrapped_demos']}, "
            f"max_labeled_demos={miprov2_config['max_labeled_demos']}, "
            f"seed={miprov2_config['seed']}, "
            f"init_temperature={miprov2_config['init_temperature']}, "
            f"num_threads={miprov2_config['num_threads']}, "
            f"minibatch={miprov2_config['minibatch']}, "
            f"minibatch_size={miprov2_config['minibatch_size']}, "
            f"minibatch_full_eval_steps={miprov2_config['minibatch_full_eval_steps']}, "
            f"view_data_batch_size={miprov2_config['view_data_batch_size']}, "
            f"track_stats={miprov2_config['track_stats']}, "
            f"log_dir={miprov2_config['log_dir']}"
        )
        compiled = optimize_with_miprov2(
            program,
            trainset,
            valset=valset,
            **miprov2_config,
        )
    else:
        print(
            "GEPA config: "
            f"config_file={args.gepa_config}, "
            f"auto={gepa_config['auto']}, "
            f"max_full_evals={gepa_config['max_full_evals']}, "
            f"max_metric_calls={gepa_config['max_metric_calls']}, "
            f"reflection_minibatch_size={gepa_config['reflection_minibatch_size']}, "
            f"candidate_selection_strategy={gepa_config['candidate_selection_strategy']}, "
            f"skip_perfect_score={gepa_config['skip_perfect_score']}, "
            f"use_merge={gepa_config['use_merge']}, "
            f"max_merge_invocations={gepa_config['max_merge_invocations']}, "
            f"num_threads={gepa_config['num_threads']}, "
            f"seed={gepa_config['seed']}, "
            f"log_dir={gepa_config['log_dir']}, "
            f"reflective_dataset_save_interval={gepa_config['reflective_dataset_save_interval']}, "
            f"track_stats={gepa_config['track_stats']}, "
            f"track_best_outputs={gepa_config['track_best_outputs']}"
        )
        compiled = optimize_prompt_target_with_gepa(
            program,
            trainset,
            reflection_lm=gepa_reflection_lm,
            valset=valset,
            **gepa_config,
        )

    _save_program(compiled, args.output_program)
    print(f"Saved optimized program to {args.output_program}")

    if args.signature_search_json is not None:
        write_signature_search_json(
            compiled,
            args.signature_search_json,
            initial_program_signatures=initial_program_signatures,
        )
        print(f"Saved signature search to {args.signature_search_json}")

    if args.trajectory_jsonl is not None and not optimization_train_only:
        write_final_trajectories_jsonl(compiled, trainset, args.trajectory_jsonl)
        print(f"Saved final trainset trajectories to {args.trajectory_jsonl}")
    elif args.trajectory_jsonl is not None:
        print("Optimization-train-only mode: skipping optimized-trainset trajectory inference.")

    optimizer_log_paths: dict[str, str] = {}
    if args.algorithm in {"copro", "miprov2", "gepa"}:
        optimizer_log_paths = write_optimizer_candidate_logs(compiled, args.algorithm, optimizer_log_dir)
    prompt_target_artifacts = write_prompt_target_artifacts(
        compiled,
        args.optimize_targets,
        args.output_program,
        optimizer_log_dir,
        args.optimized_prompt_config_yaml,
    )
    optimizer_log_paths.update(prompt_target_artifacts)

    final_eval_metadata: dict[str, Any] = {}
    if optimization_train_only:
        print("Optimization-train-only mode: skipping final-test inference.")
    elif args.final_eval_output_jsonl is not None:
        final_eval_metadata = write_batch_style_program_outputs(
            compiled,
            cuts,
            args.audio_caption_dir,
            args.final_eval_output_jsonl,
            output_dir=args.final_eval_output_dir,
            max_turns=args.max_turns,
            skip_bad_examples=args.skip_bad_examples,
            num_threads=args.final_eval_num_threads,
            batch_size=args.final_eval_batch_size,
        )

    elapsed = time.perf_counter() - started
    print(f"Total elapsed time: {elapsed:.2f}s")

    metadata_path = args.metadata_json
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "algorithm": args.algorithm,
            "optimization_train_only": optimization_train_only,
            "optimize_target": args.optimize_targets[0] if len(args.optimize_targets) == 1 else None,
            "optimize_targets": list(args.optimize_targets),
            "input_jsonl": str(args.input_jsonl),
            "trainset_jsonl": str(args.trainset_jsonl) if args.trainset_jsonl else None,
            "valset_jsonl": str(args.valset_jsonl) if args.valset_jsonl else None,
            "data_seed": args.data_seed,
            "audio_caption_dir": str(args.audio_caption_dir) if args.audio_caption_dir else None,
            "caption_cache_dir": (
                str(args.caption_cache_dir)
                if getattr(args, "caption_cache_dir", None)
                else None
            ),
            "caption_cache_scope": getattr(args, "caption_cache_scope", "all"),
            "caption_cache_coverage": dataset_info["caption_cache_coverage"],
            "daily_omni_root": (
                str(args.daily_omni_root)
                if getattr(args, "daily_omni_root", None)
                else None
            ),
            "gepa_caption_supervision": dataset_info["gepa_caption_supervision"],
            "gepa_densified_supervision": dataset_info["gepa_densified_supervision"],
            "gepa_reflection_template": gepa_reflection_template,
            "requested_audio_caption_dir": (
                str(args.requested_audio_caption_dir) if args.requested_audio_caption_dir else None
            ),
            "audio_caption_skip_reason": args.audio_caption_skip_reason,
            "output_program": str(args.output_program),
            "initial_program": str(args.initial_program) if args.initial_program else None,
            "signature_search_json": str(args.signature_search_json) if args.signature_search_json else None,
            "trajectory_jsonl": (
                str(args.trajectory_jsonl)
                if args.trajectory_jsonl is not None and not optimization_train_only
                else None
            ),
            "optimizer_log_dir": str(optimizer_log_dir),
            "optimizer_logs": optimizer_log_paths,
            "final_eval": final_eval_metadata,
            "loaded_cuts": len(cuts),
            "selected_cuts": len(selected),
            "train_source_cuts": len(dataset_info["train_source_cuts"]),
            "val_source_cuts": len(dataset_info["val_source_cuts"]),
            "train_cuts": len(dataset_info["train_cuts"]),
            "val_cuts": len(dataset_info["val_cuts"]),
            "train_selection": dataset_info["train_selection"],
            "val_selection": dataset_info["val_selection"],
            "valset_is_trainset": dataset_info["valset_is_trainset"],
            "train_examples": len(trainset),
            "val_examples": len(valset),
            "skipped_examples": skipped,
            "skipped_val_examples": skipped_val,
            "planner_model": context.planner_model,
            "perception_model": args.perception_model,
            "prompt_yaml": str(active_prompt_yaml_path()),
            "allowed_tools": list(context.allowed_tools),
            "max_turns": context.max_turns,
            "copro": {
                "breadth": args.copro_breadth,
                "depth": args.copro_depth,
                "init_temperature": args.copro_init_temperature,
            },
            "simba": {
                "bsize": args.simba_bsize,
                "num_candidates": args.simba_num_candidates,
                "max_steps": args.simba_max_steps,
                "max_demos": args.simba_max_demos,
                "num_threads": args.simba_num_threads,
                "seed": args.simba_seed,
                "temperature_for_sampling": args.simba_temperature_for_sampling,
                "temperature_for_candidates": args.simba_temperature_for_candidates,
            },
            "miprov2_config_file": str(args.miprov2_config) if args.miprov2_config else None,
            "miprov2": miprov2_config,
            "gepa_config_file": str(args.gepa_config) if args.gepa_config else None,
            "gepa": gepa_config,
            "elapsed_seconds": elapsed,
        }
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(metadata), f, ensure_ascii=False, indent=2, default=str)
        print(f"Saved metadata to {metadata_path}")
