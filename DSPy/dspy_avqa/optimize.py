"""Optimization hooks, dataset helpers, and CLI for DSPy."""

from __future__ import annotations

import argparse
import copy
import functools
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import dspy
import yaml

from .context import AVQARuntimeContext, CAPTION_PLACEMENT_CHOICES, normalize_caption_placement, resolve_allowed_tools
from .deepseek_dspy_lm import consume_planner_call_trace
from .data import build_input_state, build_result_row, maybe_dump_question_data, read_jsonl, write_results_jsonl
from .program import AVQADSPyReActProgram, normalize_option_letter
from .prompt_config import active_prompt_yaml_path, load_prompt_config, prompt_config, prompt_overrides, prompt_value
from .runner import (
    add_gemini_backend_args,
    configure_gemini_api_backend,
    extract_error_info,
    gemini_backend_log_lines,
    load_captioner_config_yaml,
    load_perception_config_yaml,
)
from .signatures import apply_prompt_config_to_signatures

logger = logging.getLogger(__name__)

GEPA_OPTIMIZER_LOG_FILENAME = "gepa_optimizer.log"
GEPA_REFLECTIVE_DATASET_DIRNAME = "reflective_datasets"
GEPA_REFLECTIVE_DATASET_STATE_FILENAME = "capture_state.json"
DEFAULT_GEPA_REFLECTIVE_DATASET_SAVE_INTERVAL = 50

GEPA_REFLECTION_TEMPERATURE_ENV = "GEPA_REFLECTION_TEMPERATURE"
GEPA_REFLECTION_MODEL_ENV = "GEPA_REFLECTION_MODEL"
GEPA_REFLECTION_REASONING_EFFORT_ENV = "GEPA_REFLECTION_REASONING_EFFORT"


def _log_gepa_stage_exceptions(stage: str):
    """Log a traceback before GEPA converts a reflection failure into no proposal."""
    def decorator(method: Any) -> Any:
        @functools.wraps(method)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return method(*args, **kwargs)
            except Exception:
                logger.exception("GEPA %s failed", stage)
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


def build_gepa_reflection_lm(planner_lm: Any) -> Any:
    """Return a GEPA reflection LM with optional model, effort, and temperature overrides."""
    overrides: dict[str, Any] = {}
    raw_model = _optional_env_value(GEPA_REFLECTION_MODEL_ENV)
    raw_effort = _optional_env_value(GEPA_REFLECTION_REASONING_EFFORT_ENV)
    raw_temperature = os.environ.get(GEPA_REFLECTION_TEMPERATURE_ENV)
    if raw_model is not None:
        overrides["model"] = _normalize_openai_model_name(raw_model)
    if raw_effort is not None:
        overrides["reasoning_effort"] = _validate_reasoning_effort(
            raw_effort,
            env_name=GEPA_REFLECTION_REASONING_EFFORT_ENV,
        )

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
    return planner_lm.copy(**overrides)


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
    """Exact-match metric with textual feedback for GEPA reflection. 例如 gold answer、predicted answer、question、options、reasoning summary 等"""
    gold = normalize_option_letter(str(example.answer))
    got = normalize_option_letter(str(getattr(pred, "answer", "")))
    score = 1.0 if gold == got else 0.0
    question = str(getattr(example, "question", "") or "").strip()
    options_json = str(getattr(example, "options_json", "") or "").strip()
    reasoning = str(getattr(pred, "reasoning_summary", "") or "").strip()
    feedback_parts = [
        f"Score: {score}. Gold answer: {gold or '<empty>'}. Predicted answer: {got or '<empty>'}.",
    ]
    if question:
        feedback_parts.append(f"Question: {question}")
    if options_json:
        feedback_parts.append(f"Options JSON: {options_json}")
    if reasoning:
        feedback_parts.append(f"Predicted reasoning summary: {reasoning}")
    if pred_name:
        feedback_parts.append(f"Predictor under reflection: {pred_name}")
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


def make_trainset(raw_items: list[dict[str, Any]]) -> list[dspy.Example]:
    """Convert raw rows to DSPy Example list."""
    trainset: list[dspy.Example] = []
    for item in raw_items:
        ex = dspy.Example(
            question=item["question"],
            options_json=json.dumps(item["options"], ensure_ascii=False),
            video_path=item["video_path"],
            audio_path=item.get("audio_path"),
            video_id=item.get("video_id"),
            video_description=item.get("video_description", ""),
            max_turns=item.get("max_turns"),
            answer=item["answer"],
        ).with_inputs(
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
    audio_caption_dir: Path,
    *,
    max_turns: int | None = None,
    skip_bad_examples: bool = False,
) -> tuple[list[dspy.Example], list[dict[str, Any]]]:
    """Convert Daily Omni cut rows into DSPy Examples."""
    raw_items: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for cut in cuts:
        cut_id = str(cut.get("id") or "")
        try:
            payload = build_input_state(cut, audio_caption_dir)
            supervisions = cut.get("supervisions") or []
            custom = (supervisions[0].get("custom") if supervisions else {}) or {}
            answer = custom.get("answer") or custom.get("Answer")
            if answer is None:
                raise ValueError(f"Missing answer in cut_id={cut_id}")
            raw_items.append(
                {
                    "question": payload["question"],
                    "options": payload["options"],
                    "video_path": payload["video_path"],
                    "audio_path": payload.get("audio_path"),
                    "video_id": payload.get("video_id") or cut_id.rsplit("-", 1)[0],
                    "video_description": payload.get("video_description", ""),
                    "max_turns": max_turns,
                    "answer": answer,
                    "cut_id": cut_id,
                }
            )
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


def _config_value(config: dict[str, Any], key: str, fallback: Any) -> Any:
    return config[key] if key in config else fallback


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


def validate_optimize_target(target_path: str) -> None:
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
    """Expose one prompt config string as the optimizer-visible component. 把一个 YAML prompt component 包成可优化 program"""

    def __init__(self, base_program: AVQADSPyReActProgram, target_path: str, initial_text: str):
        super().__init__()
        object.__setattr__(self, "_base_program", base_program)
        self.target_path = target_path
        signature = PromptComponentSignature.with_instructions(initial_text)
        self.prompt_component = dspy.Predict(signature)

    def _candidate_text(self) -> str:
        signature = getattr(self.prompt_component, "signature", None)
        return str(getattr(signature, "instructions", "") or "").strip()

    def forward(self, *args: Any, **kwargs: Any) -> dspy.Prediction:
        with prompt_overrides({_target_keys(self.target_path): self._candidate_text()}):
            return self._base_program(*args, **kwargs)

    def named_predictors(self):
        return [("prompt_component", self.prompt_component)]

    def predictors(self):
        return [self.prompt_component]


def _prompt_component_text(program: dspy.Module) -> str:
    component = getattr(program, "prompt_component", None)
    signature = getattr(component, "signature", None)
    return str(getattr(signature, "instructions", "") or "").strip()


class PromptTargetGEPAAdapter:
    """GEPA adapter for prompt components that are read but not called as predictors."""

    def __init__(
        self,
        *,
        student_module: PromptTargetProgram,
        target_path: str,
        metric_fn: Any,
        failure_score: float = 0.0,
        num_threads: int | None = None,
        rng: random.Random | None = None,
        reflection_lm: Any = None,
        reflection_minibatch_size: int | None = None,
    ):
        self.student = student_module
        self.target_path = target_path
        self.component_name = "prompt_component"
        self.metric_fn = metric_fn
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.rng = rng or random.Random(0)
        self.reflection_lm = reflection_lm
        self.reflection_minibatch_size = reflection_minibatch_size

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
        candidate_text = candidate.get(self.component_name)
        if candidate_text is not None:
            new_prog.prompt_component.signature = new_prog.prompt_component.signature.with_instructions(candidate_text)
        return new_prog

    def evaluate(self, batch: list[dspy.Example], candidate: dict[str, str], capture_traces: bool = False):
        from dspy.evaluate import Evaluate
        from gepa.core.adapter import EvaluationBatch

        program = self.build_program(candidate)
        callback_metadata = (
            {"metric_key": "eval_full"}
            if self.reflection_minibatch_size is None or len(batch) > self.reflection_minibatch_size
            else {"disable_logging": True}
        )
        if capture_traces:
            from dspy.teleprompt import bootstrap_trace as bootstrap_trace_module

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
            scores = []
            outputs = []
            for item in trajs:
                outputs.append(item["prediction"])
                score = item.get("score", self.failure_score)
                if hasattr(score, "score"):
                    score = score["score"]
                scores.append(score)
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
        result = evaluator(program)
        outputs = [row[1] for row in result.results]
        scores = [row[2] for row in result.results]
        scores = [score["score"] if hasattr(score, "score") else score for score in scores]
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=None)

    @_log_gepa_stage_exceptions("reflective dataset construction")
    def make_reflective_dataset(self, candidate: dict[str, str], eval_batch: Any, components_to_update: list[str]) -> dict[str, list[dict[str, Any]]]:
        """把 trajectories 转成 GEPA reflection dataset"""
        items: list[dict[str, Any]] = []
        trajectories = eval_batch.trajectories or []
        for data in trajectories:
            example = data.get("example")
            prediction = data.get("prediction")
            score = data.get("score", self.failure_score)
            if hasattr(score, "score"):
                score = score["score"]
            feedback = self.metric_fn(example, prediction, data.get("trace"), self.component_name, None)
            if hasattr(feedback, "feedback"):
                feedback_text = str(feedback.feedback)
            elif hasattr(feedback, "get"):
                feedback_text = str(feedback.get("feedback", ""))
            else:
                feedback_text = str(feedback)
            items.append({
                "Inputs": {
                    "Prompt target path": self.target_path,
                    "Current prompt text": candidate.get(self.component_name, ""),
                    "Question": str(getattr(example, "question", "") or ""),
                    "Options JSON": str(getattr(example, "options_json", "") or ""),
                    "Video ID": str(getattr(example, "video_id", "") or ""),
                },
                "Generated Outputs": {
                    "answer": str(getattr(prediction, "answer", "") or ""),
                    "reasoning_summary": str(getattr(prediction, "reasoning_summary", "") or ""),
                    "turn_trace": json.dumps(_json_safe(getattr(prediction, "turn_trace", [])), ensure_ascii=False),
                    "score": str(score),
                },
                "Feedback": feedback_text,
            })
        if not items:
            raise Exception(f"No valid reflective examples found for {self.component_name}")
        return {self.component_name: items}

    @_log_gepa_stage_exceptions("instruction proposal")
    def propose_new_texts(self, candidate: dict[str, str], reflective_dataset: dict[str, list[dict[str, Any]]], components_to_update: list[str]) -> dict[str, str]:
        from dspy.teleprompt.gepa.gepa_utils import InstructionProposalSignature

        results: dict[str, str] = {}
        reflection_lm = self.reflection_lm or dspy.settings.lm
        with dspy.context(lm=reflection_lm):
            for name in components_to_update:
                results[name] = InstructionProposalSignature.run(
                    lm=(lambda x: self.stripped_lm_call(x)[0]),
                    input_dict={
                        "current_instruction_doc": candidate[name],
                        "dataset_with_feedback": reflective_dataset[name],
                    },
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
    """Compile one prompt component with GEPA without adding runtime LM calls."""
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
            num_preds=1,
            num_candidates=AUTO_RUN_SETTINGS[teleprompter.auto]["n"],
            valset_size=len(valset),
        )
    elif teleprompter.max_full_evals is not None:
        teleprompter.max_metric_calls = teleprompter.max_full_evals * (len(trainset) + len(valset))
    rng = random.Random(seed)
    adapter = PromptTargetGEPAAdapter(
        student_module=program,
        target_path=program.target_path,
        metric_fn=avqa_gepa_feedback_metric,
        failure_score=teleprompter.failure_score,
        num_threads=num_threads,
        rng=rng,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=reflection_minibatch_size,
    )
    seed_candidate = {adapter.component_name: program._candidate_text()}
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
        rows.append(_drop_none_values({
            "candidate_index": idx,
            "prompt_text": candidate.get(adapter.component_name),
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
    parser.add_argument("--optimize-target", default="planner.workflow_prompt", help="Supported prompt-config dot path to optimize. Default: planner.workflow_prompt.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--trainset-jsonl", type=Path, default=None)
    parser.add_argument("--valset-jsonl", type=Path, default=None)
    parser.add_argument("--data-seed", type=int, default=int(os.environ.get("DSPY_AVQA_DATA_SEED", "0")))
    parser.add_argument("--audio-caption-dir", type=Path, required=True)
    parser.add_argument("--output-program", type=Path, required=True)
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
        help="Optional directory for per-sample JSON files from --final-eval-output-jsonl.",
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
    return parser.parse_args()


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
    audio_caption_dir: Path,
    max_turns: int,
) -> dict[str, Any]:
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
    audio_caption_dir: Path,
    *,
    max_turns: int,
    skip_bad_examples: bool,
) -> tuple[list[dspy.Example], list[dict[str, Any]]]:
    return make_trainset_from_cuts(
        cuts,
        audio_caption_dir,
        max_turns=max_turns,
        skip_bad_examples=skip_bad_examples,
    )


def resolve_optimization_datasets(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve train/val examples while preserving legacy defaults."""
    input_cuts = read_jsonl(args.input_jsonl)

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

    return {
        "input_cuts": input_cuts,
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
    }

def write_batch_style_program_outputs(
    program: dspy.Module,
    cuts: list[dict[str, Any]],
    audio_caption_dir: Path,
    output_jsonl: Path,
    *,
    output_dir: Path | None = None,
    max_turns: int,
    skip_bad_examples: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for idx, cut in enumerate(cuts):
        try:
            row = _run_program_on_cut(program, cut, audio_caption_dir, max_turns)
        except Exception as exc:
            if not skip_bad_examples:
                raise
            skipped.append({"example_index": idx, "cut_id": str(cut.get("id") or ""), "error": str(exc)})
            continue
        maybe_dump_question_data(output_dir, row)
        rows.append(row)
    write_results_jsonl(rows, output_jsonl)
    print(f"Wrote {len(rows)} optimized-program rows to {output_jsonl}")
    return {
        "final_eval_output_jsonl": str(output_jsonl),
        "final_eval_output_dir": str(output_dir) if output_dir else None,
        "final_eval_rows": len(rows),
        "final_eval_skipped_examples": skipped,
    }

def _optimized_prompt_config(target_path: str, optimized_text: str) -> dict[str, Any]:
    config = copy.deepcopy(prompt_config())
    _set_nested(config, target_path, optimized_text)
    return config


def _optimized_prompt_text_for_target(program: dspy.Module, target_path: str) -> str:
    return _prompt_component_text(program)


def _candidate_prompt_text_for_target(program: dspy.Module | None, target_path: str) -> str:
    if program is None:
        return ""
    return _prompt_component_text(program)


def _candidate_prompt_rows_from_program(program: dspy.Module, target_path: str) -> list[dict[str, Any]]:
    rows = list(getattr(program, "prompt_target_candidate_rows", []) or [])
    if rows:
        return rows
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
    target_path: str,
    output_program: Path,
    optimizer_log_dir: Path,
    optimized_prompt_config_yaml: Path | None = None,
) -> dict[str, str]:
    """Write optimized prompt component artifacts for explicit dot-path optimization."""
    safe_target = _safe_filename_part(target_path.replace(".", "_"))
    output_dir = output_program.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer_log_dir.mkdir(parents=True, exist_ok=True)
    instructions_dir = optimizer_log_dir / "instructions"
    instructions_dir.mkdir(parents=True, exist_ok=True)

    optimized_text = _optimized_prompt_text_for_target(program, target_path)
    optimized_txt = output_dir / f"optimized_{safe_target}.txt"
    optimized_txt.write_text(optimized_text, encoding="utf-8")

    optimized_yaml = optimized_prompt_config_yaml or (output_dir / f"optimized_prompt_config_{safe_target}.yaml")
    optimized_yaml.parent.mkdir(parents=True, exist_ok=True)
    with optimized_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_optimized_prompt_config(target_path, optimized_text), f, allow_unicode=True, sort_keys=False)

    candidate_rows: list[dict[str, Any]] = []
    for row in _candidate_prompt_rows_from_program(program, target_path):
        candidate_index = int(row.get("candidate_index", len(candidate_rows)))
        prompt_text = str(row.get("prompt_text") or "")
        instruction_path = instructions_dir / f"candidate_{candidate_index:04d}_{safe_target}.txt"
        instruction_path.write_text(prompt_text, encoding="utf-8")
        candidate_rows.append(_drop_none_values({
            **row,
            "optimize_target": target_path,
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


def run_optimization() -> None:
    """Entrypoint for DSPy AVQA optimization."""
    args = parse_optimize_args()
    started = time.perf_counter()

    os.environ["PERCEPTION_MODEL"] = args.perception_model
    os.environ["DSPY_AVQA_SIGNATURE_IN_SYSTEM_PROMPT"] = _env_bool(args.signature_in_system_prompt)
    os.environ["DSPY_AVQA_CAPTION_PLACEMENT"] = args.caption_placement
    load_perception_config_yaml(args.perception_config_yaml)
    load_captioner_config_yaml(args.captioner_config_yaml)
    configure_gemini_api_backend(args)
    load_prompt_config(args.prompt_yaml)
    validate_optimize_target(args.optimize_target)
    apply_prompt_config_to_signatures(apply_instructions=False)

    dataset_info = resolve_optimization_datasets(args)
    cuts = dataset_info["input_cuts"]
    selected = dataset_info["train_cuts"]
    trainset = dataset_info["trainset"]
    valset = dataset_info["valset"]
    skipped = dataset_info["skipped_train"]
    skipped_val = dataset_info["skipped_val"]

    allowed_tools = resolve_allowed_tools(args.allowed_tools)
    context = AVQARuntimeContext(
        max_turns=args.max_turns,
        allowed_tools=allowed_tools,
        caption_placement=args.caption_placement,
    )
    base_program = AVQADSPyReActProgram(context=context)
    planner_lm = dspy.settings.lm
    gepa_reflection_lm = build_gepa_reflection_lm(planner_lm) if args.algorithm == "gepa" else planner_lm
    program: dspy.Module = PromptTargetProgram(
        base_program,
        args.optimize_target,
        prompt_target_initial_text(args.optimize_target),
    )
    initial_program_signatures = _program_signature_summary(program)

    if args.initial_program is not None:
        _save_program(program, args.initial_program)
        print(f"Saved initial program to {args.initial_program}")

    print(f"Algorithm: {args.algorithm}")
    print(f"Optimize target: {args.optimize_target}")
    print(f"Signature in system prompt: {args.signature_in_system_prompt}")
    print(f"Caption placement: {context.caption_placement}")
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
    print(f"Planner model: {context.planner_model}")
    print(f"Planner temperature: {planner_lm.kwargs.get('temperature')}")
    if args.algorithm == "gepa":
        print(
            f"GEPA reflection temperature: {gepa_reflection_lm.kwargs.get('temperature')}"
        )
    print(f"Perception model: {args.perception_model}")
    for line in gemini_backend_log_lines():
        print(line)
    print(f"Prompt yaml: {active_prompt_yaml_path()}")
    print(f"Allowed tools: {','.join(context.allowed_tools)}")
    print(f"Max turns: {context.max_turns}")
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

    if args.trajectory_jsonl is not None:
        write_final_trajectories_jsonl(compiled, trainset, args.trajectory_jsonl)
        print(f"Saved final trainset trajectories to {args.trajectory_jsonl}")

    optimizer_log_paths: dict[str, str] = {}
    if args.algorithm in {"copro", "miprov2", "gepa"}:
        optimizer_log_paths = write_optimizer_candidate_logs(compiled, args.algorithm, optimizer_log_dir)
    prompt_target_artifacts = write_prompt_target_artifacts(
        compiled,
        args.optimize_target,
        args.output_program,
        optimizer_log_dir,
        args.optimized_prompt_config_yaml,
    )
    optimizer_log_paths.update(prompt_target_artifacts)

    final_eval_metadata: dict[str, Any] = {}
    if args.final_eval_output_jsonl is not None:
        final_eval_metadata = write_batch_style_program_outputs(
            compiled,
            cuts,
            args.audio_caption_dir,
            args.final_eval_output_jsonl,
            output_dir=args.final_eval_output_dir,
            max_turns=args.max_turns,
            skip_bad_examples=args.skip_bad_examples,
        )

    elapsed = time.perf_counter() - started
    print(f"Total elapsed time: {elapsed:.2f}s")

    metadata_path = args.metadata_json
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "algorithm": args.algorithm,
            "optimize_target": args.optimize_target,
            "input_jsonl": str(args.input_jsonl),
            "trainset_jsonl": str(args.trainset_jsonl) if args.trainset_jsonl else None,
            "valset_jsonl": str(args.valset_jsonl) if args.valset_jsonl else None,
            "data_seed": args.data_seed,
            "audio_caption_dir": str(args.audio_caption_dir),
            "output_program": str(args.output_program),
            "initial_program": str(args.initial_program) if args.initial_program else None,
            "signature_search_json": str(args.signature_search_json) if args.signature_search_json else None,
            "trajectory_jsonl": str(args.trajectory_jsonl) if args.trajectory_jsonl else None,
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
