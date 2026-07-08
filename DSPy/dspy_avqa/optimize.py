"""Optimization hooks, dataset helpers, and CLI for DSPy."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import dspy
import yaml

from .context import AVQARuntimeContext, resolve_allowed_tools
from .deepseek_dspy_lm import consume_planner_call_trace
from .data import build_input_state, build_result_row, maybe_dump_question_data, read_jsonl, write_results_jsonl
from .program import AVQADSPyReActProgram, normalize_option_letter
from .prompt_config import active_prompt_yaml_path, load_prompt_config
from .runner import extract_error_info, load_perception_config_yaml
from .signatures import apply_prompt_config_to_signatures


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
    """Exact-match metric with textual feedback for GEPA reflection."""
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
        "track_stats": _config_value(config, "track_stats", args.gepa_track_stats),
        "track_best_outputs": _config_value(config, "track_best_outputs", args.gepa_track_best_outputs),
    }


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
        "--prompt-yaml",
        type=Path,
        default=None,
        help="Optional YAML file with DSPy AVQA planner/perception/signature prompts.",
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


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

def run_optimization() -> None:
    """Entrypoint for DSPy AVQA optimization."""
    args = parse_optimize_args()
    started = time.perf_counter()

    os.environ["PERCEPTION_MODEL"] = args.perception_model
    load_perception_config_yaml(args.perception_config_yaml)
    load_prompt_config(args.prompt_yaml)
    apply_prompt_config_to_signatures()

    dataset_info = resolve_optimization_datasets(args)
    cuts = dataset_info["input_cuts"]
    selected = dataset_info["train_cuts"]
    trainset = dataset_info["trainset"]
    valset = dataset_info["valset"]
    skipped = dataset_info["skipped_train"]
    skipped_val = dataset_info["skipped_val"]

    allowed_tools = resolve_allowed_tools(args.allowed_tools)
    context = AVQARuntimeContext(max_turns=args.max_turns, allowed_tools=allowed_tools)
    program = AVQADSPyReActProgram(context=context)
    initial_program_signatures = _program_signature_summary(program)

    if args.initial_program is not None:
        _save_program(program, args.initial_program)
        print(f"Saved initial program to {args.initial_program}")

    print(f"Algorithm: {args.algorithm}")
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
    print(f"Perception model: {args.perception_model}")
    print(f"Prompt yaml: {active_prompt_yaml_path()}")
    print(f"Allowed tools: {','.join(context.allowed_tools)}")
    print(f"Max turns: {context.max_turns}")
    if args.perception_config_yaml:
        print(f"Perception config yaml: {args.perception_config_yaml}")

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
            f"track_stats={gepa_config['track_stats']}, "
            f"track_best_outputs={gepa_config['track_best_outputs']}"
        )
        compiled = optimize_with_gepa(
            program,
            trainset,
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
