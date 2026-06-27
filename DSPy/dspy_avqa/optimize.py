"""Optimization hooks, dataset helpers, and CLI for DSPy."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import dspy

from .context import AVQARuntimeContext, resolve_allowed_tools
from .data import build_input_state, read_jsonl
from .program import AVQADSPyReActProgram, normalize_option_letter
from .prompt_config import active_prompt_yaml_path, load_prompt_config
from .runner import load_perception_config_yaml
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


def optimize_with_copro(
    program: AVQADSPyReActProgram,
    trainset: list[dspy.Example],
    *,
    breadth: int | None = None,
    depth: int | None = None,
    init_temperature: float | None = None,
) -> dspy.Module:
    """Compile the program with COPRO."""
    from dspy.teleprompt import COPRO

    copro_kwargs = {
        "breadth": breadth,
        "depth": depth,
        "init_temperature": init_temperature,
    }
    copro_kwargs = {key: value for key, value in copro_kwargs.items() if value is not None}
    teleprompter = COPRO(metric=avqa_metric, **_filtered_kwargs(COPRO.__init__, copro_kwargs))
    compile_kwargs = _filtered_kwargs(teleprompter.compile, {"eval_kwargs": {}})
    return teleprompter.compile(student=program, trainset=trainset, **compile_kwargs)


def optimize_with_simba(
    program: AVQADSPyReActProgram,
    trainset: list[dspy.Example],
    *,
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
    compile_kwargs = _filtered_kwargs(teleprompter.compile, {"seed": seed})
    return teleprompter.compile(student=program, trainset=trainset, **compile_kwargs)


def optimize_with_miprov2(
    program: AVQADSPyReActProgram,
    trainset: list[dspy.Example],
    *,
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
    }
    mipro_kwargs = {key: value for key, value in mipro_kwargs.items() if value is not None}
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
    }
    gepa_kwargs = {key: value for key, value in gepa_kwargs.items() if value is not None}
    teleprompter = GEPA(
        metric=avqa_gepa_feedback_metric,
        **_filtered_kwargs(GEPA.__init__, gepa_kwargs),
    )
    return teleprompter.compile(student=program, trainset=trainset)


def parse_optimize_args() -> argparse.Namespace:
    """Parse optimizer CLI arguments."""
    parser = argparse.ArgumentParser(description="Optimize DSPy AVQA ReAct with DSPy optimizers.")
    parser.add_argument("--algorithm", choices=("copro", "simba", "miprov2", "gepa"), default="copro")
    parser.add_argument("--input-jsonl", type=Path, required=True)
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
    parser.add_argument("--simba-bsize", type=int, default=32)
    parser.add_argument("--simba-num-candidates", type=int, default=6)
    parser.add_argument("--simba-max-steps", type=int, default=8)
    parser.add_argument("--simba-max-demos", type=int, default=4)
    parser.add_argument("--simba-num-threads", type=int, default=None)
    parser.add_argument("--simba-seed", type=int, default=0)
    parser.add_argument("--simba-temperature-for-sampling", type=float, default=None)
    parser.add_argument("--simba-temperature-for-candidates", type=float, default=None)
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


def run_optimization() -> None:
    """Entrypoint for DSPy AVQA optimization."""
    args = parse_optimize_args()
    started = time.perf_counter()

    os.environ["PERCEPTION_MODEL"] = args.perception_model
    load_perception_config_yaml(args.perception_config_yaml)
    load_prompt_config(args.prompt_yaml)
    apply_prompt_config_to_signatures()

    cuts = read_jsonl(args.input_jsonl)
    selected = cuts[: args.debug_limit] if args.debug else cuts
    if args.train_limit is not None:
        selected = selected[: args.train_limit]

    trainset, skipped = make_trainset_from_cuts(
        selected,
        args.audio_caption_dir,
        max_turns=args.max_turns,
        skip_bad_examples=args.skip_bad_examples,
    )
    if not trainset:
        raise ValueError("No train examples were built; check input JSONL and audio-caption-dir.")

    allowed_tools = resolve_allowed_tools(args.allowed_tools)
    context = AVQARuntimeContext(max_turns=args.max_turns, allowed_tools=allowed_tools)
    program = AVQADSPyReActProgram(context=context)
    initial_program_signatures = _program_signature_summary(program)

    if args.initial_program is not None:
        _save_program(program, args.initial_program)
        print(f"Saved initial program to {args.initial_program}")

    print(f"Algorithm: {args.algorithm}")
    print(f"Loaded cuts: {len(cuts)}")
    if args.debug:
        print(f"Debug mode: first {args.debug_limit} sample(s)")
    if args.train_limit is not None:
        print(f"Train limit: {args.train_limit}")
    print(f"Train examples: {len(trainset)}")
    print(f"Skipped examples: {len(skipped)}")
    print(f"Planner model: {context.planner_model}")
    print(f"Perception model: {args.perception_model}")
    print(f"Prompt yaml: {active_prompt_yaml_path()}")
    print(f"Allowed tools: {','.join(context.allowed_tools)}")
    print(f"Max turns: {context.max_turns}")
    if args.perception_config_yaml:
        print(f"Perception config yaml: {args.perception_config_yaml}")

    if args.algorithm == "copro":
        print(
            "COPRO config: "
            f"breadth={args.copro_breadth}, "
            f"depth={args.copro_depth}, "
            f"init_temperature={args.copro_init_temperature}"
        )
        compiled = optimize_with_copro(
            program,
            trainset,
            breadth=args.copro_breadth,
            depth=args.copro_depth,
            init_temperature=args.copro_init_temperature,
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
        mipro_auto = None if args.miprov2_auto == "none" else args.miprov2_auto
        mipro_num_candidates = None if mipro_auto is not None else args.miprov2_num_candidates
        mipro_num_trials = None if mipro_auto is not None else args.miprov2_num_trials
        print(
            "MIPROv2 config: "
            f"auto={mipro_auto}, "
            f"num_candidates={mipro_num_candidates}, "
            f"num_trials={mipro_num_trials}, "
            f"max_bootstrapped_demos={args.miprov2_max_bootstrapped_demos}, "
            f"max_labeled_demos={args.miprov2_max_labeled_demos}, "
            f"seed={args.miprov2_seed}, "
            f"init_temperature={args.miprov2_init_temperature}, "
            f"num_threads={args.miprov2_num_threads}, "
            f"minibatch={not args.miprov2_no_minibatch}, "
            f"minibatch_size={args.miprov2_minibatch_size}, "
            f"minibatch_full_eval_steps={args.miprov2_minibatch_full_eval_steps}, "
            f"view_data_batch_size={args.miprov2_view_data_batch_size}"
        )
        compiled = optimize_with_miprov2(
            program,
            trainset,
            auto=mipro_auto,
            num_candidates=mipro_num_candidates,
            num_trials=mipro_num_trials,
            max_bootstrapped_demos=args.miprov2_max_bootstrapped_demos,
            max_labeled_demos=args.miprov2_max_labeled_demos,
            seed=args.miprov2_seed,
            init_temperature=args.miprov2_init_temperature,
            num_threads=args.miprov2_num_threads,
            max_errors=args.miprov2_max_errors,
            minibatch=not args.miprov2_no_minibatch,
            minibatch_size=args.miprov2_minibatch_size,
            minibatch_full_eval_steps=args.miprov2_minibatch_full_eval_steps,
            view_data_batch_size=args.miprov2_view_data_batch_size,
        )
    else:
        gepa_auto = None if args.gepa_auto == "none" else args.gepa_auto
        gepa_max_full_evals = args.gepa_max_full_evals
        if gepa_auto is not None or args.gepa_max_metric_calls is not None:
            gepa_max_full_evals = None
        print(
            "GEPA config: "
            f"auto={gepa_auto}, "
            f"max_full_evals={gepa_max_full_evals}, "
            f"max_metric_calls={args.gepa_max_metric_calls}, "
            f"reflection_minibatch_size={args.gepa_reflection_minibatch_size}, "
            f"candidate_selection_strategy={args.gepa_candidate_selection_strategy}, "
            f"skip_perfect_score={not args.gepa_dont_skip_perfect_score}, "
            f"use_merge={not args.gepa_no_merge}, "
            f"max_merge_invocations={args.gepa_max_merge_invocations}, "
            f"num_threads={args.gepa_num_threads}, "
            f"seed={args.gepa_seed}, "
            f"log_dir={args.gepa_log_dir}, "
            f"track_stats={args.gepa_track_stats}"
        )
        compiled = optimize_with_gepa(
            program,
            trainset,
            auto=gepa_auto,
            max_full_evals=gepa_max_full_evals,
            max_metric_calls=args.gepa_max_metric_calls,
            reflection_minibatch_size=args.gepa_reflection_minibatch_size,
            candidate_selection_strategy=args.gepa_candidate_selection_strategy,
            skip_perfect_score=not args.gepa_dont_skip_perfect_score,
            use_merge=not args.gepa_no_merge,
            max_merge_invocations=args.gepa_max_merge_invocations,
            num_threads=args.gepa_num_threads,
            seed=args.gepa_seed,
            log_dir=args.gepa_log_dir,
            track_stats=args.gepa_track_stats,
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

    elapsed = time.perf_counter() - started
    print(f"Total elapsed time: {elapsed:.2f}s")

    metadata_path = args.metadata_json
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "algorithm": args.algorithm,
            "input_jsonl": str(args.input_jsonl),
            "audio_caption_dir": str(args.audio_caption_dir),
            "output_program": str(args.output_program),
            "initial_program": str(args.initial_program) if args.initial_program else None,
            "signature_search_json": str(args.signature_search_json) if args.signature_search_json else None,
            "trajectory_jsonl": str(args.trajectory_jsonl) if args.trajectory_jsonl else None,
            "loaded_cuts": len(cuts),
            "selected_cuts": len(selected),
            "train_examples": len(trainset),
            "skipped_examples": skipped,
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
            "miprov2": {
                "auto": None if args.miprov2_auto == "none" else args.miprov2_auto,
                "num_candidates": None if args.miprov2_auto != "none" else args.miprov2_num_candidates,
                "num_trials": None if args.miprov2_auto != "none" else args.miprov2_num_trials,
                "max_bootstrapped_demos": args.miprov2_max_bootstrapped_demos,
                "max_labeled_demos": args.miprov2_max_labeled_demos,
                "seed": args.miprov2_seed,
                "init_temperature": args.miprov2_init_temperature,
                "num_threads": args.miprov2_num_threads,
                "max_errors": args.miprov2_max_errors,
                "minibatch": not args.miprov2_no_minibatch,
                "minibatch_size": args.miprov2_minibatch_size,
                "minibatch_full_eval_steps": args.miprov2_minibatch_full_eval_steps,
                "view_data_batch_size": args.miprov2_view_data_batch_size,
            },
            "gepa": {
                "auto": None if args.gepa_auto == "none" else args.gepa_auto,
                "max_full_evals": (
                    None
                    if args.gepa_auto != "none" or args.gepa_max_metric_calls is not None
                    else args.gepa_max_full_evals
                ),
                "max_metric_calls": args.gepa_max_metric_calls,
                "reflection_minibatch_size": args.gepa_reflection_minibatch_size,
                "candidate_selection_strategy": args.gepa_candidate_selection_strategy,
                "skip_perfect_score": not args.gepa_dont_skip_perfect_score,
                "use_merge": not args.gepa_no_merge,
                "max_merge_invocations": args.gepa_max_merge_invocations,
                "num_threads": args.gepa_num_threads,
                "seed": args.gepa_seed,
                "log_dir": args.gepa_log_dir,
                "track_stats": args.gepa_track_stats,
            },
            "elapsed_seconds": elapsed,
        }
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(metadata), f, ensure_ascii=False, indent=2, default=str)
        print(f"Saved metadata to {metadata_path}")
