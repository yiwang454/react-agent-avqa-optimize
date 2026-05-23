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


def parse_optimize_args() -> argparse.Namespace:
    """Parse optimizer CLI arguments."""
    parser = argparse.ArgumentParser(description="Optimize DSPy AVQA ReAct with COPRO/SIMBA.")
    parser.add_argument("--algorithm", choices=("copro", "simba"), default="copro")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--audio-caption-dir", type=Path, required=True)
    parser.add_argument("--output-program", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, default=None)
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
    else:
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

    _save_program(compiled, args.output_program)
    elapsed = time.perf_counter() - started
    print(f"Saved optimized program to {args.output_program}")
    print(f"Total elapsed time: {elapsed:.2f}s")

    metadata_path = args.metadata_json
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "algorithm": args.algorithm,
            "input_jsonl": str(args.input_jsonl),
            "audio_caption_dir": str(args.audio_caption_dir),
            "output_program": str(args.output_program),
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
            "elapsed_seconds": elapsed,
        }
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(metadata), f, ensure_ascii=False, indent=2, default=str)
        print(f"Saved metadata to {metadata_path}")
