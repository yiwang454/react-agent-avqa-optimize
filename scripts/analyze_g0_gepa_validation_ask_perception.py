#!/usr/bin/env python3
"""Run the C1-style saved-trajectory analysis for GPT-4.1 G0 GEPA searches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analyze_ask_perception_questions import DATA_ROOT
import analyze_c1_gepa_validation_ask_perception as validation


EXPERIMENTS = {
    "planner": {
        "run_dir": DATA_ROOT / "daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_planner_workflow_prompt_planner_gpt-4.1_seed_1234_gepa_seed18",
        "candidates": "planner_workflow_prompt_candidates.jsonl",
    },
    "captioner": {
        "run_dir": DATA_ROOT / "daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_captioner_default_caption_instruction_gpt4_1_0721_planner_gpt-4.1_seed_1234_gepa_seed18",
        "candidates": "captioner_default_caption_instruction_candidates.jsonl",
    },
    "planner_captioner": {
        "run_dir": DATA_ROOT / "daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_planner_workflow_prompt_and_captioner_default_caption_instruction_gpt4_1_0721_planner_gpt-4.1_seed_1234_gepa_seed18",
        "candidates": "planner_workflow_prompt__captioner_default_caption_instruction_candidates.jsonl",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, action="append", help="Repeat for a subset; default: all G0 targets.")
    known, remaining = parser.parse_known_args()
    selected = known.experiment or list(EXPERIMENTS)
    exit_code = 0
    for name in selected:
        config = EXPERIMENTS[name]
        run_dir = config["run_dir"]
        validation.RUN_DIR = run_dir
        validation.STATE_PATH = run_dir / "gepa_logs/gepa_state.bin"
        validation.METADATA_PATH = run_dir / "compiled_gepa_metadata.json"
        validation.CANDIDATES_PATH = run_dir / config["candidates"]
        validation.DEFAULT_OUTPUT = DATA_ROOT / "ask_perception_question_judge_gpt5_5/g0_gepa_val125" / name
        print(f"G0 {name}: source={run_dir}", flush=True)
        prior = sys.argv
        try:
            sys.argv = [prior[0], *remaining]
            exit_code = max(exit_code, validation.main())
        finally:
            sys.argv = prior
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
