"""Optimization hooks and dataset helpers for DSPy."""

from __future__ import annotations

import json
from typing import Any

import dspy

from .program import AVQADSPyReActProgram, normalize_option_letter


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
            video_description=item.get("video_description", ""),
            answer=item["answer"],
        ).with_inputs("question", "options_json", "video_path", "audio_path", "video_description")
        trainset.append(ex)
    return trainset


def optimize_with_copro(program: AVQADSPyReActProgram, trainset: list[dspy.Example]) -> dspy.Module:
    """Compile the program with COPRO."""
    from dspy.teleprompt import COPRO

    teleprompter = COPRO(metric=avqa_metric)
    return teleprompter.compile(student=program, trainset=trainset)


def optimize_with_simba(program: AVQADSPyReActProgram, trainset: list[dspy.Example]) -> dspy.Module:
    """Compile the program with SIMBA."""
    from dspy.teleprompt import SIMBA

    teleprompter = SIMBA(metric=avqa_metric, max_steps=8, max_demos=4)
    return teleprompter.compile(student=program, trainset=trainset)

