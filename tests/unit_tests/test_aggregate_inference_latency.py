from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "aggregate_inference_latency.py"
SPEC = importlib.util.spec_from_file_location("aggregate_inference_latency", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(sample_id: str, *, wall: float, adjusted: float | None) -> dict:
    return {
        "video_id": sample_id,
        "question_data": {
            "latency": {
                "wall_clock_seconds": wall,
                "retry_adjusted_seconds": adjusted,
                "successful_model_call_seconds": wall - 1.0,
                "planner_successful_calls": 2,
                "perception_successful_calls": 1,
            }
        },
    }


def test_aggregate_latency_validates_and_summarizes(tmp_path):
    input_path = tmp_path / "output_test.jsonl"
    rows = [
        _row("a", wall=2.0, adjusted=1.0),
        _row("b", wall=4.0, adjusted=None),
    ]
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = MODULE.aggregate_latency(input_path, expected_count=2)

    assert summary["row_count"] == 2
    assert summary["unique_id_count"] == 2
    assert summary["duration_seconds"]["wall_clock_seconds"] == {
        "count": 2,
        "null_count": 0,
        "sum": 6.0,
        "mean": 3.0,
        "median": 3.0,
        "p90": 3.8,
        "p95": 3.9,
        "min": 2.0,
        "max": 4.0,
    }
    assert summary["duration_seconds"]["retry_adjusted_seconds"]["count"] == 1
    assert summary["duration_seconds"]["retry_adjusted_seconds"]["null_count"] == 1
    assert summary["successful_call_counts"]["planner_successful_calls"]["sum"] == 4
