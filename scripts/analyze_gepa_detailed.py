#!/usr/bin/env python3
"""Analyze a trusted GEPA run and render candidate/evaluation diagnostics.

Run the extraction phase with the Python environment used by GEPA.  If that
environment has no matplotlib, this script automatically invokes a plotting
Python for the PNG-only rendering phase.

The acceptance terminology follows GEPA itself:
  * proposed: the child was evaluated on the comparison minibatch;
  * accepted/admitted: it passed the minibatch gate, received validation, and
    was added to the candidate pool (``new_program_idx`` exists);
  * validation-improved: its paired validation mean exceeds its parent's;
  * aggregate-best/current-frontier: post-hoc properties, not acceptance.

Use ``--live`` to take a best-effort snapshot while GEPA is still running.
That mode does not require the end-of-run metadata file: it renders all
state-only diagnostics (including candidate validation curves), but skips
question-type diagnostics unless ``--valset-jsonl`` is supplied.

Only unpickle ``gepa_state.bin`` files from trusted runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_RELATIVE_PATH = Path("gepa_logs/gepa_state.bin")
METADATA_NAME = "compiled_gepa_metadata.json"
DEFAULT_PLOT_PYTHON = "/mnt/ceph_rbd/applications/anaconda3/bin/python"

REQUESTED_TYPES = ("temporal", "audio", "visual", "why", "ordering")
AUDIO_RE = re.compile(
    r"\b(audio|auditory|sound(?:s|track)?|music(?:al)?|speech|narrat(?:or|ion|es|ed)?|"
    r"dialogue|vocal|voice|heard|spoken|speaker|commentat(?:or|ary)|lyrics?)\b",
    re.IGNORECASE,
)
VISUAL_RE = re.compile(
    r"\b(visual(?:s|ly)?|video|shown|showing|display(?:s|ed)?|screen|camera|scene|view|"
    r"visible|appear(?:s|ed|ance)?|image|graphic|shot|footage|depict(?:s|ed|ion)?)\b",
    re.IGNORECASE,
)
WHY_RE = re.compile(r"^\s*why\b", re.IGNORECASE)


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    return int(number) if number is not None and number.is_integer() else None


def mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def numeric_list(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        source = value.values()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        source = value
    else:
        source = (value,)
    return [number for item in source if (number := as_float(item)) is not None]


def indexed_scores(value: Any) -> dict[int, float]:
    output: dict[int, float] = {}
    if isinstance(value, Mapping):
        source = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        source = enumerate(value)
    else:
        return output
    for key, item in source:
        index = as_int(key)
        score = as_float(item)
        if index is not None and score is not None:
            output[index] = score
    return output


def rolling_mean(values: Sequence[float | None], window: int) -> list[float | None]:
    result: list[float | None] = []
    for index in range(len(values)):
        chunk = [v for v in values[max(0, index - window + 1) : index + 1] if v is not None]
        result.append(mean(chunk))
    return result


def load_state(path: Path, retries: int = 3, require_stable: bool = False) -> dict[str, Any]:
    """Load a GEPA state, retrying if a live writer is updating it in place."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            before = path.stat()
            with path.open("rb") as stream:
                payload = pickle.load(stream)
            after = path.stat()
            if require_stable and (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                raise RuntimeError("GEPA state changed while it was being read")
            if isinstance(payload, Mapping):
                return dict(payload)
            state_dict = getattr(payload, "__dict__", None)
            if isinstance(state_dict, Mapping):
                return dict(state_dict)
            raise TypeError(f"unsupported state object: {type(payload).__name__}")
        except (EOFError, OSError, pickle.UnpicklingError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5)
    raise RuntimeError(f"Could not read a stable GEPA state from {path}: {last_error}")


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def cut_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("id")
    if value is None:
        supervisions = row.get("supervisions")
        if isinstance(supervisions, list) and supervisions and isinstance(supervisions[0], Mapping):
            value = supervisions[0].get("id")
    return str(value) if value is not None else None


def load_val_examples(path: Path, metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(dict(payload))

    val_cuts = as_int(metadata.get("val_cuts"))
    if val_cuts is not None:
        rows = rows[:val_cuts]
    skipped_raw = metadata.get("skipped_val_examples") or []
    skipped: set[str] = set()
    if isinstance(skipped_raw, Sequence) and not isinstance(skipped_raw, (str, bytes, bytearray)):
        for value in skipped_raw:
            if isinstance(value, Mapping):
                value = value.get("id") or value.get("cut_id")
            if value is not None:
                skipped.add(str(value))
    if skipped:
        rows = [row for row in rows if cut_id(row) not in skipped]
    return rows


def example_info(row: Mapping[str, Any], val_id: int) -> dict[str, Any]:
    supervision: Mapping[str, Any] = {}
    supervisions = row.get("supervisions")
    if isinstance(supervisions, list) and supervisions and isinstance(supervisions[0], Mapping):
        supervision = supervisions[0]
    custom = supervision.get("custom")
    custom = custom if isinstance(custom, Mapping) else {}
    question = custom.get("Question") or custom.get("question") or supervision.get("text") or ""
    native_type = custom.get("task_type") or custom.get("Type") or "unknown"
    labels: list[str] = []
    # These two use authoritative dataset labels. Audio/visual are lexical
    # because Daily-Omni has no dedicated modality flags. Labels may overlap.
    if native_type == "AV Event Alignment":
        labels.append("temporal")
    if AUDIO_RE.search(str(question)):
        labels.append("audio")
    if VISUAL_RE.search(str(question)):
        labels.append("visual")
    if WHY_RE.search(str(question)):
        labels.append("why")
    if native_type == "Event Sequence":
        labels.append("ordering")
    return {
        "val_id": val_id,
        "cut_id": cut_id(row),
        "question": str(question),
        "native_type": str(native_type),
        "requested_types": labels,
    }


def parents_for(parents: Any, candidate: int) -> list[int]:
    if not isinstance(parents, Sequence) or isinstance(parents, (str, bytes, bytearray)):
        return []
    if candidate >= len(parents):
        return []
    raw = parents[candidate]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raw = [raw]
    return [index for value in raw if (index := as_int(value)) is not None]


def current_frontier_candidates(state: Mapping[str, Any]) -> set[int]:
    frontier_type = str(state.get("frontier_type") or "instance")
    if frontier_type == "objective":
        mapping = state.get("program_at_pareto_front_objectives")
    elif frontier_type == "cartesian":
        mapping = state.get("program_at_pareto_front_cartesian")
    elif frontier_type == "hybrid":
        mappings = [
            state.get("program_at_pareto_front_valset"),
            state.get("program_at_pareto_front_objectives"),
        ]
        return {
            index
            for mapping in mappings
            if isinstance(mapping, Mapping)
            for values in mapping.values()
            if isinstance(values, (set, list, tuple))
            for value in values
            if (index := as_int(value)) is not None
        }
    else:
        mapping = state.get("program_at_pareto_front_valset")
    if not isinstance(mapping, Mapping):
        return set()
    return {
        index
        for values in mapping.values()
        if isinstance(values, (set, list, tuple))
        for value in values
        if (index := as_int(value)) is not None
    }


def paired_gain(child: Mapping[int, float], parent: Mapping[int, float]) -> tuple[float | None, int]:
    common = sorted(set(child) & set(parent))
    if not common:
        return None, 0
    return statistics.fmean(child[index] - parent[index] for index in common), len(common)


def classify_trace_event(event: Mapping[str, Any], trace_index: int) -> dict[str, Any]:
    raw_iteration = as_int(event.get("i"))
    iteration = (raw_iteration + 1) if raw_iteration is not None else trace_index + 1
    child = as_int(event.get("new_program_idx"))
    selected_parent = as_int(event.get("selected_program_candidate"))
    base: dict[str, Any] = {
        "trace_index": trace_index,
        "iteration": iteration,
        "selected_parent": selected_parent,
        "child_candidate": child,
        "invoked_merge": bool(event.get("invoked_merge")),
    }

    if "new_program_subsample_scores" in event:
        new_scores = numeric_list(event.get("new_program_subsample_scores"))
        id1_scores = numeric_list(event.get("id1_subsample_scores"))
        id2_scores = numeric_list(event.get("id2_subsample_scores"))
        entities = event.get("merged_entities")
        parents: list[int] = []
        if isinstance(entities, Sequence) and not isinstance(entities, (str, bytes, bytearray)):
            parents = [index for value in list(entities)[:2] if (index := as_int(value)) is not None]
        old_means = [value for scores in (id1_scores, id2_scores) if (value := mean(scores)) is not None]
        old_mean = max(old_means) if old_means else None
        new_mean = mean(new_scores)
        return {
            **base,
            "proposal_kind": "merge",
            "proposal_status": "accepted" if child is not None else "rejected",
            "parent_candidates": parents,
            "batch_size": len(new_scores),
            "old_mean": old_mean,
            "new_mean": new_mean,
            "minibatch_gain": (new_mean - old_mean) if new_mean is not None and old_mean is not None else None,
            "old_scores_by_parent": [id1_scores, id2_scores],
        }

    old_scores = numeric_list(event.get("subsample_scores"))
    if "new_subsample_scores" in event:
        new_scores = numeric_list(event.get("new_subsample_scores"))
        old_mean = mean(old_scores)
        new_mean = mean(new_scores)
        return {
            **base,
            "proposal_kind": "reflective",
            "proposal_status": "accepted" if child is not None else "rejected",
            "parent_candidates": [selected_parent] if selected_parent is not None else [],
            "batch_size": len(new_scores),
            "old_mean": old_mean,
            "new_mean": new_mean,
            "minibatch_gain": (new_mean - old_mean) if new_mean is not None and old_mean is not None else None,
            "old_scores_by_parent": [old_scores],
        }

    return {
        **base,
        "proposal_kind": "none",
        "proposal_status": "no_completed_proposal",
        "parent_candidates": [selected_parent] if selected_parent is not None else [],
        "batch_size": len(old_scores),
        "old_mean": mean(old_scores),
        "new_mean": None,
        "minibatch_gain": None,
        "old_scores_by_parent": [old_scores] if old_scores else [],
    }


def aggregate_type_metrics(
    candidate_scores: Sequence[Mapping[int, float]], examples: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate, scores in enumerate(candidate_scores):
        groups: dict[tuple[str, str], list[float]] = {}
        for item in examples:
            val_id = int(item["val_id"])
            score = scores.get(val_id)
            if score is None:
                continue
            native = str(item["native_type"])
            groups.setdefault(("native", native), []).append(score)
            labels = item.get("requested_types") or []
            for label in labels:
                groups.setdefault(("requested", str(label)), []).append(score)
            if not labels:
                groups.setdefault(("requested", "unclassified"), []).append(score)
        for (taxonomy, question_type), values in sorted(groups.items()):
            rows.append(
                {
                    "candidate_index": candidate,
                    "taxonomy": taxonomy,
                    "question_type": question_type,
                    "score_mean": mean(values),
                    "num_examples": len(values),
                }
            )
    return rows


def analyze(run_dir: Path, valset_override: Path | None, rolling_window: int, live: bool = False) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    state_path = run_dir / STATE_RELATIVE_PATH
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing {state_path}")
    metadata = load_json_object(run_dir / METADATA_NAME)
    # GEPA writes this pickle at iteration boundaries. A stable stat check
    # prevents a live snapshot from analysing a partly replaced state file.
    state = load_state(state_path, retries=10 if live else 3, require_stable=live)

    valset_value = valset_override or (Path(str(metadata["valset_jsonl"])) if metadata.get("valset_jsonl") else None)
    if valset_value is None and not live:
        raise ValueError("No valset path in metadata; pass --valset-jsonl explicitly")
    valset_path = valset_value.expanduser().resolve() if valset_value is not None else None
    examples = (
        [example_info(row, index) for index, row in enumerate(load_val_examples(valset_path, metadata))]
        if valset_path is not None
        else []
    )

    score_values = state.get("prog_candidate_val_subscores")
    if not isinstance(score_values, Sequence) or isinstance(score_values, (str, bytes, bytearray)):
        score_values = []
    candidate_scores = [indexed_scores(value) for value in score_values]
    parents = state.get("parent_program_for_candidate")
    discovery = state.get("num_metric_calls_by_discovery")
    discovery = discovery if isinstance(discovery, Sequence) else []
    trace_raw = state.get("full_program_trace")
    trace_raw = trace_raw if isinstance(trace_raw, Sequence) else []
    events = [classify_trace_event(event, index) for index, event in enumerate(trace_raw) if isinstance(event, Mapping)]
    child_to_event = {
        int(event["child_candidate"]): event for event in events if event.get("child_candidate") is not None
    }
    frontier = current_frontier_candidates(state)

    validation_means = [mean(list(scores.values())) for scores in candidate_scores]
    finite_candidates = [index for index, value in enumerate(validation_means) if value is not None]
    best_candidate = max(finite_candidates, key=lambda index: (validation_means[index], -index)) if finite_candidates else None
    used_as_parent = Counter(parent for child in range(1, len(candidate_scores)) for parent in parents_for(parents, child))

    candidates: list[dict[str, Any]] = []
    gains: list[dict[str, Any]] = []
    for candidate, scores in enumerate(candidate_scores):
        parent_ids = parents_for(parents, candidate)
        event = child_to_event.get(candidate)
        row = {
            "candidate_index": candidate,
            "parents": parent_ids,
            "created_iteration": event.get("iteration") if event else (0 if candidate == 0 else None),
            "proposal_kind": event.get("proposal_kind") if event else ("seed" if candidate == 0 else "unknown"),
            "validation_mean": validation_means[candidate],
            "validation_examples": len(scores),
            "metric_calls_at_discovery": as_int(discovery[candidate]) if candidate < len(discovery) else None,
            "is_seed": candidate == 0,
            "is_aggregate_best": candidate == best_candidate,
            "is_on_current_pareto_front": candidate in frontier,
            "selected_as_later_parent_count": used_as_parent[candidate],
        }
        candidates.append(row)
        if candidate == 0:
            continue
        for parent_position, parent in enumerate(parent_ids):
            if parent >= len(candidate_scores):
                continue
            val_gain, val_overlap = paired_gain(scores, candidate_scores[parent])
            minibatch_gain = None
            minibatch_n = None
            if event:
                if event.get("proposal_kind") == "reflective":
                    minibatch_gain = event.get("minibatch_gain")
                    minibatch_n = event.get("batch_size")
                else:
                    old_sets = event.get("old_scores_by_parent") or []
                    old_scores = old_sets[parent_position] if parent_position < len(old_sets) else []
                    old_mean = mean(old_scores)
                    new_mean = event.get("new_mean")
                    if old_mean is not None and new_mean is not None:
                        minibatch_gain = new_mean - old_mean
                    minibatch_n = len(old_scores)
            gains.append(
                {
                    "iteration": event.get("iteration") if event else None,
                    "proposal_kind": event.get("proposal_kind") if event else "unknown",
                    "parent_candidate": parent,
                    "child_candidate": candidate,
                    "minibatch_gain": minibatch_gain,
                    "minibatch_examples": minibatch_n,
                    "validation_gain": val_gain,
                    "validation_overlap": val_overlap,
                    "validation_improved": val_gain is not None and val_gain > 0,
                    "child_is_aggregate_best": candidate == best_candidate,
                    "child_is_on_current_pareto_front": candidate in frontier,
                    "child_selected_as_later_parent": used_as_parent[candidate] > 0,
                }
            )

    old_curve = [event.get("old_mean") for event in events]
    for event, smoothed in zip(events, rolling_mean(old_curve, rolling_window)):
        event["old_mean_rolling"] = smoothed

    type_metrics = aggregate_type_metrics(candidate_scores, examples)
    label_counts = Counter(label for item in examples for label in item["requested_types"])
    native_counts = Counter(item["native_type"] for item in examples)
    proposal_counts = Counter(event["proposal_status"] for event in events)
    numeric_gains = [
        (float(row["minibatch_gain"]), float(row["validation_gain"]))
        for row in gains
        if row.get("minibatch_gain") is not None and row.get("validation_gain") is not None
    ]
    correlation = None
    if len(numeric_gains) >= 2:
        xs, ys = zip(*numeric_gains)
        if statistics.pstdev(xs) > 0 and statistics.pstdev(ys) > 0:
            correlation = statistics.correlation(xs, ys)

    state_stat = state_path.stat()
    return {
        "schema_version": 1,
        "analysis_mode": "live" if live else "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "state_path": str(state_path),
        "state_mtime_utc": datetime.fromtimestamp(state_stat.st_mtime, timezone.utc).isoformat(),
        "state_iteration_raw": as_int(state.get("i")),
        "total_metric_calls": as_int(state.get("total_num_evals")),
        "num_full_validation_evaluations": as_int(state.get("num_full_ds_evals")),
        "frontier_type": state.get("frontier_type"),
        "valset_jsonl": str(valset_path) if valset_path is not None else None,
        "valset_examples_loaded": len(examples),
        "rolling_window": rolling_window,
        "best_candidate_index": best_candidate,
        "baseline_validation_mean": validation_means[0] if validation_means else None,
        "best_validation_mean": validation_means[best_candidate] if best_candidate is not None else None,
        "candidate_count": len(candidates),
        "accepted_child_count": max(len(candidates) - 1, 0),
        "proposal_status_counts": dict(proposal_counts),
        "parent_child_gain_correlation": correlation,
        "requested_type_definition": {
            "temporal": "native task_type == 'AV Event Alignment'",
            "audio": AUDIO_RE.pattern,
            "visual": VISUAL_RE.pattern,
            "why": "question matches ^\\s*why\\b",
            "ordering": "native task_type == 'Event Sequence'",
            "note": "multi-label; categories may overlap",
        },
        "requested_type_counts": dict(label_counts),
        "native_type_counts": dict(native_counts),
        "events": events,
        "candidates": candidates,
        "parent_child_gains": gains,
        "question_type_metrics": type_metrics,
        "examples": examples,
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def compact_event_rows(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "trace_index", "iteration", "proposal_kind", "proposal_status", "selected_parent",
        "parent_candidates", "child_candidate", "batch_size", "old_mean", "new_mean",
        "minibatch_gain", "old_mean_rolling", "invoked_merge",
    )
    return [{field: event.get(field) for field in fields} for event in events]


def write_auto_report(path: Path, data: Mapping[str, Any]) -> None:
    candidates = data["candidates"]
    best = data.get("best_candidate_index")
    baseline = data.get("baseline_validation_mean")
    best_score = data.get("best_validation_mean")
    accepted = int(data.get("accepted_child_count") or 0)
    events = data["events"]
    completed = [event for event in events if event.get("proposal_kind") != "none"]
    rejected = [event for event in completed if event.get("proposal_status") == "rejected"]
    val_gains = [row.get("validation_gain") for row in data["parent_child_gains"]]
    positive = sum(value is not None and value > 0 for value in val_gains)
    zero = sum(value is not None and math.isclose(value, 0.0) for value in val_gains)
    negative = sum(value is not None and value < 0 for value in val_gains)
    has_question_type_metrics = bool(data.get("question_type_metrics"))
    question_type_definition = (
        "Requested categories are multi-label. `temporal` and `ordering` use native dataset labels; "
        "`why` uses the question prefix; `audio` and `visual` use the regexes stored in `analysis_data.json`. "
        "The native six-way task-type results are exported alongside them."
        if has_question_type_metrics
        else "Question-type diagnostics were skipped because no validation-set JSONL was available for this live snapshot. "
        "Pass `--valset-jsonl` to enable them."
    )
    question_type_file = (
        "- `question_type_analysis.png`: requested multi-label curves and native-type best-minus-baseline deltas."
        if has_question_type_metrics
        else "- `question_type_analysis.png`: not generated (no validation-set JSONL)."
    )
    lines = [
        f"# GEPA analysis: `{data['run_name']}`",
        "",
        f"Snapshot: `{data['state_mtime_utc']}`; state raw iteration `{data['state_iteration_raw']}`; "
        f"metric calls `{data['total_metric_calls']}`.",
        "",
        "## Main findings",
        "",
        f"- Completed minibatch proposals: **{len(completed)}**; admitted children: **{accepted}**; "
        f"minibatch-rejected proposals: **{len(rejected)}**.",
        f"- Aggregate validation: baseline **{baseline:.4f}**; best candidate **{best}** = **{best_score:.4f}**."
        if baseline is not None and best_score is not None else "- Aggregate validation score unavailable.",
        f"- Parent→child paired validation gain: positive/zero/negative = **{positive}/{zero}/{negative}**; "
        f"Pearson r(minibatch gain, validation gain) = **{data.get('parent_child_gain_correlation')}**.",
        "- GEPA acceptance means minibatch-gate admission into the candidate pool. It does not imply positive validation gain or final-best status.",
        "",
        "## Question-type definitions",
        "",
        question_type_definition,
        "",
        "## Files",
        "",
        "- `optimization_curves.png`: stochastic selected-parent minibatch curve and candidate validation curve.",
        "- `parent_child_gain_scatter.png`: paired minibatch gain vs paired validation gain.",
        "- `candidate_evolution.png`: proposal gate outcomes and admitted-candidate lineage.",
        question_type_file,
        "- CSV/JSON files contain exact underlying values.",
        "",
        "## Live-run caveat",
        "",
        "GEPA saves state around iteration boundaries. During a live run, this snapshot can lag the text log by an in-flight iteration.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(data_path: Path, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(data_path.read_text(encoding="utf-8"))
    events = data["events"]
    candidates = data["candidates"]
    gains = data["parent_child_gains"]
    type_rows = data["question_type_metrics"]

    # 1. Core optimization curves.
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    x = [event["iteration"] for event in events]
    old = [event.get("old_mean") for event in events]
    smooth = [event.get("old_mean_rolling") for event in events]
    axes[0].plot(x, smooth, color="#8bb8dc", linewidth=3.2, label=f"rolling mean (window={data['rolling_window']})")
    axes[0].plot(x, old, color="#085a9c", linewidth=1.3, marker="o", markersize=3, label="selected parent: raw minibatch")
    proposed = [event for event in events if event.get("new_mean") is not None]
    axes[0].scatter(
        [event["iteration"] for event in proposed], [event["new_mean"] for event in proposed],
        c=["#138a4b" if event["proposal_status"] == "accepted" else "#c94c4c" for event in proposed],
        marker="s", s=28, alpha=0.85, label="proposal (green accepted / red rejected)", zorder=4,
    )
    axes[0].set(title="GEPA discovery minibatches", xlabel="GEPA iteration", ylabel="mean score")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower right")

    metric_x = [row.get("metric_calls_at_discovery") if row.get("metric_calls_at_discovery") is not None else row["candidate_index"] for row in candidates]
    val_y = [row.get("validation_mean") for row in candidates]
    axes[1].plot(metric_x, val_y, color="#6b3fa0", marker="o", linewidth=2.0)
    for row, px, py in zip(candidates, metric_x, val_y):
        if py is not None:
            axes[1].annotate(f"c{row['candidate_index']}", (px, py), xytext=(4, 5), textcoords="offset points", fontsize=8)
    if val_y and val_y[0] is not None:
        axes[1].axhline(val_y[0], color="#777777", linestyle="--", linewidth=1, label="baseline")
    axes[1].set(title="Accepted candidates: full validation", xlabel="metric calls at discovery", ylabel="validation mean")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.savefig(output_dir / "optimization_curves.png", dpi=180)
    plt.close(fig)

    # 2. Accepted parent-child gain scatter.
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for row in gains:
        gx, gy = row.get("minibatch_gain"), row.get("validation_gain")
        if gx is None or gy is None:
            continue
        marker = "D" if row.get("proposal_kind") == "merge" else "o"
        color = "#18794e" if gy > 0 else ("#777777" if math.isclose(gy, 0.0) else "#b42318")
        size = 90 if row.get("child_is_aggregate_best") else 55
        ax.scatter(gx, gy, marker=marker, s=size, color=color, edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(
            f"c{row['parent_candidate']}→c{row['child_candidate']}", (gx, gy),
            xytext=(4, 5), textcoords="offset points", fontsize=8,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        title=f"Parent→child gain (Pearson r={data.get('parent_child_gain_correlation')})",
        xlabel="minibatch mean gain (same examples)", ylabel="validation paired mean gain (overlap)",
    )
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "parent_child_gain_scatter.png", dpi=180)
    plt.close(fig)

    # 3. Proposal dumbbells plus accepted lineage.
    fig, axes = plt.subplots(2, 1, figsize=(max(13, len(events) * 0.25), 11), constrained_layout=True)
    ax = axes[0]
    for event in proposed:
        step, old_y, new_y = event["iteration"], event.get("old_mean"), event.get("new_mean")
        if old_y is None or new_y is None:
            continue
        accepted = event["proposal_status"] == "accepted"
        ax.vlines(step, min(old_y, new_y), max(old_y, new_y), color="#9a9a9a", linewidth=1)
        ax.scatter(step, old_y, marker="o", s=45, color="#1f5f99", zorder=3)
        ax.scatter(
            step, new_y, marker="s", s=50,
            facecolor="#d1495b" if accepted else "white", edgecolor="#d1495b", linewidth=1.5, zorder=3,
        )
        if accepted:
            ax.annotate(f"c{event['child_candidate']}", (step, new_y), xytext=(3, 5), textcoords="offset points", fontsize=8)
    ax.set(title="Proposal evolution (filled child = accepted into candidate pool)", xlabel="GEPA iteration", ylabel="minibatch mean")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for row in candidates:
        child = row["candidate_index"]
        child_y = row.get("validation_mean")
        if child_y is None:
            continue
        for parent in row.get("parents") or []:
            if parent < len(candidates) and candidates[parent].get("validation_mean") is not None:
                width = 2.4 if row.get("selected_as_later_parent_count", 0) else 1.0
                ax.plot([parent, child], [candidates[parent]["validation_mean"], child_y], color="#8d99ae", linewidth=width, zorder=1)
        marker = "*" if row.get("is_aggregate_best") else "o"
        size = 180 if marker == "*" else (75 if row.get("selected_as_later_parent_count", 0) else 50)
        color = "#f4b942" if marker == "*" else ("#315f8c" if row.get("selected_as_later_parent_count", 0) else "#9bb7cf")
        ax.scatter(child, child_y, marker=marker, s=size, color=color, edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(f"c{child}", (child, child_y), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set(title="Accepted-candidate lineage (star = aggregate validation best)", xlabel="candidate index", ylabel="validation mean")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "candidate_evolution.png", dpi=180)
    plt.close(fig)

    # The core three plots need only the GEPA state. Type diagnostics need
    # the original validation examples, which a live run may not yet record.
    if not type_rows:
        return

    # 4. Requested categories over candidates + native best-baseline delta.
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    ax = axes[0]
    requested = [row for row in type_rows if row["taxonomy"] == "requested" and row["question_type"] in REQUESTED_TYPES]
    for question_type in REQUESTED_TYPES:
        rows = sorted((row for row in requested if row["question_type"] == question_type), key=lambda row: row["candidate_index"])
        if rows:
            ax.plot([row["candidate_index"] for row in rows], [row["score_mean"] for row in rows], marker="o", label=f"{question_type} (n={rows[0]['num_examples']})")
    ax.set(title="Requested question dimensions (multi-label)", xlabel="accepted candidate index", ylabel="validation mean")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)

    ax = axes[1]
    native = [row for row in type_rows if row["taxonomy"] == "native"]
    best = data.get("best_candidate_index")
    baseline_by_type = {row["question_type"]: row for row in native if row["candidate_index"] == 0}
    best_by_type = {row["question_type"]: row for row in native if row["candidate_index"] == best}
    names = sorted(set(baseline_by_type) & set(best_by_type))
    deltas = [best_by_type[name]["score_mean"] - baseline_by_type[name]["score_mean"] for name in names]
    colors = ["#18794e" if value > 0 else ("#777777" if math.isclose(value, 0) else "#b42318") for value in deltas]
    ax.barh(names, deltas, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title=f"Native task types: best c{best} minus baseline", xlabel="validation mean delta")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output_dir / "question_type_analysis.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed GEPA optimization analysis and PNG visualization")
    parser.add_argument("--gepa-result-dir", type=Path, help="GEPA result directory")
    parser.add_argument("--valset-jsonl", type=Path, default=None, help="Override validation-set JSONL")
    parser.add_argument(
        "--live", action="store_true",
        help="Snapshot an in-progress run; metadata/validation-set JSONL are optional and type diagnostics are skipped without one",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument("--plot-python", type=Path, default=Path(os.environ.get("GEPA_PLOT_PYTHON", DEFAULT_PLOT_PYTHON)))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--render-json", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.render_json is not None:
        output_dir = args.output_dir or args.render_json.parent
        render(args.render_json.resolve(), output_dir.resolve())
        return 0
    if args.gepa_result_dir is None:
        raise ValueError("--gepa-result-dir is required")
    if args.rolling_window < 1:
        raise ValueError("--rolling-window must be >= 1")
    default_output_dir = Path(__file__).resolve().parent / "analysis_results" / args.gepa_result_dir.name
    if args.live:
        default_output_dir /= "live"
    output_dir = (args.output_dir or default_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = analyze(args.gepa_result_dir, args.valset_jsonl, args.rolling_window, live=args.live)
    data_path = output_dir / "analysis_data.json"
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    write_csv(output_dir / "proposal_evolution.csv", compact_event_rows(data["events"]))
    write_csv(output_dir / "candidate_metrics.csv", data["candidates"])
    write_csv(output_dir / "parent_child_gains.csv", data["parent_child_gains"])
    write_csv(output_dir / "question_type_metrics.csv", data["question_type_metrics"])
    write_auto_report(output_dir / "analysis_report.md", data)
    if not args.no_render:
        command = [
            str(args.plot_python), str(Path(__file__).resolve()), "--render-json", str(data_path),
            "--output-dir", str(output_dir),
        ]
        subprocess.run(command, check=True)
    print(json.dumps({
        "output_dir": str(output_dir),
        "analysis_mode": data["analysis_mode"],
        "valset_jsonl": data["valset_jsonl"],
        "candidate_count": data["candidate_count"],
        "accepted_child_count": data["accepted_child_count"],
        "best_candidate_index": data["best_candidate_index"],
        "baseline_validation_mean": data["baseline_validation_mean"],
        "best_validation_mean": data["best_validation_mean"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
