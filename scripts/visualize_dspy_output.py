#!/usr/bin/env python3
"""Render DSPy AVQA ReAct JSONL output into a readable text report."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any


ANSWER_RE = re.compile(r"\b([A-F])\b")
ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-F])\s*</answer>", re.IGNORECASE)
ANSWER_PREFIX_RE = re.compile(r"^\s*([A-F])(?:[\).\]:\s]|$)", re.IGNORECASE)


def wrap_text(text: str, width: int = 110, indent: str = "") -> str:
    if not text:
        return ""
    wrapped: list[str] = []
    for line in str(text).splitlines():
        if not line.strip():
            wrapped.append("")
            continue
        wrapped.append(
            textwrap.fill(
                line,
                width=width,
                initial_indent=indent,
                subsequent_indent=indent,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n".join(wrapped)


def format_json_value(value: Any, indent: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=indent)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            parsed = json.loads(stripped)
            return json.dumps(parsed, ensure_ascii=False, indent=indent)
        except json.JSONDecodeError:
            return value
    return str(value)


def format_block(text: Any, prefix: str = "    ") -> str:
    value = format_json_value(text)
    if not value:
        return f"{prefix}<empty>"
    return "\n".join(f"{prefix}{line}" if line else prefix for line in value.splitlines())


def normalize_answer(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    tag_match = ANSWER_TAG_RE.search(text)
    if tag_match:
        return tag_match.group(1).upper()
    if len(text) == 1 and text in "ABCDEF":
        return text[0]
    prefix_match = ANSWER_PREFIX_RE.search(text)
    if prefix_match:
        return prefix_match.group(1).upper()
    match = ANSWER_RE.search(text)
    return match.group(1).upper() if match else ""


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def parse_options(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return []


def question_data(record: dict[str, Any]) -> dict[str, Any]:
    qd = record.get("question_data")
    if not isinstance(qd, dict):
        qd = record

    example = qd.get("example")
    if not isinstance(example, dict):
        return qd

    merged = dict(qd)
    for key in ("question", "question_id", "task_type", "video_id", "video_path", "audio_path"):
        value = first_non_empty(merged.get(key), example.get(key))
        if value is not None:
            merged[key] = value

    answer = first_non_empty(merged.get("answer"), merged.get("gold_answer"), example.get("answer"))
    if answer is not None:
        merged["answer"] = answer

    options = parse_options(first_non_empty(merged.get("options"), example.get("options"), example.get("options_json")))
    if options:
        merged["options"] = options

    response = first_non_empty(merged.get("response"), merged.get("reasoning_summary"))
    if response is not None:
        merged["response"] = response

    question_id = first_non_empty(merged.get("question_id"), merged.get("example_index"))
    if question_id is not None:
        merged["question_id"] = str(question_id)

    return merged


def record_video_id(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    qd = question_data(record)
    return str(first_non_empty(record.get("video_id"), metadata.get("video_id"), qd.get("video_id"), "<unknown>"))


def record_question_id(record: dict[str, Any]) -> str:
    return str(first_non_empty(question_data(record).get("question_id"), "<unknown>"))


def extract_pred_answer(record: dict[str, Any]) -> str:
    qd = question_data(record)
    pred_answer = normalize_answer(qd.get("pred_answer"))
    if pred_answer:
        return pred_answer
    trace = qd.get("turn_trace") or []
    for turn in trace:
        if str(turn.get("planner_action", "")).strip().lower() == "final":
            answer = normalize_answer(turn.get("final_answer"))
            if answer:
                return answer
    return normalize_answer(first_non_empty(qd.get("response"), qd.get("reasoning_summary")))


def is_error_record(record: dict[str, Any]) -> bool:
    qd = question_data(record)
    if qd.get("error"):
        return True
    response = str(qd.get("response") or "")
    if response.startswith("[ERROR]"):
        return True
    return any((turn.get("tool_error") for turn in qd.get("turn_trace") or []))


def format_turn(turn: dict[str, Any], turn_index: int) -> str:
    turn_id = turn.get("turn_id", turn_index + 1)
    planner_action = turn.get("planner_action")
    tool_name = turn.get("tool_name")
    backend = turn.get("perception_backend")
    final_answer = turn.get("final_answer")
    tool_error = turn.get("tool_error")

    parts: list[str] = []
    parts.append("  " + "-" * 96)
    parts.append(f"  TURN {turn_id}")
    parts.append("  " + "-" * 96)
    parts.append("  [Plan]")
    parts.append(f"    planner_action: {planner_action}")
    if tool_name:
        parts.append(f"    tool_name: {tool_name}")
    if backend:
        parts.append(f"    perception_backend: {backend}")
    if turn.get("video_id"):
        parts.append(f"    video_id: {turn.get('video_id')}")
    if turn.get("planner_raw"):
        parts.append("    planner_raw:")
        parts.append(format_block(turn.get("planner_raw"), prefix="      "))

    if turn.get("tool_args"):
        parts.append("")
        parts.append("  [Tool Args]")
        parts.append(format_block(turn.get("tool_args"), prefix="    "))

    if turn.get("tool_observation") is not None:
        parts.append("")
        parts.append("  [Tool Observation]")
        parts.append(format_block(turn.get("tool_observation"), prefix="    "))

    if turn.get("perception_thinking"):
        parts.append("")
        parts.append("  [Perception Thinking]")
        parts.append(format_block(turn.get("perception_thinking"), prefix="    "))

    if turn.get("perception_token_usage"):
        parts.append("")
        parts.append("  [Perception Token Usage]")
        parts.append(format_block(turn.get("perception_token_usage"), prefix="    "))

    if turn.get("planner_lm_response") is not None:
        parts.append("")
        parts.append("  [Planner LM Response]")
        parts.append(format_block(turn.get("planner_lm_response"), prefix="    "))

    if turn.get("planner_calls"):
        parts.append("")
        parts.append("  [Planner Calls]")
        parts.append(format_block(turn.get("planner_calls"), prefix="    "))

    if turn.get("planner_error"):
        parts.append("")
        parts.append("  [Planner Error Metadata]")
        parts.append(format_block(turn.get("planner_error"), prefix="    "))

    if tool_error:
        parts.append("")
        parts.append("  [Tool Error]")
        parts.append(format_block(str(tool_error), prefix="    "))

    if final_answer is not None and str(final_answer).strip():
        parts.append("")
        parts.append("  [Final Answer]")
        parts.append(wrap_text(str(final_answer), width=110, indent="    "))

    return "\n".join(parts)


def format_record(record: dict[str, Any], record_index: int) -> str:
    qd = question_data(record)
    trace = qd.get("turn_trace") or []

    video_id = record_video_id(record)
    question_id = record_question_id(record)
    task_type = qd.get("task_type", "")
    question = qd.get("question", "")
    options = qd.get("options") or parse_options(qd.get("options_json"))
    gt_answer = normalize_answer(first_non_empty(qd.get("answer"), qd.get("gold_answer")))
    pred_answer = extract_pred_answer(record)
    response = first_non_empty(qd.get("response"), qd.get("reasoning_summary"), "")
    score = qd.get("score")

    status = "ERROR" if is_error_record(record) else "OK"
    correctness = ""
    if gt_answer and pred_answer:
        correctness = "correct" if gt_answer == pred_answer else "wrong"

    lines: list[str] = []
    lines.append("=" * 110)
    lines.append(f"SAMPLE #{record_index + 1} [{status}]")
    lines.append("=" * 110)
    lines.append(f"video_id      : {video_id}")
    lines.append(f"question_id   : {question_id}")
    if task_type:
        lines.append(f"task_type     : {task_type}")
    if gt_answer or pred_answer:
        suffix = f" ({correctness})" if correctness else ""
        lines.append(f"answer        : gold={gt_answer or '<none>'} pred={pred_answer or '<none>'}{suffix}")
    if score is not None:
        lines.append(f"score         : {score}")
    lines.append(f"turns         : {len(trace)}")

    lines.append("")
    lines.append("Question:")
    lines.append(wrap_text(str(question), width=110, indent="  "))

    if options:
        lines.append("")
        lines.append("Options:")
        for opt in options:
            lines.append(wrap_text(str(opt), width=110, indent="  "))

    if response:
        lines.append("")
        lines.append("Model Response:")
        lines.append(wrap_text(str(response), width=110, indent="  "))

    lines.append("")
    lines.append("TURN TRACE")
    lines.append("=" * 110)
    if not trace:
        lines.append("  <no turns>")
    else:
        for idx, turn in enumerate(trace):
            lines.append(format_turn(turn, idx))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def natural_path_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def is_summary_json_file(path: Path) -> bool:
    return path.name.startswith("output_test")


def normalize_record(item: dict[str, Any], sample_id: str | None = None) -> dict[str, Any]:
    if isinstance(item.get("question_data"), dict):
        if item.get("video_id") or (item.get("metadata") or {}).get("video_id"):
            return item
        qd = question_data(item)
        video_id = str(first_non_empty(qd.get("video_id"), sample_id, qd.get("question_id"), "<unknown>"))
        metadata = dict(item.get("metadata") or {})
        metadata.setdefault("video_id", video_id)
        return {**item, "video_id": video_id, "metadata": metadata}

    qd = question_data(item)
    video_id = str(first_non_empty(item.get("video_id"), qd.get("video_id"), sample_id, qd.get("question_id"), "<unknown>"))
    return {
        "video_id": video_id,
        "metadata": {"video_id": video_id},
        "question_data": item,
    }


def load_json_file_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [normalize_record(item) for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [normalize_record(data, sample_id=path.stem)]
    raise ValueError(f"Expected JSON object or list in {path}, got {type(data).__name__}")


def load_directory_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    json_paths = sorted(
        (item for item in path.iterdir() if item.is_file() and item.suffix == ".json" and not is_summary_json_file(item)),
        key=natural_path_key,
    )
    for json_path in json_paths:
        records.extend(load_json_file_records(json_path))
    return records


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at line {line_no}, got {type(item).__name__}")
            records.append(normalize_record(item))
    return records


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        return load_directory_records(path)

    if path.suffix == ".json":
        return load_json_file_records(path)

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return [normalize_record(item) for item in data if isinstance(item, dict)]

    return load_jsonl_records(path)


def summarize(records: list[dict[str, Any]]) -> dict[str, int | float]:
    total = len(records)
    errors = sum(1 for record in records if is_error_record(record))
    with_pred = sum(1 for record in records if extract_pred_answer(record))
    total_turns = sum(len(question_data(record).get("turn_trace") or []) for record in records)
    correct = 0
    comparable = 0
    for record in records:
        qd = question_data(record)
        gold = normalize_answer(first_non_empty(qd.get("answer"), qd.get("gold_answer")))
        pred = extract_pred_answer(record)
        if gold and pred:
            comparable += 1
            correct += int(gold == pred)
    return {
        "total": total,
        "errors": errors,
        "with_pred": with_pred,
        "comparable": comparable,
        "correct": correct,
        "total_turns": total_turns,
        "avg_turns_per_question": (total_turns / total) if total else 0.0,
    }


def sample_json_path(record: dict[str, Any], input_path: Path) -> Path | None:
    video_id = record_video_id(record)
    if not video_id or video_id == "<unknown>":
        return None
    sample_dir = input_path if input_path.is_dir() else input_path.parent
    return sample_dir / f"{video_id}.json"


def delete_error_sample_jsons(records: list[dict[str, Any]], input_path: Path) -> tuple[int, int]:
    deleted = 0
    missing = 0
    for record in records:
        if not is_error_record(record):
            continue

        path = sample_json_path(record, input_path)
        if path is None:
            missing += 1
            continue
        if path.suffix != ".json" or path.resolve() == input_path.resolve():
            print(f"[SKIP] Refusing to delete non-sample file: {path}")
            continue
        if (not path.exists()) or (not path.is_file()):
            print(f"[SKIP] Refusing to delete non / missing file path: {path}")
            missing += 1
            continue

        path.unlink()
        deleted += 1

    return deleted, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Render DSPy AVQA ReAct output into readable text.")
    parser.add_argument(
        "input_jsonl",
        type=Path,
        help="Input DSPy output JSONL/JSON path, or a directory of per-sample JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output text file path. Default: <input_stem>.dspy_readable.txt, or <dir>/<dir>.dspy_readable.txt for directory input.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Render only first N samples.")
    parser.add_argument("--video-id", type=str, default=None, help="Only render this video_id.")
    parser.add_argument("--question-id", type=str, default=None, help="Only render this question_id.")
    parser.add_argument("--errors-only", action="store_true", help="Only render records with errors.")
    parser.add_argument(
        "--delete-error-sample",
        "--delete_error_sample",
        action="store_true",
        help="After writing the readable report, delete per-sample .json files for ERROR records.",
    )
    args = parser.parse_args()

    records = load_records(args.input_jsonl)
    if args.video_id:
        records = [record for record in records if record_video_id(record) == args.video_id]
    if args.question_id:
        records = [record for record in records if record_question_id(record) == args.question_id]
    if args.errors_only:
        records = [record for record in records if is_error_record(record)]
    if args.max_samples is not None:
        records = records[: args.max_samples]

    stats = summarize(records)
    if args.output:
        output_path = args.output
    elif args.input_jsonl.is_dir():
        output_path = args.input_jsonl / f"{args.input_jsonl.name}.dspy_readable.txt"
    else:
        output_path = args.input_jsonl.with_name(f"{args.input_jsonl.stem}.dspy_readable.txt")

    parts: list[str] = []
    parts.append("# Readable DSPy ReAct Output")
    parts.append(f"# Source: {args.input_jsonl}")
    parts.append(f"# Samples: {stats['total']}")
    parts.append(f"# Errors: {stats['errors']}")
    parts.append(f"# Predictions: {stats['with_pred']}")
    if stats["comparable"]:
        parts.append(f"# Comparable accuracy: {stats['correct']}/{stats['comparable']}")
    parts.append(f"# Average turns per question: {stats['avg_turns_per_question']:.2f}")
    parts.append("")

    for idx, record in enumerate(records):
        parts.append(format_record(record, idx))

    output_path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"[OK] Wrote readable DSPy output to: {output_path}")
    if args.delete_error_sample:
        deleted, missing = delete_error_sample_jsons(records, args.input_jsonl)
        print(f"[OK] Deleted {deleted} ERROR sample json file(s); {missing} expected file(s) missing.")


if __name__ == "__main__":
    main()
