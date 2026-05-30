#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import textwrap


def wrap_text(text: str, width: int = 110, indent: str = "") -> str:
    if not text:
        return ""
    lines = text.splitlines()
    wrapped = []
    for line in lines:
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


def format_json_string(value, indent: int = 2) -> str:
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


def format_multiline_block(text: str, prefix: str = "    ") -> str:
    if not text:
        return f"{prefix}<empty>"
    return "\n".join(f"{prefix}{line}" if line else prefix for line in text.splitlines())


def format_turn(turn: dict, turn_index: int) -> str:
    turn_id = turn.get("turn_id", turn_index + 1)
    planner_action = turn.get("planner_action")
    tool_name = turn.get("tool_name")
    tool_error = turn.get("tool_error")
    final_answer = turn.get("final_answer")

    planner_raw = format_json_string(turn.get("planner_raw"))
    tool_args = format_json_string(turn.get("tool_args"))
    tool_observation = format_json_string(turn.get("tool_observation"))

    parts = []
    parts.append("  " + "-" * 96)
    parts.append(f"  TURN {turn_id}")
    parts.append("  " + "-" * 96)
    parts.append("  [Plan]")
    parts.append(f"    planner_action: {planner_action}")
    if tool_name:
        parts.append(f"    tool_name: {tool_name}")
    if planner_raw:
        parts.append("    planner_raw:")
        parts.append(format_multiline_block(planner_raw, prefix="      "))

    if tool_args:
        parts.append("")
        parts.append("  [Tool Args]")
        parts.append(format_multiline_block(tool_args, prefix="    "))

    if tool_observation:
        parts.append("")
        parts.append("  [Tool Observation]")
        parts.append(format_multiline_block(tool_observation, prefix="    "))

    if tool_error:
        parts.append("")
        parts.append("  [Tool Error]")
        parts.append(format_multiline_block(str(tool_error), prefix="    "))

    if final_answer:
        parts.append("")
        parts.append("  [Final Answer]")
        parts.append(wrap_text(str(final_answer), width=110, indent="    "))

    return "\n".join(parts)


def format_record(record: dict, record_index: int) -> str:
    metadata = record.get("metadata") or {}
    qd = record.get("question_data") or {}

    video_id = record.get("video_id") or metadata.get("video_id", "<unknown>")
    question_id = qd.get("question_id", "<unknown>")
    question = qd.get("question", "")
    task_type = qd.get("task_type", "")
    answer = qd.get("answer", "")
    response = qd.get("response", "")
    options = qd.get("options") or []
    turn_trace = qd.get("turn_trace") or []

    lines = []
    lines.append("=" * 110)
    lines.append(f"SAMPLE #{record_index + 1}")
    lines.append("=" * 110)
    lines.append(f"video_id      : {video_id}")
    lines.append(f"question_id   : {question_id}")
    if task_type:
        lines.append(f"task_type     : {task_type}")
    lines.append("")
    lines.append("Question:")
    lines.append(wrap_text(str(question), width=110, indent="  "))
    if options:
        lines.append("")
        lines.append("Options:")
        for opt in options:
            lines.append(wrap_text(str(opt), width=110, indent="  "))
    if answer:
        lines.append("")
        lines.append(f"GT Answer: {answer}")
    if response:
        lines.append("Model Response:")
        lines.append(wrap_text(str(response), width=110, indent="  "))

    lines.append("")
    lines.append("TURN TRACE")
    lines.append("=" * 110)
    if not turn_trace:
        lines.append("  <no turns>")
    else:
        for idx, turn in enumerate(turn_trace):
            lines.append(format_turn(turn, idx))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def load_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {i}: {e}") from e
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Render compact react output jsonl into readable text format."
    )
    parser.add_argument("input_jsonl", type=Path, help="Input JSONL path.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output text file path. Default: <input_stem>.readable.txt",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Render only first N samples.",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Only render records with this video_id.",
    )
    parser.add_argument(
        "--question-id",
        type=str,
        default=None,
        help="Only render records with this question_id.",
    )
    args = parser.parse_args()

    input_path = args.input_jsonl
    output_path = args.output or input_path.with_suffix(".readable.txt")

    records = load_records(input_path)

    if args.video_id:
        records = [r for r in records if str(r.get("video_id")) == args.video_id]
    if args.question_id:
        records = [
            r
            for r in records
            if str((r.get("question_data") or {}).get("question_id")) == args.question_id
        ]
    if args.max_samples is not None:
        records = records[: args.max_samples]

    content_parts = []
    content_parts.append("# Readable React Output")
    content_parts.append(f"# Source: {input_path}")
    content_parts.append(f"# Samples: {len(records)}")
    content_parts.append("")

    for idx, record in enumerate(records):
        content_parts.append(format_record(record, idx))

    output_path.write_text("\n".join(content_parts).rstrip() + "\n", encoding="utf-8")
    print(f"[OK] Wrote readable output to: {output_path}")


if __name__ == "__main__":
    main()
