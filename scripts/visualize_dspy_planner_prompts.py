#!/usr/bin/env python3
"""Analyze actual DeepSeek planner prompts from DSPy AVQA output."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import textwrap
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "/mnt/ceph_rbd/data/avqa_project/daily_omni/"
    "daily_omni_dspy_qwen_v6MULTITURN_gepa_qwen_seed_1234_deepseek_seed_7_gepa_seed0/"
    "output_test.jsonl"
)
DEFAULT_SAMPLE_IDS = ("Ec_lQgZ9wlg-1", "XUWxQYmiBQY-1")
FIELD_RE = re.compile(
    r"\[\[ ## (?P<name>[^#]+?) ## \]\]\n(?P<value>.*?)(?=\n\[\[ ## [^#]+? ## \]\]|\Z)",
    re.DOTALL,
)
OBJECTIVE_MARKER = "In adhering to this structure, your objective is:"
VIDEO_MARKER = "\nVideo ID:"
ACTION_SCHEMA_START = "Return exactly one JSON object."


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(item)
    return rows


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def question_data(record: dict[str, Any]) -> dict[str, Any]:
    qd = record.get("question_data")
    return qd if isinstance(qd, dict) else record


def record_id(record: dict[str, Any]) -> str:
    qd = question_data(record)
    metadata = record.get("metadata") or {}
    return str(first_non_empty(record.get("video_id"), metadata.get("video_id"), qd.get("question_id"), qd.get("video_id"), ""))


def normalize_text(value: Any) -> str:
    return textwrap.dedent(str(value or "")).strip()


def indent_block(text: Any, prefix: str = "    ") -> str:
    value = str(text if text is not None else "").rstrip()
    if not value:
        return f"{prefix}<empty>"
    return "\n".join(f"{prefix}{line}" if line else prefix for line in value.splitlines())


def short_bool(value: bool) -> str:
    return "yes" if value else "no"


def extract_signature_instruction(program_json: dict[str, Any], predictor: str = "action_planner") -> str:
    section = program_json.get(predictor) or {}
    signature = section.get("signature") or {}
    return normalize_text(signature.get("instructions"))


def extract_signature_from_search(search_json: dict[str, Any], key: str) -> str:
    rows = search_json.get(key) or []
    if not rows:
        return ""
    signature = (rows[0].get("signature") or {}) if isinstance(rows[0], dict) else {}
    return normalize_text(signature.get("instructions"))


def get_message(call: dict[str, Any], role: str) -> str:
    for message in call.get("messages") or []:
        if message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def parse_dspy_fields(user_message: str) -> dict[str, str]:
    return {
        match.group("name").strip(): match.group("value").strip()
        for match in FIELD_RE.finditer(user_message.strip())
    }


def parse_system_message(system_message: str) -> dict[str, str]:
    if OBJECTIVE_MARKER not in system_message:
        return {
            "dspy_schema": system_message.strip(),
            "signature_instruction": "",
        }
    before, after = system_message.split(OBJECTIVE_MARKER, 1)
    return {
        "dspy_schema": before.strip(),
        "signature_instruction": normalize_text(after),
    }


def parse_task(task_text: str) -> dict[str, str]:
    task = task_text.strip()
    before_video = task
    video_and_question = ""
    marker_index = task.find(VIDEO_MARKER)
    if marker_index >= 0:
        before_video = task[:marker_index].strip()
        video_and_question = task[marker_index + 1 :].strip()

    planner_system_prompt = before_video
    planner_action_schema = ""
    action_index = before_video.find(ACTION_SCHEMA_START)
    if action_index >= 0:
        planner_system_prompt = before_video[:action_index].strip()
        planner_action_schema = before_video[action_index:].strip()

    return {
        "planner_system_prompt_in_task": planner_system_prompt,
        "planner_action_schema_in_task": planner_action_schema,
        "video_question_options_in_task": video_and_question,
    }


def iter_planner_calls(turn: dict[str, Any]) -> list[tuple[str, int, dict[str, Any]]]:
    planner_calls = turn.get("planner_calls") or {}
    rows: list[tuple[str, int, dict[str, Any]]] = []
    if isinstance(planner_calls, dict):
        for group_name, calls in planner_calls.items():
            if not isinstance(calls, list):
                continue
            for idx, call in enumerate(calls, start=1):
                if isinstance(call, dict):
                    rows.append((str(group_name), idx, call))
    elif isinstance(planner_calls, list):
        for idx, call in enumerate(planner_calls, start=1):
            if isinstance(call, dict):
                rows.append(("planner_calls", idx, call))
    return rows


def contains_block(haystack: str, needle: str) -> bool:
    normalized_haystack = normalize_text(haystack)
    normalized_needle = normalize_text(needle)
    return bool(normalized_needle) and normalized_needle in normalized_haystack


def format_unified_diff(before: str, after: str, *, fromfile: str, tofile: str) -> str:
    before_lines = normalize_text(before).splitlines()
    after_lines = normalize_text(after).splitlines()
    diff = list(difflib.unified_diff(before_lines, after_lines, fromfile=fromfile, tofile=tofile, lineterm=""))
    return "\n".join(diff) if diff else "<no text diff>"


def format_program_summary(
    *,
    initial_instruction: str,
    compiled_instruction: str,
    search_initial_instruction: str,
    search_best_instruction: str,
) -> str:
    parts: list[str] = []
    parts.append("# Program Signature Summary")
    parts.append("")
    parts.append("Initial `action_planner.signature.instructions`:")
    parts.append(indent_block(initial_instruction))
    parts.append("")
    parts.append("Compiled/optimized `action_planner.signature.instructions`:")
    parts.append(indent_block(compiled_instruction))
    parts.append("")
    if search_initial_instruction or search_best_instruction:
        parts.append("Signature search JSON cross-check:")
        parts.append(f"  initial matches initial_program: {short_bool(search_initial_instruction == initial_instruction)}")
        parts.append(f"  best matches compiled_program: {short_bool(search_best_instruction == compiled_instruction)}")
        parts.append("")
    parts.append("Initial -> compiled instruction diff:")
    parts.append(indent_block(format_unified_diff(initial_instruction, compiled_instruction, fromfile="initial", tofile="compiled")))
    parts.append("")
    return "\n".join(parts)


def format_call_analysis(
    *,
    call: dict[str, Any],
    group_name: str,
    call_index: int,
    initial_instruction: str,
    compiled_instruction: str,
    show_full_messages: bool,
) -> str:
    system_message = get_message(call, "system")
    user_message = get_message(call, "user")
    system_parts = parse_system_message(system_message)
    fields = parse_dspy_fields(user_message)
    task_parts = parse_task(fields.get("task", ""))
    actual_signature = system_parts["signature_instruction"]
    task_system = task_parts["planner_system_prompt_in_task"]
    task_action_schema = task_parts["planner_action_schema_in_task"]

    parts: list[str] = []
    parts.append(f"    Planner call: {group_name}[{call_index}]")
    parts.append(f"    model: {call.get('model', '<unknown>')}")
    if call.get("params"):
        parts.append("    params:")
        parts.append(indent_block(json.dumps(call.get("params"), ensure_ascii=False, indent=2), prefix="      "))
    usage = call.get("usage")
    if usage:
        parts.append("    usage:")
        parts.append(indent_block(json.dumps(usage, ensure_ascii=False, indent=2), prefix="      "))
    parts.append("")
    parts.append("    Layer Map")
    parts.append("      DeepSeek system message:")
    parts.append("        - DSPy schema / field descriptions")
    parts.append("        - `signatures.PlanNextAction.instructions` under the DSPy objective marker")
    parts.append("      DeepSeek user message:")
    parts.append("        - `[[ ## task ## ]]`, which contains `planner.system_prompt`, `planner.action_schema`, and sample context")
    parts.append("        - `[[ ## conversation_state ## ]]`, `turn_index`, `max_turns`")
    parts.append("")
    parts.append("    Optimization / Duplication Checks")
    parts.append(f"      actual signature == initial program signature: {short_bool(actual_signature == initial_instruction)}")
    parts.append(f"      actual signature == compiled program signature: {short_bool(actual_signature == compiled_instruction)}")
    parts.append(f"      task planner.system_prompt also appears inside actual signature: {short_bool(contains_block(actual_signature, task_system))}")
    parts.append(f"      task planner.action_schema also appears inside actual signature: {short_bool(contains_block(actual_signature, task_action_schema))}")
    parts.append("")
    parts.append("    BEFORE optimization signature layer:")
    parts.append(indent_block(initial_instruction, prefix="      "))
    parts.append("")
    parts.append("    AFTER optimization signature layer observed in DeepSeek system message:")
    parts.append(indent_block(actual_signature, prefix="      "))
    parts.append("")
    parts.append("    DeepSeek user `task` breakdown")
    parts.append("      planner.system_prompt parsed from task:")
    parts.append(indent_block(task_system, prefix="        "))
    parts.append("")
    parts.append("      planner.action_schema parsed from task:")
    parts.append(indent_block(task_action_schema, prefix="        "))
    parts.append("")
    parts.append("      video/question/options parsed from task:")
    parts.append(indent_block(task_parts["video_question_options_in_task"], prefix="        "))
    parts.append("")
    parts.append("      conversation_state:")
    parts.append(indent_block(fields.get("conversation_state", ""), prefix="        "))
    parts.append("")
    parts.append(f"      turn_index: {fields.get('turn_index', '<missing>')}")
    parts.append(f"      max_turns : {fields.get('max_turns', '<missing>')}")
    parts.append("")
    parts.append("    Planner response_text:")
    parts.append(indent_block(call.get("response_text", ""), prefix="      "))

    if show_full_messages:
        parts.append("")
        parts.append("    Full DeepSeek system message:")
        parts.append(indent_block(system_message, prefix="      "))
        parts.append("")
        parts.append("    Full DeepSeek user message:")
        parts.append(indent_block(user_message, prefix="      "))
    return "\n".join(parts)


def format_record(
    record: dict[str, Any],
    *,
    initial_instruction: str,
    compiled_instruction: str,
    show_full_messages: bool,
) -> str:
    qd = question_data(record)
    trace = qd.get("turn_trace") or []
    parts: list[str] = []
    parts.append("=" * 120)
    parts.append(f"SAMPLE {record_id(record)}")
    parts.append("=" * 120)
    parts.append(f"question: {qd.get('question', '')}")
    parts.append(f"gold answer: {qd.get('answer', qd.get('gold_answer', '<unknown>'))}")
    parts.append(f"model response: {qd.get('response', qd.get('reasoning_summary', ''))}")
    parts.append(f"turns: {len(trace)}")
    parts.append("")
    for turn_idx, turn in enumerate(trace, start=1):
        parts.append("-" * 120)
        parts.append(f"TURN {turn.get('turn_id', turn_idx)}")
        parts.append("-" * 120)
        parts.append(f"planner_action: {turn.get('planner_action')}")
        if turn.get("tool_name"):
            parts.append(f"tool_name: {turn.get('tool_name')}")
        if turn.get("tool_args"):
            parts.append("tool_args:")
            parts.append(indent_block(json.dumps(turn.get("tool_args"), ensure_ascii=False, indent=2), prefix="  "))
        if turn.get("tool_observation") is not None:
            parts.append("tool_observation:")
            parts.append(indent_block(turn.get("tool_observation"), prefix="  "))
        calls = iter_planner_calls(turn)
        if not calls:
            parts.append("  <no planner calls recorded>")
        for group_name, call_index, call in calls:
            parts.append("")
            parts.append(format_call_analysis(
                call=call,
                group_name=group_name,
                call_index=call_index,
                initial_instruction=initial_instruction,
                compiled_instruction=compiled_instruction,
                show_full_messages=show_full_messages,
            ))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def load_program_context(program_dir: Path, args: argparse.Namespace) -> dict[str, str]:
    initial_path = args.initial_program or (program_dir / "initial_gepa_program.json")
    compiled_path = args.compiled_program or (program_dir / "compiled_gepa.json")
    search_path = args.signature_search_json or (program_dir / "compiled_gepa_signature_search.json")

    initial_instruction = ""
    compiled_instruction = ""
    search_initial_instruction = ""
    search_best_instruction = ""
    if initial_path.exists():
        initial_instruction = extract_signature_instruction(read_json(initial_path))
    if compiled_path.exists():
        compiled_instruction = extract_signature_instruction(read_json(compiled_path))
    if search_path.exists():
        search_json = read_json(search_path)
        search_initial_instruction = extract_signature_from_search(search_json, "initial_program_signatures")
        search_best_instruction = extract_signature_from_search(search_json, "best_program_signatures")
        initial_instruction = initial_instruction or search_initial_instruction
        compiled_instruction = compiled_instruction or search_best_instruction

    return {
        "initial_instruction": initial_instruction,
        "compiled_instruction": compiled_instruction,
        "search_initial_instruction": search_initial_instruction,
        "search_best_instruction": search_best_instruction,
        "initial_path": str(initial_path),
        "compiled_path": str(compiled_path),
        "search_path": str(search_path),
    }


def parse_sample_ids(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize and compare actual DeepSeek planner prompts in DSPy AVQA output."
    )
    parser.add_argument("input_jsonl", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--sample-ids",
        default=",".join(DEFAULT_SAMPLE_IDS),
        help="Comma-separated sample/video ids to render.",
    )
    parser.add_argument("--program-dir", type=Path, default=None, help="Directory containing GEPA program JSON files.")
    parser.add_argument("--initial-program", type=Path, default=None)
    parser.add_argument("--compiled-program", type=Path, default=None)
    parser.add_argument("--signature-search-json", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "--no-full-messages",
        action="store_true",
        help="Omit full raw DeepSeek system/user messages after the parsed breakdown.",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input_jsonl)
    wanted_ids = parse_sample_ids(args.sample_ids)
    selected = [record for record in records if record_id(record) in wanted_ids]
    missing = sorted(wanted_ids - {record_id(record) for record in selected})

    program_dir = args.program_dir or args.input_jsonl.parent
    program_context = load_program_context(program_dir, args)
    initial_instruction = program_context["initial_instruction"]
    compiled_instruction = program_context["compiled_instruction"]

    output_path = args.output or args.input_jsonl.with_name(f"{args.input_jsonl.stem}.planner_prompt_analysis.txt")
    parts: list[str] = []
    parts.append("# DSPy AVQA Planner Prompt Analysis")
    parts.append(f"# Source output: {args.input_jsonl}")
    parts.append(f"# Program dir: {program_dir}")
    parts.append(f"# Initial program: {program_context['initial_path']}")
    parts.append(f"# Compiled program: {program_context['compiled_path']}")
    parts.append(f"# Signature search: {program_context['search_path']}")
    parts.append(f"# Requested sample ids: {', '.join(sorted(wanted_ids))}")
    if missing:
        parts.append(f"# Missing sample ids: {', '.join(missing)}")
    parts.append("")
    parts.append(format_program_summary(
        initial_instruction=initial_instruction,
        compiled_instruction=compiled_instruction,
        search_initial_instruction=program_context["search_initial_instruction"],
        search_best_instruction=program_context["search_best_instruction"],
    ))
    for record in selected:
        parts.append(format_record(
            record,
            initial_instruction=initial_instruction,
            compiled_instruction=compiled_instruction,
            show_full_messages=not args.no_full_messages,
        ))

    output_path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"[OK] Wrote planner prompt analysis to: {output_path}")
    if missing:
        print(f"[WARN] Missing requested sample id(s): {', '.join(missing)}")


if __name__ == "__main__":
    main()
