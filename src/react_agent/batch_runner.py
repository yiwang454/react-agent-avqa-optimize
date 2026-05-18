"""Batch runner for AVQA ReAct agent on cut-style JSONL input."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from react_agent import graph
from react_agent.context import Context


# DEFAULT_DEBUG_ID_LIST = [
#     "dvgrOXBj-1334",
#     # "ROEyKoiF-311",
#     # "xArSBiiu-361",
#     "rzAqzwvL-2720",
#     # "SohqvGtQ-2530",
#     # "lNQdrmMn-1410",
#     # "LhCOTGxu-185",
#     # "eapYeyoW-1809",
# ]
DEFAULT_DEBUG_ID_LIST = [
    "03aJ_RcnBko-1",
    "03aJ_RcnBko-2",
    "03aJ_RcnBko-3",
    "03zeVlBWhdY-1",
    "04ChQ0fqzxQ-1",
    "04xLK7R5wNA-1",
    "05xCYOyY1bg-1",
    "068rdc75mHM-1",
    "8EexlONNnEE-1",
    "8TNPeimqOO0-1",
    "8I2nIXrbFpo-1",
    "8IZvEF4Ui10-2",
    "SZRxEbvKih0-1",
    "DK6bKtXUE1c-1",
    "GDbG4rvL1YA-1",
    "KIg4rprmO9Y-1",
    "HGv87QeBYnc-1",
    "vBrspDt3ib0-1",
    "EQ-67udZEeg-2",
    "AnWKOvjFl8k-1",
    "_t2geYpykFo-2",
    "TF9I1GxNdJQ-3",
    "QxrYJf48s-4-2",
    "4sBpUHylq9s-1",
    "1F6g2NS_vGY-1",
    "fqRSLvhdUII-1",
    "10lWpHyN0Ok-1",
    "8TNPeimqOO0-3",
]

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Batch run AVQA ReAct over cut JSONL.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Input cut JSONL path (e.g. worldsense_test_cut.jsonl).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Output JSONL path for process_cut_task-style records.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional dir to save per-cut question_data JSON files.",
    )
    parser.add_argument(
        "--audio-caption-dir",
        type=Path,
        required=True,
        help="Directory containing caption JSON files matched by question_id prefix.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with pre-defined cut IDs.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=4,
        help="Planner max turns passed to runtime context.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=20,
        help="LangGraph recursion limit.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent graph calls.",
    )
    parser.add_argument(
        "--perception-model",
        type=str,
        choices=["gemini", "qwen"],
        default=os.environ.get("PERCEPTION_MODEL", "gemini").strip().lower(),
        help="Perception backend for tools: gemini or qwen.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows into a list of dicts."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_audio_caption_concat(audio_caption_dir: Path, question_id: str) -> str:
    """Load and concatenate caption chunks for one question id."""
    audio_caption_paths = sorted(
        audio_caption_dir.glob(f"{question_id}*.json"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else p.stem,
    )
    audio_captions: list[str] = []
    for caption_path in audio_caption_paths:
        with caption_path.open("r", encoding="utf-8") as f_caption:
            audio_caption_data = json.load(f_caption)
        audio_captions.append(audio_caption_data.get("response", ""))
    return "\n".join(audio_captions).strip()


def debug_filter(cuts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter cuts using default debug IDs."""
    filtered: list[dict[str, Any]] = []
    for cut in cuts:
        cut_id = str(cut.get("id", ""))
        if any(debug_id in cut_id or cut_id in debug_id for debug_id in DEFAULT_DEBUG_ID_LIST):
            filtered.append(cut)
    return filtered


def build_input_state(cut: dict[str, Any], audio_caption_dir: Path) -> dict[str, Any]:
    """Map cut row into InputState-compatible payload."""
    supervisions = cut.get("supervisions") or []
    if not supervisions:
        raise ValueError("Cut has no supervisions")
    supervision = supervisions[0]
    custom = supervision.get("custom") or {}

    recording = cut.get("recording") or {}
    sources = recording.get("sources") or []
    source_video = None
    if sources and isinstance(sources[0], dict):
        source_video = sources[0].get("source")

    question = supervision.get("text") or custom.get("question") or ""
    options = custom.get("options") or custom.get("Choice") or []
    video_path = custom.get("video_path") or source_video
    audio_path = custom.get("audio_path")
    if not question or not options or not video_path:
        raise ValueError(
            f"Missing required AVQA fields in cut_id={cut.get('id')}: "
            f"question={bool(question)}, options={bool(options)}, video_path={bool(video_path)}"
        )
    if os.environ.get("PERCEPTION_MODEL", "gemini").strip().lower() == "qwen" and not audio_path:
        raise ValueError(
            f"Missing required AVQA audio field in cut_id={cut.get('id')}: "
            "audio_path is required when PERCEPTION_MODEL=qwen"
        )

    question_id = str(cut.get("id") or "")
    if not question_id:
        raise ValueError("Cut id is empty, cannot match audio caption.")
    audio_caption = load_audio_caption_concat(audio_caption_dir, question_id)
    if len(audio_caption) == 0:
        raise ValueError(f"Audio caption is empty for question {question_id}")

    return {
        "question": question,
        "options": options,
        "video_path": video_path,
        "audio_path": audio_path,
        "video_id": custom.get("video_id"),
        "video_description": audio_caption,
    }


def build_result_row(cut: dict[str, Any], response_text: str) -> dict[str, Any]:
    """Build process_cut_task-style output row."""
    supervisions = cut.get("supervisions") or []
    supervision = supervisions[0]
    custom = supervision.get("custom") or {}
    cut_id = cut.get("id")

    question_data = {
        "question_id": supervision.get("id"),
        "task_type": custom.get("task_type"),
        "question": supervision.get("text"),
        "options": custom.get("options"),
        "answer": custom.get("answer"),
        "response": response_text,
        "fps": None,
    }

    metadata = {
        "video_id": cut_id,
        "duration": custom.get("duration"),
        "domain": custom.get("domain"),
        "sub_category": custom.get("sub_category"),
    }

    return {
        "video_id": cut_id,
        "metadata": metadata,
        "question_data": question_data,
    }


def _stringify_message_content(content: Any) -> str:
    """Convert message content into a stable string."""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def build_turn_trace(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """Build per-turn planner/tool trace from graph messages."""
    trace: list[dict[str, Any]] = []
    turn_id = 0
    i = 0

    while i < len(messages):
        message = messages[i]
        if not isinstance(message, AIMessage):
            i += 1
            continue

        planner_raw = message.additional_kwargs.get("planner_raw")
        planner_action = message.additional_kwargs.get("planner_action")

        if message.tool_calls:
            # Collect contiguous tool observations after the planner tool-call turn.
            j = i + 1
            tool_messages: list[ToolMessage] = []
            while j < len(messages) and not isinstance(messages[j], AIMessage):
                if isinstance(messages[j], ToolMessage):
                    tool_messages.append(messages[j])
                j += 1

            consumed_tool_idx: set[int] = set()
            for tool_call in message.tool_calls:
                turn_id += 1
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args")
                tool_call_id = tool_call.get("id")

                matched_observation: str | None = None
                for idx, tool_msg in enumerate(tool_messages):
                    if idx in consumed_tool_idx:
                        continue
                    if tool_msg.tool_call_id == tool_call_id:
                        matched_observation = _stringify_message_content(tool_msg.content)
                        consumed_tool_idx.add(idx)
                        break

                if matched_observation is None:
                    for idx, tool_msg in enumerate(tool_messages):
                        if idx in consumed_tool_idx:
                            continue
                        matched_observation = _stringify_message_content(tool_msg.content)
                        consumed_tool_idx.add(idx)
                        break

                trace.append(
                    {
                        "turn_id": turn_id,
                        "planner_raw": planner_raw,
                        "planner_action": planner_action or "tool",
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_observation": matched_observation,
                        "tool_error": None,
                        "final_answer": None,
                    }
                )
            i = j
            continue

        # Non-tool planner message is treated as final (or fallback) turn.
        turn_id += 1
        trace.append(
            {
                "turn_id": turn_id,
                "planner_raw": planner_raw,
                "planner_action": planner_action or "final",
                "tool_name": None,
                "tool_args": None,
                "tool_observation": None,
                "tool_error": None,
                "final_answer": _stringify_message_content(message.content),
            }
        )
        i += 1

    return trace


async def run_one(
    cut: dict[str, Any],
    context: Context,
    recursion_limit: int,
    audio_caption_dir: Path,
) -> tuple[dict[str, Any], str, list[AnyMessage], dict[str, Any], str | None]:
    """Run one cut through graph and return text answer."""
    payload = build_input_state(cut, audio_caption_dir)
    try:
        result = await graph.ainvoke(
            payload,
            context=context,
            config={"recursion_limit": recursion_limit},
        )
        messages = result.get("messages") or []
        response_text = str(messages[-1].content) if messages else ""
        return cut, response_text, messages, payload, None
    except Exception as exc:
        return cut, "", [], payload, str(exc)


async def run_batch(
    cuts: list[dict[str, Any]],
    context: Context,
    recursion_limit: int,
    concurrency: int,
    audio_caption_dir: Path,
) -> list[tuple[dict[str, Any], str, list[AnyMessage], dict[str, Any], str | None]]:
    """Run cuts with optional concurrency."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _guarded(
        cut: dict[str, Any]
    ) -> tuple[dict[str, Any], str, list[AnyMessage], dict[str, Any], str | None]:
        async with semaphore:
            return await run_one(cut, context, recursion_limit, audio_caption_dir)

    return await asyncio.gather(*[_guarded(cut) for cut in cuts])


def maybe_dump_question_data(
    output_dir: Path | None,
    row: dict[str, Any],
) -> None:
    """Optionally save per-cut question_data JSON."""
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    video_id = row["video_id"]
    question_data = row["question_data"]
    out_path = output_dir / f"{video_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(question_data, f, indent=2, ensure_ascii=False)


def write_results_jsonl(rows: list[dict[str, Any]], output_jsonl: Path) -> None:
    """Write process_cut_task-style rows as JSONL."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_run_config(args: argparse.Namespace, selected_cuts: list[dict[str, Any]]) -> None:
    """Print key runtime and model hyperparameters for verification."""
    planner_keys = [
        "PLANNER_TEMPERATURE",
        "PLANNER_TOP_P",
        "PLANNER_TOP_K",
        "PLANNER_REPETITION_PENALTY",
        "PLANNER_TIMEOUT",
        "PLANNER_MAX_RETRIES",
    ]
    qwen_keys = [
        "QWEN_BASE_URL",
        "QWEN_MODEL",
        "QWEN_TIMEOUT",
    ]
    gemini_keys = [
        "GEMINI_BASE_URL",
        "GEMINI_MODEL",
        "GEMINI_TIMEOUT",
    ]
    print("[run-config] ----------")
    print(f"[run-config] input_jsonl={args.input_jsonl}")
    print(f"[run-config] output_jsonl={args.output_jsonl}")
    print(f"[run-config] audio_caption_dir={args.audio_caption_dir}")
    print(f"[run-config] debug={args.debug}")
    print(f"[run-config] selected_cuts={len(selected_cuts)}")
    if selected_cuts:
        print(f"[run-config] first_cut_id={selected_cuts[0].get('id')}")
    print(f"[run-config] perception_model={args.perception_model}")
    print(f"[run-config] max_turns={args.max_turns}")
    print(f"[run-config] recursion_limit={args.recursion_limit}")
    print(f"[run-config] concurrency={args.concurrency}")
    print(f"[run-config] planner_model={os.environ.get('DEEPSEEK_MODEL') or os.environ.get('PLANNER_MODEL')}")
    for key in planner_keys + qwen_keys + gemini_keys:
        print(f"[run-config] {key}={os.environ.get(key)}")
    print("[run-config] ----------")


async def amain() -> None:
    """Async entrypoint."""
    args = parse_args()
    os.environ["PERCEPTION_MODEL"] = args.perception_model
    cuts = read_jsonl(args.input_jsonl)
    selected_cuts = cuts
    if args.debug:
        selected_cuts = debug_filter(selected_cuts)

    print(f"Loaded cuts: {len(cuts)}")
    print(f"Selected cuts: {len(selected_cuts)}")
    print(f"Perception model: {args.perception_model}")
    log_run_config(args, selected_cuts)

    context = Context(max_turns=args.max_turns)
    raw_results = await run_batch(
        selected_cuts,
        context=context,
        recursion_limit=args.recursion_limit,
        concurrency=max(1, args.concurrency),
        audio_caption_dir=args.audio_caption_dir,
    )

    rows: list[dict[str, Any]] = []
    for cut, response_text, messages, payload, error in raw_results:
        row = build_result_row(cut, response_text)
        if error is None:
            row["question_data"]["turn_trace"] = build_turn_trace(messages)
        else:
            row["question_data"]["turn_trace"] = [
                {
                    "turn_id": 1,
                    "planner_raw": None,
                    "planner_action": "error",
                    "tool_name": None,
                    "tool_args": {
                        "video_path": payload.get("video_path"),
                        "audio_path": payload.get("audio_path"),
                    },
                    "tool_observation": None,
                    "tool_error": error,
                    "final_answer": None,
                }
            ]
            row["question_data"]["response"] = f"[ERROR] {error}"
            print(
                "Cut failed:",
                {
                    "cut_id": cut.get("id"),
                    "video_path": payload.get("video_path"),
                    "audio_path": payload.get("audio_path"),
                },
            )
        maybe_dump_question_data(args.output_dir, row)
        rows.append(row)

    write_results_jsonl(rows, args.output_jsonl)
    print(f"Wrote {len(rows)} rows to {args.output_jsonl}")


def main() -> None:
    """Sync wrapper."""
    start = time.perf_counter()
    try:
        asyncio.run(amain())
    finally:
        elapsed = time.perf_counter() - start
        print(f"Total elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
