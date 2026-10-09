"""Per-question AVQA rollout metrics persisted with every result row."""

from __future__ import annotations

from collections import Counter
from typing import Any


O3_UNCACHED_INPUT_PER_MTOK = 2.0
O3_CACHED_INPUT_PER_MTOK = 0.5
O3_OUTPUT_PER_MTOK = 8.0
GEMINI_NON_AUDIO_INPUT_PER_MTOK = 0.3
GEMINI_AUDIO_INPUT_PER_MTOK = 1.0
GEMINI_OUTPUT_PER_MTOK = 2.5


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                normalized = method()
            except Exception:
                continue
            if isinstance(normalized, dict):
                return normalized
    return {}


def _planner_usage(turn_trace: list[dict[str, Any]]) -> dict[str, int]:
    totals = Counter()
    for turn in turn_trace:
        planner_groups = turn.get("planner_calls") or {}
        if not isinstance(planner_groups, dict):
            continue
        for calls in planner_groups.values():
            for call in calls if isinstance(calls, list) else []:
                usage = _mapping(call.get("usage"))
                prompt_details = _mapping(usage.get("prompt_tokens_details"))
                completion_details = _mapping(usage.get("completion_tokens_details"))
                totals["calls"] += 1
                totals["input_tokens"] += _int(
                    usage.get("prompt_tokens", usage.get("input_tokens"))
                )
                totals["cached_input_tokens"] += _int(
                    prompt_details.get("cached_tokens", usage.get("cached_input_tokens"))
                )
                totals["output_tokens"] += _int(
                    usage.get("completion_tokens", usage.get("output_tokens"))
                )
                totals["reasoning_output_tokens"] += _int(
                    completion_details.get(
                        "reasoning_tokens", usage.get("reasoning_output_tokens")
                    )
                )
    return dict(totals)


def _gemini_audio_tokens(usage: dict[str, Any]) -> int:
    details = usage.get("promptTokensDetails") or usage.get("prompt_tokens_details") or []
    if isinstance(details, dict):
        return _int(details.get("audio_tokens"))
    if not isinstance(details, list):
        return 0
    return sum(
        _int(_mapping(item).get("tokenCount", _mapping(item).get("token_count")))
        for item in details
        if str(_mapping(item).get("modality", "")).upper() == "AUDIO"
    )


def _perception_usage(turn_trace: list[dict[str, Any]]) -> dict[str, int]:
    totals = Counter()
    for turn in turn_trace:
        usage = _mapping(turn.get("perception_token_usage"))
        if not usage:
            continue
        totals["calls"] += 1
        input_tokens = _int(
            usage.get("promptTokenCount", usage.get("prompt_tokens"))
        )
        output_tokens = _int(
            usage.get("candidatesTokenCount", usage.get("completion_tokens"))
        )
        thinking_tokens = _int(
            usage.get("thoughtsTokenCount", usage.get("reasoning_tokens"))
        )
        totals["input_tokens"] += input_tokens
        totals["audio_input_tokens"] += _gemini_audio_tokens(usage)
        totals["visible_output_tokens"] += output_tokens
        totals["thinking_output_tokens"] += thinking_tokens
        totals["output_tokens"] += output_tokens + thinking_tokens
    return dict(totals)


def _estimated_cost(planner: dict[str, int], perception: dict[str, int]) -> dict[str, Any]:
    planner_input = planner.get("input_tokens", 0)
    planner_cached = min(planner_input, planner.get("cached_input_tokens", 0))
    planner_uncached = planner_input - planner_cached
    planner_cost = (
        planner_uncached * O3_UNCACHED_INPUT_PER_MTOK
        + planner_cached * O3_CACHED_INPUT_PER_MTOK
        + planner.get("output_tokens", 0) * O3_OUTPUT_PER_MTOK
    ) / 1_000_000

    perception_input = perception.get("input_tokens", 0)
    audio_input = min(perception_input, perception.get("audio_input_tokens", 0))
    perception_cost = (
        (perception_input - audio_input) * GEMINI_NON_AUDIO_INPUT_PER_MTOK
        + audio_input * GEMINI_AUDIO_INPUT_PER_MTOK
        + perception.get("output_tokens", 0) * GEMINI_OUTPUT_PER_MTOK
    ) / 1_000_000
    return {
        "currency": "USD",
        "planner_live": round(planner_cost, 8),
        "perception_live": round(perception_cost, 8),
        "live_marginal_total": round(planner_cost + perception_cost, 8),
        "caption_cache_generation_excluded": True,
        "rates_per_million_tokens": {
            "o3_uncached_input": O3_UNCACHED_INPUT_PER_MTOK,
            "o3_cached_input": O3_CACHED_INPUT_PER_MTOK,
            "o3_output_including_reasoning": O3_OUTPUT_PER_MTOK,
            "gemini_non_audio_input": GEMINI_NON_AUDIO_INPUT_PER_MTOK,
            "gemini_audio_input": GEMINI_AUDIO_INPUT_PER_MTOK,
            "gemini_output_including_thinking": GEMINI_OUTPUT_PER_MTOK,
        },
    }


def build_question_metrics(
    question_data: dict[str, Any],
    *,
    max_turns: int,
) -> dict[str, Any]:
    """Build accuracy, usage, tool, budget, and terminal-state metrics."""
    turn_trace = [
        turn for turn in (question_data.get("turn_trace") or []) if isinstance(turn, dict)
    ]
    requested_by_tool: Counter[str] = Counter()
    executed_by_tool: Counter[str] = Counter()
    invalid_calls = 0
    budget_used = 0
    for turn in turn_trace:
        if turn.get("planner_action") != "tool":
            continue
        requested_name = str(turn.get("planner_requested_tool_name") or turn.get("raw_tool_name") or "")
        requested_by_tool[requested_name or "<missing>"] += 1
        if turn.get("tool_executed", True):
            executed_by_tool[str(turn.get("tool_name") or "<unknown>")] += 1
        if turn.get("invalid_call") or turn.get("budget_exempt"):
            invalid_calls += 1
        if not turn.get("budget_exempt"):
            budget_used += 1

    final_turn = next(
        (turn for turn in reversed(turn_trace) if turn.get("planner_action") == "final"),
        None,
    )
    predicted = str((final_turn or {}).get("final_answer") or "").strip().upper()
    gold = str(question_data.get("answer") or "").strip().upper()
    terminal_status = str((final_turn or {}).get("final_answer_status") or "error")
    planner = _planner_usage(turn_trace)
    perception = _perception_usage(turn_trace)
    return {
        "evaluation": {
            "gold_answer": gold,
            "predicted_answer": predicted,
            "parseable": predicted in set("ABCDEF"),
            "correct": bool(gold and predicted == gold),
        },
        "tokens": {
            "planner": planner,
            "perception_live": perception,
            "live_total": {
                "input_tokens": planner.get("input_tokens", 0)
                + perception.get("input_tokens", 0),
                "output_tokens": planner.get("output_tokens", 0)
                + perception.get("output_tokens", 0),
            },
        },
        "estimated_cost_usd": _estimated_cost(planner, perception),
        "tool_calls": {
            "requested_total": sum(requested_by_tool.values()),
            "executed_total": sum(executed_by_tool.values()),
            "budgeted_total": budget_used,
            "invalid_total": invalid_calls,
            "requested_by_tool": dict(sorted(requested_by_tool.items())),
            "executed_by_tool": dict(sorted(executed_by_tool.items())),
        },
        "budget": {
            "limit": max_turns,
            "used": budget_used,
            "reached": bool((final_turn or {}).get("budget_reached", budget_used >= max_turns)),
        },
        "final_answer": {
            "status": terminal_status,
            "parseable": predicted in set("ABCDEF"),
        },
    }
