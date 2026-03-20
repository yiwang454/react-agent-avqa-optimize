"""Graph definition for AVQA ReAct planner + perception tools."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List, Literal, cast

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from react_agent.context import Context, get_planner_llm
from react_agent.prompts import SYSTEM_PROMPT, USER_TASK_TEMPLATE, format_options
from react_agent.state import AgentState, InputState
from react_agent.tools import TOOLS


PLANNER_ACTION_SCHEMA = """
Return exactly one JSON object.

If you need more perceptual evidence, output:
{"action":"tool","tool_name":"ask_qwen_perception","arguments":{"video_path":"...","perceptual_question":"...","start_time":null,"end_time":null}}
or
{"action":"tool","tool_name":"temporal_ground_video","arguments":{"video_path":"...","target_question":"..."}}

If evidence is sufficient, output:
{"action":"final","answer":"<single option label + concise rationale>"}

Never output markdown fences. JSON only.
""".strip()

def _build_task(state: dict[str, Any]) -> str:
    """Build task text from AVQA structured input."""
    question = state["question"]
    options = state["options"]
    video_path = state["video_path"]
    video_id = state.get("video_id") or "unknown"
    video_description = state.get("video_description") or "unknown"

    return USER_TASK_TEMPLATE.format(
        video_id=video_id,
        video_path=video_path,
        video_description=video_description,
        question=question,
        formatted_options=format_options(options),
    )


def _messages_to_text(messages: list[AnyMessage]) -> str:
    """Serialize chat history for planner API call."""
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"[human]\n{message.content}")
        elif isinstance(message, ToolMessage):
            lines.append(f"[tool:{message.name or 'unknown'}]\n{message.content}")
        elif isinstance(message, AIMessage):
            lines.append(f"[assistant]\n{message.content}")
            if message.tool_calls:
                lines.append(f"[assistant_tool_calls]\n{json.dumps(message.tool_calls, ensure_ascii=False)}")
        else:
            lines.append(f"[message]\n{str(message.content)}")
    return "\n\n".join(lines)


def _strip_fences(text: str) -> str:
    """Remove optional markdown code fences."""
    value = text.strip()
    if value.startswith("```"):
        value = value.strip("`")
        if value.startswith("json"):
            value = value[4:]
    return value.strip()


def _planner_to_ai_message(raw: str, default_video_path: str) -> AIMessage:
    """Parse planner JSON output into an AI message with optional tool call."""
    try:
        payload = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return AIMessage(
            content=raw,
            additional_kwargs={
                "planner_raw": raw,
                "planner_action": "unknown",
            },
        )
    action = payload.get("action")
    trace_meta = {
        "planner_raw": raw,
        "planner_action": action if isinstance(action, str) else "unknown",
    }
    if action == "tool":
        tool_name = payload["tool_name"]
        arguments = payload.get("arguments") or {}
        if "video_path" not in arguments:
            arguments["video_path"] = default_video_path
        return AIMessage(
            content=f"Calling tool: {tool_name}",
            additional_kwargs=trace_meta,
            tool_calls=[
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "name": tool_name,
                    "args": arguments,
                    "type": "tool_call",
                }
            ],
        )

    answer = payload.get("answer") if action == "final" else raw
    return AIMessage(content=str(answer), additional_kwargs=trace_meta)


async def call_model(
    state: AgentState, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """Call planner, then emit either tool call or final answer."""
    planner = get_planner_llm(runtime.context)
    messages = cast(list[AnyMessage], state.get("messages", []))
    user_task = _build_task(state)

    user_prompt = (
        "Task:\n"
        f"{user_task}\n\n"
        "Conversation state:\n"
        f"{_messages_to_text(messages) if messages else '<empty>'}\n\n"
        "Decide next action now."
    )
    system_prompt = f"{runtime.context.system_prompt}\n\n{PLANNER_ACTION_SCHEMA}"

    raw = await asyncio.to_thread(planner.call, system_prompt, user_prompt)
    response = _planner_to_ai_message(raw, state["video_path"])

    if state.get("is_last_step") and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    content="Reached max turns. Please provide the best final option now.",
                )
            ]
        }
    return {"messages": [response]}


builder = StateGraph(AgentState, input_schema=InputState, context_schema=Context)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge("__start__", "call_model")


def route_model_output(state: AgentState) -> Literal["__end__", "tools"]:
    """Route to tools only when planner emitted a tool call."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    if not last_message.tool_calls:
        return "__end__"
    return "tools"


builder.add_conditional_edges("call_model", route_model_output)
builder.add_edge("tools", "call_model")
graph = builder.compile(name="avqa_react_agent")