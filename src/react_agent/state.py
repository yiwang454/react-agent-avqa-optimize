"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep


from typing_extensions import TypedDict

from typing import Annotated, Any, Literal



class InputState(TypedDict):
    """User-facing input state."""

    question: str
    options: list[str]
    video_path: str
    audio_path: str | None
    video_id: str | None
    video_description: str | None


class AgentState(TypedDict, total=False):
    """Internal graph state.

    messages: chat history for the ReAct loop.
    evidence_log: structured evidence gathered from Qwen.
    eliminated_options: options ruled out by the planner.
    final_answer: final predicted option label.
    turn_count: number of planner-tool iterations so far.
    max_turns: stopping budget.
    """

    messages: Annotated[list[Any], add_messages]
    question: str
    options: list[str]
    video_path: str
    audio_path: str | None
    video_id: str | None
    video_description: str | None

    evidence_log: list[dict[str, Any]]
    eliminated_options: list[str]
    final_answer: str | None
    turn_count: int
    max_turns: int
    planner_status: Literal["continue", "answer"]

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    # Additional attributes can be added here as needed.
    # Common examples include:
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)
