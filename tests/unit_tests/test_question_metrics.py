from __future__ import annotations

from dspy_avqa.question_metrics import build_question_metrics


class Wrapper:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return self.payload


def test_question_metrics_records_usage_invalid_budget_and_terminal_status():
    question_data = {
        "answer": "B",
        "turn_trace": [
            {
                "planner_action": "tool",
                "tool_name": "ask_caption",
                "raw_tool_name": "not_a_tool",
                "planner_requested_tool_name": "not_a_tool",
                "tool_executed": False,
                "invalid_call": True,
                "budget_exempt": True,
                "planner_calls": {
                    "action_decision": [
                        {
                            "usage": {
                                "prompt_tokens": 100,
                                "completion_tokens": 20,
                                "prompt_tokens_details": Wrapper({"cached_tokens": 40}),
                                "completion_tokens_details": Wrapper({"reasoning_tokens": 10}),
                            }
                        }
                    ]
                },
            },
            {
                "planner_action": "tool",
                "tool_name": "ask_perception",
                "raw_tool_name": "ask_perception",
                "planner_requested_tool_name": "ask_perception",
                "tool_executed": True,
                "invalid_call": False,
                "budget_exempt": False,
                "perception_token_usage": {
                    "promptTokenCount": 50,
                    "candidatesTokenCount": 8,
                    "thoughtsTokenCount": 2,
                    "promptTokensDetails": [
                        {"modality": "AUDIO", "tokenCount": 30},
                        {"modality": "VIDEO", "tokenCount": 20},
                    ],
                },
                "planner_calls": {"action_decision": []},
            },
            {
                "planner_action": "final",
                "final_answer": "B",
                "final_answer_status": "natural",
                "budget_reached": False,
                "planner_calls": {"action_decision": []},
            },
        ],
    }

    metrics = build_question_metrics(question_data, max_turns=6)

    assert metrics["evaluation"]["correct"] is True
    assert metrics["tokens"]["planner"]["cached_input_tokens"] == 40
    assert metrics["tokens"]["planner"]["reasoning_output_tokens"] == 10
    assert metrics["tokens"]["perception_live"]["audio_input_tokens"] == 30
    assert metrics["tool_calls"] == {
        "requested_total": 2,
        "executed_total": 1,
        "budgeted_total": 1,
        "invalid_total": 1,
        "requested_by_tool": {"ask_perception": 1, "not_a_tool": 1},
        "executed_by_tool": {"ask_perception": 1},
    }
    assert metrics["budget"] == {"limit": 6, "used": 1, "reached": False}
    assert metrics["final_answer"] == {"status": "natural", "parseable": True}
