import importlib.util
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "DSPy" / "dspy_avqa" / "data.py"
spec = importlib.util.spec_from_file_location("dspy_avqa_data", DATA_PATH)
data_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(data_module)

maybe_dump_question_data = data_module.maybe_dump_question_data
write_results_jsonl = data_module.write_results_jsonl
build_input_state = data_module.build_input_state


class TokenDetailsWrapper:
    def __init__(self):
        self.prompt_tokens = 3

    def model_dump(self):
        return {"prompt_tokens": self.prompt_tokens}


def _row():
    return {
        "video_id": "sample-1",
        "metadata": {},
        "question_data": {
            "response": "A",
            "turn_trace": [
                {
                    "perception_token_usage": {
                        "prompt_tokens_details": TokenDetailsWrapper(),
                    }
                }
            ],
        },
    }


def test_maybe_dump_question_data_serializes_sdk_wrappers(tmp_path):
    maybe_dump_question_data(tmp_path, _row())

    data = json.loads((tmp_path / "sample-1.json").read_text())

    assert data["turn_trace"][0]["perception_token_usage"]["prompt_tokens_details"] == {
        "prompt_tokens": 3
    }


def test_write_results_jsonl_serializes_sdk_wrappers(tmp_path):
    output_jsonl = tmp_path / "output.jsonl"

    write_results_jsonl([_row()], output_jsonl)

    data = json.loads(output_jsonl.read_text().strip())
    assert data["question_data"]["turn_trace"][0]["perception_token_usage"]["prompt_tokens_details"] == {
        "prompt_tokens": 3
    }


def test_build_input_state_allows_video_only_cut_without_audio_path():
    cut = {
        "id": "worldsense-0",
        "supervisions": [
            {
                "id": "0",
                "text": "What happens?",
                "custom": {
                    "video_id": "worldsense",
                    "options": ["A. One", "B. Two"],
                    "video_path": "/videos/worldsense.mp4",
                },
            }
        ],
        "recording": {"sources": [{"source": "/videos/worldsense.mp4"}]},
    }

    state = build_input_state(cut, audio_caption_dir=None)

    assert state["video_path"] == "/videos/worldsense.mp4"
    assert state["audio_path"] is None
