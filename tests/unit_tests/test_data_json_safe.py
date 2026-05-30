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
