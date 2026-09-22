import argparse
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_ask_perception_questions.py"
SPEC = importlib.util.spec_from_file_location("ask_perception_judge_under_test", MODULE_PATH)
judge_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(judge_script)


def test_regex_distinguishes_answer_choice_from_two_option_paraphrase():
    assert judge_script.regex_category_one("Which option is correct, A or B?")
    assert judge_script.regex_category_one("Which of the following occurred? A. Rain B. Wind")
    assert judge_script.regex_category_one("Was it rain or wind?") is None


def test_no_perception_call_is_preserved_and_resume_skips_completed(tmp_path, monkeypatch):
    rows = [
        {
            "video_id": "video-1",
            "metadata": {"video_id": "video-1"},
            "question_data": {
                "question_id": "q1", "question": "What happened?", "options": ["A. Rain", "B. Wind"],
                "answer": "A", "response": "A", "turn_trace": [],
            },
        },
        {
            "video_id": "video-2",
            "metadata": {"video_id": "video-2"},
            "question_data": {
                "question_id": "q2", "question": "What happened?", "options": ["A. Rain", "B. Wind"],
                "answer": "B", "response": "B", "turn_trace": [
                    {"turn_id": 2, "tool_name": "ask_perception", "tool_args": {"perceptual_question": "Which option is correct, A or B?"}}
                ],
            },
        },
    ]
    monkeypatch.setattr(judge_script, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(judge_script, "RUNS", {"sample": "source"})
    monkeypatch.setattr(judge_script, "source_rows", lambda _: rows)
    args = argparse.Namespace(output_root=tmp_path / "results", dry_run=False, limit=None, retries=0, workers=1)

    assert judge_script.run_one("sample", args, "") == 0
    assert judge_script.run_one("sample", args, "") == 0
    output = json.loads((args.output_root / "sample" / "output_test.json").read_text())
    questions = [question for video in output for question in video["questions"]]
    assert len(questions) == 2
    assert questions[0]["ask_perception_judgments"] == []
    assert questions[1]["ask_perception_judgments"][0]["category"] == 1
    assert questions[1]["answer"] == "B"
    checkpoint = args.output_root / "sample" / "judgments_checkpoint.jsonl"
    assert sum(1 for _ in checkpoint.open()) == 2
