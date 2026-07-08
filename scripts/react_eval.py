import argparse
import json
import re
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_INPUT_FILE = (
    "/mnt/ceph_rbd/data/avqa_project/daily_omni/"
    "daily_omni_per_question_promptV2/output_test.jsonl"
)


def parse_answer_tag(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"<answer>\s*([^<]+?)\s*</answer>", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().upper()
    stripped = text.strip().upper()
    if re.fullmatch(r"[A-F]", stripped):
        return stripped
    match = re.match(r"\s*([A-F])(?:[\).\]:\s]|$)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def get_question_data(sample: Dict[str, Any]) -> Dict[str, Any]:
    question_data = sample.get("question_data")
    if isinstance(question_data, dict):
        return question_data
    return {}


def get_turn_trace(sample: Dict[str, Any], question_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    trace = question_data.get("turn_trace")
    if isinstance(trace, list):
        return [item for item in trace if isinstance(item, dict)]
    trace = sample.get("turn_trace")
    if isinstance(trace, list):
        return [item for item in trace if isinstance(item, dict)]
    return []


def extract_pred_answer(turn_trace: List[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    final_answer_text: Optional[str] = None
    for turn in turn_trace:
        if str(turn.get("planner_action", "")).strip().lower() != "final":
            continue
        candidate = turn.get("final_answer")
        if candidate is None:
            continue
        candidate_text = str(candidate).strip()
        if candidate_text:
            final_answer_text = candidate_text
    if final_answer_text is None:
        return "", None
    return parse_answer_tag(final_answer_text), final_answer_text


def evaluate_jsonl(path: str) -> Dict[str, Any]:
    total_samples = 0
    valid_samples = 0
    correct = 0
    parse_fail = 0
    missing_final = 0
    missing_final_question_ids: List[str] = []
    wrong_examples: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            total_samples += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(sample, dict):
                continue

            question_data = get_question_data(sample)
            answer = str(question_data.get("answer", "")).strip().upper()
            if not answer:
                continue

            valid_samples += 1
            turn_trace = get_turn_trace(sample, question_data)
            pred_answer, final_answer_text = extract_pred_answer(turn_trace)

            if final_answer_text is None:
                missing_final += 1
                missing_final_question_ids.append(
                    str(question_data.get("question_id", "")).strip()
                )
                continue
            if not pred_answer:
                parse_fail += 1
                continue

            if pred_answer == answer:
                correct += 1
            else:
                wrong_examples.append(
                    {
                        "line": line_idx,
                        "question_id": question_data.get("question_id", ""),
                        "gold": answer,
                        "pred": pred_answer,
                        "final_answer": final_answer_text,
                    }
                )

    accuracy = (correct / valid_samples * 100.0) if valid_samples else 0.0
    parsed_pred = valid_samples - parse_fail - missing_final

    return {
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "parsed_pred": parsed_pred,
        "correct": correct,
        "accuracy": accuracy,
        "missing_final": missing_final,
        "missing_final_question_ids": missing_final_question_ids,
        "parse_fail": parse_fail,
        "wrong_examples": wrong_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate react-agent output accuracy from JSONL."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help="Path to output_test.jsonl",
    )
    parser.add_argument(
        "--show_wrong",
        type=int,
        default=5,
        help="Number of wrong examples to print.",
    )
    args = parser.parse_args()

    metrics = evaluate_jsonl(args.input_file)

    print("=====================================")
    print("React Eval")
    print("=====================================")
    print(f"Input file: {args.input_file}")
    print(f"Total lines (non-empty): {metrics['total_samples']}")
    print(f"Valid samples: {metrics['valid_samples']}")
    print(f"Parsed predictions: {metrics['parsed_pred']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print("-------------------------------------")
    print(f"Missing final answer: {metrics['missing_final']}")
    print(f"Failed to parse <answer>: {metrics['parse_fail']}")
    if metrics["missing_final_question_ids"]:
        print("Missing final answer question_ids:")
        for qid in metrics["missing_final_question_ids"]:
            print(f"- {qid}")

    show_n = max(args.show_wrong, 0)
    if show_n > 0 and metrics["wrong_examples"]:
        print("-------------------------------------")
        print(f"Wrong examples (top {min(show_n, len(metrics['wrong_examples']))})")
        for item in metrics["wrong_examples"][:show_n]:
            print(
                f"- line={item['line']} question_id={item['question_id']} "
                f"gold={item['gold']} pred={item['pred']}"
            )


if __name__ == "__main__":
    main()
