from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "DSPy" / "dspy_avqa" / "yamls" / "modality_merge_ablation"

EXPECTED_TOOLS = {
    "b0_omniagent_cqm.yaml": [
        "audio_global_caption", "audio_qa", "video_global_qa", "video_clip_qa", "video_metadata"
    ],
    "b1_no_crosscheck_cqm.yaml": [
        "audio_global_caption", "audio_qa", "video_global_qa", "video_clip_qa", "video_metadata"
    ],
    "b2_new_workflow_cqm.yaml": [
        "audio_global_caption", "audio_qa", "video_global_qa", "video_clip_qa", "video_metadata"
    ],
    "b3_omni_caption_specialized_qm.yaml": [
        "ask_caption", "audio_qa", "video_global_qa", "video_clip_qa", "video_metadata"
    ],
    "b4_omni_caption_specialized_qm_caption_first.yaml": [
        "ask_caption", "audio_qa", "video_global_qa", "video_clip_qa", "video_metadata"
    ],
    "b5_flex_omni_caption_global_clip_qm.yaml": [
        "ask_caption", "ask_perception", "omni_clip_perception", "video_metadata"
    ],
    "b5_omni_caption_global_clip_qm_caption_first.yaml": [
        "ask_caption", "ask_perception", "omni_clip_perception", "video_metadata"
    ],
    "b6_omni_caption_global_clip_q_no_metadata_caption_first.yaml": [
        "ask_caption", "ask_perception", "omni_clip_perception"
    ],
    "b7_omni_caption_global_qm_no_clip_caption_first.yaml": [
        "ask_caption", "ask_perception", "video_metadata"
    ],
    "b8_omni_caption_global_q_no_clip_no_metadata_caption_first.yaml": [
        "ask_caption", "ask_perception"
    ],
}

CAPTION_FIRST_CONFIGS = {
    "b4_omni_caption_specialized_qm_caption_first.yaml",
    "b5_omni_caption_global_clip_qm_caption_first.yaml",
    "b6_omni_caption_global_clip_q_no_metadata_caption_first.yaml",
    "b7_omni_caption_global_qm_no_clip_caption_first.yaml",
    "b8_omni_caption_global_q_no_clip_no_metadata_caption_first.yaml",
}


def _raw(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def _merged(name: str) -> dict:
    payload = _raw(name)
    parent = payload.pop("extends", None)
    if not parent:
        return payload
    base = _merged(parent)

    def merge(left: dict, right: dict) -> dict:
        result = dict(left)
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = value
        return result

    return merge(base, payload)


def test_all_ablation_configs_expose_exact_tools_and_fixed_controls():
    for name, expected in EXPECTED_TOOLS.items():
        config = _merged(name)
        assert config["tools"]["allowed"] == expected
        assert config["tools"]["strict_validation"] is True
        assert config["planner"]["compress_history"] is False
        assert config["planner"]["optimization_guidance"] == ""
        expected_first = "ask_caption" if name in CAPTION_FIRST_CONFIGS else None
        assert config["tools"].get("required_first") == expected_first


def test_revised_path_changes_only_the_intended_tool_factors():
    b3 = _merged("b3_omni_caption_specialized_qm.yaml")["tools"]
    b4 = _merged("b4_omni_caption_specialized_qm_caption_first.yaml")["tools"]
    b5 = _merged("b5_omni_caption_global_clip_qm_caption_first.yaml")["tools"]
    b5_flex = _merged("b5_flex_omni_caption_global_clip_qm.yaml")["tools"]
    b6 = _merged("b6_omni_caption_global_clip_q_no_metadata_caption_first.yaml")["tools"]
    b7 = _merged("b7_omni_caption_global_qm_no_clip_caption_first.yaml")["tools"]
    b8 = _merged("b8_omni_caption_global_q_no_clip_no_metadata_caption_first.yaml")["tools"]

    assert b4["allowed"] == b3["allowed"]
    assert b5["allowed"] == b5_flex["allowed"]
    assert b5["required_first"] == "ask_caption"
    assert "required_first" not in b5_flex
    assert b6["allowed"] == [tool for tool in b5["allowed"] if tool != "video_metadata"]
    assert b7["allowed"] == [tool for tool in b5["allowed"] if tool != "omni_clip_perception"]
    assert b8["allowed"] == [
        tool for tool in b5["allowed"]
        if tool not in {"omni_clip_perception", "video_metadata"}
    ]


def test_b1_only_removes_crosscheck_sentence_from_b0_workflow():
    b0 = _merged("b0_omniagent_cqm.yaml")["planner"]["workflow_prompt"]
    b1 = _merged("b1_no_crosscheck_cqm.yaml")["planner"]["workflow_prompt"]
    sentence = " Cross-check and verify important information using multiple tools if needed."
    assert sentence in b0
    assert sentence not in b1
    assert b0.replace(sentence, "") == b1


def test_audio_global_caption_prompt_preserves_omniagent_trailing_space():
    prompt = _merged("b0_omniagent_cqm.yaml")["specialized_tools"]["audio_global_caption_prompt"]
    assert prompt == (
        "Provide a high-level summary of the audio. Focus on the main topics, "
        "key events, and the overall atmosphere, "
    )
