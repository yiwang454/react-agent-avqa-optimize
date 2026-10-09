#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"

export INPUT_JSONL="${INPUT_JSONL:-/mnt/ceph_rbd/data/avqa_project/avut/mllm_instruct_opt/avut_gemini2.5flash_direct_safe_duration_sorted_input/avut_full1734_duration_ascending_safe_videos.jsonl}"
export CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR:-/mnt/ceph_rbd/data/avqa_project/avut/avut_caption_gemini25flash_whole_video_timestamp3/caption_cache}"
export EXPECTED_COUNT="${EXPECTED_COUNT:-1734}"
export INFERENCE_NUM_THREADS="${INFERENCE_NUM_THREADS:-4}"
export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-4}"
export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception,omni_clip_caption,omni_clip_perception"
export PROMPT_YAML_OVERRIDE="${PROMPT_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v11_free_react_omni_clip_caption_perception_in_task.yaml}"
export OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR_OVERRIDE:-/mnt/ceph_rbd/data/avqa_project/avut/avut_dspy_free_react_o3_gemini25flash_both_omni_clip_tools_whole_video_timestamp3_maxturns6}"
export LOG_FILE_OVERRIDE="${LOG_FILE_OVERRIDE:-${REPO_DIR}/scripts/logs/${SCRIPT_NAME}.log}"

exec "${SCRIPT_DIR}/12_run_dspy_react_agent_daily_o3_gemini25flash_omni_clip_perception_free_react_6turn.sh" "$@"
