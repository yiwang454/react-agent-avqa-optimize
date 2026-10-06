#!/usr/bin/env bash
set -euo pipefail

# Run the WorldSense benchmark with the exact Experiment 11 workflow from
# README.md: o3 (medium), ask_caption + ask_perception, a six-tool-call budget,
# cached first caption, and live Gemini 2.5 Flash for later tool calls.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"

export INPUT_JSONL="${INPUT_JSONL:-/mnt/ceph_rbd/data/avqa_project/WorldSense/worldsense_reencoded_over30m_max18m/worldsense_test_cut_reencoded_over30m_max18m.jsonl}"
export CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR:-/mnt/ceph_rbd/data/avqa_project/WorldSense/worldsense_caption_gemini25flash_split_avs_plan_a_timestamp5/caption_cache}"
export PROMPT_YAML_OVERRIDE="${PROMPT_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_free_react_caption_in_task_timestamp5.yaml}"
export EXPECTED_COUNT="${EXPECTED_COUNT:-3172}"
export OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR_OVERRIDE:-/mnt/ceph_rbd/data/avqa_project/WorldSense/worldsense_dspy_free_react_o3_gemini25flash_basic_tools_split_avs_timestamp3_maxturns6}"
export LOG_FILE_OVERRIDE="${LOG_FILE_OVERRIDE:-${REPO_DIR}/scripts/logs/${SCRIPT_NAME}.log}"

exec "${SCRIPT_DIR}/11_run_dspy_react_agent_daily_o3_gemini25flash_basic_tools_free_react_6turn.sh" "$@"
