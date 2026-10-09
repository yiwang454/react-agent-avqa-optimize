#!/usr/bin/env bash
set -euo pipefail

# Controlled comparison with Experiment 11: only the tool-call budget changes
# from 6 to 30. The selected Val125 input uses the original muxed MP4 files,
# each of which contains its embedded audio stream.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAILY_OMNI_ROOT="/mnt/ceph_rbd/data/avqa_project/daily_omni"

export INPUT_JSONL="${INPUT_JSONL:-${DAILY_OMNI_ROOT}/daily_omni_cuts_selectedVal125.jsonl}"
export EXPECTED_COUNT="${EXPECTED_COUNT:-125}"
export MAX_TURNS="${MAX_TURNS:-30}"
export INFERENCE_NUM_THREADS="${INFERENCE_NUM_THREADS:-4}"
export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-${INFERENCE_NUM_THREADS}}"
export OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR_OVERRIDE:-${DAILY_OMNI_ROOT}/daily_omni_dspy_free_react_o3_gemini25flash_basic_tools_selectedVal125_maxturns30_repeat1}"
export OUTPUT_JSONL_OVERRIDE="${OUTPUT_JSONL_OVERRIDE:-${OUTPUT_DIR_OVERRIDE}/output_test.jsonl}"
export LOG_FILE_OVERRIDE="${LOG_FILE_OVERRIDE:-${OUTPUT_DIR_OVERRIDE}/run.log}"

exec "${SCRIPT_DIR}/11_run_dspy_react_agent_daily_o3_gemini25flash_basic_tools_free_react_6turn.sh" "$@"
