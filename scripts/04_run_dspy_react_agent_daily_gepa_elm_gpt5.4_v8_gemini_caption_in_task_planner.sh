#!/usr/bin/env bash
set -euo pipefail

export GEPA_LAUNCHER_PATH="./scripts/04_run_dspy_react_agent_daily_gepa_elm_gpt5.4_v8_gemini_caption_in_task_planner.sh"
export GEPA_PLANNER_CONFIG_YAML="./DSPy/dspy_avqa/yamls/reasoner_elm_gpt5_4_medium.yaml"
export GEPA_REFLECTION_CONFIG_YAML="./DSPy/dspy_avqa/yamls/reasoner_elm_gpt5_4_medium.yaml"
export GEPA_RUN_DIR="/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_planner_workflow_prompt_gpt5_4_medium_planner_gpt-5.4_seed_1234_gepa_seed18"
export GEPA_SEED=18

exec ./scripts/run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common.sh "$@"
