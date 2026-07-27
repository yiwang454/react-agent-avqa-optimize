#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
: "${GEPA_SUPERVISION_TIER:?GEPA_SUPERVISION_TIER must be G0 or G3}"

densified_args=()
case "${GEPA_SUPERVISION_TIER}" in
  G0)
    unset GEPA_DENSIFIED_LABEL_DIR
    export GEPA_REFLECTION_TEMPLATE_VERSION="${GEPA_REFLECTION_TEMPLATE_VERSION:-original}"
    ;;
  G3)
    export GEPA_DENSIFIED_LABEL_DIR="${GEPA_DENSIFIED_LABEL_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_densified_labels_selectedTrain125_gpt-5.4_high}"
    export GEPA_REFLECTION_TEMPLATE_VERSION="${GEPA_REFLECTION_TEMPLATE_VERSION:-densified_key_evidence}"
    densified_args=(--gepa-densified-label-dir "${GEPA_DENSIFIED_LABEL_DIR}")
    ;;
  *)
    echo "Unsupported GEPA_SUPERVISION_TIER=${GEPA_SUPERVISION_TIER}; expected G0 or G3." >&2
    exit 2
    ;;
esac

export OPTIMIZE_TARGETS_CSV="planner.workflow_prompt"
export PROMPT_YAML_OVERRIDE="${REPO_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_free_caption_in_task.yaml"
export GEPA_OPTIMIZED_PROMPT_CONFIG_BASENAME="daily_qa_prompt_v8_free_caption_in_task.yaml"
export GEPA_CAPTION_SUPERVISION="none"
export PLANNER_REASONING_EFFORT_OVERRIDE="none"
export PLANNER_SEED_OVERRIDE="${PLANNER_SEED_OVERRIDE:-1234}"
export GEPA_REFLECTION_MODEL="gpt-5.4"
export GEPA_REFLECTION_REASONING_EFFORT="medium"
unset GEPA_REFLECTION_TEMPERATURE

export GEPA_MAX_FULL_EVALS="${GEPA_MAX_FULL_EVALS:-16}"
export GEPA_REFLECTION_MINIBATCH_SIZE="${GEPA_REFLECTION_MINIBATCH_SIZE:-16}"
export GEPA_TRAIN_LIMIT="${GEPA_TRAIN_LIMIT:-64}"
export GEPA_SEEDS_OVERRIDE="${GEPA_SEEDS_OVERRIDE:-18}"

exec "${SCRIPT_DIR}/run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common_0721.sh" \
  --ignore-audio-caption-dir \
  "${densified_args[@]}" \
  "$@"
