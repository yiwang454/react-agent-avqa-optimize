#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAILY_OMNI_DIR="${GEPA_DAILY_OMNI_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni}"
EVENT_SEQUENCE_TRAINSET_JSONL="${GEPA_EVENT_SEQUENCE_TRAINSET_JSONL:-${DAILY_OMNI_DIR}/daily_omni_cuts_selectedTrain125_event_sequence.jsonl}"
EVENT_SEQUENCE_VALSET_JSONL="${GEPA_EVENT_SEQUENCE_VALSET_JSONL:-${DAILY_OMNI_DIR}/daily_omni_cuts_selectedVal125_event_sequence.jsonl}"
EVENT_SEQUENCE_INPUT_JSONL="${GEPA_INPUT_JSONL:-${DAILY_OMNI_DIR}/daily_omni_cuts_v3_Event_Sequence.jsonl}"

export GEPA_REFLECTION_MINIBATCH_SIZE="${GEPA_REFLECTION_MINIBATCH_SIZE:-12}"
export GEPA_MAX_FULL_EVALS="${GEPA_MAX_FULL_EVALS:-22}"

export GEPA_EXPERIMENT_NAME="planner_workflow_prompt_event_sequence"
export OPTIMIZE_TARGETS_CSV="planner.workflow_prompt"
export GEPA_TRAINSET_JSONL="${EVENT_SEQUENCE_TRAINSET_JSONL}"
export GEPA_VALSET_JSONL="${EVENT_SEQUENCE_VALSET_JSONL}"
# input-jsonl drives post-optimization evaluation; make it task-specific too.
export GEPA_INPUT_JSONL="${EVENT_SEQUENCE_INPUT_JSONL}"

exec "${SCRIPT_DIR}/run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common.sh" "$@"
