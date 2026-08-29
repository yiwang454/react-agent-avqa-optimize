#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/env_files/.env_dspy_react_agent_daily_qwen_v8}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logs/complete_unanswered_gpt4_1.log}"
COMMON_SCRIPT="${SCRIPT_DIR}/run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common.sh"
DATA_ROOT="/mnt/ceph_rbd/data/avqa_project/daily_omni"
DENSIFIED_LABEL_DIR="${DATA_ROOT}/daily_omni_densified_labels_selectedTrain125_gpt-5.4_high"
SCRIPT_ARGS=("$@")

mkdir -p "$(dirname "${LOG_FILE}")"
exec >> "${LOG_FILE}" 2>&1

if [ ! -f "${ENV_FILE}" ]; then
  echo "Missing env file: ${ENV_FILE}"
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

export PROJECT_DIR_OVERRIDE="${PROJECT_DIR_OVERRIDE:-${REPO_DIR}}"
export PLANNER_API_KEY="${PLANNER_API_KEY:-${ELM_API_KEY:-}}"
if [ -z "${PLANNER_API_KEY}" ]; then
  echo "Set PLANNER_API_KEY or ELM_API_KEY before running."
  exit 2
fi
if [ -z "${PYTHON_BIN:-}" ] && [ -n "${ENV_PREFIX:-}" ] && [ -x "${ENV_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${ENV_PREFIX}/bin/python"
fi
export PYTHON_BIN="${PYTHON_BIN:-python}"

failures=()

run_completion() (
  set -euo pipefail

  local label="$1"
  local original_script="$2"
  local experiment_name="$3"
  local optimize_targets_csv="$4"
  local caption_supervision="$5"
  local densified="$6"
  local run_dir="${DATA_ROOT}/daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_${experiment_name}_planner_gpt-4.1_seed_1234_gepa_seed18"
  local safe_targets="${optimize_targets_csv//[,._]/_}"
  local extra_args=()
  local eval_output
  local invalid_count

  unset GEPA_RUN_DIR GEPA_BASE_ENV_OUTPUT_DIR GEPA_DENSIFIED_LABEL_DIR
  unset GEPA_REFLECTION_TEMPERATURE_OVERRIDE

  export GEPA_EXPERIMENT_NAME="${experiment_name}"
  export OPTIMIZE_TARGETS_CSV="${optimize_targets_csv}"
  export GEPA_SEEDS_OVERRIDE="18"
  export PLANNER_MODEL_OVERRIDE="gpt-4.1"
  export PLANNER_REASONING_EFFORT_OVERRIDE="none"
  export PLANNER_TEMPERATURE_OVERRIDE="0.0"
  export PLANNER_SEED_OVERRIDE="1234"
  export GEPA_REFLECTION_MODEL="gpt-4.1"
  export GEPA_REFLECTION_REASONING_EFFORT="none"
  export GEPA_REFLECTION_TEMPERATURE="0.0"
  export GEPA_MAX_FULL_EVALS="16"
  export GEPA_REFLECTION_MINIBATCH_SIZE="16"
  export GEPA_CAPTION_SUPERVISION="${caption_supervision}"

  if [ "${densified}" = "true" ]; then
    extra_args+=(--gepa-densified-label-dir "${DENSIFIED_LABEL_DIR}")
  fi
  extra_args+=(--gepa-caption-supervision "${caption_supervision}")

  echo
  echo "================================================================"
  echo "START ${label}"
  echo "original_experiment_script=${original_script}"
  echo "run_dir=${run_dir}"
  echo "targets=${optimize_targets_csv}; caption_supervision=${caption_supervision}; densified=${densified}"
  echo "================================================================"

  if [ ! -f "${run_dir}/compiled_gepa.json" ]; then
    echo "Missing compiled GEPA program: ${run_dir}/compiled_gepa.json"
    exit 1
  fi

  "${COMMON_SCRIPT}" \
    "${extra_args[@]}" \
    --inference-only \
    "${SCRIPT_ARGS[@]}"

  eval_output="$(
    "${PYTHON_BIN}" "${REPO_DIR}/scripts/react_eval.py" \
      --input_file "${run_dir}/output_test.jsonl" \
      --show_wrong 0
  )"
  printf '%s\n' "${eval_output}"
  invalid_count="$(
    printf '%s\n' "${eval_output}" |
      awk '/^Missing final answer:|^Failed to parse <answer>:/ { count += $NF } END { print count + 0 }'
  )"
  if [ "${invalid_count}" -ne 0 ]; then
    echo "${label} still has ${invalid_count} unanswered/unparseable sample(s)."
    exit 1
  fi

  echo "DONE ${label}; optimized_prompt_artifact=${run_dir}/${safe_targets}_GEPA_daily_qa_prompt_v8_caption_in_task.yaml"
)

run_and_record() {
  local label="$1"
  shift
  if ! run_completion "${label}" "$@"; then
    failures+=("${label}")
    echo "FAILED ${label}; continuing with the next experiment."
  fi
}

run_baseline_completion() (
  set -euo pipefail

  local label="$1"
  local original_script="$2"
  local project_dir="${PROJECT_DIR_OVERRIDE}"
  local output_dir="${DATA_ROOT}/daily_omni_dspy_GPT_v8GeminiCaptionInTask_planner_gpt-4.1_seed_1234"
  local output_jsonl="${output_dir}/output_test.jsonl"
  local runtime_config="/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml"
  local eval_output
  local invalid_count
  local cmd

  # The baseline has no compiled GEPA program. The batch runner resumes from
  # per-question cache files and reruns only cache entries without a valid final answer.
  export PLANNER_PROVIDER="elm_gpt"
  export PLANNER_MODEL="gpt-4.1"
  export PLANNER_REASONING_EFFORT="none"
  export PLANNER_TEMPERATURE="0.0"
  export PLANNER_OUTPUT_SEQ_LEN="32768"
  export PLANNER_SEED="1234"
  export DEEPSEEK_API_KEY="${PLANNER_API_KEY}"
  unset DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE
  export PERCEPTION_MODEL="gemini"
  export CAPTIONER_MODEL="gemini"
  export PERCEPTION_CONFIG_YAML="${runtime_config}"
  export CAPTIONER_CONFIG_YAML="${runtime_config}"
  export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception"
  export GEMINI_API_BACKEND="legacy"
  export GEMINI_RESPONSE_ERROR_SENSITIVE="false"

  echo
  echo "================================================================"
  echo "START ${label}"
  echo "original_experiment_script=${original_script}"
  echo "run_dir=${output_dir}"
  echo "mode=batch-resume; expected_unanswered_before_run=9"
  echo "================================================================"

  if [ ! -f "${output_jsonl}" ]; then
    echo "Missing baseline output: ${output_jsonl}"
    exit 1
  fi

  cd "${project_dir}"
  cmd=(
    "${PYTHON_BIN}" DSPy/avqa_dspy_impl.py
    --input-jsonl "${DATA_ROOT}/daily_omni_cuts_v3.jsonl"
    --output-jsonl "${output_jsonl}"
    --output-dir "${output_dir}"
    --max-turns 3
    --perception-model gemini
    --perception-config-yaml "${runtime_config}"
    --captioner-config-yaml "${runtime_config}"
    --prompt-yaml "${project_dir}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_caption_in_task.yaml"
    --allowed-tools ask_caption,ask_perception
    --gemini-api-backend legacy
    --signature-in-system-prompt
    --caption-placement task
  )
  cmd+=("${SCRIPT_ARGS[@]}")
  "${cmd[@]}"

  eval_output="$(
    "${PYTHON_BIN}" "${REPO_DIR}/scripts/react_eval.py" \
      --input_file "${output_jsonl}" \
      --show_wrong 0
  )"
  printf '%s\n' "${eval_output}"
  invalid_count="$(
    printf '%s\n' "${eval_output}" |
      awk '/^Missing final answer:|^Failed to parse <answer>:/ { count += $NF } END { print count + 0 }'
  )"
  if [ "${invalid_count}" -ne 0 ]; then
    echo "${label} still has ${invalid_count} unanswered/unparseable sample(s)."
    exit 1
  fi

  echo "DONE ${label}"
)

run_baseline_and_record() {
  local label="$1"
  shift
  if ! run_baseline_completion "${label}" "$@"; then
    failures+=("${label}")
    echo "FAILED ${label}; continuing with the next experiment."
  fi
}

echo "GPT-4.1 unanswered-sample completion started at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "log=${LOG_FILE}"

# Original experiment script:
# scripts/run_dspy_react_agent_daily_qwen_v8_GPT4.1_gemini_captionInTask.sh
run_baseline_and_record \
  "GPT-4.1 G0 baseline" \
  "scripts/run_dspy_react_agent_daily_qwen_v8_GPT4.1_gemini_captionInTask.sh"

# # Original experiment script:
# # scripts/run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_planner.sh
# run_and_record \
#   "GPT-4.1 G0 planner" \
#   "scripts/run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_planner.sh" \
#   "planner_workflow_prompt" \
#   "planner.workflow_prompt" \
#   "auto" \
#   "false"

# # Original experiment script:
# # scripts/run_dspy_react_agent_daily_gepa_elm_gpt4.1_v8_gemini_caption_in_task_captioner_common_0721.sh
# run_and_record \
#   "GPT-4.1 G0 captioner" \
#   "scripts/run_dspy_react_agent_daily_gepa_elm_gpt4.1_v8_gemini_caption_in_task_captioner_common_0721.sh" \
#   "captioner_default_caption_instruction_gpt4_1_0721" \
#   "captioner.default_caption_instruction" \
#   "none" \
#   "false"

# # Original experiment script:
# # scripts/run_dspy_react_agent_daily_gepa_elm_gpt4.1_v8_gemini_caption_in_task_planner_captioner_common_0721.sh
# run_and_record \
#   "GPT-4.1 G0 planner + captioner" \
#   "scripts/run_dspy_react_agent_daily_gepa_elm_gpt4.1_v8_gemini_caption_in_task_planner_captioner_common_0721.sh" \
#   "planner_workflow_prompt_and_captioner_default_caption_instruction_gpt4_1_0721" \
#   "planner.workflow_prompt,captioner.default_caption_instruction" \
#   "auto" \
#   "false"

# # Original experiment script:
# # scripts/run_dspy_react_agent_daily_gepa_densified_gpt4_1_planner.sh
# run_and_record \
#   "GPT-4.1 G3 planner" \
#   "scripts/run_dspy_react_agent_daily_gepa_densified_gpt4_1_planner.sh" \
#   "densified_planner_workflow_prompt_gpt4_1" \
#   "planner.workflow_prompt" \
#   "none" \
#   "true"

# # Original experiment script:
# # scripts/run_dspy_react_agent_daily_gepa_densified_gpt4_1_planner_captioner.sh
# run_and_record \
#   "GPT-4.1 G3 planner + captioner" \
#   "scripts/run_dspy_react_agent_daily_gepa_densified_gpt4_1_planner_captioner.sh" \
#   "densified_planner_workflow_prompt_and_captioner_default_caption_instruction_gpt4_1" \
#   "planner.workflow_prompt,captioner.default_caption_instruction" \
#   "none" \
#   "true"

# # Original experiment script:
# # scripts/run_dspy_react_agent_daily_gepa_densified_gpt4_1_planner_repeat.sh
# run_and_record \
#   "GPT-4.1 G3 planner repeat" \
#   "scripts/run_dspy_react_agent_daily_gepa_densified_gpt4_1_planner_repeat.sh" \
#   "densified_planner_workflow_prompt_gpt4_1_repeat" \
#   "planner.workflow_prompt" \
#   "none" \
#   "true"

echo
if [ "${#failures[@]}" -gt 0 ]; then
  echo "Completion finished with ${#failures[@]} failed experiment(s): ${failures[*]}"
  exit 1
fi
echo "All GPT-4.1 completion experiments finished successfully."
