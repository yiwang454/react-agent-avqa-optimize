#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

cd "${PROJECT_DIR_OVERRIDE:-${REPO_DIR}}"

PLANNER_CONFIG_YAML="${PLANNER_CONFIG_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_gpt4_1_none.yaml}"
REFLECTION_CONFIG_YAML="${REFLECTION_CONFIG_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_gpt4_1_none.yaml}"
GEMINI_CONFIG_YAML="${GEMINI_CONFIG_YAML_OVERRIDE:-/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml}"
PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-${GEMINI_CONFIG_YAML}}"
CAPTIONER_CONFIG_YAML="${CAPTIONER_CONFIG_YAML_OVERRIDE:-${GEMINI_CONFIG_YAML}}"
PROMPT_YAML="${PROMPT_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_free_react_caption_in_task.yaml}"
REFLECTION_TEMPLATE_YAML="${REFLECTION_TEMPLATE_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/DSPy/reflection_template.yaml}"
INPUT_JSONL="${INPUT_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3.jsonl}"
TRAINSET_JSONL="${TRAINSET_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedTrain125.jsonl}"
VALSET_JSONL="${VALSET_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedVal125.jsonl}"
DAILY_OMNI_ROOT="${DAILY_OMNI_ROOT:-/mnt/ceph_rbd/data/avqa_project/daily_omni}"
CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_caption_cache_v8_gemini2.5flash_seed27}"

EXPECTED_COUNT="${EXPECTED_COUNT:-1197}"
EXPECTED_TRAIN_COUNT=125
EXPECTED_VAL_COUNT=125
MAX_TURNS=4
: "${GEPA_SEED:?GEPA_SEED must be set to 42 or 2 by a seed-specific launcher.}"
case "${GEPA_SEED}" in
  42|2) ;;
  *)
    echo "This C1 launcher only accepts GEPA_SEED=42 or GEPA_SEED=2; got ${GEPA_SEED}." >&2
    exit 2
    ;;
esac
GEPA_CALL_BUDGET=2500
INFERENCE_REPEATS=3
FINAL_EVAL_NUM_THREADS="${FINAL_EVAL_NUM_THREADS:-4}"
FINAL_EVAL_BATCH_SIZE="${FINAL_EVAL_BATCH_SIZE:-${FINAL_EVAL_NUM_THREADS}}"

GEPA_RUN_DIR="${GEPA_RUN_DIR_OVERRIDE:-${DAILY_OMNI_ROOT}/daily_omni_dspy_free_react_gpt4_1_gemini_first_cached_then_live_caption_gepa_planner_maxturns4_seed${GEPA_SEED}_calls${GEPA_CALL_BUDGET}}"
COMPILED_PROGRAM="${GEPA_RUN_DIR}/compiled_gepa.json"
CANDIDATES_JSONL="${GEPA_RUN_DIR}/planner_workflow_prompt_candidates.jsonl"
GATE_REPORT="${GEPA_RUN_DIR}/validation_improvement_gate.json"
ORCHESTRATOR_LOG="${LOG_FILE_OVERRIDE:-${GEPA_RUN_DIR}/c1_seed${GEPA_SEED}_conditional_3repeat.log}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ceph_rbd/applications/anaconda3/envs/react-avqa-dspy/bin/python}"

mkdir -p "${GEPA_RUN_DIR}"
exec > >(tee -a "${ORCHESTRATOR_LOG}") 2>&1

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
: "${ELM_API_KEY:?Set ELM_API_KEY for the GPT-4.1 planner and reflection model.}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY for live Gemini captioning and perception.}"

export PLANNER_API_KEY="${ELM_API_KEY}"
unset DEEPSEEK_API_KEY DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE
export PERCEPTION_MODEL="gemini"
export CAPTIONER_MODEL="gemini"
export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception"
export GEMINI_API_BACKEND="${GEMINI_API_BACKEND:-legacy}"
export GEMINI_RESPONSE_ERROR_SENSITIVE="false"

case "${GEMINI_API_BACKEND}" in
  legacy|dspy) ;;
  *)
    echo "GEMINI_API_BACKEND must be either legacy or dspy." >&2
    exit 2
    ;;
esac

for required_file in \
  "${PLANNER_CONFIG_YAML}" \
  "${REFLECTION_CONFIG_YAML}" \
  "${PERCEPTION_CONFIG_YAML}" \
  "${CAPTIONER_CONFIG_YAML}" \
  "${PROMPT_YAML}" \
  "${REFLECTION_TEMPLATE_YAML}" \
  "${INPUT_JSONL}" \
  "${TRAINSET_JSONL}" \
  "${VALSET_JSONL}" \
  "${CAPTION_CACHE_DIR}/manifest.json" \
  "${SCRIPT_DIR}/check_gepa_validation_improvement.py"; do
  if [ ! -f "${required_file}" ]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done

count_rows() {
  awk 'NF {count++} END {print count+0}' "$1"
}

actual_count="$(count_rows "${INPUT_JSONL}")"
actual_train_count="$(count_rows "${TRAINSET_JSONL}")"
actual_val_count="$(count_rows "${VALSET_JSONL}")"
if [ "${actual_count}" -ne "${EXPECTED_COUNT}" ] || \
   [ "${actual_train_count}" -ne "${EXPECTED_TRAIN_COUNT}" ] || \
   [ "${actual_val_count}" -ne "${EXPECTED_VAL_COUNT}" ]; then
  echo "Dataset counts must be full-v3/Train125/Val125; got ${actual_count}/${actual_train_count}/${actual_val_count}." >&2
  exit 2
fi
if [ "${FINAL_EVAL_NUM_THREADS}" -lt 1 ] || [ "${FINAL_EVAL_BATCH_SIZE}" -lt 1 ]; then
  echo "FINAL_EVAL_NUM_THREADS and FINAL_EVAL_BATCH_SIZE must be >= 1." >&2
  exit 2
fi

common_runtime_args=(
  --input-jsonl "${INPUT_JSONL}"
  --max-turns "${MAX_TURNS}"
  --planner-config-yaml "${PLANNER_CONFIG_YAML}"
  --gepa-reflection-config-yaml "${REFLECTION_CONFIG_YAML}"
  --perception-model "${PERCEPTION_MODEL}"
  --perception-config-yaml "${PERCEPTION_CONFIG_YAML}"
  --captioner-config-yaml "${CAPTIONER_CONFIG_YAML}"
  --prompt-yaml "${PROMPT_YAML}"
  --allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}"
  --caption-cache-dir "${CAPTION_CACHE_DIR}"
  --caption-cache-scope first_call_only
  --ignore-audio-caption-dir
  --gemini-api-backend "${GEMINI_API_BACKEND}"
  --signature-in-system-prompt
  --caption-placement task
  --print-config
)

search_cmd=(
  "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
  --algorithm gepa
  --optimization-train-only
  --trainset-jsonl "${TRAINSET_JSONL}"
  --valset-jsonl "${VALSET_JSONL}"
  --daily-omni-root "${DAILY_OMNI_ROOT}"
  --output-program "${COMPILED_PROGRAM}"
  --initial-program "${GEPA_RUN_DIR}/initial_gepa_program.json"
  --metadata-json "${GEPA_RUN_DIR}/compiled_gepa_metadata.json"
  --signature-search-json "${GEPA_RUN_DIR}/compiled_gepa_signature_search.json"
  --trajectory-jsonl "${GEPA_RUN_DIR}/optimized_trainset_trajectories.jsonl"
  --optimizer-log-dir "${GEPA_RUN_DIR}/optimizer_logs"
  --optimized-prompt-config-yaml "${GEPA_RUN_DIR}/planner_workflow_prompt_GEPA_daily_qa_prompt_v8_free_react_caption_in_task.yaml"
  --optimize-target planner.workflow_prompt
  --gepa-seed "${GEPA_SEED}"
  --gepa-max-metric-calls "${GEPA_CALL_BUDGET}"
  --gepa-reflection-minibatch-size 16
  --gepa-candidate-selection-strategy pareto
  --gepa-max-merge-invocations 5
  --gepa-reflection-template-yaml "${REFLECTION_TEMPLATE_YAML}"
  --gepa-reflection-template-version auto
  --gepa-log-dir "${GEPA_RUN_DIR}/gepa_logs"
  --gepa-track-stats
  --gepa-track-best-outputs
  "${common_runtime_args[@]}"
)

inference_command() {
  local repeat_index="$1"
  local output_dir="${GEPA_RUN_DIR}/inference_repeat${repeat_index}"
  local -n command_ref="$2"
  command_ref=(
    "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
    --algorithm gepa
    --inference-only
    --output-program "${COMPILED_PROGRAM}"
    --metadata-json "${output_dir}/inference_only_metadata.json"
    --final-eval-output-jsonl "${output_dir}/output_test.jsonl"
    --final-eval-output-dir "${output_dir}"
    --final-eval-num-threads "${FINAL_EVAL_NUM_THREADS}"
    --final-eval-batch-size "${FINAL_EVAL_BATCH_SIZE}"
    "${common_runtime_args[@]}"
  )
}

echo "C1 conditional run: seed=${GEPA_SEED}; search budget=${GEPA_CALL_BUDGET}; inference repeats=${INFERENCE_REPEATS}"
echo "Search directory: ${GEPA_RUN_DIR}"
echo "Orchestrator log: ${ORCHESTRATOR_LOG}"
echo "Validation gate: selected full-Val125 accuracy must be strictly greater than candidate 0."

if [ "${DRY_RUN:-false}" = "true" ]; then
  printf 'Search command:'
  printf ' %q' "${search_cmd[@]}"
  printf '\n'
  for repeat_index in $(seq 1 "${INFERENCE_REPEATS}"); do
    inference_command "${repeat_index}" repeat_cmd
    printf 'Inference repeat %s command:' "${repeat_index}"
    printf ' %q' "${repeat_cmd[@]}"
    printf '\n'
  done
  exit 0
fi

echo "Search started: $(date -Is)"
"${search_cmd[@]}"
echo "Search finished: $(date -Is)"

for artifact in "${COMPILED_PROGRAM}" "${CANDIDATES_JSONL}" "${GEPA_RUN_DIR}/compiled_gepa_metadata.json"; do
  if [ ! -s "${artifact}" ]; then
    echo "Search did not produce required artifact: ${artifact}" >&2
    exit 1
  fi
done

set +e
"${PYTHON_BIN}" "${SCRIPT_DIR}/check_gepa_validation_improvement.py" \
  --candidates-jsonl "${CANDIDATES_JSONL}" \
  --output-json "${GATE_REPORT}" \
  --expected-validation-examples "${EXPECTED_VAL_COUNT}"
gate_status=$?
set -e
if [ "${gate_status}" -eq 10 ]; then
  echo "Inference skipped: selected candidate did not strictly improve over candidate 0."
  echo "Gate report: ${GATE_REPORT}"
  exit 0
fi
if [ "${gate_status}" -ne 0 ]; then
  echo "Inference blocked because the validation gate could not be audited." >&2
  exit "${gate_status}"
fi

echo "Validation improved; starting three inference repeats in parallel: $(date -Is)"
pids=()
for repeat_index in $(seq 1 "${INFERENCE_REPEATS}"); do
  output_dir="${GEPA_RUN_DIR}/inference_repeat${repeat_index}"
  mkdir -p "${output_dir}"
  inference_command "${repeat_index}" repeat_cmd
  (
    echo "Inference repeat ${repeat_index} started: $(date -Is)"
    "${repeat_cmd[@]}"
    echo "Inference repeat ${repeat_index} finished: $(date -Is)"
  ) > >(tee -a "${output_dir}/run.log") 2>&1 &
  pids+=("$!")
done

failures=0
for index in "${!pids[@]}"; do
  repeat_index=$((index + 1))
  if wait "${pids[index]}"; then
    echo "Inference repeat ${repeat_index} completed successfully."
  else
    echo "Inference repeat ${repeat_index} failed; see inference_repeat${repeat_index}/run.log." >&2
    failures=$((failures + 1))
  fi
done
if [ "${failures}" -ne 0 ]; then
  echo "${failures} of ${INFERENCE_REPEATS} inference repeats failed." >&2
  exit 1
fi
echo "All ${INFERENCE_REPEATS} inference repeats finished: $(date -Is)"
