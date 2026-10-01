#!/usr/bin/env bash
set -euo pipefail

# C1 replacement: keep the candidate-0 ReAct workflow contract fixed and let
# GEPA optimize only planner.optimization_guidance. Run seeds 18/42/2 in
# parallel on Train125/Val125, with no full-v3 inference.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

cd "${PROJECT_DIR_OVERRIDE:-${REPO_DIR}}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/ceph_rbd/applications/anaconda3/envs/react-avqa-dspy/bin/python}"
PLANNER_CONFIG_YAML="${PLANNER_CONFIG_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_gpt4_1_none.yaml}"
REFLECTION_CONFIG_YAML="${REFLECTION_CONFIG_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_gpt4_1_none.yaml}"
GEMINI_CONFIG_YAML="${GEMINI_CONFIG_YAML_OVERRIDE:-/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml}"
PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-${GEMINI_CONFIG_YAML}}"
CAPTIONER_CONFIG_YAML="${CAPTIONER_CONFIG_YAML_OVERRIDE:-${GEMINI_CONFIG_YAML}}"
PROMPT_YAML="${PROMPT_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_free_react_guidance_caption_in_task.yaml}"
REFLECTION_TEMPLATE_YAML="${REFLECTION_TEMPLATE_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/DSPy/reflection_template.yaml}"

DAILY_OMNI_ROOT="${DAILY_OMNI_ROOT:-/mnt/ceph_rbd/data/avqa_project/daily_omni}"
INPUT_JSONL="${INPUT_JSONL:-${DAILY_OMNI_ROOT}/daily_omni_cuts_v3.jsonl}"
TRAINSET_JSONL="${TRAINSET_JSONL:-${DAILY_OMNI_ROOT}/daily_omni_cuts_selectedTrain125.jsonl}"
VALSET_JSONL="${VALSET_JSONL:-${DAILY_OMNI_ROOT}/daily_omni_cuts_selectedVal125.jsonl}"
CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR:-${DAILY_OMNI_ROOT}/daily_omni_caption_cache_v8_gemini2.5flash_seed27}"
RUN_PREFIX="${GEPA_RUN_PREFIX_OVERRIDE:-${DAILY_OMNI_ROOT}/daily_omni_dspy_free_react_gpt4_1_gemini_first_cached_then_live_caption_gepa_guidance_maxturns4}"

EXPECTED_COUNT=1197
EXPECTED_TRAIN_COUNT=125
EXPECTED_VAL_COUNT=125
MAX_TURNS=4
GEPA_CALL_BUDGET=2500
GEPA_SEEDS=(18 42 2)

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
  "${CAPTION_CACHE_DIR}/manifest.json"; do
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

build_search_command() {
  local seed="$1"
  local run_dir="$2"
  local -n command_ref="$3"
  command_ref=(
    "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
    --algorithm gepa
    --optimization-train-only
    --trainset-jsonl "${TRAINSET_JSONL}"
    --valset-jsonl "${VALSET_JSONL}"
    --daily-omni-root "${DAILY_OMNI_ROOT}"
    --output-program "${run_dir}/compiled_gepa.json"
    --initial-program "${run_dir}/initial_gepa_program.json"
    --metadata-json "${run_dir}/compiled_gepa_metadata.json"
    --signature-search-json "${run_dir}/compiled_gepa_signature_search.json"
    --trajectory-jsonl "${run_dir}/optimized_trainset_trajectories.jsonl"
    --optimizer-log-dir "${run_dir}/optimizer_logs"
    --optimized-prompt-config-yaml "${run_dir}/planner_optimization_guidance_GEPA_daily_qa_prompt_v8_free_react_guidance_caption_in_task.yaml"
    --optimize-target planner.optimization_guidance
    --gepa-seed "${seed}"
    --gepa-max-metric-calls "${GEPA_CALL_BUDGET}"
    --gepa-reflection-minibatch-size 16
    --gepa-candidate-selection-strategy pareto
    --gepa-max-merge-invocations 5
    --gepa-reflection-template-yaml "${REFLECTION_TEMPLATE_YAML}"
    --gepa-reflection-template-version auto
    --gepa-log-dir "${run_dir}/gepa_logs"
    --gepa-track-stats
    --gepa-track-best-outputs
    "${common_runtime_args[@]}"
  )
}

echo "C1 guidance-only GEPA: parallel seeds=${GEPA_SEEDS[*]}"
echo "Optimize target: planner.optimization_guidance"
echo "Fixed contract: planner.workflow_contract"
echo "Planner/reflection: elm_gpt/gpt-4.1; reasoning_effort=none; model seed=1234"
echo "Train/validation: ${TRAINSET_JSONL} (${actual_train_count}) / ${VALSET_JSONL} (${actual_val_count})"
echo "Rollout: max_turns=${MAX_TURNS}; first caption exact-cache, later captions/perception live Gemini"
echo "GEPA: max_metric_calls=${GEPA_CALL_BUDGET}; minibatch=16; selection=pareto; max_merges=5"
echo "Optimization-train-only: true; full-v3 inference will not run"

if [ "${DRY_RUN:-false}" = "true" ]; then
  for seed in "${GEPA_SEEDS[@]}"; do
    run_dir="${RUN_PREFIX}_seed${seed}_calls${GEPA_CALL_BUDGET}"
    build_search_command "${seed}" "${run_dir}" search_cmd
    printf 'Seed %s run directory: %s\nCommand:' "${seed}" "${run_dir}"
    printf ' %q' "${search_cmd[@]}" "$@"
    printf '\n'
  done
  exit 0
fi

pids=()
run_dirs=()
cleanup_children() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup_children EXIT INT TERM

for seed in "${GEPA_SEEDS[@]}"; do
  run_dir="${RUN_PREFIX}_seed${seed}_calls${GEPA_CALL_BUDGET}"
  if [ -e "${run_dir}/compiled_gepa_metadata.json" ]; then
    echo "Refusing to overwrite completed metadata: ${run_dir}/compiled_gepa_metadata.json" >&2
    exit 2
  fi
done

for seed in "${GEPA_SEEDS[@]}"; do
  run_dir="${RUN_PREFIX}_seed${seed}_calls${GEPA_CALL_BUDGET}"
  mkdir -p "${run_dir}"
  build_search_command "${seed}" "${run_dir}" search_cmd
  (
    echo "Seed ${seed} started: $(date -Is)"
    echo "Run directory: ${run_dir}"
    "${search_cmd[@]}" "$@"
    echo "Seed ${seed} finished: $(date -Is)"
  ) > >(tee -a "${run_dir}/run.log") 2>&1 &
  pids+=("$!")
  run_dirs+=("${run_dir}")
done

failures=0
for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then
    echo "Completed: ${run_dirs[index]}"
  else
    echo "Failed: ${run_dirs[index]} (see run.log)" >&2
    failures=$((failures + 1))
  fi
done
trap - EXIT INT TERM

if [ "${failures}" -ne 0 ]; then
  echo "${failures} of ${#GEPA_SEEDS[@]} GEPA searches failed." >&2
  exit 1
fi
echo "All ${#GEPA_SEEDS[@]} guidance-only GEPA searches completed. No inference was run."
