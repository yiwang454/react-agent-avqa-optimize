#!/usr/bin/env bash
set -euo pipefail

# Run two additional independent full-v3 inference repeats of the selected
# seed-2 guidance-only GEPA program (candidate 2). The repeats run concurrently,
# with exactly three final-evaluation workers each.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

cd "${PROJECT_DIR_OVERRIDE:-${REPO_DIR}}"

if [ "$#" -ne 0 ]; then
  echo "This launcher accepts no positional arguments." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/ceph_rbd/applications/anaconda3/envs/react-avqa-dspy/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/ceph_rbd/data/avqa_project/daily_omni}"
SOURCE_RUN="${SOURCE_RUN_OVERRIDE:-${DATA_ROOT}/daily_omni_dspy_free_react_gpt4_1_gemini_first_cached_then_live_caption_gepa_guidance_maxturns4_seed2_calls2500}"
OUTPUT_PREFIX="${OUTPUT_PREFIX_OVERRIDE:-${SOURCE_RUN}_best_full_v3}"

INPUT_JSONL="${INPUT_JSONL:-${DATA_ROOT}/daily_omni_cuts_v3.jsonl}"
TRAINSET_JSONL="${TRAINSET_JSONL:-${DATA_ROOT}/daily_omni_cuts_selectedTrain125.jsonl}"
VALSET_JSONL="${VALSET_JSONL:-${DATA_ROOT}/daily_omni_cuts_selectedVal125.jsonl}"
CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR:-${DATA_ROOT}/daily_omni_caption_cache_v8_gemini2.5flash_seed27}"
PLANNER_CONFIG_YAML="${PLANNER_CONFIG_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_gpt4_1_none.yaml}"
GEMINI_CONFIG_YAML="${GEMINI_CONFIG_YAML_OVERRIDE:-/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml}"
OPTIMIZED_PROMPT_YAML="${SOURCE_RUN}/planner_optimization_guidance_GEPA_daily_qa_prompt_v8_free_react_guidance_caption_in_task.yaml"
SOURCE_PROGRAM="${SOURCE_RUN}/compiled_gepa.json"
SOURCE_METADATA="${SOURCE_RUN}/compiled_gepa_metadata.json"

NUM_WORKERS=3
MAX_TURNS=4
REPEATS=(repeat2 repeat3)

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
: "${ELM_API_KEY:?Set ELM_API_KEY for the GPT-4.1 planner.}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY for live Gemini captioning and perception.}"

export PLANNER_API_KEY="${ELM_API_KEY}"
unset DEEPSEEK_API_KEY DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE
export PERCEPTION_MODEL=gemini
export CAPTIONER_MODEL=gemini
export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception"
export GEMINI_API_BACKEND="${GEMINI_API_BACKEND:-legacy}"
export GEMINI_RESPONSE_ERROR_SENSITIVE=false

case "${GEMINI_API_BACKEND}" in
  legacy|dspy) ;;
  *)
    echo "GEMINI_API_BACKEND must be either legacy or dspy." >&2
    exit 2
    ;;
esac

for required_file in \
  "${SOURCE_PROGRAM}" \
  "${SOURCE_METADATA}" \
  "${SOURCE_RUN}/optimized_planner_optimization_guidance.txt" \
  "${SOURCE_RUN}/planner_optimization_guidance_candidates.jsonl" \
  "${OPTIMIZED_PROMPT_YAML}" \
  "${INPUT_JSONL}" \
  "${TRAINSET_JSONL}" \
  "${VALSET_JSONL}" \
  "${CAPTION_CACHE_DIR}/manifest.json" \
  "${PLANNER_CONFIG_YAML}" \
  "${GEMINI_CONFIG_YAML}"; do
  if [ ! -f "${required_file}" ]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" - \
  "${SOURCE_PROGRAM}" \
  "${SOURCE_METADATA}" \
  "${OPTIMIZED_PROMPT_YAML}" \
  "${SOURCE_RUN}/optimized_planner_optimization_guidance.txt" \
  "${SOURCE_RUN}/planner_optimization_guidance_candidates.jsonl" \
  "${INPUT_JSONL}" \
  "${TRAINSET_JSONL}" \
  "${VALSET_JSONL}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

(
    program_path,
    metadata_path,
    prompt_path,
    guidance_path,
    candidates_path,
    input_path,
    train_path,
    val_path,
) = map(Path, sys.argv[1:])

program = json.loads(program_path.read_text())
metadata = json.loads(metadata_path.read_text())
prompt = yaml.safe_load(prompt_path.read_text())
guidance = guidance_path.read_text().strip()
compiled_guidance = str(program["prompt_component_0"]["signature"]["instructions"]).strip()
prompt_guidance = str(prompt["planner"]["optimization_guidance"]).strip()
contract = str(prompt["planner"]["workflow_contract"]).strip()
template = str(prompt["planner"]["task_prompt_template"])
candidates = [
    json.loads(line)
    for line in candidates_path.read_text().splitlines()
    if line.strip()
]
selected = [row for row in candidates if row.get("whether_selected_as_best") is True]

if metadata.get("optimization_train_only") is not True:
    raise SystemExit("Source metadata is not optimization_train_only.")
if metadata.get("optimize_target") != "planner.optimization_guidance":
    raise SystemExit("Source metadata has the wrong optimize target.")
if not contract:
    raise SystemExit("Optimized prompt YAML has an empty workflow contract.")
if compiled_guidance != guidance or compiled_guidance != prompt_guidance:
    raise SystemExit("Compiled, persisted, and YAML optimization guidance differ.")
if len(selected) != 1 or selected[0].get("candidate_index") != 2:
    raise SystemExit("Source program is not uniquely associated with selected candidate 2.")
if str(selected[0].get("prompt_text", "")).strip() != compiled_guidance:
    raise SystemExit("Selected candidate 2 does not match the compiled program.")
if not (
    "{workflow_contract}" in template
    and "Optimization guidance:" in template
    and "{optimization_guidance}" in template
    and template.index("{workflow_contract}")
    < template.index("Optimization guidance:")
    < template.index("{optimization_guidance}")
):
    raise SystemExit("Runtime task template does not concatenate contract then guidance.")

for path, expected in ((input_path, 1197), (train_path, 125), (val_path, 125)):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    row_ids = [str(row.get("id") or "") for row in rows]
    if len(rows) != expected or any(not item for item in row_ids) or len(set(row_ids)) != expected:
        raise SystemExit(f"Expected {expected} rows with unique IDs in {path}.")

print(
    "Verified seed2 candidate2 source: "
    f"Val125={selected[0].get('full_valset_score')}, "
    "runtime prompt=fixed workflow_contract + optimized guidance"
)
PY

build_command() {
  local output_dir="$1"
  local -n command_ref="$2"
  command_ref=(
    "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
    --algorithm gepa
    --inference-only
    --input-jsonl "${INPUT_JSONL}"
    --trainset-jsonl "${TRAINSET_JSONL}"
    --valset-jsonl "${VALSET_JSONL}"
    --daily-omni-root "${DATA_ROOT}"
    --max-turns "${MAX_TURNS}"
    --planner-config-yaml "${PLANNER_CONFIG_YAML}"
    --gepa-reflection-config-yaml "${PLANNER_CONFIG_YAML}"
    --perception-model gemini
    --perception-config-yaml "${GEMINI_CONFIG_YAML}"
    --captioner-config-yaml "${GEMINI_CONFIG_YAML}"
    --prompt-yaml "${OPTIMIZED_PROMPT_YAML}"
    --optimize-target planner.optimization_guidance
    --allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}"
    --caption-cache-dir "${CAPTION_CACHE_DIR}"
    --caption-cache-scope first_call_only
    --ignore-audio-caption-dir
    --gemini-api-backend "${GEMINI_API_BACKEND}"
    --signature-in-system-prompt
    --caption-placement task
    --output-program "${output_dir}/compiled_gepa.json"
    --metadata-json "${output_dir}/inference_only_metadata.json"
    --final-eval-output-jsonl "${output_dir}/output_test.jsonl"
    --final-eval-output-dir "${output_dir}"
    --final-eval-num-threads "${NUM_WORKERS}"
    --final-eval-batch-size "${NUM_WORKERS}"
    --print-config
  )
}

echo "Seed2 candidate2 repeats: ${REPEATS[*]}"
echo "Parallel jobs: ${#REPEATS[@]}; workers per job: ${NUM_WORKERS}"
echo "Source program: ${SOURCE_PROGRAM}"

if [ "${DRY_RUN:-false}" = true ]; then
  for repeat in "${REPEATS[@]}"; do
    output_dir="${OUTPUT_PREFIX}_${repeat}"
    build_command "${output_dir}" inference_cmd
    printf '%s output directory: %s\nCommand:' "${repeat}" "${output_dir}"
    printf ' %q' "${inference_cmd[@]}"
    printf '\n'
  done
  exit 0
fi

pids=()
output_dirs=()
cleanup_children() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup_children EXIT INT TERM

for repeat in "${REPEATS[@]}"; do
  output_dir="${OUTPUT_PREFIX}_${repeat}"
  copied_program="${output_dir}/compiled_gepa.json"
  mkdir -p "${output_dir}"
  if [ -f "${copied_program}" ]; then
    if ! cmp -s "${SOURCE_PROGRAM}" "${copied_program}"; then
      echo "Refusing to resume ${repeat}: copied program differs from source." >&2
      exit 2
    fi
  else
    cp "${SOURCE_PROGRAM}" "${copied_program}"
  fi
  build_command "${output_dir}" inference_cmd
  (
    echo "${repeat} started: $(date -Is)"
    echo "Log file: ${output_dir}/run.log"
    echo "Output directory: ${output_dir}"
    echo "Workers: ${NUM_WORKERS}"
    "${inference_cmd[@]}"
    echo "${repeat} finished: $(date -Is)"
  ) > >(tee -a "${output_dir}/run.log") 2>&1 &
  pids+=("$!")
  output_dirs+=("${output_dir}")
done

failures=0
for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then
    echo "Completed: ${output_dirs[index]}"
  else
    echo "Failed: ${output_dirs[index]} (see run.log)" >&2
    failures=$((failures + 1))
  fi
done
trap - EXIT INT TERM

if [ "${failures}" -ne 0 ]; then
  echo "${failures} of ${#REPEATS[@]} inference repeats failed." >&2
  exit 1
fi
echo "Both seed2 candidate2 inference repeats completed."
