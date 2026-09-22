#!/usr/bin/env bash
# Run full-v3 inference with GEPA candidate 7 from the completed C1 seed18 search.
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

PYTHON_BIN="${PYTHON_BIN:-/mnt/ceph_rbd/applications/anaconda3/envs/react-avqa-dspy/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/ceph_rbd/data/avqa_project/daily_omni}"
SOURCE_RUN="${SOURCE_RUN:-${DATA_ROOT}/daily_omni_dspy_free_react_gpt4_1_gemini_first_cached_then_live_caption_gepa_planner_maxturns4_seed18_calls2500}"
CANDIDATE_INDEX=7
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-${SOURCE_RUN}_candidate${CANDIDATE_INDEX}_full_v3}"
DERIVED_PROGRAM="${OUTPUT_DIR}/compiled_gepa_candidate${CANDIDATE_INDEX}.json"

INPUT_JSONL="${INPUT_JSONL:-${DATA_ROOT}/daily_omni_cuts_v3.jsonl}"
TRAINSET_JSONL="${TRAINSET_JSONL:-${DATA_ROOT}/daily_omni_cuts_selectedTrain125.jsonl}"
VALSET_JSONL="${VALSET_JSONL:-${DATA_ROOT}/daily_omni_cuts_selectedVal125.jsonl}"
CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR:-${DATA_ROOT}/daily_omni_caption_cache_v8_gemini2.5flash_seed27}"
PLANNER_CONFIG_YAML="${PLANNER_CONFIG_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_gpt4_1_none.yaml}"
GEMINI_CONFIG_YAML="${GEMINI_CONFIG_YAML_OVERRIDE:-/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml}"
PROMPT_YAML="${PROMPT_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_free_react_caption_in_task.yaml}"
FINAL_EVAL_NUM_THREADS="${FINAL_EVAL_NUM_THREADS:-4}"
FINAL_EVAL_BATCH_SIZE="${FINAL_EVAL_BATCH_SIZE:-${FINAL_EVAL_NUM_THREADS}}"

for required in \
  "${PYTHON_BIN}" \
  "${SOURCE_RUN}/compiled_gepa.json" \
  "${SOURCE_RUN}/gepa_logs/gepa_state.bin" \
  "${SOURCE_RUN}/optimizer_logs/instructions/candidate_0007_planner_workflow_prompt.txt" \
  "${INPUT_JSONL}" "${TRAINSET_JSONL}" "${VALSET_JSONL}" \
  "${CAPTION_CACHE_DIR}/manifest.json" \
  "${PLANNER_CONFIG_YAML}" "${GEMINI_CONFIG_YAML}" "${PROMPT_YAML}"; do
  [ -f "${required}" ] || { echo "Required file not found: ${required}" >&2; exit 1; }
done

for expected in "${INPUT_JSONL}:1197" "${TRAINSET_JSONL}:125" "${VALSET_JSONL}:125"; do
  file="${expected%:*}"; expected_count="${expected##*:}"
  count="$(awk 'NF {n++} END {print n+0}' "${file}")"
  [ "${count}" = "${expected_count}" ] || {
    echo "Expected ${expected_count} rows in ${file}, found ${count}." >&2
    exit 2
  }
done

: "${ELM_API_KEY:?Set ELM_API_KEY for the GPT-4.1 planner.}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY for live Gemini perception.}"
export PLANNER_API_KEY="${ELM_API_KEY}"
unset DEEPSEEK_API_KEY DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE
export PERCEPTION_MODEL=gemini
export CAPTIONER_MODEL=gemini
export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception"
export GEMINI_API_BACKEND="${GEMINI_API_BACKEND:-legacy}"
export GEMINI_RESPONSE_ERROR_SENSITIVE=false

mkdir -p "${OUTPUT_DIR}"
PYTHONPATH="$PWD/DSPy" "${PYTHON_BIN}" - \
  "${SOURCE_RUN}/compiled_gepa.json" \
  "${SOURCE_RUN}/gepa_logs/gepa_state.bin" \
  "${SOURCE_RUN}/optimizer_logs/instructions/candidate_0007_planner_workflow_prompt.txt" \
  "${DERIVED_PROGRAM}" \
  "${OUTPUT_DIR}/candidate7_provenance.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import cloudpickle

source_program, state_path, prompt_path, output_program, provenance_path = map(Path, sys.argv[1:])
state = cloudpickle.load(state_path.open("rb"))
candidate_index = 7
candidate_prompt = state["program_candidates"][candidate_index]["prompt_component_0"]
persisted_prompt = prompt_path.read_text().rstrip("\n")
if candidate_prompt != persisted_prompt:
    raise SystemExit("Persisted candidate_0007 prompt does not match gepa_state.bin candidate 7.")

program = json.loads(source_program.read_text())
program["prompt_component_0"]["signature"]["instructions"] = candidate_prompt
output_program.write_text(json.dumps(program, ensure_ascii=False, indent=2) + "\n")
provenance = {
    "source_run": str(source_program.parent),
    "candidate_index": candidate_index,
    "candidate_prompt_sha256": hashlib.sha256(candidate_prompt.encode()).hexdigest(),
    "source_compiled_program": str(source_program),
    "source_gepa_state": str(state_path),
    "derived_program": str(output_program),
}
provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(provenance, indent=2))
PY

cmd=(
  "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
  --algorithm gepa
  --inference-only
  --input-jsonl "${INPUT_JSONL}"
  --trainset-jsonl "${TRAINSET_JSONL}"
  --valset-jsonl "${VALSET_JSONL}"
  --daily-omni-root "${DATA_ROOT}"
  --max-turns 4
  --planner-config-yaml "${PLANNER_CONFIG_YAML}"
  --gepa-reflection-config-yaml "${PLANNER_CONFIG_YAML}"
  --perception-model gemini
  --perception-config-yaml "${GEMINI_CONFIG_YAML}"
  --captioner-config-yaml "${GEMINI_CONFIG_YAML}"
  --prompt-yaml "${PROMPT_YAML}"
  --allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}"
  --caption-cache-dir "${CAPTION_CACHE_DIR}"
  --caption-cache-scope first_call_only
  --ignore-audio-caption-dir
  --gemini-api-backend "${GEMINI_API_BACKEND}"
  --signature-in-system-prompt
  --caption-placement task
  --output-program "${DERIVED_PROGRAM}"
  --metadata-json "${OUTPUT_DIR}/inference_only_metadata.json"
  --final-eval-output-jsonl "${OUTPUT_DIR}/output_test.jsonl"
  --final-eval-output-dir "${OUTPUT_DIR}"
  --final-eval-num-threads "${FINAL_EVAL_NUM_THREADS}"
  --final-eval-batch-size "${FINAL_EVAL_BATCH_SIZE}"
  --print-config
)

if [ "${DRY_RUN:-false}" = true ]; then
  printf 'Candidate 7 full-v3 command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

echo "Candidate 7 prompt: ${SOURCE_RUN}/optimizer_logs/instructions/candidate_0007_planner_workflow_prompt.txt"
echo "Output directory: ${OUTPUT_DIR}"
echo "Started: $(date -Is)"
"${cmd[@]}"
echo "Finished: $(date -Is)"
