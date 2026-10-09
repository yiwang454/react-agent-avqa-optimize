#!/usr/bin/env bash
set -euo pipefail

# Run B0-B7 as one parallel group, then run the three groups sequentially.
# Each underlying launcher keeps its normal per-question checkpoint/resume behavior.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_selectedVal125_repeats}"
INFERENCE_NUM_THREADS_PER_RUN="${INFERENCE_NUM_THREADS_PER_RUN:-1}"
INFERENCE_BATCH_SIZE_PER_RUN="${INFERENCE_BATCH_SIZE_PER_RUN:-${INFERENCE_NUM_THREADS_PER_RUN}}"

ABLATIONS=(b0 b1 b2 b3 b4 b5 b6 b7)
REPEATS=(repeat1 repeat2 repeat3)
child_pids=()
child_ids=()

if ! [[ "${INFERENCE_NUM_THREADS_PER_RUN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INFERENCE_NUM_THREADS_PER_RUN must be a positive integer." >&2
  exit 2
fi
if ! [[ "${INFERENCE_BATCH_SIZE_PER_RUN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INFERENCE_BATCH_SIZE_PER_RUN must be a positive integer." >&2
  exit 2
fi

cleanup_children() {
  local pid
  local active_pids=()
  trap - INT TERM EXIT
  for pid in "${child_pids[@]}"; do
    if [ -n "${pid}" ]; then
      active_pids+=("${pid}")
    fi
  done
  if [ "${#active_pids[@]}" -gt 0 ]; then
    kill -TERM "${active_pids[@]}" 2>/dev/null || true
    wait "${active_pids[@]}" 2>/dev/null || true
  fi
}

run_repeat() {
  local repeat_name="$1"
  shift
  local repeat_root="${OUTPUT_BASE}/${repeat_name}"
  local ablation_id launcher output_dir launcher_log
  local index status failed=0

  mkdir -p "${repeat_root}"
  child_pids=()
  child_ids=()

  echo "Starting ${repeat_name}: B0-B7 in parallel"
  echo "Output root: ${repeat_root}"
  echo "Per-run concurrency: threads=${INFERENCE_NUM_THREADS_PER_RUN}, batch=${INFERENCE_BATCH_SIZE_PER_RUN}"

  for ablation_id in "${ABLATIONS[@]}"; do
    launcher="${SCRIPT_DIR}/run_${ablation_id}_val125.sh"
    output_dir="${repeat_root}/${ablation_id}"
    launcher_log="${output_dir}/launcher.log"
    mkdir -p "${output_dir}"

    echo "Launching ${repeat_name}/${ablation_id^^}; log=${launcher_log}"
    OUTPUT_ROOT="${repeat_root}" \
      INFERENCE_NUM_THREADS="${INFERENCE_NUM_THREADS_PER_RUN}" \
      INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE_PER_RUN}" \
      "${launcher}" "$@" > "${launcher_log}" 2>&1 &
    child_pids+=("$!")
    child_ids+=("${ablation_id}")
  done

  for index in "${!child_pids[@]}"; do
    ablation_id="${child_ids[index]}"
    launcher_log="${repeat_root}/${ablation_id}/launcher.log"
    if wait "${child_pids[index]}"; then
      echo "Completed ${repeat_name}/${ablation_id^^}"
    else
      status=$?
      failed=1
      echo "FAILED ${repeat_name}/${ablation_id^^} (exit ${status}); log=${launcher_log}" >&2
      tail -n 40 "${launcher_log}" >&2 || true
    fi
    child_pids[index]=""
  done

  child_pids=()
  child_ids=()
  if [ "${failed}" -ne 0 ]; then
    echo "Stopping before the next repeat because ${repeat_name} had failures." >&2
    return 1
  fi
  echo "Finished ${repeat_name}"
}

trap 'cleanup_children; exit 130' INT
trap 'cleanup_children; exit 143' TERM
trap 'cleanup_children' EXIT

for repeat_name in "${REPEATS[@]}"; do
  run_repeat "${repeat_name}" "$@"
done

trap - INT TERM EXIT
echo "Finished repeat1, repeat2, and repeat3."
