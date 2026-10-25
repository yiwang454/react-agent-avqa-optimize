#!/usr/bin/env bash
set -euo pipefail

# Rerun the five Stage B v2 configurations selected by the leave-one-run-out
# review. All five launch concurrently and write to new directories, leaving
# the original repeat1-repeat3 outputs untouched.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_outlier_reruns}"
INFERENCE_NUM_THREADS_PER_RUN="${INFERENCE_NUM_THREADS_PER_RUN:-3}"
INFERENCE_BATCH_SIZE_PER_RUN="${INFERENCE_BATCH_SIZE_PER_RUN:-${INFERENCE_NUM_THREADS_PER_RUN}}"

# Format: launcher ID:the original run this rerun is intended to replace.
RERUN_SPECS=(
  "b0:repeat3"
  "b1:repeat2"
  "b2:repeat2"
  "b3:repeat1"
  "b5_flex:repeat2"
)

child_pids=()
child_labels=()
child_logs=()

if [ "$#" -ne 0 ]; then
  echo "Use environment variables for this controlled launcher; positional runner arguments are not accepted." >&2
  exit 2
fi

if ! [[ "${INFERENCE_NUM_THREADS_PER_RUN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INFERENCE_NUM_THREADS_PER_RUN must be a positive integer." >&2
  exit 2
fi
if ! [[ "${INFERENCE_BATCH_SIZE_PER_RUN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INFERENCE_BATCH_SIZE_PER_RUN must be a positive integer." >&2
  exit 2
fi

cleanup_children() {
  local pid attempt any_active
  local active_pids=()
  trap - INT TERM EXIT
  for pid in "${child_pids[@]}"; do
    if [ -n "${pid}" ]; then
      active_pids+=("${pid}")
    fi
  done
  if [ "${#active_pids[@]}" -gt 0 ]; then
    for pid in "${active_pids[@]}"; do
      kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    for attempt in {1..20}; do
      any_active=0
      for pid in "${active_pids[@]}"; do
        if kill -0 -- "-${pid}" 2>/dev/null; then
          any_active=1
          break
        fi
      done
      if [ "${any_active}" -eq 0 ]; then
        break
      fi
      sleep 0.25
    done
    for pid in "${active_pids[@]}"; do
      if kill -0 -- "-${pid}" 2>/dev/null; then
        kill -KILL -- "-${pid}" 2>/dev/null || true
      fi
    done
    wait "${active_pids[@]}" 2>/dev/null || true
  fi
}

trap 'cleanup_children; exit 130' INT
trap 'cleanup_children; exit 143' TERM
trap 'cleanup_children' EXIT

echo "Starting five Stage B v2 outlier reruns in parallel"
echo "Output root: ${OUTPUT_BASE}"
echo "Per-run concurrency: threads=${INFERENCE_NUM_THREADS_PER_RUN}, batch=${INFERENCE_BATCH_SIZE_PER_RUN}"

for spec in "${RERUN_SPECS[@]}"; do
  ablation_id="${spec%%:*}"
  source_repeat="${spec##*:}"
  display_id="${ablation_id/_/-}"
  label="${display_id}-${source_repeat}-rerun"
  launcher="${SCRIPT_DIR}/run_${ablation_id}_val125.sh"
  output_dir="${OUTPUT_BASE}/${label}"
  launcher_log="${output_dir}/launcher.log"
  mkdir -p "${output_dir}"

  echo "Launching ${display_id^^} replacement for ${source_repeat}; log=${launcher_log}"
  setsid --wait env \
    OUTPUT_ROOT="${OUTPUT_BASE}" \
    OUTPUT_DIR_OVERRIDE="${output_dir}" \
    INFERENCE_NUM_THREADS="${INFERENCE_NUM_THREADS_PER_RUN}" \
    INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE_PER_RUN}" \
    "${launcher}" > "${launcher_log}" 2>&1 &
  child_pids+=("$!")
  child_labels+=("${label}")
  child_logs+=("${launcher_log}")
done

failed=0
for index in "${!child_pids[@]}"; do
  if wait "${child_pids[index]}"; then
    echo "Completed ${child_labels[index]}"
  else
    status=$?
    failed=1
    echo "FAILED ${child_labels[index]} (exit ${status}); log=${child_logs[index]}" >&2
    tail -n 40 "${child_logs[index]}" >&2 || true
  fi
  child_pids[index]=""
done

child_pids=()
child_labels=()
child_logs=()
trap - INT TERM EXIT

if [ "${failed}" -ne 0 ]; then
  echo "One or more outlier reruns failed." >&2
  exit 1
fi
echo "Finished all five Stage B v2 outlier reruns."
