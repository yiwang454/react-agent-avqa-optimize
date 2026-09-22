#!/usr/bin/env bash
# Copy the complete C0/C1 experiment folders to neuralnoise_spark.
# A single rsync process copies all directories through the configured SSH jump host.
set -euo pipefail

LOCAL_ROOT="/mnt/ceph_rbd/data/avqa_project/daily_omni"
REMOTE="neuralnoise_spark"
REMOTE_ROOT="/home/yiwang/data/avqa_project/daily_omni"

RUN_DIRS=(
  "mllm_instruct_opt/daily_gemini2.5flash_single_mllm_legacy_thinking_store_qa_baseline_full_v3"
)

for run_name in "${RUN_DIRS[@]}"; do
  [ -d "${LOCAL_ROOT}/${run_name}" ] || { echo "Missing: ${LOCAL_ROOT}/${run_name}" >&2; exit 1; }
done

echo "Syncing C0 and C1 folders to ${REMOTE}:${REMOTE_ROOT}"
echo "Enter the jump-host and remote passwords once each if prompted."
rsync -avhP \
  "${LOCAL_ROOT}/${RUN_DIRS[0]}" \
  "${REMOTE}:${REMOTE_ROOT}/"

echo "Done: ${REMOTE}:${REMOTE_ROOT}"
