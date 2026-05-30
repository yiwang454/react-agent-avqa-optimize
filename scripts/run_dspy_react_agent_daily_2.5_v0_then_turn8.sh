#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_dspy_react_agent_daily_controltool_sanity.sh"

ENV_FILES=(
  "${SCRIPT_DIR}/env_files/.env_dspy_react_agent_daily_2.5_v0"
  "${SCRIPT_DIR}/env_files/.env_dspy_react_agent_daily_2.5_v0_turn8"
)

if [ ! -x "${RUNNER}" ]; then
  echo "Missing or non-executable runner: ${RUNNER}" >&2
  exit 1
fi

for env_file in "${ENV_FILES[@]}"; do
  if [ ! -f "${env_file}" ]; then
    echo "Missing env file: ${env_file}" >&2
    exit 1
  fi
done

for env_file in "${ENV_FILES[@]}"; do
  echo "============================================================"
  echo "Running DSPy ReAct with env: ${env_file}"
  echo "============================================================"
  ENV_FILE="${env_file}" "${RUNNER}" "$@"
done
