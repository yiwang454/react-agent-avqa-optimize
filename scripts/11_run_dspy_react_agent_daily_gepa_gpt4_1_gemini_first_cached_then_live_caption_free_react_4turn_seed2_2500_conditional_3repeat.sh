#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GEPA_SEED=2
exec "${SCRIPT_DIR}/run_c1_gepa_conditional_3repeat_common.sh" "$@"
