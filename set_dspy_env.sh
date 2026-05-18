#!/usr/bin/env bash
set -euo pipefail

# Install environment for DSPy-based ReAct implementation on local ceph workspace.
PROJECT_ROOT="/mnt/ceph_rbd/workspace/avqa_project/general_scripts/react-agent-avqa"
CONDA_ENVS_DIR="/mnt/ceph_rbd/applications/anaconda3/envs"
ENV_PREFIX="${CONDA_ENVS_DIR}/react-avqa-dspy"
PYTHON_VERSION="3.11"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH. Please install/initialize conda first."
  exit 1
fi

eval "$(conda shell.bash hook)"

create_env() {
  conda create -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" -y
}

if [ -d "${ENV_PREFIX}" ] && [ ! -x "${ENV_PREFIX}/bin/python" ]; then
  echo "Existing ENV_PREFIX looks incomplete: ${ENV_PREFIX}" >&2
  echo "Remove it and rerun this script:" >&2
  echo "  rm -rf ${ENV_PREFIX}" >&2
  exit 1
fi

if [ ! -d "${ENV_PREFIX}" ]; then
  if ! create_env; then
    echo "conda create failed; cleaning package cache and retrying once..." >&2
    conda clean --packages --tarballs --index-cache -y
    create_env
  fi
fi

conda activate "${ENV_PREFIX}"

python -m pip install -U pip setuptools wheel
python -m pip install -U uv

cd "${PROJECT_ROOT}"

# Keep local secrets/config if .env already exists.
if [ ! -f .env ]; then
  cp .env.example .env
fi

# Install project deps first, then DSPy-specific deps.
uv sync

# DSPy depends on LiteLLM; keep LiteLLM pinned because newer releases are not allowed here.
uv pip install -U dspy-ai requests python-dotenv openai pyyaml "litellm==1.82.6"
python - <<'PY_CHECK_LITELLM'
from importlib.metadata import version

expected = "1.82.6"
installed = version("litellm")
if installed != expected:
    raise SystemExit(f"LiteLLM version must be exactly {expected}; installed {installed}")
print(f"LiteLLM pinned version verified: {installed}")
PY_CHECK_LITELLM

echo "DSPy ReAct environment is ready."
echo "Activate with:"
echo "  conda activate ${ENV_PREFIX}"
echo "Run daily DSPy ReAct with:"
echo "  cd ${PROJECT_ROOT}"
echo "  scripts/run_dspy_react_agent_daily.sh"
echo "Or inspect the DSPy entrypoint with:"
echo "  uv run --no-sync python DSPy/avqa_dspy_impl.py --help"
