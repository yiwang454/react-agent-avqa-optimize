# 1) 新建 conda 环境（推荐 Python 3.11）
CONDA_ENVS_DIR=/mnt/ceph_rbd/applications/anaconda3/envs
ENV_PREFIX="${CONDA_ENVS_DIR}/react-avqa"

conda init
# 2) 激活环境
conda activate "${ENV_PREFIX}"

#  进入项目并安装依赖
cd /mnt/ceph_rbd/workspace/avqa_project/general_scripts/react-agent-avqa

# Network/proxy for uv/pip package index access?

# Do not overwrite existing local secrets/config.
if [ ! -f .env ]; then
  cp .env.example .env
fi

# Load project environment variables before running.
set -a
source .env
set +a

# Ensure uv is installed before usage.
if ! command -v uv >/dev/null 2>&1; then
  python -m pip install -U uv
fi

# Ensure requests is installed before usage.
if ! python -c "import requests" >/dev/null 2>&1; then
  python -m pip install -U requests
fi

# uv run langgraph dev
# # Safari 场景可用：
# # uv run langgraph dev --tunnel

# Print related tokens to verify env loading.
echo "DEEPSEEK_TOKEN=${DEEPSEEK_TOKEN}"
echo "PLANNER_API_KEY=${PLANNER_API_KEY}"
echo "QWEN_API_KEY=${QWEN_API_KEY}"
echo "GEMINI_API_KEY=${GEMINI_API_KEY}"

echo "Running batch runner..."
uv run --no-sync python -m react_agent.batch_runner \
  --input-jsonl /mnt/ceph_rbd/data/avqa_project/worldsense/worldsense_test_cut.jsonl \
  --audio-caption-dir /mnt/ceph_rbd/data/avqa_project/worldsense/worldsense_captioner_instruct_trial/ \
  --output-jsonl /mnt/ceph_rbd/data/avqa_project/worldsense/worldsense_test_react_gemini2/output_test.jsonl \
  --output-dir /mnt/ceph_rbd/data/avqa_project/worldsense/worldsense_test_react_gemini2 \
  --perception-model "${PERCEPTION_MODEL:-gemini}" \
  --max-turns 4 \
  --recursion-limit 20 \
  --concurrency 1 \
  --debug