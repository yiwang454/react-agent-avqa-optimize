# 1) 新建 conda 环境（推荐 Python 3.11）
CONDA_ENVS_DIR=/mnt/ceph_rbd/applications/anaconda3/envs
ENV_PREFIX="${CONDA_ENVS_DIR}/react-avqa-dspy"

conda init
# 2) 激活环境
conda activate "${ENV_PREFIX}"

# 进入项目目录
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

# # Ensure uv is installed before usage.
# if ! command -v uv >/dev/null 2>&1; then
#   python -m pip install -U uv
# fi

# # Ensure runtime deps for DSPy runner exist.
# if ! python -c "import dspy, requests" >/dev/null 2>&1; then
#   python -m pip install -U dspy-ai requests
# fi

# Print related tokens to verify env loading.
echo "DEEPSEEK_TOKEN=${DEEPSEEK_TOKEN}"
echo "PLANNER_API_KEY=${PLANNER_API_KEY}"
echo "QWEN_API_KEY=${QWEN_API_KEY}"
echo "PERCEPTION_MODEL=${PERCEPTION_MODEL:-qwen}"

echo "Running DSPy batch runner..."
export LITELLM_LOCAL_MODEL_COST_MAP=true; python DSPy/avqa_dspy_impl.py \
  --input-jsonl /mnt/ceph_rbd/data/avqa_project/worldsense/worldsense_test_cut.jsonl \
  --audio-caption-dir /mnt/ceph_rbd/data/avqa_project/worldsense/worldsense_captioner_instruct_trial/ \
  --output-jsonl /mnt/ceph_rbd/data/avqa_project/worldsense/worldsense_test_cut_per_question_audio_dspy/output_test.jsonl \
  --output-dir /mnt/ceph_rbd/data/avqa_project/worldsense/worldsense_test_cut_per_question_audio_dspy \
  --max-turns 4 \
  --perception-model "${PERCEPTION_MODEL:-qwen}" \
  --debug
