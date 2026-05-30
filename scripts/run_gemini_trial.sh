# 1) 新建 conda 环境（推荐 Python 3.11）
CONDA_ENVS_DIR=/mnt/ceph_rbd/applications/anaconda3/envs
ENV_PREFIX="${CONDA_ENVS_DIR}/react-avqa"

# 2) 激活环境

conda init; conda activate "${ENV_PREFIX}"

# Ensure requests is installed before usage.
if ! python -c "import requests" >/dev/null 2>&1; then
  python -m pip install -U requests
fi

# Network/proxy for uv/pip package index access?

export GATEWAY_URL_THINK=http://29.232.225.71:8000
# python /mnt/ceph_rbd/workspace/avqa_project/general_scripts/gemini_api/test.py
python /mnt/ceph_rbd/workspace/avqa_project/general_scripts/gemini_api/qwen_api.py