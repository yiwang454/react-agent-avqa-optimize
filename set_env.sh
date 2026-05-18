# python -m pip install --user uv

# 1) 新建 conda 环境（推荐 Python 3.11）
CONDA_ENVS_DIR=/mnt/ceph_rbd/applications/anaconda3/envs
ENV_PREFIX="${CONDA_ENVS_DIR}/react-avqa"
conda create -p "${ENV_PREFIX}" python=3.11 -y

conda init
# 2) 激活环境
conda activate "${ENV_PREFIX}"

# 3) 在环境里安装 uv（可选但推荐）
python -m pip install -U uv

# 4) 进入项目并安装依赖
cd /mnt/ceph_rbd/workspace/avqa_project/general_scripts/react-agent-avqa

# 安装依赖（包含 dev 组里 langgraph-cli[inmem]）
uv sync

cp .env.example .env

# uv run langgraph dev
# # Safari 场景可用：
# # uv run langgraph dev --tunnel

# uv run python -m react_agent.batch_runner \
#   --input-jsonl /apdcephfs_cq10/share_1603164/user/yiizoewang/data/worldsense/worldsense_test_cut.jsonl \
#   --audio-caption-dir /apdcephfs_cq10/share_1603164/user/yiizoewang/data/worldsense/worldsense_captioner_instruct_trial/ \
#   --output-jsonl /apdcephfs_cq10/share_1603164/user/yiizoewang/data/worldsense/worldsense_test_cut_per_question/output_test.json \
#   --output-dir /apdcephfs_cq10/share_1603164/user/yiizoewang/data/worldsense/worldsense_test_cut_per_question \
#   --max-turns 4 \
#   --recursion-limit 20 \
#   --concurrency 1 \
#   --debug