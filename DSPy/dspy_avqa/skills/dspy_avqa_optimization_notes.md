# DSPy AVQA Optimization Notes

本文档概括本轮围绕 `DSPy/dspy_avqa` 的 COPRO/SIMBA optimizer、DSPy LM backend、trajectory/signature 记录、cold env 与 DeepSeek thinking response 的问题和解决方案。

## 背景

已有的 DSPy AVQA ReAct baseline 可以通过类似下面的脚本跑通：

- `scripts/run_dspy_react_agent_daily_controltool_sanity.sh`
- `DSPy/avqa_dspy_impl.py`
- `DSPy/dspy_avqa/runner.py`

本轮目标是把 `DSPy/dspy_avqa/optimize.py` 里的 DSPy optimizer 跑在 Daily Omni 数据上，并能保存可复盘的 prompt/signature search、final trajectories，以及支持 DeepSeek OpenAI-compatible backend。

主要数据：

- Input JSONL: `/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selected250.jsonl`
- Audio caption dir: `/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_captioner_instruct/`

## 相关脚本

主要 COPRO 脚本：

- `scripts/run_dspy_react_agent_daily_copro.sh`
- `scripts/run_dspy_react_agent_daily_copro_sets.sh`

`run_dspy_react_agent_daily_copro_sets.sh` 当前用于串行跑两组设置：

- pilot
- recommended

当前 cold 版本 source 的 env 文件是：

- `scripts/env_files/.env_dspy_react_agent_daily_2 copy.5_v0_corpo_cold`

并输出到 cold 专用目录，避免覆盖旧结果：

- `/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_gemini2.5_copro_cold_pilot`
- `/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_gemini2.5_copro_cold_recommended`

SIMBA 脚本：

- `scripts/run_dspy_react_agent_daily_simba.sh`

## Hyperparameter 总览

优化相关参数可以分成四层：

- Dataset scope: 控制参与 optimizer search 的样本数量，例如 `TRAIN_LIMIT` / `COPRO_TRAIN_LIMIT` / `SIMBA_TRAIN_LIMIT`。
- Search budget: 控制 optimizer 搜索宽度、深度、步数，例如 COPRO 的 `BREADTH/DEPTH`，SIMBA 的 `BSIZE/NUM_CANDIDATES/MAX_STEPS`。
- Runtime budget: 控制每个 AVQA program rollout 内部最多走多少 ReAct turn，例如 `MAX_TURNS`。
- Sampling behavior: 控制 planner/perception/candidate generation 的采样随机性，例如 `PLANNER_TEMPERATURE`、Gemini YAML 的 `sampling_params.temperature`、`COPRO_INIT_TEMPERATURE`、SIMBA 的两个 temperature。

记号：

```text
T = train_limit，即 optimizer search 用的样本数
B = COPRO_BREADTH
D = COPRO_DEPTH
S = SIMBA_MAX_STEPS
C = SIMBA_NUM_CANDIDATES
K = SIMBA_BSIZE
M = MAX_TURNS
```

`T` 当前是 deterministic slicing：代码使用 `selected[:T]`，不是 random sample。每个 candidate/step 用的是同一份 selected trainset 或 SIMBA shuffle 后的 mini-batch 序列，不会每次 candidate evaluate 随机换 T 个样本。

## COPRO 参数详解

常用参数：

- `COPRO_TRAIN_LIMIT` / `--train-limit`: 参与 COPRO search 的样本数 T。当前逻辑取 input JSONL 的前 T 条。
- `COPRO_BREADTH` / `--copro-breadth`: 每轮候选 prompt/signature 的数量 B。初始轮会把原始 prompt 加入候选；后续轮基于已评估结果继续生成新候选。
- `COPRO_DEPTH` / `--copro-depth`: 迭代轮数 D。
- `COPRO_INIT_TEMPERATURE` / `--copro-init-temperature`: COPRO 调用 prompt model 生成候选 instruction/prefix 时的 temperature。

COPRO 在当前 AVQA program 中主要优化 `action_planner` predictor 的 signature：

- signature instructions
- 最后一个 output field 的 prefix

它不会自动优化任意 runtime 字符串，例如 `system_prompt`、`action_schema`、tool 实现逻辑或 perception prompt，除非这些内容被设计进 DSPy signature 或 optimizer 可见字段。

### COPRO 调用量估算

在当前单 predictor 程序下，COPRO 的 candidate evaluate 数量大致是：

```text
candidate_evaluate_calls = B * D
```

每次 candidate evaluate 会在 T 个样本上跑一次 AVQA program，所以样本级 program rollout 约为：

```text
search_program_rollouts = T * B * D
```

如果开启 `--trajectory-jsonl`，optimization 完成后还会用 best compiled program 重新跑一遍 selected trainset：

```text
final_trajectory_rollouts = T
```

总 program rollout 估算：

```text
total_program_rollouts ~= T * B * D + T
```

每个 program rollout 内部还会产生若干 planner/perception API call。粗略上限受 `MAX_TURNS=M` 影响：

```text
DeepSeek planner calls per rollout: 通常 1..M，可能包含 final fallback
Gemini/Qwen perception calls per rollout: 0..M，取决于 planner 是否调用 tool
```

因此真实外部 API call 数会显著高于 program rollout 数。

### COPRO 建议配置

当前建议保留四档：

| Preset | T | B | D | COPRO_INIT_TEMPERATURE | Candidate evaluate calls | Program rollouts with final trajectory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 1 | 1 | 1 | 1.0 | 1 | 2 |
| pilot | 16 | 2 | 2 | 1.0 | 4 | 80 |
| recommended | 64 | 5 | 3 | 1.0 | 15 | 1024 |
| full / expensive | 250 | 10 | 3 | 1.0 | 30 | 7750 |

解释：

- smoke 只验证 pipeline、文件输出、API 参数是否能跑通，不适合看 optimizer 质量。
- pilot 用来快速观察 COPRO 是否能找到比 initial signature 更好的候选。
- recommended 是当前更平衡的正式实验设置。
- full / expensive 更接近完整搜索，但成本高，且如果 backend/perception 非确定性高，额外样本不一定线性提升可信度。

当前 `run_dspy_react_agent_daily_copro_sets.sh` 串行跑：

```text
pilot:       T=16, B=2, D=2, COPRO_INIT_TEMPERATURE=1.0
recommended: T=64, B=5, D=3, COPRO_INIT_TEMPERATURE=1.0
```

### COPRO_INIT_TEMPERATURE 的实际影响

`COPRO_INIT_TEMPERATURE` 控制 COPRO 生成候选 instruction/prefix 的随机性。它不直接控制 final compiled program 在 `output_test.jsonl` rerun 时的 answer sampling。

但有两个 caveat：

- 如果 DeepSeek thinking mode 下服务端忽略 `temperature/top_p`，那么这个参数对候选生成的实际影响可能有限。
- 当前 cold env 中未显式设置 `PLANNER_TEMPERATURE` / `PLANNER_TOP_P`，native LM wrapper 会移除这些 sampling 参数，包括 COPRO per-call 传入的 `temperature=1.0`。也就是说 cold run 里 DeepSeek candidate generation 会尽量不带 sampling 参数，交给服务端默认行为。

## SIMBA 参数详解

常用参数：

- `SIMBA_TRAIN_LIMIT` / `--train-limit`: 参与 SIMBA optimization 的样本数 T。
- `SIMBA_BSIZE` / `--simba-bsize`: 每个 optimization step 的 mini-batch size K。必须满足 `T >= K`，否则 SIMBA 会 assert fail。
- `SIMBA_NUM_CANDIDATES` / `--simba-num-candidates`: 每步用于 trajectory sampling 的模型/候选数量 C，并影响每步最多生成的新 candidate program 数。
- `SIMBA_MAX_STEPS` / `--simba-max-steps`: optimization steps 数 S。
- `SIMBA_MAX_DEMOS` / `--simba-max-demos`: 每个 predictor 最多保留/构造 demos 的目标上限；`0` 会禁用 append-demo strategy，只保留 rule strategy。
- `SIMBA_NUM_THREADS` / `--simba-num-threads`: DSPy parallel runner 的线程数。空值使用 DSPy 默认。
- `SIMBA_SEED` / `--simba-seed`: 控制 trainset shuffle、program sampling、strategy choice 等随机过程。
- `SIMBA_TEMPERATURE_FOR_SAMPLING` / `--simba-temperature-for-sampling`: 从 program pool 中 softmax 选择 program 来采样 trajectory 的温度。DSPy 默认 `0.2`。
- `SIMBA_TEMPERATURE_FOR_CANDIDATES` / `--simba-temperature-for-candidates`: 选择 source program 来构造新 candidate 的温度。DSPy 默认 `0.2`。

当前 `scripts/run_dspy_react_agent_daily_simba.sh` 默认：

```text
SIMBA_TRAIN_LIMIT=250
SIMBA_BSIZE=32
SIMBA_NUM_CANDIDATES=6
SIMBA_MAX_STEPS=8
SIMBA_MAX_DEMOS=4
SIMBA_NUM_THREADS=<dspy-default>
SIMBA_SEED=0
SIMBA_TEMPERATURE_FOR_SAMPLING=<dspy-default 0.2>
SIMBA_TEMPERATURE_FOR_CANDIDATES=<dspy-default 0.2>
```

### SIMBA 调用量估算

按当前安装版 DSPy SIMBA 实现，每个 step 主要包含两段 program execution：

```text
trajectory sampling: K * C
candidate evaluation: up to K * (C + 1)
```

每 step program rollout 约为：

```text
K * (2C + 1)
```

S steps 后，还会做一次 full-trainset validation，候选 program 数最多约为 `C + 1`：

```text
validation_rollouts <= T * (C + 1)
```

因此 SIMBA optimization 的 program rollout 粗略上限：

```text
simba_program_rollouts <= S * K * (2C + 1) + T * (C + 1)
```

如果额外开启 final trajectory dump，再加：

```text
+ T
```

注意：candidate strategy 可能失败，实际 candidate evaluation 会少于上限；但每个 rollout 内仍可能包含多次 planner/perception API call。

### SIMBA 建议配置

| Preset | T | K / bsize | C / num_candidates | S / max_steps | max_demos | Estimated optimization rollouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 8 | 2 | 2 | 1 | 1 | <= 34 |
| pilot | 32 | 8 | 3 | 2 | 2 | <= 240 |
| recommended | 64 | 16 | 4 | 4 | 4 | <= 896 |
| full / script default | 250 | 32 | 6 | 8 | 4 | <= 5078 |

计算方式：

```text
smoke:      1*2*(2*2+1) + 8*(2+1) = 10 + 24 = 34
pilot:      2*8*(2*3+1) + 32*(3+1) = 112 + 128 = 240
recommended: 4*16*(2*4+1) + 64*(4+1) = 576 + 320 = 896
full:       8*32*(2*6+1) + 250*(6+1) = 3328 + 1750 = 5078
```

推荐用法：

- smoke: 验证 SIMBA 路径、assert、输出能否正常完成。
- pilot: 初步看 SIMBA 是否能产生 candidate program、trial logs 是否合理。
- recommended: 比较适合和 COPRO recommended 做成本接近的对照。
- full / script default: 成本明显更高，适合确认 recommended 结果稳定后再跑。

## Runtime / backend hyperparameters

### Dataset 与 turn budget

常用运行参数：

- `INPUT_JSONL`: optimizer 读取的数据文件。
- `AUDIO_CAPTION_DIR`: Daily Omni caption/audio side information 目录。
- `MAX_TURNS`: 每个 AVQA ReAct rollout 的最大 planner/tool turn 数，当前 COPRO sets 用 `MAX_TURNS=4`。
- `PERCEPTION_MODEL`: 当前 cold COPRO env 使用 `gemini`。
- `PERCEPTION_CONFIG_YAML`: perception backend 的模型、temperature、top_p、max_tokens 等配置。
- `PROMPT_YAML`: DSPy AVQA planner/perception/signature prompt 配置。
- `DSPY_AVQA_ALLOWED_TOOLS`: 可选工具集合；如果不显式传，会从 prompt YAML 或默认工具集合解析。

### DeepSeek planner sampling

历史默认来自 `AVQARuntimeContext`：

```text
PLANNER_TEMPERATURE default = 0.2
PLANNER_TOP_P default = 0.6
PLANNER_TOP_K default = 20
PLANNER_OUTPUT_SEQ_LEN default = 2048
```

当前 cold env 设置：

```text
PLANNER_OUTPUT_SEQ_LEN=16384
# PLANNER_TEMPERATURE is commented out
# PLANNER_TOP_P is commented out
# PLANNER_TOP_K is commented out
```

当前 native DSPy LM wrapper 的行为：

- 如果 env 未显式设置 `PLANNER_TEMPERATURE`，不向 DeepSeek 请求传 `temperature`。
- 如果 env 未显式设置 `PLANNER_TOP_P`，不向 DeepSeek 请求传 `top_p`。
- 即使 COPRO/SIMBA per-call 传入 `temperature`，cold env 下 wrapper 也会移除它。
- `max_tokens`、`api_base`、`api_key`、`timeout` 仍会传。

### Gemini perception sampling

Gemini/Qwen perception sampling 由 `PERCEPTION_CONFIG_YAML` 写入环境变量，再由 `tools.py` 使用。

普通 Gemini YAML 曾使用：

```yaml
sampling_params:
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  max_tokens: 4096
```

cold env 当前指向：

```text
/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold.yaml
```

应在该 YAML 中确认 cold sampling，例如 temperature 是否为 `0.0`。如果 COPRO evaluate 和 final trajectory 仍不一致，优先检查 perception YAML 是否真的被脚本 source 并加载。

### DeepSeek thinking mode

当前没有显式传：

```text
extra_body={"thinking": {"type": "enabled"}}
reasoning_effort="high/max"
```

如果 DeepSeek 服务端默认 thinking enabled，则 planner 可能处于 thinking mode。当前 wrapper 会捕获 response 中的 `reasoning_content` 到 trajectory/debug，但不会把 thinking 内容混入 `planner_raw`。

## 输出文件含义

COPRO run 通常输出：

- `initial_copro_program.json`
- `compiled_copro.json`
- `compiled_copro_metadata.json`
- `compiled_copro_signature_search.json`
- `output_test.jsonl` 或 `compiled_copro_train_trajectories.jsonl`

含义如下。

### initial_copro_program.json

optimization 之前的原始 DSPy program。用于和 compiled program 对比。

### compiled_copro.json

COPRO search 后选出的 best program。里面的 signature 是训练集 metric 分数最高的 candidate 对应的 signature。

### compiled_copro_signature_search.json

用于复盘 signature search。包含：

- `total_evaluate_calls`: optimizer 实际评估了多少个 candidate program。
- `initial_program_signatures`: optimize 起点 signature。
- `best_program_signatures`: 最终 best compiled program 的 signature。
- `candidate_programs`: 所有被评估过的 candidate，按 score 从高到低排序。

注意：`candidate_index` 是排序后的 rank，不是生成时间顺序。`candidate_index=0` 是当前记录里的最高分 candidate。

### output_test.jsonl / compiled_copro_train_trajectories.jsonl

这是 optimization 完成后，用最终 `compiled` program 重新跑 selected trainset 得到的 trajectory。它不是 held-out test set，也不是 COPRO evaluate 过程中的原始日志。

因此：

```text
output_test.jsonl 使用的是 compiled_copro.json 里的 best signature。
```

但它是 fresh rerun。如果 planner/perception backend 有随机性，final rerun score 可能和 COPRO evaluate 时 best candidate 的 score 不一致。

## 为什么 COPRO evaluate score 和 final trajectory score 不一致

主要原因：

1. COPRO evaluate 阶段和 final trajectory 阶段是两次独立调用。
2. `output_test.jsonl` 不是复用 evaluate 阶段的 prediction，而是用 best compiled program 重新跑。
3. DeepSeek planner、Gemini perception、服务端自身都可能有非确定性。
4. 如果 Gemini temperature 较高，perception observation 会变，planner 后续 action/final answer 也可能变。
5. 即使 planner temperature 设低，DeepSeek thinking mode 下 sampling 参数可能被服务端忽略。

## DeepSeek OpenAI-compatible backend

采用 route B：把 DeepSeek endpoint 当 OpenAI-compatible backend 给 DSPy native `dspy.LM` 使用。

模型名处理：

```text
deepseek-v4-pro -> openai/deepseek-v4-pro
```

因为 LiteLLM/OpenAI-compatible endpoint 曾报错：

```text
Invalid n value (currently only n = 1 is supported)
```

所以加入了 `SerialNOpenAICompatibleLM`：

- 当 DSPy/COPRO 请求 `n=1`，正常调用。
- 当 DSPy/COPRO 请求 `n>1`，拆成多次串行 `n=1` 调用，再合并 choices。

这让 COPRO breadth 可以继续工作，但成本和时间会按候选数线性增加。

## DeepSeek thinking response

DeepSeek thinking mode 可能在 response 中返回类似：

```text
message.reasoning_content
```

当前 `SerialNOpenAICompatibleLM` 会捕获：

- `response_text`: 最终 content，用于 DSPy adapter 解析 action JSON。
- `reasoning_content`: DeepSeek thinking 内容，仅用于 trajectory/debug。
- `choices[*].reasoning_content`: 多 choice 情况下每个 choice 的 thinking 内容。

trajectory 中可在类似下面位置看到：

```text
turn_trace[*].planner_calls.action_decision[*].reasoning_content
turn_trace[*].planner_calls.final_fallback[*].reasoning_content
```

重要原则：thinking 内容不混入 `planner_raw`，避免破坏 action JSON parsing。

## 常见坑

### 1. Train limit 不是随机采样

当前 `TRAIN_LIMIT=T` 使用前 T 个 sample：

```python
selected = selected[: args.train_limit]
```

每个 candidate 都评估同一批 T 个 sample，不会每步换随机 batch。

### 2. output_test.jsonl 名字容易误导

它通常是 trainset final trajectories，不是 held-out test。更清楚的名字是：

```text
compiled_copro_train_trajectories.jsonl
```

### 3. COPRO candidate_index 不是时间顺序

`compiled_copro_signature_search.json` 里的 `candidate_programs` 是按 score 排序后的列表。

### 4. env 文件名有空格时必须 quote

当前 cold env 文件名包含空格：

```text
.env_dspy_react_agent_daily_2 copy.5_v0_corpo_cold
```

脚本里必须用：

```bash
source "${ENV_FILE}"
```

### 5. bash env assignment 不能有空格

错误：

```bash
PLANNER_TEMPERATURE = 0.0
```

正确：

```bash
PLANNER_TEMPERATURE=0.0
```

### 6. scripts 目录可能被 gitignore

`git status` 可能不显示 scripts 下的改动。复盘脚本内容时直接打开文件确认。

## 当前验证过的检查

常用 sanity checks：

```bash
python -m py_compile DSPy/dspy_avqa/context.py DSPy/dspy_avqa/deepseek_dspy_lm.py DSPy/dspy_avqa/optimize.py DSPy/avqa_dspy_optimize.py
bash -n scripts/run_dspy_react_agent_daily_copro_sets.sh
bash -n scripts/run_dspy_react_agent_daily_copro.sh
bash -n "scripts/env_files/.env_dspy_react_agent_daily_2 copy.5_v0_corpo_cold"
```

本轮还做过本地 monkeypatch tests，确认：

- `n>1` 会拆成多次 `n=1`。
- DeepSeek `reasoning_content` 会进入 `planner_calls`。
- cold env 下 native LM constructor 不携带 `temperature/top_p`。
