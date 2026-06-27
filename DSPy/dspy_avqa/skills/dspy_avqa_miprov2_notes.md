# DSPy AVQA MIPROv2 Optimization Notes

本文档对应脚本：

- `scripts/run_dspy_react_agent_daily_miprov2.sh`

参考 DSPy 文档：`https://dspy.ai/api/optimizers/MIPROv2/`。当前本地环境为 DSPy `3.2.1`，脚本按该版本的 `MIPROv2.__init__` / `compile` 参数写。

## 背景

MIPROv2 同时搜索两类 prompt 参数：

- instruction candidates：给 `action_planner` predictor 的 signature instruction 生成候选。
- few-shot demo candidates：从 trainset bootstrap / labeled examples 构造 demo 候选。

当前 AVQA 程序只有一个主要 predictor：`action_planner`。MIPROv2 不会直接优化 tool 实现、`system_prompt`、perception prompt 或 runtime 控制逻辑，除非这些内容被显式放进 DSPy signature / predictor 可见字段。

脚本默认复用 cold COPRO env：

```text
scripts/env_files/.env_dspy_react_agent_daily_2.5_v0_corpo_cold
```

默认输出目录：

```text
/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_gemini2.5_miprov2_recommended
```

## 参数详解

### Dataset / runtime 参数

- `MIPROV2_TRAIN_LIMIT` / `--train-limit`: 选入 MIPROv2 的样本数 T。代码先取 input JSONL 前 T 条。
- `MIPROV2_MAX_TURNS` / `--max-turns`: 单个 AVQA ReAct rollout 最多 planner/tool turn 数。
- `MIPROV2_PERCEPTION_MODEL` / `--perception-model`: perception backend，通常沿用 cold env 的 `gemini`。
- `PERCEPTION_CONFIG_YAML`: perception backend 的模型、temperature、top_p、max_tokens 等配置。
- `PROMPT_YAML`: DSPy AVQA planner/perception/signature prompt 配置。
- `DSPY_AVQA_ALLOWED_TOOLS`: 可选工具集合。

注意：本地 MIPROv2 如果不单独传 `valset`，会把传入 trainset 再切分：约前 20% 用于 bootstrap / proposal train，后 80% 用于 validation / optimization score。也就是说脚本里的 T 不是 MIPROv2 内部 bootstrap train size。

近似：

```text
V = max(1, floor(0.8 * T))
T_bootstrap = T - V
```

### MIPROv2 search 参数

- `MIPROV2_AUTO` / `--miprov2-auto`: `none`、`light`、`medium`、`heavy`。脚本默认 `none`，使用显式 `num_candidates` 和 `num_trials`，调用量更可控。若改成 `light/medium/heavy`，代码会忽略显式 `num_candidates/num_trials`。
- `MIPROV2_NUM_CANDIDATES` / `--miprov2-num-candidates`: instruction/demo 候选数量 C。`auto=none` 时必须提供。
- `MIPROV2_NUM_TRIALS` / `--miprov2-num-trials`: Bayesian optimization trial 数 N。`auto=none` 时必须提供。
- `MIPROV2_MAX_BOOTSTRAPPED_DEMOS`: 每个 predictor 最多 bootstrap 成功 demo 数。
- `MIPROV2_MAX_LABELED_DEMOS`: 每个 predictor 最多 labeled demo 数。
- `MIPROV2_INIT_TEMPERATURE`: 生成 instruction candidates 时 prompt model 的 temperature。
- `MIPROV2_SEED`: 控制 candidate sampling、Optuna sampler 等随机过程。
- `MIPROV2_NUM_THREADS`: DSPy evaluator 线程数；空值使用 DSPy 默认。
- `MIPROV2_MAX_ERRORS`: evaluator 最大错误数；空值使用 DSPy 默认。

### Minibatch 参数

- `MIPROV2_MINIBATCH`: 默认 `true`。设为非 `true` 会加 `--miprov2-no-minibatch`，每个 trial 都跑完整 V。
- `MIPROV2_MINIBATCH_SIZE`: 每个 trial 评估的 validation minibatch 大小 M。必须小于等于 V。
- `MIPROV2_MINIBATCH_FULL_EVAL_STEPS`: 每隔若干 minibatch trial 做一次 full validation。
- `MIPROV2_VIEW_DATA_BATCH_SIZE`: proposer 查看数据样本数，用于 data-aware instruction proposal。

## 调用量估算

记号：

```text
T = MIPROV2_TRAIN_LIMIT
V = max(1, floor(0.8 * T))
C = MIPROV2_NUM_CANDIDATES
N = MIPROV2_NUM_TRIALS
M = MIPROV2_MINIBATCH_SIZE
F = MIPROV2_MINIBATCH_FULL_EVAL_STEPS
```

MIPROv2 optimization 主要 program rollout 来自：

1. bootstrap few-shot demo candidates：调用量和 C、`T_bootstrap`、demo 上限、metric_threshold、teacher 行为相关，实际比较依赖 DSPy 内部实现。
2. default program full validation：约 `V`。
3. N 个 optimization trials：minibatch 时约 `N * M`；非 minibatch 时约 `N * V`。
4. 周期性 full validation：粗略按 `ceil(N / F) * V` 估算。
5. 脚本 `--trajectory-jsonl` final rerun：额外 `T`。

因此脚本默认 minibatch 模式下，样本级 program rollout 粗略估算：

```text
search_eval_rollouts ~= V + N * M + ceil(N / F) * V
final_trajectory_rollouts = T
known_rollouts ~= search_eval_rollouts + T
```

bootstrap / proposer 还会额外消耗 prompt-model / task-model 调用，尤其 `max_bootstrapped_demos > 0` 时。每个 AVQA program rollout 内部还会产生多次 DeepSeek planner 和 Gemini/Qwen perception call，受 `MAX_TURNS` 和 planner 是否调用工具影响。

## 建议配置

| Preset | T | C | N | boot demos | labeled demos | minibatch M | full eval step F | known rollouts with final trajectory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 8 | 2 | 2 | 1 | 1 | 2 | 1 | ~= 30 |
| pilot | 32 | 4 | 6 | 2 | 2 | 8 | 3 | ~= 155 |
| recommended | 64 | 6 | 10 | 2 | 2 | 16 | 5 | ~= 377 |

计算示例：

```text
smoke:      V=6,  6 + 2*2  + ceil(2/1)*6  + 8  = 30
pilot:      V=25, 25 + 6*8 + ceil(6/3)*25 + 32 = 155
recommended: V=51, 51 + 10*16 + ceil(10/5)*51 + 64 = 377
```

推荐用法：

- smoke: 验证 MIPROv2 API、输出文件、signature search JSON 和 trajectory JSONL 是否能跑通。
- pilot: 初步观察 MIPROv2 是否能产生有意义的 instruction/demo 组合。
- recommended: 当前建议正式版；成本和已有 COPRO/SIMBA recommended 大致处于同一量级，但会有 bootstrap/proposer 额外调用。

脚本默认就是 recommended：

```bash
bash scripts/run_dspy_react_agent_daily_miprov2.sh
```

跑 smoke 示例：

```bash
MIPROV2_RUN_DIR=/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_gemini2.5_miprov2_smoke \
MIPROV2_TRAIN_LIMIT=8 \
MIPROV2_NUM_CANDIDATES=2 \
MIPROV2_NUM_TRIALS=2 \
MIPROV2_MAX_BOOTSTRAPPED_DEMOS=1 \
MIPROV2_MAX_LABELED_DEMOS=1 \
MIPROV2_MINIBATCH_SIZE=2 \
MIPROV2_MINIBATCH_FULL_EVAL_STEPS=1 \
bash scripts/run_dspy_react_agent_daily_miprov2.sh
```

