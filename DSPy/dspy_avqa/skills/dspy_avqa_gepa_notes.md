# DSPy AVQA GEPA Optimization Notes

本文档对应脚本：

- `scripts/run_dspy_react_agent_daily_gepa.sh`

参考 DSPy 文档：`https://dspy.ai/tutorials/gepa_ai_program/` 和 `https://dspy.ai/api/optimizers/GEPA/`。当前本地环境为 DSPy `3.2.1`，脚本按该版本的 `GEPA.__init__` / `compile` 参数写。

## 背景

GEPA 通过执行 program、收集 metric feedback、让 reflection LM 反思错误轨迹，然后提出新的 instruction。和普通 exact-match metric 不同，GEPA metric 必须能返回 textual feedback。

本实现新增了：

```text
avqa_gepa_feedback_metric(example, pred, trace, pred_name, pred_trace)
```

它仍然用 option-letter exact match 打分，但会把 gold/pred answer、question、options、reasoning summary 和针对错误样本的改写建议写进 feedback。GEPA 的 `reflection_lm` 默认复用当前 DSPy 全局 planner LM，因此需要保证 env 中的 DeepSeek/OpenAI-compatible planner backend 可用于反思生成。

脚本默认复用 cold COPRO env：

```text
scripts/env_files/.env_dspy_react_agent_daily_2.5_v0_corpo_cold
```

默认输出目录：

```text
/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_gemini2.5_gepa_recommended
```

## 参数详解

### Dataset / runtime 参数

- `GEPA_TRAIN_LIMIT` / `--train-limit`: 参与 GEPA optimization 的样本数 T。代码取 input JSONL 前 T 条。
- `GEPA_MAX_TURNS` / `--max-turns`: 单个 AVQA ReAct rollout 最多 planner/tool turn 数。
- `GEPA_PERCEPTION_MODEL` / `--perception-model`: perception backend，通常沿用 cold env 的 `gemini`。
- `PERCEPTION_CONFIG_YAML`: perception backend 的模型、temperature、top_p、max_tokens 等配置。
- `PROMPT_YAML`: DSPy AVQA planner/perception/signature prompt 配置。
- `DSPY_AVQA_ALLOWED_TOOLS`: 可选工具集合。

当前 CLI 未单独传 GEPA valset，因此 GEPA 会把同一份 trainset 同时用于 reflective updates 和 Pareto score tracking。这适合先做 inference-time / trainset-bound prompt search；如果要验证泛化，后续应扩展 CLI 传独立 valset。

### GEPA budget 参数

GEPA 要求以下三者只能有一个生效：

- `GEPA_AUTO` / `--gepa-auto`: `none`、`light`、`medium`、`heavy`。脚本默认 `none`。
- `GEPA_MAX_FULL_EVALS` / `--gepa-max-full-evals`: 以完整 trainset evaluation 次数表达预算。脚本默认 `6`。
- `GEPA_MAX_METRIC_CALLS` / `--gepa-max-metric-calls`: 直接指定 metric call 上限；如果设置了它，脚本会优先使用它并忽略 `GEPA_MAX_FULL_EVALS`。

脚本默认使用 `max_full_evals`，因为对 AVQA 这种昂贵 rollout 更容易估算。

### GEPA reflection / selection 参数

- `GEPA_REFLECTION_MINIBATCH_SIZE`: 每次 reflection 使用的失败/样本 minibatch 大小。
- `GEPA_CANDIDATE_SELECTION_STRATEGY`: `pareto` 或 `current_best`。默认 `pareto`，更符合 GEPA 的多候选 Pareto tracking。
- `GEPA_SKIP_PERFECT_SCORE`: 默认 `true`，满分样本不送去反思，节省 reflection 调用。
- `GEPA_USE_MERGE`: 默认 `true`，允许 GEPA merge 候选 instruction。
- `GEPA_MAX_MERGE_INVOCATIONS`: merge 调用上限。
- `GEPA_NUM_THREADS`: evaluator 线程数；空值使用 DSPy 默认。
- `GEPA_SEED`: GEPA 内部随机种子。
- `GEPA_LOG_DIR`: GEPA 运行日志目录。
- `GEPA_TRACK_STATS`: 是否把详细 GEPA 结果附到 compiled program。

## 调用量估算

记号：

```text
T = GEPA_TRAIN_LIMIT
E = GEPA_MAX_FULL_EVALS
B = GEPA_MAX_METRIC_CALLS
```

当前没有传独立 valset；在 DSPy 3.2.1 中：

```text
if max_full_evals is used:
    max_metric_calls = E * T
if max_metric_calls is used:
    max_metric_calls = B
if auto is used:
    max_metric_calls is computed by DSPy auto_budget(...)
```

脚本还会在 optimization 完成后用 best compiled program 重新跑 selected trainset 写 `output_test.jsonl`：

```text
final_trajectory_rollouts = T
```

因此使用 `GEPA_MAX_FULL_EVALS` 时，样本级 program rollout 上限粗略是：

```text
optimization_rollouts <= E * T
known_rollouts_with_final_trajectory <= E * T + T
```

使用 `GEPA_MAX_METRIC_CALLS` 时：

```text
known_rollouts_with_final_trajectory <= B + T
```

另外还要单独考虑 reflection LM 调用。GEPA 会基于 feedback 让 reflection LM 提出 instruction 改写；这部分不是 AVQA program rollout，但会消耗 planner/reflection 模型 token。`skip_perfect_score=true` 可以减少满分样本的反思开销。

每个 AVQA program rollout 内部仍可能包含多次 DeepSeek planner 和 Gemini/Qwen perception call，受 `MAX_TURNS` 和 tool usage 影响。

## 建议配置

| Preset | T | GEPA_MAX_FULL_EVALS E | reflection minibatch | selection | merge | known rollouts with final trajectory |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| smoke | 8 | 2 | 2 | pareto | true | <= 24 |
| pilot | 32 | 4 | 3 | pareto | true | <= 160 |
| recommended | 64 | 6 | 3 | pareto | true | <= 448 |

计算方式：

```text
smoke:      2*8  + 8  = 24
pilot:      4*32 + 32 = 160
recommended: 6*64 + 64 = 448
```

推荐用法：

- smoke: 验证 GEPA feedback metric、reflection LM、日志目录和输出文件能否跑通。
- pilot: 初步观察 GEPA 是否能根据错误轨迹提出有效 instruction 改写。
- recommended: 当前建议正式版；预算仍可控，并适合和 MIPROv2 / COPRO / SIMBA recommended 做第一轮对照。

脚本默认就是 recommended：

```bash
bash scripts/run_dspy_react_agent_daily_gepa.sh
```

跑 smoke 示例：

```bash
GEPA_RUN_DIR=/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_gemini2.5_gepa_smoke \
GEPA_TRAIN_LIMIT=8 \
GEPA_MAX_FULL_EVALS=2 \
GEPA_REFLECTION_MINIBATCH_SIZE=2 \
bash scripts/run_dspy_react_agent_daily_gepa.sh
```

