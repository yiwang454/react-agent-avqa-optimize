# DSPy AVQA ReAct

这个目录是基于 DSPy 重写的 AVQA ReAct 版本，结构上尽量对齐 `src/react_agent/`，把原来单文件实现拆成独立模块，方便维护和后续做 COPRO/SIMBA 优化。

## 目录结构

```text
DSPy/
├── README.md
├── skeleton.py
├── avqa_dspy_impl.py
└── dspy_avqa/
    ├── __init__.py
    ├── context.py
    ├── signatures.py
    ├── tools.py
    ├── program.py
    ├── data.py
    ├── runner.py
    └── optimize.py
```

## 模块职责

- `dspy_avqa/context.py`
  - 运行时上下文配置（DeepSeek planner 参数、`max_turns`、系统提示词）。
  - 负责 `configure_deepseek_lm()`，把 DSPy LM 配成 DeepSeek planner。

- `dspy_avqa/tools.py`
  - 感知工具层，主要调用 Qwen3-omni。
  - 提供：
    - `ask_qwen_perception()`：常规感知问题。
    - `temporal_ground_video()`：时间定位/grounding 问题。

- `dspy_avqa/signatures.py`
  - 显式 planner 子任务的 DSPy signatures：
    - 起草下一个 perceptual question
    - 根据 observation 更新 belief
    - 决策继续调用工具还是输出最终答案
    - 达到最大轮次后的 fallback final answer

- `dspy_avqa/program.py`
  - 核心 ReAct 循环实现（显式多轮交互）：
    - `DraftPerceptualQuestion -> Tool -> UpdateBeliefFromObservation -> DecideFinalAction`
  - 产出 `turn_trace`，用于分析 planner/tool 交互过程。

- `dspy_avqa/data.py`
  - 数据加载与输出格式拼装。
  - 对齐原 `batch_runner` 的输入输出风格（`question_data` / `metadata` / `turn_trace`）。

- `dspy_avqa/runner.py`
  - CLI 入口逻辑与批量执行。
  - 提供 `run_batch()` 和单样本 `run_one()`。

- `dspy_avqa/optimize.py`
  - DSPy 优化相关接口：
    - `avqa_metric`
    - `make_trainset`
    - `optimize_with_copro`
    - `optimize_with_simba`

- `dspy_avqa/__init__.py`
  - 对外统一导出，方便从包级别直接 import。

- `avqa_dspy_impl.py`
  - 兼容入口（wrapper），转发到 `dspy_avqa` 包，不再放业务实现。

- `skeleton.py`
  - 兼容入口（保留旧调用路径）。

## 与 `src/react_agent/` 的对应关系

- `src/react_agent/context.py` -> `DSPy/dspy_avqa/context.py`
- `src/react_agent/tools.py` -> `DSPy/dspy_avqa/tools.py`
- `src/react_agent/graph.py`（planner-tool 循环）-> `DSPy/dspy_avqa/program.py`
- `src/react_agent/batch_runner.py` -> `DSPy/dspy_avqa/data.py` + `DSPy/dspy_avqa/runner.py`

## 运行方式

推荐直接用脚本：

- `scripts/run_dspy_react_agent.sh`

也可以手动运行：

```bash
uv run --no-sync python DSPy/avqa_dspy_impl.py \
  --input-jsonl <input.jsonl> \
  --audio-caption-dir <caption_dir> \
  --output-jsonl <output.jsonl> \
  --output-dir <output_dir> \
  --max-turns 4 \
  --concurrency 1 \
  --debug
```

## ELM GPT planner

Use the University of Edinburgh ELM GPT endpoint as the planner and GEPA
reflection model by setting the following values in a local, untracked env
file. `PLANNER_PROVIDER=elm_gpt` deliberately ignores every DeepSeek/base-URL
setting and lets the OpenAI-compatible SDK use its default endpoint.

```bash
PLANNER_PROVIDER=elm_gpt
PLANNER_MODEL=gpt-5-mini
PLANNER_API_KEY=...
PLANNER_REASONING_EFFORT=medium
PLANNER_SEED=7
PLANNER_TEMPERATURE=0.0
GEPA_REFLECTION_TEMPERATURE=1.0
```

`DEEPSEEK_API_KEY` remains a compatibility alias for `PLANNER_API_KEY` (the
existing precedence is `DEEPSEEK_API_KEY`, `DEEPSEEK_TOKEN`, then
`PLANNER_API_KEY`). Do not set or depend on `DEEPSEEK_BASE_URL`,
`DEEPSEEK_API_BASE`, or `PLANNER_API_BASE` in this mode. Planner rollout uses
`PLANNER_TEMPERATURE`; GEPA copies the same LM and only overrides its
temperature with `GEPA_REFLECTION_TEMPERATURE`. In the currently pinned DSPy
version, dash-named GPT-5 variants such as `gpt-5-mini` accept only `0.0` or
`1.0` as an explicit temperature and require `PLANNER_OUTPUT_SEQ_LEN >= 16000`;
`PLANNER_REASONING_EFFORT` and `PLANNER_SEED` are forwarded only in `elm_gpt`
mode.

## Gemini legacy / DSPy Vertex 后端

Gemini 默认走现有 `legacy` generateContent 路径。若要通过 DSPy/LiteLLM 调用官方 Vertex Gemini，必须同时显式提供本地媒体根与对应 GCS 根；`GEMINI_API_KEY`、`GEMINI_BASE_URL` 在该模式下不会使用。

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
python DSPy/avqa_dspy_impl.py ... \
  --gemini-api-backend dspy \
  --vertex-project decoupled-avqa-501420 \
  --vertex-location global \
  --gemini-local-data-root /local/dataset/root \
  --gemini-gcs-data-root gs://bucket/dataset-prefix
```

`--gemini-api-backend dspy` 同时覆盖 Gemini perception 与 Gemini captioner。Vertex Gemini 的 `sampling_params.top_k` 必须为 `1..64`；旧配置中的 `0` 请改为例如 `64`。认证由 ADC/service account 完成。

`--response-error-sensitive` 会仅对 Gemini 启用进程级熔断：空响应、`[ERROR]` 响应或最终 Gemini/LiteLLM 调用异常会立即停止当前运行，不再写成普通失败样本。默认关闭；可用 `--no-response-error-sensitive` 显式关闭。

## Final test inference 断点续跑

Optimization 结束后，可以在原命令后追加 `--inference-only`，直接加载
`--output-program` 并运行或继续 final test，而不会重新执行 optimizer：

```bash
python DSPy/avqa_dspy_optimize.py \
  <与原 optimization 相同的参数> \
  --inference-only
```

该模式要求提供 `--final-eval-output-jsonl`。如果没有指定
`--final-eval-output-dir`，默认使用 compiled program 所在目录。每个完成的样本会保存为
`<sample_id>.json`；重新运行时会跳过其中包含非空 `response` 的样本，并在全部完成后按
input JSONL 顺序重建汇总 JSONL。`[ERROR]` response 也视为已完成。一个 cache 目录只能对应
一个 compiled program，不要在更换 program 后复用旧目录。

`--inference-only` 会在开始前验证 compiled program 存在且能够加载。final test 使用目录锁，
因此同一 output directory 不能被两个 inference 进程同时写入。

预计算的 `--audio-caption-dir` 现在是可选的：只有当 `--caption-placement task` 且
active `planner.task_prompt_template` 引用了 `{video_description}` 时才会读取它；其他情况
即使传入也会忽略。对于首轮 `video_description` 只是占位、由 `ask_caption` 提供真实 caption
的任务，可显式传入 `--ignore-audio-caption-dir`。

## Caveat:

小 caveat：如果不是走 run_batch()，而是在别的 Python 代码里直接 new AVQARuntimeContext() / AVQADSPyReActProgram()，想用自定义 YAML，需要先调用 load_prompt_config(custom_path)，再创建 context/program；否则会用默认 v0。
