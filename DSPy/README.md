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

## Caveat:

小 caveat：如果不是走 run_batch()，而是在别的 Python 代码里直接 new AVQARuntimeContext() / AVQADSPyReActProgram()，想用自定义 YAML，需要先调用 load_prompt_config(custom_path)，再创建 context/program；否则会用默认 v0。
