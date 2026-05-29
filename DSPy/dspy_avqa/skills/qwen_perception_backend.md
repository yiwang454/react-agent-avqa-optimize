# Qwen Perception Backend 接入笔记

## 背景

这次修改的目标是在 `dspy_avqa` 里把 Qwen/Qwen3-Omni 接成 `ask_perception` 的另一个可选 perception backend，同时保持 planner 可见的 tool surface 不变。也就是说，planner 仍然只调用 `ask_perception`，具体走 Gemini 还是 Qwen 由 `PERCEPTION_MODEL` 或 runner 参数 `--perception-model` 决定。

用户已经提供了一个在其他脚本里测试可用的 Qwen helper：`DSPy/dspy_avqa/qwen3omni_api.py`。同时提供了一个 demos 里的 Qwen YAML config，后来复制并精简为 `DSPy/dspy_avqa/yamls/config_AVUT_qa_localqwen_api_baseline.yaml`。

## 问题

原来的 DSPy perception 路径需要支持 Qwen，但不能新增 planner 需要显式调用的 tool，例如 `ask_qwen_perception`。工具层仍应只暴露 `ask_perception` 这个统一入口。

Qwen YAML 里原本有一些属于 standalone demo 脚本的数据集/选择逻辑字段，例如 `input_jsonl`、`caption_dir`、`caption_based`、`select_criteria`、`select_value`、`qa_template_name`。这些值不应该被 DSPy 的 perception config loader 读取，否则容易和 DSPy runner 自己的 CLI 参数产生概念上的冲突。

另一个问题是 retry 间隔。`qwen3omni_api.py` 已经有 `max_retries` 重试机制，但 retry 之间的 sleep 曾经是硬编码 `time.sleep(1)`。用户希望沿用其他代码里的命名 `qwen_delay_s`，用它控制 retry 之间的间隔。

## 解决方案

保留统一工具入口：`ask_perception`。

Backend 选择方式：

- `PERCEPTION_MODEL=qwen` 使用 Qwen backend。
- `PERCEPTION_MODEL=gemini` 使用 Gemini backend。
- runner 里也可以用 `--perception-model qwen|gemini` 设置。

`DSPy/dspy_avqa/tools.py` 的 Qwen backend 现在直接调用 `qwen3omni_api.call_qwen_messages()`，而不是自己拼 HTTP request。它会构造包含 `audio_url`、`video_url` 和 perception prompt 的 Qwen message，并记录 metadata，例如 token usage 和 thinking text。

`DSPy/dspy_avqa/runner.py` 的 `--perception-config-yaml` 现在不再是 Gemini-only，而是 perception backend config。它会把 Qwen YAML 里的相关字段映射到 `QWEN_*` 环境变量。

`DSPy/dspy_avqa/qwen3omni_api.py` 现在给 `call_qwen_messages()`、`call_qwen()`、`call_qwen3omni()` 都加了 `retry_delay_s` 参数。retry loop 失败后会 `time.sleep(retry_delay_s)`，不再硬编码 1 秒。

## 关键文件

### `DSPy/dspy_avqa/tools.py`

- `selected_perception_model()` 负责读取 `PERCEPTION_MODEL`。
- `call_qwen_perception()` 是 DSPy perception tool 走 Qwen 的实际入口。
- `call_qwen_perception()` 从 `qwen3omni_api.py` import `call_qwen_messages()` 和 `to_data_url()`。
- Qwen 调用参数来自这些环境变量：
  - `QWEN_MODEL`
  - `QWEN_API_KEY`
  - `QWEN_BASE_URL`
  - `QWEN_TIMEOUT`
  - `QWEN_MAX_RETRIES`
  - `QWEN_DELAY_S`
  - `QWEN_TEMPERATURE`
  - `QWEN_TOP_P`
  - `QWEN_TOP_K`
  - `QWEN_MAX_TOKENS`
  - `QWEN_FPS`
  - `QWEN_MAX_FRAMES`
  - `QWEN_ENABLE_THINKING`
  - `QWEN_VIDEO_ONLY`

### `DSPy/dspy_avqa/runner.py`

- `load_perception_config_yaml()` 读取 `--perception-config-yaml`。
- 它把 Qwen config 映射成 `QWEN_*` 环境变量。
- 当 `--perception-model qwen` 时，runner 会打印当前 Qwen config，方便确认 model/base_url/retry delay 等配置。

### `DSPy/dspy_avqa/qwen3omni_api.py`

- `call_qwen_messages()` 包含 OpenAI-compatible SDK 请求和 retry loop。
- `retry_delay_s` 控制失败 retry 之间的 sleep 时间。
- `call_qwen()` 和 `call_qwen3omni()` 也支持并向下传递 `retry_delay_s`。

### `DSPy/dspy_avqa/yamls/config_AVUT_qa_localqwen_api_baseline.yaml`

这是当前放在 repo 内的 Qwen perception config。当前保留的字段主要用于 Qwen backend runtime/sampling 参数。

## DSPy 会读取的 YAML 字段

Qwen 相关字段：

- `task.video_only` -> `QWEN_VIDEO_ONLY`
- `task.enable_thinking` -> `QWEN_ENABLE_THINKING`
- `task.batch_size` -> `QWEN_BATCH_SIZE`
- `task.timeout` -> `QWEN_TIMEOUT`
- `task.max_retries` -> `QWEN_MAX_RETRIES`
- `task.qwen_delay_s` -> `QWEN_DELAY_S`
- `model.qwen_model` -> `QWEN_MODEL`
- `model.qwen_base_url` -> `QWEN_BASE_URL`
- `model.qwen_api_key` -> `QWEN_API_KEY`
- `sampling_params.temperature` -> `QWEN_TEMPERATURE`
- `sampling_params.top_p` -> `QWEN_TOP_P`
- `sampling_params.top_k` -> `QWEN_TOP_K`
- `sampling_params.max_tokens` -> `QWEN_MAX_TOKENS`
- `sampling_params.fps` -> `QWEN_FPS`
- `sampling_params.max_frames` -> `QWEN_MAX_FRAMES`

Gemini 相关字段仍然会映射到 `GEMINI_*`，例如 `task.retry_delay_s` -> `GEMINI_RETRY_DELAY_S`。

DSPy 不会从 perception config 读取这些 standalone demo 字段：

- `input_jsonl`
- `caption_dir`
- `caption_based`
- `select_criteria`
- `select_value`
- `qa_template_name`

这些应该继续由 DSPy runner 的 CLI 参数控制，例如 `--input-jsonl` 和 `--audio-caption-dir`。

## Retry 语义

Qwen backend 的 retry 配置现在是：

- `task.max_retries` 控制最多尝试次数。
- `task.qwen_delay_s` 控制失败 retry 之间的间隔秒数。
- YAML loader 把 `task.qwen_delay_s` 写成 `QWEN_DELAY_S`。
- `tools.py` 把 `QWEN_DELAY_S` 作为 `retry_delay_s` 传给 `call_qwen_messages()`。
- `call_qwen_messages()` 在失败且还有剩余 attempt 时执行 `time.sleep(retry_delay_s)`。

注意：`qwen3omni_api.py` 里也有 `call_qwen_serial(..., delay_s=...)`，它原本表示多个 prompt/batch 之间的间隔。但在 DSPy 的 `ask_perception -> qwen` 路径里不走 `call_qwen_serial()`，因此这里明确复用 `qwen_delay_s` 作为 retry delay，符合用户其他代码里的命名习惯。

## 示例命令

```bash
PYTHONPATH=DSPy \
PERCEPTION_MODEL=qwen \
python -m dspy_avqa.runner \
  --input-jsonl /path/to/input.jsonl \
  --output-jsonl /path/to/output.jsonl \
  --audio-caption-dir /path/to/audio_captions \
  --perception-model qwen \
  --perception-config-yaml DSPy/dspy_avqa/yamls/config_AVUT_qa_localqwen_api_baseline.yaml
```

如果运行环境已经把 `DSPy` 加进了 Python path，可以省略 `PYTHONPATH=DSPy`。

## 已做验证

语法检查：

```bash
python -m py_compile DSPy/dspy_avqa/qwen3omni_api.py DSPy/dspy_avqa/tools.py DSPy/dspy_avqa/runner.py
```

无网络 retry smoke test：mock Qwen client 连续失败，确认 `retry_delay_s=0.25` 且 `max_retries=2` 时，实际 sleep 记录为 `[0.25]`。

Config loader smoke test：确认 YAML 中 `task.qwen_delay_s: 1.0` 会映射成 `QWEN_DELAY_S=1.0`。
