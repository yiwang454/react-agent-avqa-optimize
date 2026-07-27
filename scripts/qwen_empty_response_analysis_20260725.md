# Qwen3-Omni ReACT 空返回并发消融

日期：2026-07-25

## 实验目标

比较以下三种 Qwen3-Omni 输入方式在并发请求中的空返回概率：

1. `use_audio_in_video=true`，只发送 `video_url`
2. `use_audio_in_video=false`，发送 `audio_url`、`video_url`
3. `use_audio_in_video=false`，发送 `video_url`、`audio_url`

实验期间没有应用层 retry，也关闭了 OpenAI SDK 自带的 transport retry。所有请求均为非流式响应。

## 实验设置

- Endpoint：`http://10.62.186.8:8000/v1`
- Model：`qwen3-omni-instruct`
- `temperature=0.0`
- `top_p=1.0`
- `top_k=0`
- `max_tokens=4096`
- `fps=2.0`
- `max_frames=128`
- `seed=1234`
- `enable_thinking=false`
- 并发度：3
- 每个样本、每种输入方式同时请求 3 次
- 样本数：12
- 总请求数：108

12 个样本从 166 个历史空返回中分层选取，覆盖历史短空、历史 4096-token 长空，以及 low、mid、high 三档 prompt 长度。因此下面是“在历史失败样本上的条件空率”，用于比较输入方式，不代表全数据集总体空率。

## 总结果

| 输入方式 | 空返回 | 请求数 | 条件空率 | HTTP 错误 |
|---|---:|---:|---:|---:|
| 视频内音轨 | 4 | 36 | 11.1% | 0 |
| 独立音频在前 | 25 | 36 | 69.4% | 0 |
| 独立视频在前 | 11 | 36 | 30.6% | 0 |

- 独立音频在前的空率约为视频内音轨的 6.25 倍。
- 独立视频在前比独立音频在前稳定，但仍不如视频内音轨。
- 视频内音轨不是绝对可靠，仍有 4/36 空返回。

按历史空返回类型拆分：

| 输入方式 | 历史短空样本 | 历史长空样本 |
|---|---:|---:|
| 视频内音轨 | 1/18 | 3/18 |
| 独立音频在前 | 12/18 | 13/18 |
| 独立视频在前 | 5/18 | 6/18 |

## 逐样本结果

| Sample | 视频内音轨 | 音频在前 | 视频在前 |
|---|---:|---:|---:|
| `00FBAdjlF4g-2` | 0/3 | 3/3 | 0/3 |
| `-spqqACI6UQ-3` | 1/3 | 1/3 | 2/3 |
| `-xfgovG6-KU-1` | 0/3 | 1/3 | 2/3 |
| `02XbmweOzOY-3` | 0/3 | 2/3 | 1/3 |
| `-yWE13LbFd4-1` | 0/3 | 2/3 | 0/3 |
| `0JjgoicpYkU-1` | 0/3 | 3/3 | 0/3 |
| `0NW8rsCj6xQ-1` | 1/3 | 2/3 | 3/3 |
| `0qZ54ovyEWQ-1` | 1/3 | 3/3 | 1/3 |
| `h7ljgWAgb0E-2` | 0/3 | 1/3 | 0/3 |
| `-oC3FVOx62g-1` | 1/3 | 2/3 | 2/3 |
| `068rdc75mHM-1` | 0/3 | 3/3 | 0/3 |
| `2LDriAWltwM-2` | 0/3 | 2/3 | 0/3 |

## 空返回的实际内容

空返回不是 HTTP body 缺失，也不是客户端解析丢失。所有 40 个空返回都包含纯换行：

- 短空通常是 4-7 completion tokens，对应 1-2 个 `\n`，然后 `finish_reason=stop`。
- 长空通常是 3842-4096 completion tokens，对应约 1920-2048 个 `\n`；部分以 `length` 结束，部分生成 EOS 后以 `stop` 结束。
- `reasoning_content` 为空。
- `tool_observation` 在 `.strip()` 后变成空字符串。

所以直接原因是模型 decoder 进入“只生成换行”的退化状态，而不是结果保存代码把正常文本清空。

## 并发非确定性

同一组内的三次请求具有相同 prompt、媒体、seed、temperature 和 sampling 参数，但经常出现一次正常文本、一次 5-token 空返回、一次 4096-token 换行循环。三个输入方式中，出现不同输出内容的 sample group 数分别为：

- 视频内音轨：12/12
- 独立音频在前：10/12
- 独立视频在前：12/12

因此 `temperature=0` 和固定 seed 没有让这个 vLLM-Omni endpoint 在并发下保持确定性。输入媒体没有缺失，因为完全相同的请求可以在同一组中正常返回。

## 视频内音轨的 token 分叉

`use_audio_in_video=true` 时，每个 sample 的三次并发请求都出现两档 `prompt_tokens`：1 次较低、2 次较高。高低差随音频时长变化，例如 `h7ljgWAgb0E-2` 为 7093 与 7483。

独立音频的两种顺序在全部 sample 中 token 数都完全稳定，且两种顺序的 token 总数相同。

视频内音轨的 12 次低-token 请求中有 4 次为空，24 次高-token 请求中没有空返回。结合之前“关闭视频音轨提取”和“单独发送 WAV”的单请求 token ablation，高低差约等于一份音频 features 的 token 数。最可能的解释是：

- 并发视频音轨预处理存在共享状态或缓存竞争。
- 同一视频的音频 features 在不同请求中被不同次数地计入。
- 高-token 分支很可能重复计入了一份音频 features；低-token 分支更接近单份音频输入。

最后一点是基于 token 差值的推断，需要 vLLM-Omni server 日志或 processor-level tracing 才能完全确认。但“同一请求的 multimodal prompt 长度在并发下不稳定”是直接观测结果。

## 原因判断

按证据强度排序：

1. **Decoder 换行退化**：是 `tool_observation=""` 的直接原因。
2. **媒体顺序敏感**：独立音频在前显著提高退化概率；视频在前能降低但不能消除。
3. **并发调度非确定性**：相同 seed 和输入在同组内分叉，说明问题位于服务端 multimodal preprocessing、scheduler 或模型状态，而不是样本内容。
4. **视频音轨预处理不稳定**：`use_audio_in_video=true` 的 prompt token 数在每组中规律性分叉，提示共享 processor/cache 状态竞争。
5. **不是 context overflow**：最长 prompt 约 15K，加 4096 输出仍低于 endpoint 报告的 `max_model_len=28672`。
6. **不是网络或 retry 伪影**：正式实验 108/108 HTTP 成功，且两层 retry 均关闭。

## 对 ReACT 实验的建议

1. 固定 `use_audio_in_video=true`、不发送独立 `audio_url`，在当前三组中空率最低，也避免 audio/video 消息顺序变化。
2. 保持多个 dataset repeat 串行运行。并发下相同请求不确定，且曾触发过 endpoint 进程退出。
3. 保留 whitespace-only 检测；不能只检查 `content is None`。
4. 记录每次请求的 `prompt_tokens`、`completion_tokens`、`finish_reason` 和媒体配置。
5. Production run 可保留空响应 retry，但 retry 必须被记录。当前 `use_audio_in_video=true` 只有一个 video item，`video_first` reorder 本身不会改变请求。
6. 可以考虑降低 perception JSON 的 `max_tokens`，减少 4096-token 换行循环的时间和算力浪费；这只能降低代价，不能修复根因。
7. 空率和 accuracy 要分开评估。视频内音轨的低空率部分伴随疑似重复音频 features，可能解释为什么它不一定带来更高 accuracy。

## 结果文件

- `results.jsonl`：`/mnt/ceph_rbd/data/avqa_project/daily_omni/qwen_av_order_concurrency_tests/20260725T225657Z/results.jsonl`
- `manifest.json`：`/mnt/ceph_rbd/data/avqa_project/daily_omni/qwen_av_order_concurrency_tests/20260725T225657Z/manifest.json`
- 可复现实验脚本：`scripts/test_qwen_av_order_concurrency.py`
