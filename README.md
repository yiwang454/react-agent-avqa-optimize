# DailyOmni Free-ReAct Experiments

This branch contains four inference experiments for evaluating an o3 planner in a
Free-ReAct audio-visual question answering workflow. The full OmniAgent o3
system is included as a baseline. A Free-ReAct planner cannot inspect a video
directly: it must first obtain a whole-video caption and can then decide whether
to request more evidence or return a final answer.

## Results

All systems use the 1,197-question DailyOmni evaluation set. The four Free-ReAct
experiments use Gemini 2.5 Flash for captioning and perception and allow at most
six tool calls per question.

| No. | Experiment name | Reasoning effort | Available tools | Input tok. | Thinking output tok. | Non-thinking output tok. | Latency/question | Cost/question | Acc. | Avg. tool calls/question |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | Full OmniAgent (o3) | not persisted | `Audio_EventList`, `Audio_EventLocation`, `audio_ASR`, `audio_global_caption`, `audio_qa`, `video_clip_qa`, `video_global_qa`, `video_metadata` | 62.7k | 3.6k† | 1.3k† | 71.00 s | $0.0674 | 77.53% | 7.27 |
| 10 | o3 + Omni Clip Caption | medium | `ask_caption`, `ask_perception`, `omni_clip_caption` | 9.0k | 0.81k | 0.30k | 11.35 s | $0.0173 | 78.11% | 1.59 |
| 11 | o3 Basic Tools | medium | `ask_caption`, `ask_perception` | 7.8k | 0.78k | 0.28k | 11.58 s | $0.0157 | 78.86% | 1.55 |
| 12 | o3 + Omni Clip Perception | medium | `ask_caption`, `ask_perception`, `omni_clip_perception` | 8.1k | 0.79k | 0.30k | 12.50 s | $0.0171 | 79.03% | 1.58 |
| 11b | o3 Basic Tools (High Reasoning) | high | `ask_caption`, `ask_perception` | 9.9k | 1.61k | 0.33k | 14.38 s | $0.0243 | 79.53% | 1.63 |

Rows 10 and 12 were refreshed from their completed aggregate outputs on
2026-10-07. Experiment 10 has 935/1,197 overall accuracy and 935/1,196
answered accuracy; Experiment 12 has 946/1,197 for both. Their latency values
remain the original full-run throughput figures: the final repair launches
reused 1,168 and 1,103 existing checkpoints, respectively, so dividing those
incremental repair times by the full benchmark would not be comparable.

`Avg. tool calls/question` uses one definition for every row: it counts only
turns whose planner action is `tool`. The final decision round performed by the
reasoner is excluded. The Full OmniAgent value is therefore 8,700 tool calls / 1,197
questions = 7.27, computed from `OmniAgent_repeat1/output_test.jsonl`.

† The Full OmniAgent summary reports 4.9k total output tokens per question. Its
trace preserves the o3 planner split (3.55k reasoning and 0.69k visible output)
but does not preserve Gemini tool-call thinking-token metadata (because the original OmniAgent repo doesn't have it). The table assigns
the remaining reported output to non-thinking output, so the baseline split is
an approximation. The Free-ReAct rows use the exact persisted o3 reasoning-token
and Gemini `thoughtsTokenCount` fields.

OmniAgent replication is run with https://github.com/yiwang454/OmniAgent_replicate.git, which is cloned from https://github.com/KD-TAO/OmniAgent.git .

### Why high reasoning has more output tokens

The earlier combined `Out tok.` value included hidden reasoning/thinking tokens.
Experiment 11b does not make many more tool calls than Experiment 11 (1.63 versus
1.55 per question), but o3 produces substantially more hidden reasoning on every
planner decision, including the final decision round. Its thinking output rises
from 0.78k to 1.61k tokens per question, while visible non-thinking output rises
only from 0.28k to 0.33k. The increase is therefore primarily reasoning tokens,
not additional tool calls or longer visible answers.

## Experiment configurations

### Baseline. Full OmniAgent (o3)

The baseline uses the original multi-tool OmniAgent workflow. Its accuracy and
tool-call count were verified from:

```text
/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_repeat1/output_test.jsonl
```

It answers 928 of 1,197 questions correctly (77.53%) and records 8,700 actual
tool actions. Final decision rounds are not included in that tool count.

### 10. o3 + Omni Clip Caption

Run with:

```bash
bash scripts/10_run_dspy_react_agent_daily_o3_gemini25flash_omni_clip_caption_free_react_6turn.sh
```

In addition to whole-video captioning and perception, the planner can select a
time range and ask the captioner for a targeted description of that short clip.
The planner uses the o3 `medium` reasoning configuration.

### 11. o3 Basic Tools

Run with:

```bash
bash scripts/11_run_dspy_react_agent_daily_o3_gemini25flash_basic_tools_free_react_6turn.sh
```

This is the basic-tools baseline. The planner can request a whole-video caption
or ask the perception model a question about the full video. The planner uses
the o3 `medium` reasoning configuration.

### 11b. o3 Basic Tools with High Reasoning

Run with:

```bash
bash scripts/11b_run_dspy_react_agent_daily_o3_gemini25flash_basic_tools_free_react_6turn_highreasoning.sh
```

This experiment keeps the same tools and prompt as Experiment 11 while changing
the o3 reasoning effort from `medium` to `high`. It isolates the effect of
additional planner reasoning without introducing another perception tool.

### 12. o3 + Omni Clip Perception

Run with:

```bash
bash scripts/12_run_dspy_react_agent_daily_o3_gemini25flash_omni_clip_perception_free_react_6turn.sh
```

In addition to the basic tools, the planner can select a time range and ask the
perception model a targeted audio-visual question about that short clip. The
planner uses the o3 `medium` reasoning configuration.

## Shared workflow

Each experiment follows the same high-level protocol:

1. The first tool call must be `ask_caption`.
2. The first whole-video caption is read from the same exact caption cache.
3. Later caption or perception requests use live Gemini 2.5 Flash calls.
4. After observing the first caption, o3 may call any enabled tool repeatedly or
   return a final answer.
5. The maximum budget is six tool calls. A final decision is a planner round but
   does not consume a tool call.

The planner configurations are:

- `DSPy/dspy_avqa/yamls/reasoner_elm_o3_medium.yaml`
- `DSPy/dspy_avqa/yamls/reasoner_elm_o3_high.yaml`

Both use deterministic sampling (`temperature: 0`, `seed: 1234`) and a maximum
output length of 32,768 tokens.

## Metric definitions

- **Input tok.**: average input tokens per question. For the Free-ReAct rows,
  this is persisted planner input plus live Gemini tool input. The cached first
  caption is excluded because reading it does not issue a new model request.
- **Thinking output tok.**: hidden o3 reasoning tokens plus Gemini thinking
  tokens, when the backend persisted both fields.
- **Non-thinking output tok.**: output tokens excluding the recorded hidden
  reasoning/thinking tokens.
- **Latency/question**: total wall-clock run time divided by 1,197 questions. It
  represents concurrent experiment throughput, including retries and waiting,
  rather than the latency of a single serial request.
- **Cost/question**: estimated standard API cost of the persisted live calls.
  The calculation uses $2.00/M input and $8.00/M output tokens for o3, and
  $0.30/M text/image/video input, $1.00/M audio input, and $2.50/M output tokens
  for Gemini 2.5 Flash. It excludes the historical cost of generating the
  shared caption cache, cache storage, and unpersisted failed retries.
- **Acc.**: exact multiple-choice accuracy over all 1,197 questions.
- **Avg. tool calls/question**: total recorded tool actions divided by 1,197;
  final-answer planner rounds performed by the reasoner are excluded for every
  system, including Full OmniAgent.

## Summary

The high-reasoning basic-tools run achieved the highest accuracy at 79.53%, with
higher thinking-token usage, latency, and cost. The medium-reasoning basic-tools
run was the least expensive system. All four Free-ReAct variants used far fewer
tool calls than Full OmniAgent. Clip captioning did not improve aggregate
accuracy over the medium basic-tools run; clip perception produced a small
improvement (79.03% versus 78.86%) while remaining below high reasoning.

## GPT-4.1 G0 workflow ablations

The simplified and quartered GPT-4.1 workflow runs completed on 2026-10-06.
Each `output_test.jsonl` contains exactly 1,197 unique DailyOmni question IDs,
with no missing or extra IDs relative to `daily_omni_cuts_v3.jsonl`. Both runs
also have 1,197 per-question checkpoint files, and their shared launcher log
ends with `Wrote 1197 rows` and a total elapsed time for each run.

The results below use the same evaluation protocol as the historical GPT-4.1
G0 Planner result in
`scripts/experiment_records/g0_g3_gpt5_4_gpt4_1_results.md`: the unchanged
`avqa_reasoning_datasets/daily_omni/enhanced_eval_results_dspy.py` evaluator,
with overall accuracy computed over all 1,197 questions and held-out accuracy
computed by `clean_heldout_exclude_gemini_unusable_and_train_val`. The 250-ID
exclusion manifest matched all 250 IDs in every run, and no Gemini-unusable
sample was excluded from these three results.

| Workflow | Overall accuracy | Answered accuracy | Not answered | Held-out accuracy | Overall delta vs. G0 Planner | Held-out delta vs. G0 Planner |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical GPT-4.1 G0 Planner | 75.61% (905/1197) | 75.61% (905/1197) | 0 | 75.08% (711/947) | baseline | baseline |
| Simplified | 71.85% (860/1197) | 72.33% (860/1189) | 8 | 71.38% (676/947) | -3.76 pp | -3.70 pp |
| Quartered | 71.43% (855/1197) | 72.21% (855/1184) | 13 | 71.17% (674/947) | -4.18 pp | -3.91 pp |

All parseable final answers in the two ablations were already single option
letters (1,189 simplified and 1,184 quartered). The remaining 8 and 13 rows had
an empty `final_answer` and non-option responses such as `Insufficient evidence`
or `None`, so no additional answer recovery was applied.

### Token usage, cost, and tool calls

This table reports question-average token usage across all persisted planner
and live Gemini calls, including the final planner-only turn. Accuracy is
temporarily reported as **answered accuracy**, not accuracy over all rows. The
held-out column excludes the same 250 Train125/Val125 IDs as the evaluation
above; its full scope contains 947 questions. The answered denominators are
945 for no-GEPA, 947 for G0 Planner, 941 for simplified, and 937 for quartered.
Latency is intentionally omitted pending a corrected latency analysis.

| No. | Experiment name | Reasoning effort | Available tools | Input tok. | Thinking output tok. | Non-thinking output tok. | Latency/question | Cost/question | Full-set answered acc. | Held-out answered acc. | Avg. tool calls/question |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | GPT-4.1 no-GEPA | none | `ask_caption`, `ask_perception` | 13.304k | 0.101k | 0.752k | — | $0.01358 | 71.80% (858/1195) | 71.11% (672/945) | 2.000 |
| G0 | GPT-4.1 G0 Planner | none | `ask_caption`, `ask_perception` | 28.603k | 0.851k | 1.138k | — | $0.03167 | 75.61% (905/1197) | 75.08% (711/947) | 1.796 |
| 14-S | GPT-4.1 G0 Simplified | none | `ask_caption`, `ask_perception` | 13.545k | 0.063k | 0.531k | — | $0.01570 | 72.33% (860/1189) | 71.84% (676/941) | 1.842 |
| 14-Q | GPT-4.1 G0 Quartered | none | `ask_caption`, `ask_perception` | 15.783k | 0.309k | 0.700k | — | $0.01547 | 72.21% (855/1184) | 71.93% (674/937) | 1.777 |

The calculation follows the protocol in
`scripts/experiment_records/g0_gpt4_1_gpt5_4_token_usage_and_latency.md`:

- Input tokens are planner `prompt_tokens` plus live Gemini
  `promptTokenCount`.
- Thinking output is planner `reasoning_tokens` plus Gemini
  `thoughtsTokenCount`. Planner reasoning tokens are zero in all four runs.
- Non-thinking output is the remaining persisted planner and Gemini output;
  Gemini total output is `totalTokenCount - promptTokenCount` before splitting
  out thinking tokens.
- Tool calls count only turns whose planner action is `tool`; the final planner
  decision is excluded. No-GEPA, G0 Planner, simplified, and quartered record
  2,394, 2,150, 2,205, and 2,127 tool calls respectively, divided by 1,197
  questions above.
- Cost uses the same regular-rate scenario as the older record: GPT-4.1 at
  $2.00/M input and $8.00/M output, and Gemini 2.5 Flash at $0.30/M
  text/image/video input, $1.00/M audio input, and $2.50/M output including
  thinking. Gemini calls without modality details receive the observed audio
  fraction for their tool type. Cache discounts are not applied.

Only successfully persisted usage is included. Failed retries without a saved
usage object and provider-specific gateway fees cannot be recovered from these
artifacts. As a protocol check, applying the same calculation to the historical
GPT-4.1 G0 Planner output exactly reproduces its recorded 34,237,267 input
tokens, 2,381,207 total output tokens, and $0.03167/question regular-rate cost.

## AVUT and WorldSense experiment snapshots

The following results are a checkpoint snapshot taken on 2026-10-07 at
13:40 BST. Both AVUT runs were complete. The WorldSense OmniAgent and ReAct
runs had 1,732/3,172 and 2,820/3,172 persisted question checkpoints,
respectively. Partial-run accuracy is therefore reported as **answered
accuracy** (`correct / answered`) and is not directly comparable to full-set
accuracy until those runs finish.

The token columns follow the same live-call protocol as the DailyOmni table:
persisted planner calls plus live Gemini tool calls, excluding a cached first
caption read. Thinking output is o3 reasoning plus Gemini
`thoughtsTokenCount`; non-thinking output is the remaining output. The averages
divide the persisted token totals by all checkpoint questions in that row.

| No. | Experiment name | Reasoning effort | Available tools | Input tok. | Thinking output tok. | Non-thinking output tok. | Latency/question | Cost/question | Acc. | Avg. tool calls/question |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AVUT-O | AVUT Full OmniAgent (o3) | not persisted | `Audio_EventList`, `Audio_EventLocation`, `audio_ASR`, `audio_global_caption`, `audio_qa`, `video_clip_qa`, `video_global_qa`, `video_metadata` | 33.898k | 3.266k | 1.006k | 27.81 s | $0.08312 | 78.57% (1338/1703) | 6.382 |
| AVUT-13 | AVUT o3 + Both Omni Clip Tools | medium | `ask_caption`, `ask_perception`, `omni_clip_caption`, `omni_clip_perception` | 9.426k | 0.671k | 0.293k | 12.13 s | $0.01884 | 79.24% (1374/1734) | 1.943 |
| WS-O | WorldSense Full OmniAgent (o3), partial 1732/3172 | not persisted | `Audio_EventList`, `Audio_EventLocation`, `audio_ASR`, `audio_global_caption`, `audio_qa`, `video_clip_qa`, `video_global_qa`, `video_metadata` | 60.361k | 3.857k | 1.233k | 38.03 s | $0.10299 | 59.05% (979/1658) | 6.990 |
| WS-13 | WorldSense o3 + Both Omni Clip Tools, partial 2820/3172 | medium | `ask_caption`, `ask_perception`, `omni_clip_caption`, `omni_clip_perception` | 20.422k | 0.865k | 0.354k | 16.26 s | $0.02504 | 58.42% (1647/2819) | 2.077 |

For completed runs, latency is full concurrent run wall time divided by the
number of questions. For the two live WorldSense runs it is elapsed time from
process start to the snapshot divided by the checkpoint count, so it is a
provisional throughput figure rather than serial request latency. Persisted
token metadata is present for 1,705/1,734 AVUT OmniAgent checkpoints,
1,662/1,732 WorldSense OmniAgent checkpoints, and every ReAct checkpoint.
Missing usage and failed retries are not reconstructed. Cost uses o3 at
$2.00/M input and $8.00/M output and Gemini 2.5 Flash at $0.30/M non-audio
input, $1.00/M audio input, and $2.50/M output; Gemini calls without modality
details receive the observed audio fraction for the same tool type.

### AVUT by video duration

Clip calls are split by tool. `Clip caption` means `omni_clip_caption`.
`Clip perception` means `omni_clip_perception` for ReAct and `video_clip_qa`
for OmniAgent, which has no clip-caption tool. Each `called (%)` column is the
share of checkpoint questions that used that specific tool, and each mean is
that tool's calls divided by all questions in the duration bucket. Accuracy
remains `correct / answered`.

#### Full OmniAgent (o3)

| Duration | #Q | Clip caption called (%) | Mean clip-caption calls | Clip perception called (%) | Mean clip-perception calls | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤1 min | 1125 | 0.00% | 0.000 | 60.44% | 2.588 | 77.19% (853/1105) |
| 1–3 min | 571 | 0.00% | 0.000 | 48.16% | 2.384 | 80.71% (456/565) |
| 3–5 min | 7 | 0.00% | 0.000 | 57.14% | 4.000 | 100.00% (6/6) |
| >5 min | 31 | 0.00% | 0.000 | 45.16% | 2.419 | 85.19% (23/27) |

#### o3 + Both Omni Clip Tools

| Duration | #Q | Clip caption called (%) | Mean clip-caption calls | Clip perception called (%) | Mean clip-perception calls | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤1 min | 1125 | 7.11% | 0.083 | 24.44% | 0.371 | 77.60% (873/1125) |
| 1–3 min | 571 | 10.16% | 0.147 | 22.24% | 0.315 | 82.31% (470/571) |
| 3–5 min | 7 | 28.57% | 0.286 | 14.29% | 0.143 | 85.71% (6/7) |
| >5 min | 31 | 3.23% | 0.032 | 9.68% | 0.161 | 80.65% (25/31) |

### WorldSense by video duration

#### Full OmniAgent (o3), partial 1732/3172

| Duration | #Q | Clip caption called (%) | Mean clip-caption calls | Clip perception called (%) | Mean clip-perception calls | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤1 min | 680 | 0.00% | 0.000 | 36.03% | 2.543 | 56.89% (384/675) |
| 1–3 min | 720 | 0.00% | 0.000 | 40.56% | 2.926 | 62.10% (444/715) |
| 3–5 min | 172 | 0.00% | 0.000 | 38.95% | 2.523 | 53.61% (89/166) |
| >5 min | 160 | 0.00% | 0.000 | 31.25% | 2.219 | 60.78% (62/102) |

#### o3 + Both Omni Clip Tools, partial 2820/3172

| Duration | #Q | Clip caption called (%) | Mean clip-caption calls | Clip perception called (%) | Mean clip-perception calls | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤1 min | 973 | 5.65% | 0.073 | 16.96% | 0.232 | 57.45% (559/973) |
| 1–3 min | 1226 | 5.46% | 0.068 | 16.07% | 0.230 | 58.86% (721/1225) |
| 3–5 min | 295 | 6.44% | 0.095 | 16.27% | 0.258 | 55.59% (164/295) |
| >5 min | 326 | 5.21% | 0.058 | 17.18% | 0.242 | 62.27% (203/326) |

The duration evaluator uses source-cut seconds, with a 0.5-second tolerance at
the 1/3/5-minute boundaries to keep nominal boundary videos in their intended
bucket after transcoding. Benchmark task categories are read dynamically rather
than using DailyOmni's six fixed categories. Reproduce the reports with:

```bash
python /mnt/ceph_rbd/workspace/avqa_project/avqa_reasoning_datasets/AVUT/enhanced_eval_results_agents.py
python /mnt/ceph_rbd/workspace/avqa_project/avqa_reasoning_datasets/worldsense/enhanced_eval_results_agents.py
```
