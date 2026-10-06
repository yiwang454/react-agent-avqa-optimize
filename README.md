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
| 10 | o3 + Omni Clip Caption | medium | `ask_caption`, `ask_perception`, `omni_clip_caption` | 9.2k | 0.83k | 0.30k | 11.35 s | $0.0175 | 78.36% | 1.61 |
| 11 | o3 Basic Tools | medium | `ask_caption`, `ask_perception` | 7.8k | 0.78k | 0.28k | 11.58 s | $0.0157 | 78.86% | 1.55 |
| 12 | o3 + Omni Clip Perception | medium | `ask_caption`, `ask_perception`, `omni_clip_perception` | 8.7k | 0.93k | 0.32k | 12.50 s | $0.0187 | 78.20% | 1.66 |
| 11b | o3 Basic Tools (High Reasoning) | high | `ask_caption`, `ask_perception` | 9.9k | 1.61k | 0.33k | 14.38 s | $0.0243 | 79.53% | 1.63 |

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
tool calls than Full OmniAgent. Adding clip-level caption or perception tools did
not improve aggregate accuracy in these runs, although those tools provide
additional traces for studying time-localized evidence requests.

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
