# Free-ReAct Audio-Visual QA Experiments

## Main contributions
- The proposed agentic AVQA workflow outperforms (AVUT/DailyOmni) or obtains comparative performance compared with the SOTA agentic AVQA method OmniAgent, with substantially lower cost (and shorter latency), suggesting better efficiency.
- The proposed agentic workflow slightly outperformed the direct OmniLLM inference results on AVQA benches (by 1.8% on DailyOmni, 1.2 on AVUT and 1.9% on WorldSense), but showed significant better accuracy on >= 5min long videos than direct inference.
- We experimented instruction optimization to enhance the performance when using less strengthened reasoning model (GPT 4.1) but didn't generalize such gain on more powerful reasoning model (GPT o3) - future work is to extend this instruction optimization on long video understanding where agentic framework shows larger benefit. 

## Main benchmark results

All token, tool-call, and cost columns are averages per question. `Latency` is
the mean of the persisted per-question `retry_adjusted_seconds`: it starts at
the first planner model request for that question, ends after its final answer,
and removes failed request attempts and their retry backoff. It is independent
of the benchmark worker count and is not full-benchmark wall time divided by the
number of questions. `—` means that the corresponding output does not persist
this question-level latency schema; all Full OmniAgent latency cells are
therefore intentionally blank.

### DailyOmni

| No. | Experiment | Output | Reasoning effort | Available tools | Accuracy | Input tok. | Output thinking tok. | Output non-thinking tok. | Tool calls | Cost | Latency |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D | Gemini 2.5 Flash Direct (paper baseline) | [JSONL](artifacts/results/daily_omni/direct_gemini_2_5_flash/output_test.jsonl) | Gemini default thinking | none (direct video QA) | 77.03% (922/1197) | 5.048k | 0.004k | 0.002k | 0.000 | $0.00154 | — |
| O | Full OmniAgent (o3) | [JSONL.gz](artifacts/results/daily_omni/omniagent_replication/output_test.jsonl.gz) | high | `Audio_EventList`, `Audio_EventLocation`, `audio_ASR`, `audio_global_caption`, `audio_qa`, `video_clip_qa`, `video_global_qa`, `video_metadata` | 77.53% (928/1197) | 62.7k | 3.6k† | 1.3k† | 7.27 | $0.0674 | — |
| 10 | o3 + Omni Clip Caption | [JSONL](artifacts/results/daily_omni/experiment_10_omni_clip_caption/output_test.jsonl) | medium | `ask_caption`, `ask_perception`, `omni_clip_caption` | 78.11% (935/1197) | 9.0k | 0.81k | 0.30k | 1.59 | $0.0173 | — |
| 11 | o3 Basic Tools | [JSONL](artifacts/results/daily_omni/experiment_11_basic_tools/output_test.jsonl) | medium | `ask_caption`, `ask_perception` | **78.86% (944/1197)** | 7.8k | 0.78k | 0.28k | 1.55 | $0.0157 | — |
| 12 | o3 + Omni Clip Perception | — | medium | `ask_caption`, `ask_perception`, `omni_clip_perception` | 79.03% (946/1197) | 8.1k | 0.79k | 0.30k | 1.58 | $0.0171 | — |
<!-- | 11b | o3 Basic Tools (High Reasoning) | — | high | `ask_caption`, `ask_perception` | 79.53% (952/1197) | 9.9k | 1.61k | 0.33k | 1.63 | $0.0243 | — | -->

DailyOmni Direct uses the exact paper-baseline run requested from the sibling
experiment ledger: `daily_omni_seed27_repeat3_gemini-2.5-flash_QA_PROMPT_TEMPLATE_0.0`.
Its accuracy is the ledger's reported overall accuracy; the token and cost
columns are recomputed from that run's persisted usage. Experiment 13 has 1,197
unique output IDs, 1,197 parsed answers, complete inference-only metadata, and
1,197 non-null retry-adjusted latency values.

### AVUT

| No. | Experiment | Reasoning effort | Available tools | Accuracy | Input tok. | Output thinking tok. | Output non-thinking tok. | Tool calls | Cost | Latency |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D | Gemini 2.5 Flash Direct OmniLLM | Gemini default thinking | none (direct video QA) | 79.35% (1376/1734) | 11.457k | 0.143k | 0.006k | 0.000 | $0.00437 | — |
| O | Full OmniAgent (o3) | high | `Audio_EventList`, `Audio_EventLocation`, `audio_ASR`, `audio_global_caption`, `audio_qa`, `video_clip_qa`, `video_global_qa`, `video_metadata` | 77.16% (1338/1734) | 33.898k | 3.266k | 1.006k | 6.382 | $0.08312 | — |
| 11 | o3 Basic Tools | medium | `ask_caption`, `ask_perception` | **80.57% (1397/1734)** | 9.081k | 0.490k | 0.263k | 1.777 | $0.01594 | 27.55 s‡ |
| 13 | o3 + Both Omni Clip Tools | medium | `ask_caption`, `ask_perception`, `omni_clip_caption`, `omni_clip_perception` | 79.24% (1374/1734) | 9.426k | 0.671k | 0.293k | 1.943 | $0.01884 | — |

‡ AVUT-11 has 1,733 non-null retry-adjusted latency values. Its one terminal
failure retains wall-clock and successful-call timing but correctly stores
`retry_adjusted_seconds: null`; 27.55 s is the mean over the 1,733 defined
question latencies.

### WorldSense

| No. | Experiment | Output | Reasoning effort | Available tools | Accuracy | Input tok. | Output thinking tok. | Output non-thinking tok. | Tool calls | Cost | Latency |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D | Gemini 2.5 Flash Direct OmniLLM | [JSONL](artifacts/results/worldsense/direct_gemini_2_5_flash/output_test.jsonl) | Gemini default thinking | none (direct video QA) | 56.43% (1790/3172) | 22.517k | 0.289k | 0.014k | 0.000 | $0.00874 | — |
| O | Full OmniAgent (o3), repair pending; 2852/3172 answered | [JSONL.gz](artifacts/results/worldsense/omniagent_replication/output_test.jsonl.gz) | high | `Audio_EventList`, `Audio_EventLocation`, `audio_ASR`, `audio_global_caption`, `audio_qa`, `video_clip_qa`, `video_global_qa`, `video_metadata` | **59.57% (1699/2852)†** | 62.003k | 3.820k | 1.211k | 6.652 | $0.10155 | — |
| 11 | o3 Basic Tools | [JSONL.gz](artifacts/results/worldsense/experiment_11_basic_tools/output_test.jsonl.gz) | medium | `ask_caption`, `ask_perception` | **58.39% (1852/3172)** | 17.721k | 0.629k | 0.313k | 1.986 | $0.02099 | 31.29 s |
| 13 | o3 + Both Omni Clip Tools | [JSONL.gz](artifacts/results/worldsense/experiment_13_both_omni_clip_tools/output_test.jsonl.gz) | medium | `ask_caption`, `ask_perception`, `omni_clip_caption`, `omni_clip_perception` | 57.88% (1836/3172) | 20.307k | 0.859k | 0.354k | 2.087 | $0.02493 | — |

WorldSense-11 has a non-null retry-adjusted latency for every one of its 3,172
questions. Full-set accuracy is used throughout the three main tables except
for the repair-pending WorldSense Full OmniAgent row. AVUT Full OmniAgent
additionally has 78.57% answered accuracy (1,338/1,703).

† WorldSense Full OmniAgent has attempted all 3,172 questions, but 320 error
rows are still unanswered and require repair. Its current result therefore uses
answered accuracy (`correct / answered`), not final full-set accuracy.
<!-- † The DailyOmni Full OmniAgent summary reports 4.9k total output tokens per
question. Its trace preserves the o3 planner split (3.55k reasoning and 0.69k
visible output) but does not preserve Gemini tool-call thinking-token metadata.
The remaining reported output is assigned to non-thinking output, so this split
is approximate. The Free-ReAct rows use exact persisted o3 reasoning-token and
Gemini `thoughtsTokenCount` fields. -->

OmniAgent replication uses
https://github.com/yiwang454/OmniAgent_replicate.git, cloned from
https://github.com/KD-TAO/OmniAgent.git.

## DailyOmni experiment details

This branch evaluates an o3 planner in a Free-ReAct audio-visual question
answering workflow. A Free-ReAct planner cannot inspect a video directly: it
must first obtain a whole-video caption and can then decide whether to request
more evidence or return a final answer. The Free-ReAct experiments use Gemini
2.5 Flash for captioning and perception and allow at most six tool calls per
question.

Rows 10 and 12 were refreshed from their completed aggregate outputs on
2026-10-07. Experiment 10 has 935/1,197 overall accuracy and 935/1,196
answered accuracy; Experiment 12 has 946/1,197 for both. Their repair launches
reused 1,168 and 1,103 existing checkpoints, respectively, and their older
timing values were benchmark-throughput figures, so their main-table latency is
left blank rather than mixed with question-level retry-adjusted latency.

`Tool calls` uses one definition for every row: it counts only turns whose
planner action is `tool`. The final decision round performed by the reasoner is
excluded. The DailyOmni Full OmniAgent value is therefore 8,700 tool calls /
1,197 questions = 7.27, computed from
`OmniAgent_repeat1/output_test.jsonl`.

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

### 13. o3 Basic Tools + GEPA Guidance

The full-v3 inference loads the seed-2 compiled GEPA guidance with:

```bash
bash scripts/run_c1_seed2_candidate6_full_inference.sh
```

It keeps the Experiment 11 tool set and medium-reasoning planner, and appends
the optimized `planner.optimization_guidance`. The completed output and latency
summary are in:

```text
/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_free_react_o3_gemini25flash_guidance_seed2_candidate6_full_v3/
```

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
- **Output thinking tok.**: hidden o3 reasoning tokens plus Gemini thinking
  tokens, when the backend persisted both fields.
- **Output non-thinking tok.**: output tokens excluding the recorded hidden
  reasoning/thinking tokens.
- **Latency**: mean per-question `retry_adjusted_seconds`. Each question is
  measured independently from its first planner request through its final
  answer. Failed request durations and the following retry backoff are removed;
  worker concurrency and full-benchmark wall time do not enter the value.
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

The high-reasoning basic-tools run achieved the highest DailyOmni accuracy at
79.53%, with higher thinking-token usage and cost. The medium-reasoning
basic-tools run was the least expensive o3 Free-ReAct system. All Free-ReAct
variants used far fewer tool calls than Full OmniAgent. Clip captioning did not
improve aggregate accuracy over the medium basic-tools run; clip perception
produced a small improvement (79.03% versus 78.86%) while remaining below high
reasoning. The GEPA-guidance full-v3 run reached 77.94%.

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
Only the simplified and quartered outputs persist the question-level latency
schema; their latency cells use the same retry-adjusted mean as the main tables.

| No. | Experiment name | Reasoning effort | Available tools | Full-set answered acc. | Held-out answered acc. | Input tok. | Output thinking tok. | Output non-thinking tok. | Tool calls | Cost | Latency |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | GPT-4.1 no-GEPA | none | `ask_caption`, `ask_perception` | 71.80% (858/1195) | 71.11% (672/945) | 13.304k | 0.101k | 0.752k | 2.000 | $0.01358 | — |
| G0 | GPT-4.1 G0 Planner | none | `ask_caption`, `ask_perception` | 75.61% (905/1197) | 75.08% (711/947) | 28.603k | 0.851k | 1.138k | 1.796 | $0.03167 | — |
| 14-S | GPT-4.1 G0 Simplified | none | `ask_caption`, `ask_perception` | 72.33% (860/1189) | 71.84% (676/941) | 13.545k | 0.063k | 0.531k | 1.842 | $0.01570 | 34.56 s |
| 14-Q | GPT-4.1 G0 Quartered | none | `ask_caption`, `ask_perception` | 72.21% (855/1184) | 71.93% (674/937) | 15.783k | 0.309k | 0.700k | 1.777 | $0.01547 | 24.87 s |

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

## AVUT and WorldSense experiment details

The main tables at the top were audited on 2026-10-08. The Direct Gemini, Basic
Tools, AVUT Full OmniAgent, and Both Omni Clip Tools rows are complete full-set
runs with exact input-ID coverage. They report full-set accuracy (`correct / all
questions`).
WorldSense OmniAgent has attempted all 3,172 questions, but 320 error rows are
still unanswered and require repair. Its current accuracy is therefore reported
as **answered accuracy** (`correct / answered`), not as final full-set accuracy.

For ReAct rows, the token columns follow the same live-call protocol as the
DailyOmni table: persisted planner calls plus live Gemini tool calls, excluding
a cached first-caption read. Thinking output is o3 reasoning plus Gemini
`thoughtsTokenCount`; non-thinking output is the remaining output. Direct rows
contain the single Gemini video-QA call and have no tool calls. Averages divide
the persisted token totals by the full row count. Only Basic Tools currently
persists comparable question-level retry-adjusted latency; older throughput
figures are not mixed into the main tables.

AVUT Direct includes the original pass plus the 88-question re-encoded
recovery. WorldSense Both Omni Clip Tools completed all 3,172 questions. Its
one previously unparseable sample, `HRXUIIaw-2164`, was rerun on
2026-10-08 and now has a valid answer (A); it is incorrect against gold C, so
the final accuracy remains 57.88% (1,836/3,172). WorldSense OmniAgent currently
has 1,699 correct and 2,852 answered questions, giving 59.57% answered accuracy.
The remaining 320 rows are not counted as wrong for this live/repair-pending
result. Persisted token metadata is present for 1,705/1,734 AVUT OmniAgent checkpoints,
2,861/3,172 WorldSense OmniAgent
checkpoints, every Basic Tools checkpoint, every AVUT Direct row, and 3,070 of
3,172 WorldSense Direct rows. Missing usage and failed retries are not
reconstructed. Cost uses o3 at $2.00/M input and $8.00/M output and Gemini 2.5
Flash at $0.30/M non-audio input, $1.00/M audio input, and $2.50/M output;
Gemini calls without modality details receive the observed audio fraction for
the same tool type.

AVUT Direct combines 1,646 original rows with 88 successful re-encoded retries.
Applying the canonical long-response parser gives 1,720 answered and 1,376
correct rows; the table reports full-set accuracy. For WorldSense Direct, the
same parser recovers 91 of 138 originally `invalid_answer` rows (53 correct),
leaving 47 truly unparsed/truncated rows, 102 transport errors, and one empty
response. Its 1,790/3,172 figure is likewise full-set accuracy. The audited
artifacts are:

- Basic Tools: `/mnt/ceph_rbd/data/avqa_project/avut/avut_dspy_free_react_o3_gemini25flash_basic_tools_whole_video_timestamp3_maxturns6/output_test.jsonl` and `/mnt/ceph_rbd/data/avqa_project/WorldSense/worldsense_dspy_free_react_o3_gemini25flash_basic_tools_whole_video_timestamp3_maxturns6/output_test.jsonl`.
- Direct OmniLLM: `/mnt/ceph_rbd/data/avqa_project/avut/mllm_instruct_opt/avut_gemini2.5flash_direct_baseline_full_reencoded_recovery/output_test.jsonl` and `/mnt/ceph_rbd/data/avqa_project/WorldSense/mllm_instruct_opt/worldsense_gemini2.5flash_direct_baseline_full/output_test.jsonl`.

### AVUT by video duration

The consolidated table below uses full-set accuracy in every bucket, so an
unanswered or unparseable question counts as incorrect. The denominators are
the exact source-manifest bucket sizes and sum to all 1,734 AVUT questions.

| No. | Experiment | Overall | ≤1 min (n=1125) | 1–3 min (n=571) | 3–5 min (n=7) | >5 min (n=31) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| D | Gemini 2.5 Flash Direct OmniLLM | 79.35% (1376/1734) | 78.22% (880/1125) | 81.79% (467/571) | 85.71% (6/7) | 74.19% (23/31) |
| O | Full OmniAgent (o3) | 77.16% (1338/1734) | 75.82% (853/1125) | 79.86% (456/571) | 85.71% (6/7) | 74.19% (23/31) |
| 11 | o3 Basic Tools | **80.57% (1397/1734)** | **79.11% (890/1125)** | **83.19% (475/571)** | **85.71% (6/7)** | **83.87% (26/31)** |
| 13 | o3 + Both Omni Clip Tools | 79.24% (1374/1734) | 77.60% (873/1125) | 82.31% (470/571) | 85.71% (6/7) | 80.65% (25/31) |

#### Clip-tool detail: Full OmniAgent (o3)

Clip calls are split by tool. `Clip caption` means `omni_clip_caption`.
`Clip perception` means `omni_clip_perception` for ReAct and `video_clip_qa`
for OmniAgent, which has no clip-caption tool. Each `called (%)` column is the
share of checkpoint questions that used that specific tool, and each mean is
that tool's calls divided by all questions in the duration bucket. Accuracy
in the two tool-detail tables remains `correct / answered`, unlike the
full-set consolidated table above.

| Duration | #Q | Clip caption called (%) | Mean clip-caption calls | Clip perception called (%) | Mean clip-perception calls | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤1 min | 1125 | 0.00% | 0.000 | 60.44% | 2.588 | 77.19% (853/1105) |
| 1–3 min | 571 | 0.00% | 0.000 | 48.16% | 2.384 | 80.71% (456/565) |
| 3–5 min | 7 | 0.00% | 0.000 | 57.14% | 4.000 | 100.00% (6/6) |
| >5 min | 31 | 0.00% | 0.000 | 45.16% | 2.419 | 85.19% (23/27) |

#### Clip-tool detail: o3 + Both Omni Clip Tools

| Duration | #Q | Clip caption called (%) | Mean clip-caption calls | Clip perception called (%) | Mean clip-perception calls | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤1 min | 1125 | 7.11% | 0.083 | 24.44% | 0.371 | 77.60% (873/1125) |
| 1–3 min | 571 | 10.16% | 0.147 | 22.24% | 0.315 | 82.31% (470/571) |
| 3–5 min | 7 | 28.57% | 0.286 | 14.29% | 0.143 | 85.71% (6/7) |
| >5 min | 31 | 3.23% | 0.032 | 9.68% | 0.161 | 80.65% (25/31) |

### WorldSense by video duration

The WorldSense split uses the same full-set protocol. Its four source-manifest
buckets contain all 3,172 questions. Experiment O here is the completed
3,172-checkpoint audit used by the tool-detail tables below; the main table at
the top separately shows a still-incomplete OmniAgent aggregation snapshot.

| No. | Experiment | Overall | ≤1 min (n=1059) | 1–3 min (n=1386) | 3–5 min (n=348) | >5 min (n=379) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| D | Gemini 2.5 Flash Direct OmniLLM | 56.43% (1790/3172) | **57.70% (611/1059)** | 58.80% (815/1386) | 53.45% (186/348) | 46.97% (178/379) |
| O | Full OmniAgent (o3) | 53.56% (1699/3172) | 54.39% (576/1059) | 57.29% (794/1386) | 50.00% (174/348) | 40.90% (155/379) |
| 11 | o3 Basic Tools | **58.39% (1852/3172)** | 55.43% (587/1059) | **59.52% (825/1386)** | **56.90% (198/348)** | **63.85% (242/379)** |
| 13 | o3 + Both Omni Clip Tools | 57.88% (1836/3172) | 57.22% (606/1059) | 58.51% (811/1386) | 53.16% (185/348) | 61.74% (234/379) |

WorldSense-11 is strongest in every bucket above one minute. The largest
separation is on videos over five minutes: 63.85% for experiment 11 versus
46.97% for Direct and 40.90% for Full OmniAgent; experiment 13 reaches 61.74%.

#### Clip-tool detail: Full OmniAgent (o3), repair pending; 2852/3172 answered

| Duration | #Q | Clip caption called (%) | Mean clip-caption calls | Clip perception called (%) | Mean clip-perception calls | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤1 min | 1059 | 0.00% | 0.000 | 37.02% | 2.701 | 57.49% (576/1002) |
| 1–3 min | 1386 | 0.00% | 0.000 | 38.60% | 2.907 | 61.22% (794/1297) |
| 3–5 min | 348 | 0.00% | 0.000 | 37.07% | 2.316 | 55.41% (174/314) |
| >5 min | 379 | 0.00% | 0.000 | 25.33% | 1.958 | 64.85% (155/239) |

#### Clip-tool detail: o3 + Both Omni Clip Tools

| Duration | #Q | Clip caption called (%) | Mean clip-caption calls | Clip perception called (%) | Mean clip-perception calls | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤1 min | 1059 | 5.95% | 0.076 | 17.28% | 0.240 | 57.22% (606/1059) |
| 1–3 min | 1386 | 5.63% | 0.069 | 15.80% | 0.224 | 58.51% (811/1386) |
| 3–5 min | 348 | 6.32% | 0.092 | 17.24% | 0.282 | 53.16% (185/348) |
| >5 min | 379 | 5.28% | 0.058 | 17.41% | 0.243 | 61.74% (234/379) |

The duration evaluator uses source-cut seconds, with a 0.5-second tolerance at
the 1/3/5-minute boundaries to keep nominal boundary videos in their intended
bucket after transcoding. Benchmark task categories are read dynamically rather
than using DailyOmni's six fixed categories. Direct parsing follows the same
audited protocol as the main tables: AVUT applies the canonical long-response
parser to every row; WorldSense retains valid persisted predictions and applies
that parser to `invalid_answer` responses. Failed and still-unparsed questions
remain in the full-set denominator. The duration joins use the Direct and Basic
Tools artifacts listed above together with
`/mnt/ceph_rbd/data/avqa_project/avut/mllm_instruct_opt/avut_gemini2.5flash_direct_safe_duration_sorted_input/avut_full1734_duration_ascending_safe_videos.jsonl`
and
`/mnt/ceph_rbd/data/avqa_project/WorldSense/worldsense_test_cut_old.jsonl`;
all four Direct/Basic-Tools joins have exact ID coverage with no missing or
extra IDs.
