# Stage B v2: rerun-substituted sensitivity analysis

## Outline

- [Experiment path](#experiment-path)
- [Replacement audit](#replacement-audit)
- [Results](#results)
- [Fixed controls and provenance](#fixed-controls-and-provenance)
- [Interpretation limits](#interpretation-limits)

## Experiment path

This document is a sensitivity analysis of the completed Stage B v2 study in
[README_ablation.md](README_ablation.md). It substitutes four newly generated
runs into their selected logical repeat slots while leaving the other 20 runs
unchanged. No original output was overwritten. The B2 repeat2 rerun was
reviewed but rejected as a replacement, so the original B2 repeat2 remains in
all aggregate and logical-repeat tables.

| Logical slot replaced | Configuration | Replacement output |
| --- | --- | --- |
| B0 repeat3 | OmniAgent workflow, specialized C+Q+M, original Gemini sampling | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_outlier_reruns/b0-repeat3-rerun/output_test.jsonl) |
| B1 repeat2 | B0 with Experiment 11 Gemini sampling | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_outlier_reruns/b1-repeat2-rerun/output_test.jsonl) |
| B3 repeat1 | B2 with `ask_caption` replacing audio-only captioning | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_outlier_reruns/b3-repeat1-rerun/output_test.jsonl) |
| B5-flex repeat2 | Omni caption and QA, free order, conversation-state caption | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_outlier_reruns/b5-flex-repeat2-rerun/output_test.jsonl) |

Rejected candidate: [B2 repeat2 rerun](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_outlier_reruns/b2-repeat2-rerun/output_test.jsonl)
scored 66.4%, farther from the other B2 runs than the original repeat2. It is
retained as an audit artifact but is not used below.

The complete configuration path remains:

| ID | Change from prior run | Caption | QA | Planner-visible M | First call / caption placement |
| --- | --- | --- | --- | --- | --- |
| B0 | Move OmniAgent −A−E into DSPy with original workflow and specialized tools | `audio_global_caption` | Specialized Q | Yes | Free / conversation state |
| B1 | Gemini sampling only: 0.6/0.95/20 → 0/1/0 | Same | Same | Yes | Same |
| B2 | Replace the whole OmniAgent workflow with the Experiment 11 Basic Tools workflow, adapted for C+Q+M and free order | Same | Same | Yes | Free / conversation state |
| B3 | Replace audio-only caption with `ask_caption` | `ask_caption` | Specialized Q | Yes | Free / conversation state |
| B4 | Require first-round `ask_caption` and move its full observation into the task | `ask_caption` | Specialized Q | Yes | Required first / task |
| B5 | Replace specialized Q with global and clip omni QA | `ask_caption` | `ask_perception` + `omni_clip_perception` | Yes | Required first / task |
| B5-flex | Same tools as B5, but use free order and keep caption in conversation state | Same | Same | Yes | Free / conversation state |
| B6 | Hide `video_metadata` from the planner; retain clip QA | `ask_caption` | `ask_perception` + `omni_clip_perception` | No | Required first / task |

Specialized Q is `audio_qa + video_global_qa + video_clip_qa`.

## Replacement audit

All five candidate rerun outputs have 125 rows and the same 125 unique
question IDs as the original study. Three candidates have 125 natural final
answers. The accepted B0 and B1 replacements each contain one non-parseable
API-error row, detailed below; their reported accuracy therefore counts that
row as incorrect.

| Logical slot | Original → replacement accuracy | Calls / question | Live cost / question | Latency / question | Replacement final status |
| --- | ---: | ---: | ---: | ---: | --- |
| B0 repeat3 | 72.0% → 74.4% | 4.776 → 4.136 | $0.08995 → $0.07397 | 64.93 → 60.69 s | 124 natural; 1 error |
| B1 repeat2 | 78.4% → 77.6% | 4.104 → 4.144 | $0.07915 → $0.07499 | 58.16 → 60.09 s | 124 natural; 1 error |
| B2 repeat2 (rejected) | 73.6% → 66.4% | 3.376 → 3.032 | $0.04937 → $0.03631 | 40.24 → 37.81 s | 125 natural; original retained |
| B3 repeat1 | 77.6% → 74.4% | 2.728 → 2.576 | $0.05229 → $0.04967 | 46.75 → 35.67 s | 125 natural |
| B5-flex repeat2 | 81.6% → 80.0% | 1.920 → 1.792 | $0.03209 → $0.02570 | 24.31 → 25.51 s | 125 natural |

The B0 error is question `6_Foh42T5XM-1`; the B1 error is
`SfZrnoo1GPM-1`. Both stopped at turn 1 after an upstream Gemini HTTP 400
“Request contains an invalid argument” response. They did not reach a natural
or forced final answer. For those two runs, zero recorded tokens/cost from the
error row lower the 125-question token and cost averages slightly, while
latency is averaged over the 124 rows with retry-adjusted latency.

## Results

These tables use each accepted replacement in the logical repeat slot listed
above. The other 20 cells, including B2 repeat2, use the original outputs. A
dagger marks a replacement rather than an original repeat result. Live
marginal cost excludes caption-generation cost for observations read from
cache.

### Three-run mean and standard deviation after substitution

Values are mean ± **sample standard deviation** across the three logical
repeat slots (`n=3`). Per-question measures are first averaged within each
125-question run and then summarized across the three runs.

| ID | Change from prior run | Accuracy | Calls / question | Live input tokens / question | Live output tokens / question | Live cost / question | Latency / question |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | Move OmniAgent −A−E into DSPy with its original workflow and specialized C+Q+M | 75.47% ± 1.22% | 4.309 ± 0.205 | 18,788 ± 1,206 | 6,358 ± 293 | $0.07934 ± $0.00542 | 64.73 ± 8.64 s |
| B1 | Gemini sampling only: 0.6/0.95/20 → 0/1/0 | 76.00% ± 1.60% | 4.232 ± 0.081 | 18,752 ± 158 | 6,204 ± 329 | $0.07683 ± $0.00224 | 61.84 ± 7.02 s |
| B2 | Replace the OmniAgent workflow with the Experiment 11 workflow adapted to specialized C+Q+M | 72.27% ± 1.22% | 3.005 ± 0.408 | 10,832 ± 1,547 | 3,519 ± 662 | $0.04235 ± $0.00759 | 39.33 ± 9.55 s |
| B3 | Replace `audio_global_caption` with `ask_caption` | 74.67% ± 0.46% | 2.496 ± 0.167 | 10,303 ± 891 | 3,767 ± 387 | $0.04602 ± $0.00508 | 32.68 ± 3.87 s |
| B4 | Add mandatory caption-first ordering and task placement | 77.87% ± 0.46% | 2.397 ± 0.107 | **7,983 ± 326** | 2,988 ± 106 | $0.03586 ± $0.00128 | 29.35 ± 6.65 s |
| B5 | Replace specialized Q with `ask_perception` + `omni_clip_perception` | 74.40% ± 1.39% | 2.013 ± 0.072 | 8,550 ± 475 | 2,188 ± 280 | $0.02829 ± $0.00352 | 24.50 ± 6.45 s |
| B5-flex | B5 control: remove caption-first and keep caption in conversation state | 77.33% ± 2.81% | **1.661 ± 0.146** | 8,292 ± 861 | **1,718 ± 277** | **$0.02329 ± $0.00317** | 22.99 ± 3.15 s |
| B6 | From B5, hide planner-visible `video_metadata` | **78.40% ± 1.60%** | 1.941 ± 0.103 | 8,048 ± 623 | 1,962 ± 273 | $0.02555 ± $0.00340 | **21.43 ± 2.95 s** |

| ID | Median calls | P95 calls | Maximum calls | Questions over 6 / 125 | Invalid calls across 3 runs | Reached budget across 3 runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 3.00 ± 0.00 | 12.00 ± 1.00 | 27.67 ± 2.31 | 20.67 ± 2.08 | 13 | 0 |
| B1 | 3.00 ± 0.00 | 10.67 ± 0.58 | 25.00 ± 3.00 | 20.00 ± 1.73 | 18 | 0 |
| B2 | 2.00 ± 0.00 | 9.00 ± 2.00 | 24.67 ± 3.79 | 10.33 ± 4.16 | 1 | 0 |
| B3 | 2.00 ± 0.00 | 7.00 ± 1.00 | 21.67 ± 6.03 | 8.67 ± 2.31 | 7 | 0 |
| B4 | 2.00 ± 0.00 | 6.00 ± 1.00 | 19.00 ± 8.54 | 4.67 ± 2.89 | 1 | 0 |
| B5 | 2.00 ± 0.00 | 4.00 ± 0.00 | 20.67 ± 8.08 | 3.00 ± 1.00 | 0 | 0 |
| B5-flex | **1.00 ± 0.00** | **3.33 ± 0.58** | **13.67 ± 8.08** | **1.67 ± 1.53** | 0 | 0 |
| B6 | 2.00 ± 0.00 | 3.67 ± 0.58 | 18.33 ± 8.50 | 2.00 ± 0.00 | 0 | 0 |

Bold identifies the best mean in each primary metric column, not a claim of
statistical significance. The observed maximum still exceeds six calls for
every configuration, including B5-flex and B6.

### Logical repeat 1

| ID | Correct / 125 | Accuracy | Total tool calls | Calls / question | Live input tokens / question | Live output tokens / question | Live cost / question | Latency / question |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [B0](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat1/b0/output_test.jsonl) | 96 | 76.8% | 567 | 4.536 | 20,131 | 6,654 | $0.08481 | 74.65 s |
| [B1](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat1/b1/output_test.jsonl) | 93 | 74.4% | 531 | 4.248 | 18,731 | 6,481 | $0.07932 | 69.58 s |
| [B2](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat1/b2/output_test.jsonl) | 90 | 72.0% | 384 | 3.072 | 11,041 | 3,604 | $0.04339 | 48.39 s |
| [B3†](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_outlier_reruns/b3-repeat1-rerun/output_test.jsonl) | 93 | 74.4% | 322 | 2.576 | 11,270 | 4,012 | $0.04967 | 35.67 s |
| [B4](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat1/b4/output_test.jsonl) | 97 | 77.6% | 315 | 2.520 | 8,260 | 3,074 | $0.03727 | 37.00 s |
| [B5](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat1/b5/output_test.jsonl) | 92 | 73.6% | 247 | 1.976 | 8,612 | 2,143 | $0.02818 | 31.61 s |
| [B5-flex](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat1/b5-flex/output_test.jsonl) | 97 | 77.6% | 188 | 1.504 | 7,389 | 1,398 | $0.01969 | 24.01 s |
| [B6](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat1/b6/output_test.jsonl) | 96 | 76.8% | 232 | 1.856 | 7,603 | 1,717 | $0.02306 | 24.36 s |

| ID | Executed tool calls | Median / P95 / max calls | Questions over 6 | Invalid calls | Reached budget | Final status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| B0 | `audio_global_caption` 92; `audio_qa` 200; `video_clip_qa` 193; `video_global_qa` 57; `video_metadata` 25 | 3 / 13 / 29 | 23 | 5 | 0 | 125 natural |
| B1 | `audio_global_caption` 93; `audio_qa` 180; `video_clip_qa` 169; `video_global_qa` 65; `video_metadata` 24 | 3 / 11 / 28 | 19 | 4 | 0 | 125 natural |
| B2 | `audio_global_caption` 51; `audio_qa` 129; `video_clip_qa` 84; `video_global_qa` 103; `video_metadata` 17 | 2 / 9 / 29 | 9 | 0 | 0 | 125 natural |
| B3† | `ask_caption` 109; `audio_qa` 99; `video_clip_qa` 69; `video_global_qa` 39; `video_metadata` 6 | 2 / 7 / 28 | 10 | 4 | 0 | 125 natural |
| B4 | `ask_caption` 141; `audio_qa` 76; `video_clip_qa` 68; `video_global_qa` 30 | 2 / 7 / 18 | 8 | 0 | 0 | 125 natural |
| B5 | `ask_caption` 130; `ask_perception` 92; `omni_clip_perception` 25 | 2 / 4 / 22 | 2 | 0 | 0 | 125 natural |
| B5-flex | `ask_caption` 69; `ask_perception` 104; `omni_clip_perception` 13; `video_metadata` 2 | 1 / 3 / 5 | 0 | 0 | 0 | 125 natural |
| B6 | `ask_caption` 125; `ask_perception` 90; `omni_clip_perception` 17 | 2 / 3 / 12 | 2 | 0 | 0 | 125 natural |

### Logical repeat 2

| ID | Correct / 125 | Accuracy | Total tool calls | Calls / question | Live input tokens / question | Live output tokens / question | Live cost / question | Latency / question |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [B0](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat2/b0/output_test.jsonl) | 94 | 75.2% | 532 | 4.256 | 18,432 | 6,352 | $0.07924 | 58.85 s |
| [B1†](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_outlier_reruns/b1-repeat2-rerun/output_test.jsonl) | 97 | 77.6% | 518 | 4.144 | 18,919 | 5,840 | $0.07499 | 60.09 s |
| [B2](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat2/b2/output_test.jsonl) | 92 | 73.6% | 422 | 3.376 | 12,263 | 4,135 | $0.04937 | 40.24 s |
| [B3](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat2/b3/output_test.jsonl) | 94 | 75.2% | 326 | 2.608 | 10,126 | 3,966 | $0.04818 | 34.06 s |
| [B4](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat2/b4/output_test.jsonl) | 97 | 77.6% | 291 | 2.328 | 7,623 | 2,870 | $0.03477 | 25.07 s |
| [B5](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat2/b5/output_test.jsonl) | 92 | 73.6% | 262 | 2.096 | 8,991 | 2,487 | $0.03187 | 22.85 s |
| [B5-flex†](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_outlier_reruns/b5-flex-repeat2-rerun/output_test.jsonl) | 100 | 80.0% | 224 | 1.792 | 9,105 | 1,886 | $0.02570 | 25.51 s |
| [B6](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat2/b6/output_test.jsonl) | 100 | 80.0% | 257 | 2.056 | 8,760 | 2,256 | $0.02942 | 21.47 s |

| ID | Executed tool calls | Median / P95 / max calls | Questions over 6 | Invalid calls | Reached budget | Final status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| B0 | `audio_global_caption` 92; `audio_qa` 207; `video_clip_qa` 152; `video_global_qa` 57; `video_metadata` 24 | 3 / 11 / 29 | 19 | 3 | 0 | 125 natural |
| B1† | `audio_global_caption` 90; `audio_qa` 193; `video_clip_qa` 158; `video_global_qa` 50; `video_metadata` 27 | 3 / 10 / 25 | 19 | 8 | 0 | 124 natural; 1 error |
| B2 | `audio_global_caption` 54; `audio_qa` 165; `video_clip_qa` 81; `video_global_qa` 105; `video_metadata` 17 | 2 / 11 / 23 | 15 | 0 | 0 | 125 natural |
| B3 | `ask_caption` 111; `audio_qa` 105; `video_clip_qa` 69; `video_global_qa` 33; `video_metadata` 8 | 2 / 8 / 21 | 10 | 0 | 0 | 125 natural |
| B4 | `ask_caption` 130; `audio_qa` 66; `video_clip_qa` 69; `video_global_qa` 26 | 2 / 6 / 11 | 3 | 0 | 0 | 125 natural |
| B5 | `ask_caption` 131; `ask_perception` 99; `omni_clip_perception` 31; `video_metadata` 1 | 2 / 4 / 28 | 4 | 0 | 0 | 125 natural |
| B5-flex† | `ask_caption` 77; `ask_perception` 119; `omni_clip_perception` 26; `video_metadata` 2 | 1 / 3 / 21 | 2 | 0 | 0 | 125 natural |
| B6 | `ask_caption` 134; `ask_perception` 105; `omni_clip_perception` 18 | 2 / 4 / 28 | 2 | 0 | 0 | 125 natural |

### Logical repeat 3

| ID | Correct / 125 | Accuracy | Total tool calls | Calls / question | Live input tokens / question | Live output tokens / question | Live cost / question | Latency / question |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [B0†](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_outlier_reruns/b0-repeat3-rerun/output_test.jsonl) | 93 | 74.4% | 517 | 4.136 | 17,801 | 6,069 | $0.07397 | 60.69 s |
| [B1](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat3/b1/output_test.jsonl) | 95 | 76.0% | 538 | 4.304 | 18,605 | 6,292 | $0.07617 | 55.87 s |
| [B2](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat3/b2/output_test.jsonl) | 89 | 71.2% | 321 | 2.568 | 9,192 | 2,819 | $0.03430 | 29.36 s |
| [B3](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat3/b3/output_test.jsonl) | 93 | 74.4% | 288 | 2.304 | 9,514 | 3,321 | $0.04023 | 28.31 s |
| [B4](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat3/b4/output_test.jsonl) | 98 | 78.4% | 293 | 2.344 | 8,066 | 3,019 | $0.03552 | 25.96 s |
| [B5](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat3/b5/output_test.jsonl) | 95 | 76.0% | 246 | 1.968 | 8,048 | 1,933 | $0.02482 | 19.03 s |
| [B5-flex](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat3/b5-flex/output_test.jsonl) | 93 | 74.4% | 211 | 1.688 | 8,382 | 1,870 | $0.02446 | 19.47 s |
| [B6](/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats/repeat3/b6/output_test.jsonl) | 98 | 78.4% | 239 | 1.912 | 7,780 | 1,913 | $0.02416 | 18.46 s |

| ID | Executed tool calls | Median / P95 / max calls | Questions over 6 | Invalid calls | Reached budget | Final status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| B0† | `audio_global_caption` 86; `audio_qa` 198; `video_clip_qa` 151; `video_global_qa` 59; `video_metadata` 23 | 3 / 12 / 25 | 20 | 5 | 0 | 124 natural; 1 error |
| B1 | `audio_global_caption` 91; `audio_qa` 193; `video_clip_qa` 155; `video_global_qa` 68; `video_metadata` 31 | 3 / 11 / 22 | 22 | 6 | 0 | 125 natural |
| B2 | `audio_global_caption` 39; `audio_qa` 105; `video_clip_qa` 76; `video_global_qa` 86; `video_metadata` 15 | 2 / 7 / 22 | 7 | 1 | 0 | 125 natural |
| B3 | `ask_caption` 110; `audio_qa` 89; `video_clip_qa` 48; `video_global_qa` 37; `video_metadata` 4 | 2 / 6 / 16 | 6 | 3 | 0 | 125 natural |
| B4 | `ask_caption` 134; `audio_qa` 72; `video_clip_qa` 56; `video_global_qa` 31 | 2 / 5 / 28 | 3 | 1 | 0 | 125 natural |
| B5 | `ask_caption` 126; `ask_perception` 95; `omni_clip_perception` 25 | 2 / 4 / 12 | 3 | 0 | 0 | 125 natural |
| B5-flex | `ask_caption` 72; `ask_perception` 111; `omni_clip_perception` 25; `video_metadata` 3 | 1 / 4 / 15 | 3 | 0 | 0 | 125 natural |
| B6 | `ask_caption` 130; `ask_perception` 93; `omni_clip_perception` 16 | 2 / 4 / 15 | 2 | 0 | 0 | 125 natural |

## Fixed controls and provenance

The substitutions use the same configuration files and launchers as their
original logical slots: o3 high reasoning, seed 1234, 30 valid tool calls plus
one forced-final opportunity, uncompressed history, Experiment 11 retry
settings, and caption-cache scope `all`. Specialized tools use sepAV media;
omni tools use the original MP4 with embedded audio. See the main
[README_ablation.md](README_ablation.md) for the complete prompt, media,
cache, and execution documentation.

Original run root:

```text
/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_repeats
```

Replacement run root:

```text
/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_v2_selectedVal125_outlier_reruns
```

## Interpretation limits

This is a **post-hoc sensitivity analysis**, not an unbiased replacement for
the pre-specified three-repeat estimate. The candidate slots were selected
after examining their outcomes, and replacing only selected runs can
introduce selection bias. The original table in [README_ablation.md](README_ablation.md)
should remain the primary report; this document shows how conclusions change
under the proposed substitutions.

The B2 replacement candidate scored 66.4% and was rejected; B2 therefore
retains its original 72.27% ± 1.22% result. B3 becomes more stable in accuracy
(74.67% ± 0.46%) after replacing its high-scoring repeat1. B5-flex remains
competitive at 77.33% ± 2.81% while retaining the lowest mean calls and cost.
B6 is unchanged and remains the highest mean-accuracy configuration at
78.40% ± 1.60%.

Because B0 and B1 each contain one API-error row, their substituted means are
not fully clean three-run estimates. Re-running only those failed question IDs
would be preferable before using this sensitivity table as a final reported
result.
