# `ask_perception` question classification

Run from this repository:

```bash
scripts/run_ask_perception_question_judge.sh --dry-run
scripts/run_ask_perception_question_judge.sh --workers 8
```

The wrapper loads `.env` and uses the `react-avqa-dspy` Python environment. It
routes `openai/gpt-5.5` through LiteLLM with `ELM_API_KEY`, matching the
`elm_gpt` provider in `DSPy/dspy_avqa/context.py`. The final script requests
`reasoning_effort=none`, JSON response format, and a 512-token response cap.
The full run resumes from
`judgments_checkpoint.jsonl`; rerunning it retries failed judgments and skips
completed ones. `--experiment NAME` selects a subset; repeat the flag to run
several. `--limit N` is for a partial live check and exits nonzero until the
selected experiment has all 1,197 questions.

The default five sources are the first completed C0 max-turn-4 run and the
GPT-4.1 no-GEPA / G0 planner / G0 captioner / G0 planner+captioner runs in the
two experiment records named by the request. The G0 directories use seed 18.
The ledger counts the old C1 seed18 selected-c0 result under C0. The new C1
seed2/seed42 validation gates both selected candidate 0 with no strict gain
and recorded `decision: skip_inference`, so neither has `output_test.jsonl`.
The old result is available through explicit
`--experiment c1_prior_seed18` and was selected for this analysis by the user.

Classification is per `turn_trace` entry whose `tool_name` is
`ask_perception`. Category 1 asks the tool to return or choose the original
answer option; obvious requests are screened with a regular expression.
Category 2 asks the original answer target with a narrower scope or added
specification, including the original event's timestamp or two paraphrased
option descriptions. Category 3 asks for intermediate evidence that still
needs further combination or comparison. Category 4 covers other cases.
The GPT judge returns `category` and brief `evidence`; invalid answers are
recorded as `judge_error` and retried on the next run.

By default, outputs are under
`/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/NAME/`:

- `output_test.json`: array of `video_id`, `metadata`, and `questions` records,
  including the original QA, final response, and a judgment for every
  `ask_perception` call. Questions with no such call have an empty list.
- `manifest.json`: selected source, model, total questions, total calls, and
  counts by category or error status.
- `judgments_checkpoint.jsonl`: append-only per-question resume data. For a
  repeated question ID, the last line is authoritative.

The model sees only the original question, answer options, and the
`perceptual_question`, with no video, tool observation, gold answer, or final
ReAct response. The source `output_test.jsonl` is read only.

## Completed results

The denominator for categories is the number of `ask_perception` calls. Every
listed run has 1,197 original question rows; calls from the same question are
counted separately. All output rows and question IDs were matched against
their source files, and all judgments have `status: ok`.

| Run | Calls | Questions with no call | Category 1 | Category 2 | Category 3 | Category 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 max-turn 4 | 873 | 326 | 524 (60.0%) | 346 (39.6%) | 3 (0.3%) | 0 |
| C1 old seed18 full benchmark | 867 | 333 | 548 (63.2%) | 317 (36.6%) | 2 (0.2%) | 0 |
| GPT-4.1 no GEPA | 1,197 | 0 | 88 (7.4%) | 987 (82.5%) | 122 (10.2%) | 0 |
| GPT-4.1 G0 planner | 953 | 244 | 37 (3.9%) | 792 (83.1%) | 124 (13.0%) | 0 |
| GPT-4.1 G0 captioner | 1,197 | 0 | 76 (6.3%) | 1,004 (83.9%) | 117 (9.8%) | 0 |
| GPT-4.1 G0 planner + captioner | 966 | 231 | 64 (6.6%) | 794 (82.2%) | 108 (11.2%) | 0 |

Full-run JSON: [C0](/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/c0/output_test.json),
[C1 seed18](/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/c1_prior_seed18/output_test.json),
[GPT-4.1 no GEPA](/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/gpt41_none/output_test.json),
[G0 planner](/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/gpt41_g0_planner/output_test.json),
[G0 captioner](/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/gpt41_g0_captioner/output_test.json),
and [G0 planner + captioner](/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/gpt41_g0_planner_captioner/output_test.json).

C0 and the old C1 seed18 result have many more direct answer-option requests
than the four v8 GPT-4.1 runs.
These are descriptive differences across saved workflows, not a controlled
effect of GEPA. C0 and GPT-4.1 no-GEPA were judged with the ELM provider's
default reasoning effort; C1 seed18 used explicit `none`. During the G0 live run, a few calls returned empty
content despite retries; the remaining and failed G0 judgments were run with
explicit `reasoning_effort=none`. The checkpoint does not store this request
parameter per call, so the G0 labels use both settings. The category rubric
and GPT-5.5 model name were unchanged.

## C1 GEPA Val125 search

Run the saved-trajectory classification and plot:

```bash
scripts/run_c1_gepa_validation_ask_perception_judge.sh --dry-run
scripts/run_c1_gepa_validation_ask_perception_judge.sh --workers 8
/mnt/ceph_rbd/applications/anaconda3/bin/python scripts/plot_c1_gepa_validation_ask_perception.py /mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/c1_gepa_val125/trend.csv
```

The archived `gepa_state.bin` has all 125 validation scores for each of 12
full-validated candidates (candidate 0 is the starting prompt). The state also has
35 optimizer trace events; 24 did not add a full-Val125 candidate, so the
table reports the 12 full validation steps. Its
`best_outputs_valset` contains only Pareto-retained ReAct predictions, so the
question-level files mark the others `trajectory_status: not_saved`.

| Candidate | Discovery calls | Full Val125 | Saved trajectories | Observed calls | Category 1 | Category 2 | Category 3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 100/125 | 111 | 75 | 51 | 24 | 0 |
| 1 | 157 | 90/125 | 101 | 41 | 9 | 31 | 1 |
| 2 | 314 | 94/125 | 105 | 59 | 38 | 21 | 0 |
| 3 | 503 | 91/125 | 102 | 41 | 18 | 23 | 0 |
| 4 | 660 | 91/125 | 102 | 41 | 27 | 14 | 0 |
| 5 | 881 | 91/125 | 102 | 46 | 18 | 28 | 0 |
| 6 | 1,038 | 87/125 | 98 | 33 | 17 | 16 | 0 |
| 7 | 1,195 | 94/125 | 105 | 47 | 29 | 18 | 0 |
| 8 | 1,352 | 86/125 | 97 | 33 | 14 | 19 | 0 |
| 9 | 1,765 | 92/125 | 103 | 35 | 29 | 6 | 0 |
| 10 | 1,986 | 90/125 | 101 | 24 | 14 | 10 | 0 |
| 11 | 2,495 | 92/125 | 103 | 36 | 30 | 6 | 0 |

Category 4 was zero at every step. The plotted direct-option share varies
sharply (51/75 at candidate 0, 9/41 at candidate 1, 30/36 at candidate 11),
without a monotonic trend. Observed perception-call rate falls from 75/111
saved trajectories at candidate 0 to 36/103 at candidate 11, but no later
candidate beats candidate 0's 100/125 full validation score.

Coverage is selected by outcome: for each candidate the archive contains
**all correct Val predictions plus exactly 11 incorrect predictions**. Every
missing trajectory is incorrect, so the observed question-category shares
cannot be projected to the full 125 questions. The plot and CSV use the
number of saved perception calls as the category denominator and show saved
trajectory coverage separately. Replaying candidates would produce new
trajectories rather than the exact historical search record.

Artifacts: [trend CSV](/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/c1_gepa_val125/trend.csv),
[trend plot](/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/c1_gepa_val125/trend.png),
and per-candidate `output_test.json` and `manifest.json` in the same directory.

## G0 GEPA Val125 search

The same GPT-5.5 judge was run on all saved trajectories from the three
GPT-4.1 G0 searches. Their candidate counts are planner 20, captioner 19, and
planner + captioner 18. Each `trend.csv` has one row per full Val125 candidate;
category shares use the saved `ask_perception` calls in that row as denominator.

| Search target | Candidate 0 | Selected candidate | Selected Val125 | Selected call categories (1 / 2 / 3) |
| --- | ---: | ---: | ---: | --- |
| Planner | 96/125 | 8 | 104/125 | 7.8% / 83.1% / 9.1% (n=77) |
| Captioner | 91/125 | 3 | 100/125 | 9.5% / 81.9% / 8.6% (n=105) |
| Planner + captioner | 87/125 | 5 | 96/125 | 8.5% / 85.9% / 5.6% (n=71) |

The line plots provide the full step-by-step percentage changes:
[planner](/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/category_percentage_plots/g0_planner_category_percentages.png),
[captioner](/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/category_percentage_plots/g0_captioner_category_percentages.png),
[planner + captioner](/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/category_percentage_plots/g0_planner_captioner_category_percentages.png),
and [C1 seed18](/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/category_percentage_plots/c1_seed18_category_percentages.png).

The G0 trajectories are outcome-selected: the state retains all correct Val
predictions plus exactly 5 incorrect predictions per candidate for planner and
captioner, or exactly 6 for planner + captioner; every missing
trajectory is incorrect. Thus the plots faithfully describe question types in
the saved frontier trajectories, but their percentages must not be extrapolated
to all 125 validation questions. Per-target step data and per-question outputs
are in [planner](/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/g0_gepa_val125/planner/trend.csv),
[captioner](/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/g0_gepa_val125/captioner/trend.csv), and
[planner + captioner](/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/g0_gepa_val125/planner_captioner/trend.csv).

## Relationship with accuracy

Full inference percentages below are per `ask_perception` call. Overall
accuracy uses all 1,197 questions, including those with no perception call.
The strict final-answer parser in `scripts/react_eval.py` reproduced the
recorded correct counts for all six runs.

| Run | Overall accuracy | No-call questions | Category 1 | Category 2 | Category 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0 | 919/1197 (76.78%) | 326 | 60.02% | 39.63% | 0.34% |
| C1 old seed18 | 911/1197 (76.11%) | 333 | 63.21% | 36.56% | 0.23% |
| GPT-4.1 no GEPA | 858/1197 (71.68%) | 0 | 7.35% | 82.46% | 10.19% |
| G0 planner | 905/1197 (75.61%) | 244 | 3.88% | 83.11% | 13.01% |
| G0 captioner | 874/1197 (73.02%) | 0 | 6.35% | 83.88% | 9.77% |
| G0 planner + captioner | 870/1197 (72.68%) | 231 | 6.63% | 82.19% | 11.18% |

Across the six heterogeneous runs, Pearson correlations of overall accuracy
with category 1/2/3 call shares are respectively +0.753/-0.772/-0.663.
Within the four GPT-4.1 v8 runs they are -0.998/+0.360/+0.857.
These contrasting signs show that the pooled six-run correlation is dominated
by the C versus G workflow difference and cannot support a causal claim.
The four-run coefficient is also too sensitive to individual runs for a
stable conclusion.

Question-level accuracy, grouping multi-call questions by their first
`ask_perception` call, gives a different view. In G0 planner, no-call
questions are 223/244 (91.39%) correct, category 1 is 32/37 (86.49%),
category 2 is 565/792 (71.34%), and category 3 is 85/124 (68.55%).
The corresponding no-call questions were already 213/244 (87.30%) correct in
the no-GEPA run, so the high no-call accuracy partly reflects which questions
the planner chose to skip. Against the same question IDs in the baseline,
G0 planner's overall +47 correct answers decompose into +10 no-call,
+2 category 1, +34 category 2, and +1 category 3. Categories are assigned
from the G0 run, so these are descriptive strata, not causal effects.

For validation, the x variable is each candidate's category share among
**saved** perception calls and the y variable is its exact full Val125
accuracy. Correlations were calculated within each search, without mixing
the search targets.

| Search | Candidates | Category 1 Pearson r (p) | Category 2 Pearson r (p) | Category 3 Pearson r (p) |
| --- | ---: | ---: | ---: | ---: |
| C1 seed18 | 12 | +0.444 (0.148) | -0.449 (0.143) | -0.132 (0.683) |
| G0 planner | 20 | +0.315 (0.176) | -0.055 (0.818) | -0.212 (0.371) |
| G0 captioner | 19 | +0.028 (0.909) | -0.123 (0.617) | +0.169 (0.489) |
| G0 planner + captioner | 18 | +0.233 (0.353) | -0.020 (0.938) | -0.199 (0.428) |

C1 category 1 has Spearman rho +0.662 (p=0.019), but its Pearson result is
weaker and the selected candidate 0 scored 100/125 with 68.0% category 1,
while later candidates reached 82.9%-83.3% with only 92/125. G0 has no
consistent category-share/accuracy relationship. In particular, G0 planner
candidate 8 scored 104/125 with 7.8% category 1, while candidate 16 scored
95/125 with 15.0% category 1. The candidate tests are exploratory: prompts
share the same 125 validation questions, successive candidates are dependent,
and the many category tests are not independent. More importantly, saved
trajectory coverage is determined by accuracy because all correct predictions
but only a fixed number of incorrect predictions were kept. These correlations
describe the saved records and cannot estimate an effect of changing the
perception question type on accuracy.

## C0/C1 Val125 accuracy by question group

`scripts/analyze_c0_c1_validation_category_accuracy.py` joins the C0 full
benchmark to the exact 125 C1 validation IDs, and writes the reproducible
per-group results to [CSV](/mnt/ceph_rbd/data/avqa_project/daily_omni/ask_perception_question_judge_gpt5_5/c0_c1_validation_category_accuracy.csv).
There is at most one `ask_perception` call per Val125 question in these data.

The C0 full benchmark has complete trajectories, so these are exact
conditional accuracies on all 125 validation questions:

| Run | Overall | No `ask_perception` | Category 1 | Category 2 | Category 3/4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0 full benchmark | 95/125 (76.0%) | 29/34 (85.3%) | 39/53 (73.6%) | 27/38 (71.1%) | no questions |

For C1, category membership is available only when GEPA saved the ReAct
trajectory. The state saved every correct trajectory and exactly 11 incorrect
ones at every candidate; the remaining 14--28 trajectories per candidate are
known to be wrong but their call status is unknown. They must not be counted
as no-call questions. The following accuracies therefore condition on saved
trajectories only:

| C1 candidate | Full Val125 | Saved no-call | Saved category 1 | Saved category 2 | Saved category 3 | Unsaved, known wrong |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100/125 | 30/36 (83.3%) | 47/51 (92.2%) | 23/24 (95.8%) | — | 0/14 |
| 1 | 90/125 | 52/60 (86.7%) | 9/9 (100.0%) | 28/31 (90.3%) | 1/1 (100.0%) | 0/24 |
| 2 | 94/125 | 40/46 (87.0%) | 34/38 (89.5%) | 20/21 (95.2%) | — | 0/20 |
| 3 | 91/125 | 53/61 (86.9%) | 17/18 (94.4%) | 21/23 (91.3%) | — | 0/23 |
| 4 | 91/125 | 52/61 (85.2%) | 26/27 (96.3%) | 13/14 (92.9%) | — | 0/23 |
| 5 | 91/125 | 49/56 (87.5%) | 17/18 (94.4%) | 25/28 (89.3%) | — | 0/23 |
| 6 | 87/125 | 55/65 (84.6%) | 17/17 (100.0%) | 15/16 (93.8%) | — | 0/27 |
| 7 | 94/125 | 50/58 (86.2%) | 27/29 (93.1%) | 17/18 (94.4%) | — | 0/20 |
| 8 | 86/125 | 55/64 (85.9%) | 14/14 (100.0%) | 17/19 (89.5%) | — | 0/28 |
| 9 | 92/125 | 58/68 (85.3%) | 28/29 (96.6%) | 6/6 (100.0%) | — | 0/22 |
| 10 | 90/125 | 67/77 (87.0%) | 14/14 (100.0%) | 9/10 (90.0%) | — | 0/24 |
| 11 | 92/125 | 57/67 (85.1%) | 29/30 (96.7%) | 6/6 (100.0%) | — | 0/22 |

Across the saved C1 trajectories, no-call is 618/719 (86.0%), category 1 is
279/294 (94.9%), category 2 is 200/216 (92.6%), and category 3 is 1/1.
The high category 1/2 values are expected to be upward biased by the
outcome-selected archive, and cannot be compared directly to C0's complete
Val125 values or interpreted as the effect of making those call types.
