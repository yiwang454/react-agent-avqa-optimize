# Modality-merge ablation on DailyOmni selected Val125

This suite contains the revised B0-through-B8 path plus B5-flex: ten runs in
total. The launchers define the runs but never start the full Val125 suite
automatically.

## Fixed controls

Every launcher fixes the following values:

- input: `daily_omni_cuts_selectedVal125.jsonl` (125 questions);
- planner: `elm_gpt/o3`, medium reasoning, temperature 0, seed 1234;
- perception model: Gemini 2.5 Flash with the materialized OmniAgent-compatible
  sampling/retry config;
- six valid tool calls, with retryable validation errors excluded from budget;
- one planner-only forced-final round after budget exhaustion;
- full, uncompressed planner/tool history and no optimization guidance;
- question-scoped caption-cache reads for every corresponding caption call;
- the same source MP4/WAV paths and the existing exact clip-boundary handling.

`video_metadata` removal only changes planner visibility. Clip tools may still
read media duration internally to legalize the requested interval.

## Configurations

| ID | Prompt / control | Planner-visible tools | First tool |
| --- | --- | --- | --- |
| B0 | OmniAgent instruction and C+Q+M guidance | `audio_global_caption`, `audio_qa`, `video_global_qa`, `video_clip_qa`, `video_metadata` | free |
| B1 | B0 minus only the cross-check sentence | same as B0 | free |
| B2 | Experiment-11-style workflow adapted to C+Q+M | same as B0 | free |
| B3 | B2 with omni-modal caption | `ask_caption`, specialized Q, `video_metadata` | free |
| B4 | B3 with caption-first only | `ask_caption`, specialized Q, `video_metadata` | forced `ask_caption` |
| B5-flex | Merged global/clip omni Q, without caption-first | `ask_caption`, `ask_perception`, `omni_clip_perception`, `video_metadata` | free |
| B5 | B4 with specialized Q replaced by global/clip omni Q | `ask_caption`, `ask_perception`, `omni_clip_perception`, `video_metadata` | forced `ask_caption` |
| B6 | B5 without planner-visible metadata | `ask_caption`, `ask_perception`, `omni_clip_perception` | forced `ask_caption` |
| B7 | B5 without omni clip Q, retaining metadata | `ask_caption`, `ask_perception`, `video_metadata` | forced `ask_caption` |
| B8 | B7 without planner-visible metadata | `ask_caption`, `ask_perception` | forced `ask_caption` |

B0/B1/B2 use the exact audio-only OmniAgent caption cache:

```text
/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_audio_global_caption_cache_omniagent_repeat1_completed
```

B3/B4/B5-flex/B5/B6/B7/B8 use the existing Experiment 11 audio-visual caption cache:

```text
/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_caption_cache_v8_gemini_from3repeats
```

## Launching

Run one configuration at a time. Existing complete per-question JSON files in
its output directory are loaded as checkpoints, so rerunning the same command
resumes incomplete work.

```bash
bash scripts/modality_merge_ablation/run_b0_val125.sh
bash scripts/modality_merge_ablation/run_b1_val125.sh
bash scripts/modality_merge_ablation/run_b2_val125.sh
bash scripts/modality_merge_ablation/run_b3_val125.sh
bash scripts/modality_merge_ablation/run_b4_val125.sh
bash scripts/modality_merge_ablation/run_b5_flex_val125.sh
bash scripts/modality_merge_ablation/run_b5_val125.sh
bash scripts/modality_merge_ablation/run_b6_val125.sh
bash scripts/modality_merge_ablation/run_b7_val125.sh
bash scripts/modality_merge_ablation/run_b8_val125.sh
```

Set `DRY_RUN=true` to validate and print a launcher without model requests. Set
`OUTPUT_ROOT`, `INFERENCE_NUM_THREADS`, or `INFERENCE_BATCH_SIZE` to override
operational paths/concurrency without changing the controlled model settings.
For a non-destructive subset smoke run, set comma-separated `SAMPLE_IDS`; the
default output root then changes to `modality_merge_ablation_smoke`.

Each run writes `output_test.jsonl`, `metrics_summary.json`,
`resolved_experiment_config.yaml`, per-question checkpoints, and `run.log`.
Every question row records correctness, planner/perception tokens, estimated
live cost, retry-adjusted latency, requested/executed tool counts, invalid
calls, budget use/exhaustion, and final-answer status.

### Three sequential repeats of the B0-B7 path

The aggregate launcher runs B0-B7 concurrently within each repeat, waits for
all eight runs to succeed, and only then starts the next repeat:

```bash
bash scripts/modality_merge_ablation/run_b0-7_val125.sh
```

Outputs default to
`modality_merge_ablation_selectedVal125_repeats/repeat{1,2,3}/b{0..7}`.
Rerunning the aggregate launcher resumes the existing per-question
checkpoints. B5-flex and B8 are intentionally not part of this eight-run path.
The default is one inference thread per run, giving eight active workers per
repeat. Override it with `INFERENCE_NUM_THREADS_PER_RUN` and optionally
`INFERENCE_BATCH_SIZE_PER_RUN`; override the parent output location with
`OUTPUT_BASE`. Extra command-line arguments are forwarded to every B0-B7
launcher.

## Known migration differences

B0 preserves the OmniAgent instruction, tool names, exact specialized-tool
prompts, modality split, and stopping intent, but it runs inside this repo's
DSPy compact-JSON Free-ReAct loop. It therefore does not reproduce LangChain's
native OpenAI tool-call wire format, message classes, or 30-iteration default.
The controlled suite uses six valid calls so every B configuration shares the
Experiment 11 budget and forced-final mechanism. The final answer is normalized
to an option label instead of relying on OmniAgent's `<answer>` tag parser.

The specialized tools are thin adapters over this repo's instrumented Gemini
transport. Audio tools send the preprocessed WAV only; global/clip video tools
send visual video only; clip QA retains OmniAgent's 5-fps request hint and
visual-only clip preprocessing. The legacy Gemini transport matches the
OmniAgent inline-media request most closely; the optional Vertex/DSPy transport
does not carry the legacy `videoMetadata.fps` hint.
