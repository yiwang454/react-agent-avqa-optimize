# Recent experiment script notes

Last checked: 2026-07-25. This file summarizes the recent scripts shown by `ls -lt scripts/ | head -n 20`.

## Common defaults not repeated in the table

Most GEPA wrappers eventually call `run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common.sh`, usually through the `*_common_0721.sh` shim. Unless a row says otherwise, GEPA runs use:

- train set: `/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedTrain125.jsonl`
- val set: `/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedVal125.jsonl`
- final eval input: `/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3.jsonl`
- planner seed: `1234`
- GEPA seed: `18`
- train limit: `64`
- max turns: `3`
- GEPA max full evals: `16`
- GEPA reflection minibatch size: `16`
- perception and captioner runtime config: `/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml`
- allowed tools: `ask_caption,ask_perception`
- caption placement: `task`
- signature in system prompt: enabled

## Experiment variants and run status

| Script | Purpose / main comparison | Changed parameters | Output / log name | Run status |
| --- | --- | --- | --- | --- |
| `run_dspy_react_agent_daily_qwen_v8_gemini_captionInTask_seed.sh` | Non-GEPA baseline: Qwen perception plus Gemini captioner, repeated evaluation. | `QWEN_SEED=1234`; `DEEPSEEK_SEED=7`; `RUN_REPEATS=3`; prompt `daily_qa_prompt_v8_caption_in_task.yaml`; perception config `config_localqwen_api_instruct.yaml`; captioner model `gemini`. | `daily_omni_dspy_qwen_v8GeminiCaptionInTask_qwen_seed_1234_deepseek_seed_7_repeat{1,2,3}` | Completed. Repeat 1/2/3 each have `output_test.jsonl` with 1197 rows. |
| `run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_planner_captioner_0721.sh` | GEPA optimize both planner workflow prompt and captioner default instruction, using the 0721 budget shim. | `GEPA_EXPERIMENT_NAME=planner_workflow_prompt_and_captioner_default_caption_instruction_0721`; `OPTIMIZE_TARGETS_CSV=planner.workflow_prompt,captioner.default_caption_instruction`; `GEPA_CAPTION_SUPERVISION=none`; default planner/reflection model `gpt-5.4`. | `daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_planner_workflow_prompt_and_captioner_default_caption_instruction_0721_planner_gpt-5.4_seed_1234_gepa_seed18`; log `planner_workflow_prompt_and_captioner_default_caption_instruction_0721.log` | Completed. Has `compiled_gepa.json`, metadata, and `output_test.jsonl` with 1197 rows. |
| `run_dspy_react_agent_daily_gepa_densified_gpt4_1_planner.sh` | Densified-label GEPA, planner-only, GPT-4.1 planner/reflection. | `OPTIMIZE_TARGETS_CSV=planner.workflow_prompt`; `PLANNER_MODEL_OVERRIDE=gpt-4.1`; `GEPA_REFLECTION_MODEL=gpt-4.1`; `GEPA_REFLECTION_REASONING_EFFORT=none`; `GEPA_REFLECTION_TEMPERATURE=0.0`; densified label dir enabled. | `daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_densified_planner_workflow_prompt_gpt4_1_planner_gpt-4.1_seed_1234_gepa_seed18`; log `densified_planner_workflow_prompt_gpt4_1.log` | Completed. Has `compiled_gepa.json`, metadata, and `output_test.jsonl` with 1197 rows. |
| `run_dspy_react_agent_daily_gepa_densified_gpt4_1_planner_captioner.sh` | Densified-label GEPA, planner plus captioner, GPT-4.1 planner/reflection. | `OPTIMIZE_TARGETS_CSV=planner.workflow_prompt,captioner.default_caption_instruction`; `PLANNER_MODEL_OVERRIDE=gpt-4.1`; `GEPA_REFLECTION_MODEL=gpt-4.1`; `GEPA_REFLECTION_REASONING_EFFORT=none`; `GEPA_REFLECTION_TEMPERATURE=0.0`; densified label dir enabled. | `daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_densified_planner_workflow_prompt_and_captioner_default_caption_instruction_gpt4_1_planner_gpt-4.1_seed_1234_gepa_seed18`; log `densified_planner_workflow_prompt_and_captioner_default_caption_instruction_gpt4_1.log` | Completed. Has `compiled_gepa.json`, metadata, and `output_test.jsonl` with 1197 rows. |
| `run_dspy_react_agent_daily_gepa_densified_gpt5_4_planner.sh` | Densified-label GEPA, planner-only, GPT-5.4 planner/reflection. | `OPTIMIZE_TARGETS_CSV=planner.workflow_prompt`; `PLANNER_MODEL_OVERRIDE=gpt-5.4`; `GEPA_REFLECTION_MODEL=gpt-5.4`; `GEPA_REFLECTION_REASONING_EFFORT=medium`; reflection temperature unset; densified label dir enabled. | `daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_densified_planner_workflow_prompt_gpt5_4_medium_planner_gpt-5.4_seed_1234_gepa_seed18`; log `densified_planner_workflow_prompt_gpt5_4_medium.log` | Completed. Has `compiled_gepa.json`, metadata, and `output_test.jsonl` with 1197 rows. |
| `run_dspy_react_agent_daily_gepa_densified_gpt5_4_planner_captioner.sh` | Densified-label GEPA, planner plus captioner, GPT-5.4 planner/reflection. | `OPTIMIZE_TARGETS_CSV=planner.workflow_prompt,captioner.default_caption_instruction`; `PLANNER_MODEL_OVERRIDE=gpt-5.4`; `GEPA_REFLECTION_MODEL=gpt-5.4`; `GEPA_REFLECTION_REASONING_EFFORT=medium`; reflection temperature unset; densified label dir enabled. | `daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_densified_planner_workflow_prompt_and_captioner_default_caption_instruction_gpt5_4_medium_planner_gpt-5.4_seed_1234_gepa_seed18`; log `densified_planner_workflow_prompt_and_captioner_default_caption_instruction_gpt5_4_medium.log` | Partially completed. It saved `compiled_gepa.json` and signature search, but there is no metadata and no `output_test.jsonl`; log reached final eval with 1197 remaining samples. |
| `run_dspy_react_agent_daily_gepa_v8free_gemini_captionInTask_g0_gpt4_1_planner.sh` | Free-caption-in-task prompt, no densified supervision, planner-only GPT-4.1. | `GEPA_EXPERIMENT_NAME=v8free_g0_planner_gpt4_1`; `GEPA_SUPERVISION_TIER=G0`; `PLANNER_MODEL_OVERRIDE=gpt-4.1`; prompt `daily_qa_prompt_v8_free_caption_in_task.yaml`; `GEPA_REFLECTION_TEMPLATE_VERSION=original`; `GEPA_CAPTION_SUPERVISION=none`; `--ignore-audio-caption-dir`. | `daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_v8free_g0_planner_gpt4_1_planner_gpt-4.1_seed_1234_gepa_seed18`; log `run_dspy_react_agent_daily_gepa_v8free_gemini_captionInTask_g0_gpt4_1_planner.log` | Running at check time. Python process was active; output dir exists, but no `compiled_gepa.json`, metadata, or final `output_test.jsonl` yet. |
| `run_dspy_react_agent_daily_gepa_v8free_gemini_captionInTask_g3_gpt4_1_planner.sh` | Free-caption-in-task prompt with G3 densified key-evidence supervision, planner-only GPT-4.1. | `GEPA_EXPERIMENT_NAME=v8free_g3_key_evidence_planner_gpt4_1`; `GEPA_SUPERVISION_TIER=G3`; `PLANNER_MODEL_OVERRIDE=gpt-4.1`; prompt `daily_qa_prompt_v8_free_caption_in_task.yaml`; densified label dir enabled; `GEPA_REFLECTION_TEMPLATE_VERSION=densified_key_evidence`; `GEPA_CAPTION_SUPERVISION=none`; `--ignore-audio-caption-dir`. | `daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_v8free_g3_key_evidence_planner_gpt4_1_planner_gpt-4.1_seed_1234_gepa_seed18`; log `v8free_g3_key_evidence_planner_gpt4_1.log` | Running at check time. Python process was active; output dir exists, but no `compiled_gepa.json`, metadata, or final `output_test.jsonl` yet. |
| `run_dspy_react_agent_daily_gepa_v8free_gemini_captionInTask_g0_gpt5_4_planner.sh` | Free-caption-in-task prompt, no densified supervision, planner-only GPT-5.4. | Same as G0 GPT-4.1, except `GEPA_EXPERIMENT_NAME=v8free_g0_planner_gpt5_4` and `PLANNER_MODEL_OVERRIDE=gpt-5.4`; reflection model remains `gpt-5.4` with medium effort from common wrapper. | Expected `daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_v8free_g0_planner_gpt5_4_planner_gpt-5.4_seed_1234_gepa_seed18`; expected log `v8free_g0_planner_gpt5_4.log` | Not found / likely not run yet. No matching output dir or log was found. |
| `run_dspy_react_agent_daily_gepa_v8free_gemini_captionInTask_g3_gpt5_4_planner.sh` | Free-caption-in-task prompt with G3 densified key-evidence supervision, planner-only GPT-5.4. | Same as G3 GPT-4.1, except `GEPA_EXPERIMENT_NAME=v8free_g3_key_evidence_planner_gpt5_4` and `PLANNER_MODEL_OVERRIDE=gpt-5.4`; reflection model remains `gpt-5.4` with medium effort from common wrapper. | Expected `daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_v8free_g3_key_evidence_planner_gpt5_4_planner_gpt-5.4_seed_1234_gepa_seed18`; expected log `v8free_g3_key_evidence_planner_gpt5_4.log` | Not found / likely not run yet. No matching output dir or log was found. |

## Shared helper scripts

| Script | Role | Notes |
| --- | --- | --- |
| `run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common.sh` | Main GEPA command builder. | Sets planner provider/model defaults, shared data paths, prompt/config defaults, GEPA output layout, and invokes `DSPy/avqa_dspy_optimize.py`. Not meant to be a standalone experiment unless the caller sets `GEPA_EXPERIMENT_NAME` and `OPTIMIZE_TARGETS_CSV`. |
| `run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common_0721.sh` | Budget/caption-supervision shim. | Sets `GEPA_MAX_FULL_EVALS=16`, `GEPA_REFLECTION_MINIBATCH_SIZE=16`, forwards `--gepa-caption-supervision`, then calls the main common script. |
| `run_dspy_react_agent_daily_gepa_densified_common.sh` | Densified-label shim. | Adds `--gepa-densified-label-dir` and forces `GEPA_CAPTION_SUPERVISION=none`, then calls the 0721 common shim. |
| `run_dspy_react_agent_daily_gepa_v8free_gemini_captionInTask_common.sh` | Free-caption-in-task G0/G3 shim. | Switches prompt to `daily_qa_prompt_v8_free_caption_in_task.yaml`; optimizes only `planner.workflow_prompt`; chooses G0 original reflection vs G3 densified key-evidence reflection; disables caption supervision and ignores audio caption dir. |

## Status criteria used here

- `Completed`: `output_test.jsonl` exists with 1197 rows, and GEPA rows also have `compiled_gepa.json` plus `compiled_gepa_metadata.json`.
- `Partially completed`: optimization artifacts exist, but final eval output or metadata is missing.
- `Running`: matching Python process was active when checked.
- `Not found / likely not run yet`: no matching log or output directory was found.
