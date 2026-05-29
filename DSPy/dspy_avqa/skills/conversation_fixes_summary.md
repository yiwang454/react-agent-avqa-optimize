# Conversation Fixes Summary

This note summarizes the issues handled in this conversation and the implemented solutions.

## 1. Delete ERROR Sample JSON Files

**Problem**

`scripts/visualize_dspy_output.py` could render DSPy outputs into a readable text file, but there was no way to clean up failed per-sample JSON files after identifying ERROR samples.

**Solution**

Added a CLI flag:

```bash
--delete-error-sample
--delete_error_sample
```

The flag is implemented with `action="store_true"`. After the readable DSPy report is generated, the script deletes only per-sample `.json` files for records classified as ERROR.

**Safety Behavior**

- Does not delete `.jsonl` files.
- Does not delete `.txt` report files.
- Refuses to delete the input file itself.
- Only targets sample-level `.json` paths inferred from the input directory or input file parent directory.

## 2. DSPy Runner Resume From Per-Sample JSON

**Problem**

`DSPy/dspy_avqa/runner.py` needed breakpoint resume support. The desired behavior was:

- Load samples that were already run.
- Skip completed samples.
- Run only missing samples.
- Write a fresh combined `output_test.jsonl` containing cached and newly generated rows.

The first implementation considered loading `output_test.jsonl`, but the requirement was then refined: do not use `output_test.jsonl` as the resume cache.

**Solution**

Resume now uses only per-sample JSON files under `--output-dir`, such as:

```text
<output_dir>/<cut_id>.json
```

For each selected cut:

- If `<cut_id>.json` exists and contains a valid non-empty response, it is treated as cached.
- Cached rows are reconstructed with the current row-building logic.
- Missing or invalid samples are rerun.
- Final output JSONL is rewritten in the selected input order, combining cached and newly generated results.

**Important Detail**

`output_test.jsonl` is no longer used to decide whether a sample has already run.

## 3. Visualize DSPy Directory Without output_test.jsonl

**Problem**

`scripts/visualize_dspy_output.py` originally expected an input JSONL or JSON file. Some output directories may contain only per-sample `.json` files and no `output_test.jsonl`.

**Solution**

Added directory input support.

When the input path is a directory, the script:

- Reads first-level `*.json` files.
- Skips summary-like files whose names start with `output_test`.
- Sorts sample JSON files using natural filename order.
- Treats the directory as a virtual JSONL by loading each JSON sample in order.
- Supports both full result rows and question-data-only JSON files.

Default output for a directory is:

```text
<input_dir>/<input_dir_name>.dspy_readable.txt
```

## 4. DSPy + COPRO Output Compatibility

**Problem**

`scripts/visualize_dspy_output.py` could parse the `turn_trace` portion of DSPy + COPRO output, but could not correctly extract:

- `video_id`
- `gold_answer`
- `pred_answer`
- question text
- options from `options_json`

As a result, accuracy statistics were missing or wrong.

The COPRO format observed in:

```text
/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_gemini2.5_copro_recommended/output_test.jsonl
```

stores fields mostly at the top level and inside `example`, for example:

```json
{
  "example_index": 0,
  "example": {
    "question": "...",
    "options_json": "[...]",
    "video_id": "...",
    "answer": "D"
  },
  "gold_answer": "D",
  "pred_answer": "B",
  "score": 0.0,
  "reasoning_summary": "...",
  "turn_trace": [...]
}
```

**Solution**

Chose the minimal unified-compatibility approach instead of adding a manual `mode` flag.

The script now normalizes both formats through shared helper logic:

- Old format: reads from `question_data`.
- COPRO format: reads from top-level fields and `example`.

The unified parser now extracts:

- `video_id` from `record.video_id`, `metadata.video_id`, or `example.video_id`.
- gold answer from `question_data.answer`, `gold_answer`, or `example.answer`.
- prediction from `pred_answer`, final `turn_trace`, or response text.
- options from list-style `options` or JSON string `options_json`.
- model response from `response` or `reasoning_summary`.

`--video-id` and `--question-id` filtering were also updated to use the unified field accessors.

**Answer Parsing Fix**

`normalize_answer` was improved to prefer:

- `<answer>B</answer>` tags
- exact single-letter answers
- answer prefixes like `B.`
- standalone option letters

This avoids misreading explanation text such as `During ...` as answer `D`.

## Verification

The updated DSPy visualization script passed:

```bash
python3 -m py_compile general_scripts/react-agent-avqa/scripts/visualize_dspy_output.py
```

COPRO validation on the full sample file produced:

```text
# Samples: 64
# Errors: 0
# Predictions: 63
# Comparable accuracy: 45/63
```

Plain old-format validation also worked on:

```text
/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_gemini2.5/output_test.jsonl
```

with expected readable output and comparable accuracy statistics.

