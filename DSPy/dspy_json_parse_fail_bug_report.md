# DSPy ReAct JSON Parse Failure Bug Report

## Summary

The DSPy AVQA ReAct runner can receive a meaningful DeepSeek planner response but still fail the whole sample because DSPy's `JSONAdapter` requires every output field in the active `dspy.Signature` to be present.

Observed result: `output_test.jsonl` contains `[ERROR] Adapter JSONAdapter failed to parse the LM response`, and `turn_trace` has only one synthetic error turn even though `MAX_TURNS=4`.

This is not a missing-LM-response problem. It is a strict structured-output parsing problem.

## Observed Example

For sample `Me4W36_lUcI-1`, the output row contains:

```text
[ERROR] Adapter JSONAdapter failed to parse the LM response.

LM Response: {
  "reasoning": "Current evidence shows a panoramic view of Earth ...",
  "action": "continue",
  "answer": "",
  "reasoning_summary": "The exact audio ...

Expected to find output fields in the LM response: [reasoning, action, answer, reasoning_summary, evidence_summary]
Actual output fields parsed from the LM response: [reasoning, action, answer, reasoning_summary]
```

The raw LM response includes a useful `action: continue`, but it lacks `evidence_summary`, so DSPy refuses to return a `Prediction` object.

## Why `MAX_TURNS=4` Still Produces Only One Trace Entry

`MAX_TURNS=4` controls the loop in:

```text
DSPy/dspy_avqa/program.py
AVQADSPyReActProgram.forward()
```

The loop is:

```python
for turn_idx in range(1, max_iters + 1):
```

However, if any DSPy submodule raises an exception, the loop exits immediately. The exception is caught in:

```text
DSPy/dspy_avqa/runner.py
run_one()
```

Then `run_batch()` writes a synthetic trace entry:

```python
{
    "turn_id": 1,
    "planner_action": "error",
    "tool_name": None,
    "tool_observation": None,
    "tool_error": error,
}
```

So the single turn in `turn_trace` is not a real ReAct turn. It is an error placeholder created by the runner.

## Where The LM Response Comes From

Call path:

```text
program.py
  self.action_decider(...)
    dspy.ChainOfThought(DecideFinalAction)
      DSPy JSONAdapter
        deepseek_dspy_lm.py: DeepSeekDSPyLM.forward()
          self.client.call_messages(...)
            deepseek_api.py: DeepSeekPlannerClient.call_messages()
              deepseek_api_new.py: call_deepseek()
                OpenAI(...).chat.completions.create(...)
                return resp.choices[0].message.content
```

The `LM Response` shown in the error is the raw DeepSeek `message.content` returned by `deepseek_api_new.call_deepseek()`.

## Where `planner_action` And `tool_name` Should Come From

In normal execution:

- `tool_name` comes from the first planner substep:

```python
draft = self.question_drafter(...)
tool_name = str(draft.tool_name).strip()
```

- `planner_action` comes from the action-decision substep:

```python
action = self.action_decider(...)
action_name = str(action.action).strip().lower()
```

Current problem: `turn_trace.append(...)` happens only after `action_decider` succeeds. If `action_decider` parse fails, the already selected `tool_name` and tool observation are lost from the final trace.

## Is `evidence_summary` Required By DSPy Or The Optimizers?

No.

`evidence_summary` is not required by DSPy, COPRO, or SIMBA. It is only a local custom output field in:

```text
DSPy/dspy_avqa/signatures.py
```

Current definitions:

```python
class DecideFinalAction(dspy.Signature):
    action = dspy.OutputField(desc="continue or final")
    answer = dspy.OutputField(desc="Option letter if action=final, else empty")
    reasoning_summary = dspy.OutputField()
    evidence_summary = dspy.OutputField()

class FinalAnswerFallback(dspy.Signature):
    answer = dspy.OutputField()
    reasoning_summary = dspy.OutputField()
    evidence_summary = dspy.OutputField()
```

The optimizer metric only uses `pred.answer`:

```python
def avqa_metric(example, pred, trace=None):
    gold = normalize_option_letter(str(example.answer))
    got = normalize_option_letter(str(pred.answer))
    return 1.0 if gold == got else 0.0
```

Therefore, `evidence_summary` can be removed if the downstream code is adjusted.

## Minimal Fix Option A: Remove `evidence_summary`

This is the simplest fix for the observed failure.

Change `DSPy/dspy_avqa/signatures.py`:

```python
class DecideFinalAction(dspy.Signature):
    action = dspy.OutputField(desc="continue or final")
    answer = dspy.OutputField(desc="Option letter if action=final, else empty")
    reasoning_summary = dspy.OutputField()

class FinalAnswerFallback(dspy.Signature):
    answer = dspy.OutputField()
    reasoning_summary = dspy.OutputField()
```

Then change `DSPy/dspy_avqa/program.py` so it does not access:

```python
action.evidence_summary
fallback.evidence_summary
```

Use `prior_evidence` / `accumulated_evidence` as the returned evidence summary instead:

```python
evidence_summary=prior_evidence
```

This keeps DSPy optimization compatible because:

- the program remains a `dspy.Module`,
- the optimizer still compiles the same top-level program,
- the metric still uses `pred.answer`,
- fewer required output fields makes JSONAdapter less brittle.

## More Robust Fix Option B: Keep Field But Add Safe Parsing

If `evidence_summary` is useful for debugging or analysis, keep it but add a safe wrapper around DSPy predictions.

Catch:

```python
from dspy.utils.exceptions import AdapterParseError
```

Then recover partial outputs from `exc.parsed_result` or `exc.lm_response`, filling defaults for missing fields.

For this exact bug:

```python
if evidence_summary is missing:
    evidence_summary = reasoning_summary or prior_evidence
```

This resembles the old LangGraph implementation, which parses planner JSON manually and does not crash the entire run when parsing is imperfect.

## What To Borrow From The Old `src/react_agent` Implementation

Reference:

```text
src/react_agent/graph.py
```

The old implementation uses a smaller action schema:

```json
{"action":"tool","tool_name":"ask_qwen_perception","arguments":{"perceptual_question":"..."}}
```

or:

```json
{"action":"final","answer":"..."}
```

It then manually parses with:

```python
payload = json.loads(_strip_fences(raw))
```

If parsing fails, it preserves raw planner output instead of throwing away the whole turn:

```python
"planner_raw": raw,
"planner_action": "unknown"
```

DSPy version should keep DSPy modules for optimization, but adopt the old implementation's robustness principle:

1. Keep planner output schema small.
2. Do not require nonessential fields.
3. Preserve raw planner output in traces.
4. Append turn trace before fragile final parsing, then update it when parsing succeeds.

## Recommended First Patch

Start with Option A:

1. Remove `evidence_summary` from `DecideFinalAction` and `FinalAnswerFallback`.
2. Replace `action.evidence_summary` / `fallback.evidence_summary` in `program.py` with `prior_evidence`.
3. Move `turn_trace.append(...)` earlier, immediately after the tool observation is available.
4. Later, optionally add `AdapterParseError` recovery for even more robustness.

This should fix the current parse failure while preserving DSPy/COPRO/SIMBA compatibility.
