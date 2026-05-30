# DeepSeek Planner / Optimizer LM Caveat

This note records the current DSPy AVQA DeepSeek LM wiring after adding env-controlled thinking mode.

## Current behavior

`AVQADSPyReActProgram.__init__()` calls `configure_deepseek_lm(context)`, which configures a single DSPy global LM backed by DeepSeek. The normal AVQA planner calls use this LM.

COPRO/SIMBA optimization currently does not configure a separate optimizer/prompt LM. In `DSPy/dspy_avqa/optimize.py`, the teleprompter is created and compiled with the same student program, for example:

```python
teleprompter = COPRO(metric=avqa_metric, ...)
compiled = teleprompter.compile(student=program, trainset=trainset, ...)
```

Because no separate `prompt_model`, optimizer LM, or second `dspy.configure(...)` is provided, optimizer-side DeepSeek calls also use the same global DSPy LM.

## Env parameters affected

The following env parameters should be understood as applying to the shared DeepSeek DSPy LM in optimization runs:

```bash
PLANNER_THINKING_MODE=disabled|enabled|auto
PLANNER_TEMPERATURE=0.0
PLANNER_TOP_P=1.0
```

For plain daily batch inference, these effectively control only the planner DeepSeek calls because COPRO/SIMBA are not running.

For optimization runs, the same settings affect both:

- AVQA program rollout planner calls.
- COPRO/SIMBA candidate-generation or prompt-model calls that rely on DSPy's global LM.

## Temperature caveat

`PLANNER_THINKING_MODE=disabled` is attached to the LM configuration as `extra_body`, so it should consistently request non-thinking mode for the shared DeepSeek LM.

Temperature/top-p are slightly more subtle. The native LM wrapper only removes `temperature` and `top_p` when the corresponding env variable is not set. If `PLANNER_TEMPERATURE` or `PLANNER_TOP_P` is set, those values are configured on the LM. However, DSPy optimizers may pass per-call temperature kwargs, such as COPRO `init_temperature` or SIMBA candidate temperatures. Those per-call kwargs may override the configured defaults depending on the installed DSPy version and optimizer path.

So with the current shared-LM design:

- Thinking mode is shared between planner and optimizer LM use.
- Planner and optimizer sampling parameters are not cleanly separated.
- Strictly forcing optimizer temperature to `0.0` may require additional guarding if DSPy passes per-call overrides.

## Future separation

If future experiments need different behavior, split the LM roles explicitly:

- Planner rollout LM: DeepSeek non-thinking, deterministic sampling.
- Optimizer/prompt LM: its own DeepSeek config, possibly thinking enabled or higher temperature.

That likely means adding separate env names such as `OPTIMIZER_THINKING_MODE`, `OPTIMIZER_TEMPERATURE`, and `OPTIMIZER_TOP_P`, then passing a dedicated optimizer/prompt LM to COPRO/SIMBA where the installed DSPy API supports it.
