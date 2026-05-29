# DSPy AVQA Prompt YAML, Tool Control, and Recovery Summary

This note summarizes the main questions and answers from the conversation about
`DSPy/dspy_avqa`, especially prompt YAML loading, tool selection, runner scripts,
and recovery after an accidental `git reset --hard`.

## 1. Prompt Templates: Hardcoded or YAML-Loaded?

**Question**

Can the current DSPy repository load prompts from:

```text
DSPy/dspy_avqa/yamls/daily_qa_prompt_v0.yaml
```

instead of hardcoding prompt templates in Python scripts?

**Answer**

Yes, after the prompt-config changes, the main planner/perception/signature prompt
text is loaded through YAML.

The loading path is:

```text
DSPy/avqa_dspy_impl.py
  -> run_batch()
  -> DSPy/dspy_avqa/runner.py
  -> load_prompt_config(args.prompt_yaml)
  -> DSPy/dspy_avqa/prompt_config.py
  -> DEFAULT_PROMPT_YAML = DSPy/dspy_avqa/yamls/daily_qa_prompt_v0.yaml
```

`avqa_dspy_impl.py` does not directly pass the YAML path. It imports the package
and calls `run_batch()`. The actual CLI/default prompt loading happens in
`runner.py` and `prompt_config.py`.

Important files:

- `prompt_config.py`: YAML loader and prompt renderer.
- `context.py`: planner system prompt comes from YAML.
- `program.py`: task prompt, action schema, turn templates, fallback instruction,
  default perceptual question, and truncation marker come from YAML.
- `tools.py`: perception system/evidence/temporal-grounding prompts come from YAML.
- `signatures.py`: DSPy Signature instructions and field descriptions come from YAML.
- `runner.py`: supports `--prompt-yaml` and refreshes signatures after loading YAML.

## 2. Can Tool Selection Be Controlled Only by `.env` or Prompt YAML?

**Question**

Can we choose which tools are involved using only an env file such as:

```text
scripts/env_files/.env_dspy_react_agent_daily_tool3
```

or a prompt YAML such as:

```text
DSPy/dspy_avqa/yamls/daily_qa_prompt_v1.yaml
```

For example, can one run use only `ask_perception` and disable
`temporal_ground_video`?

**Initial Answer**

Before the tool-control code change, no. Removing `temporal_ground_video` from
the prompt only discouraged the planner from choosing it. The Python program
could still execute `temporal_ground_video` if the planner emitted that tool name.

**Implemented Answer**

After the change, yes. Tool selection can be controlled by:

1. CLI argument: `--allowed-tools`
2. Env variable: `DSPY_AVQA_ALLOWED_TOOLS`
3. Prompt YAML field: `tools.allowed`
4. Default fallback: all supported tools

Priority order:

```text
--allowed-tools / DSPY_AVQA_ALLOWED_TOOLS
  > prompt YAML tools.allowed
  > all tools: ask_perception, temporal_ground_video
```

The runtime guard is in `program.py`: even if the planner outputs a disabled
tool, the program will not dispatch that disabled tool. If only `ask_perception`
is allowed, `temporal_ground_video` will not actually be called.

## 3. Tool-Only v1 Prompt Setup

**Goal**

Make one controlled version where the only tool is:

```text
ask_perception
```

and `temporal_ground_video` is disabled.

**Implemented Configuration**

`daily_qa_prompt_v1.yaml` contains:

```yaml
tools:
  allowed:
    - ask_perception
```

Its planner action schema only advertises:

```json
{"action":"tool","tool_name":"ask_perception","arguments":{"perceptual_question":"..."}}
```

The `temporal_grounding_prompt_template` was removed from v1.

The env file:

```text
scripts/env_files/.env_dspy_react_agent_daily_tool3
```

sets:

```bash
PROMPT_YAML=/mnt/ceph_rbd/workspace/avqa_project/general_scripts/react-agent-avqa/DSPy/dspy_avqa/yamls/daily_qa_prompt_v1.yaml
DSPY_AVQA_ALLOWED_TOOLS=ask_perception
```

The runner script:

```text
scripts/run_dspy_react_agent_daily_controltool.sh
```

passes both values through to `DSPy/avqa_dspy_impl.py` as:

```bash
--prompt-yaml "${PROMPT_YAML}"
--allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}"
```

## 4. Can v0 Still Use Temporal Grounding?

**Question**

If `PROMPT_YAML` is set to:

```text
/mnt/ceph_rbd/workspace/avqa_project/general_scripts/react-agent-avqa/DSPy/dspy_avqa/yamls/daily_qa_prompt_v0.yaml
```

can temporal grounding still be used?

**Answer**

Yes, as long as `DSPY_AVQA_ALLOWED_TOOLS` or `--allowed-tools` does not restrict
the tool list to `ask_perception`.

`daily_qa_prompt_v0.yaml` has no `tools.allowed` section, so the fallback is all
tools:

```text
ask_perception, temporal_ground_video
```

However, if the env still contains:

```bash
DSPY_AVQA_ALLOWED_TOOLS=ask_perception
```

then v0's prompt text alone is not enough to enable temporal grounding. The env
restriction wins.

## 5. `controltool` vs `controltool_sanity`

**Question**

Are these the only differences between:

```text
scripts/run_dspy_react_agent_daily_controltool.sh
scripts/run_dspy_react_agent_daily_controltool_sanity.sh
```

- `controltool.sh`: only `ask_perception`, uses v1 YAML
- `controltool_sanity.sh`: both tools, uses v0 YAML
- API key differs

**Answer**

The scripts themselves are almost identical. The only script-body difference is
the default env file:

```text
controltool.sh
  -> scripts/env_files/.env_dspy_react_agent_daily_tool3

controltool_sanity.sh
  -> scripts/env_files/.env_dspy_react_agent_daily_tool3_sanitycheck
```

The effective runtime configuration differs in four ways:

1. Tool set:
   - `tool3`: `DSPY_AVQA_ALLOWED_TOOLS=ask_perception`
   - `sanitycheck`: no `DSPY_AVQA_ALLOWED_TOOLS`, so v0 falls back to both tools
2. Prompt YAML:
   - `tool3`: `daily_qa_prompt_v1.yaml`
   - `sanitycheck`: `daily_qa_prompt_v0.yaml`
3. Gemini API key:
   - `GEMINI_API_KEY` differs
   - `DEEPSEEK_API_KEY` is the same
4. Output path:
   - `tool3`: writes to a `daily_omni_dspy_examples_tool3` output directory
   - `sanitycheck`: writes to a `daily_omni_dspy_examples_originalSanity` output directory

Other key settings are the same:

- `PERCEPTION_CONFIG_YAML`
- `GEMINI_BASE_URL`
- `GEMINI_MODEL`
- `PERCEPTION_MODEL`
- `INPUT_JSONL`
- `AUDIO_CAPTION_DIR`
- `MAX_TURNS`
- `CONCURRENCY`
- `DEBUG`

## 6. Recovery After Accidental `git reset --hard`

**Problem**

The prompt YAML/tool-control changes were accidentally removed by:

```bash
git reset --hard HEAD~1
```

Before the reset, the recent DSPy changes had been added to Git's index.

**Recovery Approach**

Used:

```bash
git fsck --lost-found --no-reflogs
```

to find dangling blobs created from the previously staged files. Those blobs were
mapped back to their corresponding files and written back into the working tree.

Recovered files included:

- `DSPy/README.md`
- `DSPy/dspy_avqa/__init__.py`
- `DSPy/dspy_avqa/context.py`
- `DSPy/dspy_avqa/program.py`
- `DSPy/dspy_avqa/runner.py`
- `DSPy/dspy_avqa/signatures.py`
- `DSPy/dspy_avqa/tools.py`
- `DSPy/dspy_avqa/prompt_config.py`
- `DSPy/dspy_avqa/yamls/daily_qa_prompt_v0.yaml`
- `DSPy/dspy_avqa/yamls/daily_qa_prompt_v1.yaml`

The run scripts and env files under `scripts/` were still present because
`scripts/` is ignored by `.gitignore`, so `git reset --hard` did not manage those
ignored files.

## 7. Verification Performed

After recovery, the following checks passed:

```bash
python -m compileall DSPy/dspy_avqa DSPy/avqa_dspy_impl.py
bash -n scripts/run_dspy_react_agent_daily_controltool.sh scripts/run_dspy_react_agent_daily_controltool_sanity.sh
```

YAML parsing confirmed:

```text
daily_qa_prompt_v0.yaml:
  tools.allowed = None
  temporal_grounding_prompt_template exists

daily_qa_prompt_v1.yaml:
  tools.allowed = ['ask_perception']
  temporal_grounding_prompt_template does not exist
```

## 8. Current Operational Rules

Use v1/tool3 when the run should only call `ask_perception`:

```bash
PROMPT_YAML=.../daily_qa_prompt_v1.yaml
DSPY_AVQA_ALLOWED_TOOLS=ask_perception
```

Use v0/sanity when the run should allow both tools:

```bash
PROMPT_YAML=.../daily_qa_prompt_v0.yaml
# leave DSPY_AVQA_ALLOWED_TOOLS unset
```

Remember: if `DSPY_AVQA_ALLOWED_TOOLS=ask_perception` is set, it overrides v0 and
disables `temporal_ground_video` even when v0 prompt text mentions it.
