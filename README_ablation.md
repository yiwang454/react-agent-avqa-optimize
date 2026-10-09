# Stage 2 Modality-Merge Ablation


B0 first reproduces the OmniAgent setup inside our framework, B1 removes only
the cross-check sentence, and the later configurations introduce the new
workflow, omni captioning, caption-first ordering, and modality merging one
factor at a time.

## Terminology

"Specialized Q" denotes all three original QA tools:

- `audio_qa`;
- `video_global_qa`;
- `video_clip_qa`.

The original five C+Q+M tools, previously referred to as tools A-E, are:

| Label | Tool | Role |
| --- | --- | --- |
| A | `audio_global_caption` | Whole-audio caption |
| B | `audio_qa` | Audio-only question answering |
| C | `video_global_qa` | Whole-video visual question answering |
| D | `video_clip_qa` | Clip-level visual question answering |
| E | `video_metadata` | Duration, FPS, frame count, and resolution |

Removing metadata means only that `video_metadata` is hidden from the planner.
Clip tools may still read duration internally to validate clip boundaries.

## Ablation configurations

| ID | Caption | QA | Metadata | Caption-first | Research question |
| --- | --- | --- | --- | --- | --- |
| **B0** | Specialized audio caption | Specialized Q | Yes | No | Can the framework migration reproduce the OmniAgent C+Q+M setup? |
| **B1** | Specialized audio caption | Specialized Q | Yes | No | Does removing the cross-check sentence reduce tool calls and latency? |
| **B2** | Specialized audio caption | Specialized Q | Yes | No | What is the effect of replacing the OmniAgent workflow with the new Free-ReAct workflow? |
| **B3** | `ask_caption` | Specialized Q | Yes | No | What is the isolated effect of replacing audio-only captioning with omni-modal captioning? |
| **B4** | `ask_caption` | Specialized Q | Yes | **Yes** | What is the isolated effect of mandatory caption-first ordering? |
| **B5** | `ask_caption` | Global omni Q + clip omni Q | Yes | **Yes** | What is the effect of replacing specialized QA tools with unified omni-modal QA tools? |
| **B6** | `ask_caption` | Global omni Q + clip omni Q | **No** | **Yes** | Is planner-visible metadata still necessary when clip-level omni QA remains available? |
| **B7** | `ask_caption` | Global omni Q only | Yes | **Yes** | Is clip-level omni QA necessary when metadata remains available? |
| **B8** | `ask_caption` | Global omni Q only | **No** | **Yes** | Can both metadata and clip-level omni QA be removed, leaving only caption plus global omni QA? |

`B5-flex` is an additional control. It is identical to B5 in its caption, QA,
metadata, models, and budget, but it does not enforce caption-first ordering.
It tests caption-first ordering again after the QA tools have been merged.

## Intended comparisons

- **B0 → B1:** Delete only the sentence "Cross-check and verify important
  information using multiple tools if needed." The preceding "Be selective:
  tools may be noisy or incomplete." sentence remains. 
- **B1 → B2:** Replace the OmniAgent workflow prompt with our workflow prompt.
  The new prompt removes the explicitly prescribed
  `THINK → ACT → OBSERVE → REFLECT` wording, describes the available tool
  types directly, and tells the planner to stop when the evidence is
  sufficient.
- **B2 → B3:** Replace audio-only captioning with omni-modal captioning.
- **B3 → B4:** **B4 enforces the planner calling `ask_caption` in the first
  round, but the planner may also call `ask_caption` again later. B3 may call
  `ask_caption` at any time and does not enforce it as the first call.**
- **B4 → B5:** Replace the three specialized QA tools with global and
  clip-level omni-modal QA tools.
- **B5 → B6:** Remove planner-visible metadata while retaining clip-level
  omni-modal QA.
- **B5 → B7:** Remove clip-level omni-modal QA while retaining metadata.
- **B7 → B8:** Remove metadata as well, leaving only `ask_caption` and
  `ask_perception`.
- **B5-flex → B5:** Isolate caption-first ordering when the omni-modal QA tools
  are already enabled.

## Clarification: B0

B0 does **not** remove the cross-check sentence. It uses tools A-E and the
OmniAgent-derived instruction. Its intended experimental change is the
framework migration: OmniAgent used LangChain, while B0 runs in our DSPy
compact-JSON Free-ReAct implementation. B1 is the configuration that removes
the sentence.

B0 preserves the tool split, specialized-tool prompts, and workflow intent as
closely as possible, but it cannot reproduce LangChain's native tool-call wire
format and message classes exactly. The ablation suite also fixes a common
six-valid-call budget and common final-answer mechanism across all
configurations.

## Clarification: B2 is still ReAct

B2 is **not** a non-ReAct baseline. It continues to use the same Free-ReAct
execution loop: at every round the planner either issues one tool action or
returns a final answer, receives the observation, and then makes the next
decision. What changes is the workflow wording, not the agent architecture.

The two workflow sources are:

- [OmniAgent-derived workflow used by B0](DSPy/dspy_avqa/yamls/modality_merge_ablation/b0_omniagent_cqm.yaml)
- [New Free-ReAct workflow used by B2](DSPy/dspy_avqa/yamls/modality_merge_ablation/b2_new_workflow_cqm.yaml)

### OmniAgent-derived workflow

```text
You are the central reasoning brain of an audio-video analysis agent.

Your role:
- Answer the user's question about a given video by intelligently using the available tools (audio and video analysis).
- Follow a THINK → ACT → OBSERVE → REFLECT loop:
- THOUGHT: Reason step by step about what to do next.
- ACTION: Call exactly one tool that moves you closer to the answer.
- OBSERVATION: Read and interpret the tool's output, update your beliefs.
- REFLECTION: Reflect on the previous steps and the overall process.

General rules:
- Use both AUDIO and VIDEO information whenever they can help. Prefer to listen first, then look.
- Do not invent timestamps, file paths, or other arguments. Use values taken from the user input or from previous tool outputs.
- Be selective: tools may be noisy or incomplete. Cross-check and verify important information using multiple tools if needed.
- Stop calling tools once you have enough evidence to answer confidently.
- If a tool reports a retryable invalid-input error, correct the arguments and call it again. Such validation failures do not use the tool-call budget.

Tool usage guidelines for this task:
- For a high-level understanding of the audio, you can use audio_global_caption.
- For detailed questions about what is said or heard, you can use audio_qa.
- For visual understanding of the whole video, use video_global_qa.
- For fine-grained visual details in a short time range, use video_clip_qa.
- If you need to choose or validate time ranges, call video_metadata.
- Use audio to find time and content first whenever possible, then inspect the corresponding visuals: from listen to look.
- Plan your tool calls, but you are free to adjust the plan based on what you observe from previous tools.

Final answer style:
- When you are done with tools, return a single option label and a brief high-level reasoning summary.
- Do not expose raw tool-call traces or long chain-of-thought.
```

### New Free-ReAct workflow

```text
You are the planning/reasoning model in an audio-visual question answering ReAct system.

You CANNOT watch or listen to the video directly.
You can only reason over:
1. the user question and options,
2. observations returned by the enabled caption, QA, and metadata tools.

There are five different tools:
- audio_global_caption: obtains a factual high-level caption of the whole audio track.
- audio_qa: asks a free-form question about the audio track.
- video_global_qa: asks a free-form question about the visual content of the whole video.
- video_clip_qa: asks a visual question about a selected time range.
- video_metadata: returns duration, FPS, frame count, and resolution for validating clip ranges.

Required workflow:
1. Freely choose any enabled tool based on the evidence you need. You may call any enabled tool multiple times.
2. Use audio evidence to identify content or time when useful, then inspect the corresponding visuals when the question requires it.
3. Return the final answer as soon as the available evidence is sufficient.

Tool-call freedom:
- Tool questions may quote, reuse, or directly ask the original user question and options.
- Tool questions do not have to be narrowly targeted or limited to missing or ambiguous evidence.
- video_clip_qa requires time_range [start_seconds, end_seconds] with end greater than start.
- If a tool returns a retryable invalid-input error, correct the call and continue; validation failures do not consume budget.

Budget:
- Each decision includes turn_index and max_turns.
- max_turns is the total valid tool-call budget.
- If the budget is exhausted, return a final answer using the evidence already collected.

Constraints:
- Do not invent timestamps or paths.
- Do not output markdown fences.
```

The main difference is therefore instruction style: the OmniAgent-derived
prompt explicitly names a four-stage reasoning loop and encourages cross-tool
verification, whereas the new workflow retains ReAct execution but uses a
simpler evidence-driven policy with explicit tool descriptions and budget
semantics.

## Why B4 is needed

B4 is required because changing the caption tool in B3 does not guarantee that
the planner will call it first. A planner may call a specialized QA tool first,
call `ask_caption` later, or answer without calling it. Comparing B3 with B4
isolates the causal effect of always obtaining the same broad audio-visual
context before subsequent planning. B5-flex provides the corresponding
ordering control after modality merging.

## Fixed controls

Across the suite, planner and perception models, reasoning effort, valid
tool-call budget, retry behavior, final-answer behavior, history handling,
media preprocessing, and caption-cache policy remain fixed. Each result records
accuracy, tokens, estimated cost, latency, per-tool call counts, invalid calls,
budget exhaustion, and final-answer status.
