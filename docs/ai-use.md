# AI use in this prototype

There are two kinds of AI use here:

1. **AI inside the prototype**: the models the code calls at run time.
2. **AI used to write the code**: coding assistants that changed files.

## 1. Models the prototype calls

| Use | Where | Back-end |
|---|---|---|
| Object detection (open vocabulary) | `webcam_scene_graph.py`, YOLO-World with CLIP text embeddings | local GPU |
| Single VLM annotation (`v`) | `vlm_annotate` / `_vlm_text` | `anthropic` or `ollama` |
| Assertion verification (`c`) | `verify_assertions` / `_vlm_text` | `anthropic` or `ollama` |
| Risk agent with tools (`a`) | `risk_agent.run_agent`, model call via `call_model` | `anthropic` or `ollama` |

Back-ends:

- **`anthropic`**: frames and graph text are sent to the Anthropic API.
- **`ollama`**: everything stays on the machine, and requests go only to the
  local Ollama server. Models tested on Ollama 0.40.14:
  - **`qwen3-vl:2b-instruct`** (digest `afc393e3…`): the build to use.
  - **`qwen3-vl:2b` and `qwen3-vl:4b`** (Q4_K_M, thinking builds): not
    usable with these output limits.
  - **`qwen3-vl:4b-instruct`** (digest `84f7ed80…`): one agent run on
    2026-11-06; it wrote accepted annotations but is about 2.7× slower on
    this machine.

  Each Ollama result records:
  - model tag and digest
  - `num_ctx`, `num_predict`, temperature (0.2) and seed (42)
  - token counts, including cached tokens
  - a validity label (`valid`, `invalid: truncated`, `invalid: context
    overflow`, `invalid: failed`)
  - for the agent: crop and frame sizes, crop limit per node, image
    dropping, context budget, `stopped_by` and tokens per turn

  The seed gives identical output only when the prompt-cache state is the
  same, so treat local runs as nearly, not fully, repeatable.

All VLM output is advisory research output, not a safety function:

- The risk agent's annotations pass a validator that requires cited tool calls
  as evidence.
- Each saved result records the model name and back-end settings, so runs can
  be compared.

## 2. AI-assisted development

Claude Code (Anthropic) was used to build parts of the prototype:

- **Environment setup.** It set up the environment and the CLIP dependency.
- **Vocabulary suggestion (since replaced).** It added an earlier
  vocabulary-suggestion feature. That feature is no longer in the current
  `webcam_scene_graph.py`.
- **Ollama back-end.** It wrote `ollama_backend.py` and
  `test_ollama_backend.py`, and added the back-end options to
  `webcam_scene_graph.py`.
- **Run records.** It added `--max-tokens`, `--temperature` and `--seed`,
  truncation and failure detection, and the run records in `run_meta.json`
  and the agent log.
- **Agent context options.** It added context-overflow handling (partial log
  kept), the crop and frame size options, the crop limit per node, dropping
  of old crop images, and the context budget. They were later run with the
  real models (see `docs/claude-code-notes.md`).
- **Prompt and loop changes.** It moved the vocabulary and severity scale
  into the agent's system prompt and the `v` prompt, added the repeated-lookup
  note and the no-progress reminder and stop, and made the `c` check skip
  relations outside the vocabulary. The prompt versions are now
  `risk-agent-0.2` and `vlm-0.2`. The vocabulary itself and the validator
  rules are unchanged; they remain the author's design decisions.

How the Ollama back-end was checked:

- **API fields:** checked against the installed Ollama 0.40.14 with test calls
  and the Ollama API docs, not assumed.
- **Undocumented behaviour:** images on tool messages, silent truncation and
  thinking were observed in test calls and are noted in the code.
- **Results:** the smoke-test results in `docs/claude-code-notes.md` come from
  real runs, with the method and limits stated.

The author reviews every change. Design decisions remain the author's: the
validator rules, the vocabularies, and which back-end to use for experiments.
