# Webcam scene graph prototype

2D scene graphs from a webcam, video or image folder (YOLO-World + tracker +
rule-based relations), with optional VLM features:

| Key | What it does | Model calls |
|---|---|---|
| `v` | single VLM call: facts and hazards for the frame (baseline, no tools) | 1 |
| `c` | turn the graph into yes/no assertions and let the VLM verify them | 1 |
| `a` | risk agent with tools, evidence and a validator (`risk_agent.py`) | up to `--agent-steps` |
| `s` / `q` | save graph + frame / quit | none |

Files: `webcam_scene_graph.py` (pipeline), `risk_agent.py` (agent),
`ollama_backend.py` (local back-end), `test_*.py` (pytest).

## Setup

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q
```

## Model back-ends

The VLM features (`v`, `c`, `a`) can use one of two back-ends:

| `--backend` | Needs | Data leaves the machine |
|---|---|---|
| `anthropic` (default) | `ANTHROPIC_API_KEY` in the environment | yes, frames go to the Anthropic API |
| `ollama` | a running Ollama server with a vision model | no, requests go only to `--ollama-host` |

Options:

- `--vlm-model`: model name for the chosen back-end, for example `claude-sonnet-5-5` or `qwen3-vl:2b`
- `--ollama-host`: default `http://localhost:11434`
- `--num-ctx`: Ollama context window in tokens, default 4096
- `--max-tokens`: Ollama only; output limit per call (`num_predict`), default 1024.
  The Anthropic back-end keeps its fixed limits (2000 for `v`/`c`, 1500 per agent turn).
- `--temperature` (default 0.2) and `--seed` (default 42): Ollama only
- Agent and image options, Ollama only (the Anthropic back-end keeps the original behaviour):
  - `--crop-max-side` (512) and `--frame-max-side` (640): longest image side sent to the model
  - `--max-crops-per-node` (2): further crops of that node return an error telling the agent
    to write the annotation now, or `unknown`
  - `--drop-old-images` / `--no-drop-old-images` (on): crop images from earlier turns are
    replaced by a text placeholder naming the node and call_id; the camera frame is kept
  - `--budget-remind` (0.70) and `--budget-stop` (0.85): share of `num_ctx` at which the agent
    is told to annotate and finish, and at which the run stops (`stopped_by: "context_budget"`)
  - `--progress-remind` (4) and `--progress-stop` (8): if the agent has made no
    `write_annotation` call after that many turns, it gets a reminder listing the inspected
    nodes, and then the run stops (`stopped_by: "no_progress"`)

The allowed vocabulary (hazard types, predicates, fact attributes, enumerated values) and
the S0–S4 severity scale are in the agent's system prompt and in the `v` prompt
(`risk_agent.vocabulary_text()`), so the agent and the `v` baseline use the same terms.
`lookup_knowledge` covers object hazard knowledge and GHS pictograms only; a repeated
lookup of the same term returns the earlier answer with a note to write the annotation.

Use the **`-instruct`** builds (`qwen3-vl:2b-instruct`, `qwen3-vl:4b-instruct`).
The plain `qwen3-vl:2b` / `:4b` tags are thinking builds that always think and
used up the output limit in the earlier tests (see `docs/claude-code-notes.md`).

Run commands (Ollama):

```powershell
# live camera, keys v / c / a use the local model
.\.venv\Scripts\python webcam_scene_graph.py --backend ollama --vlm-model qwen3-vl:2b-instruct --num-ctx 8192

# headless, risk agent on every saved graph
.\.venv\Scripts\python webcam_scene_graph.py --backend ollama --vlm-model qwen3-vl:2b-instruct `
    --num-ctx 8192 --source clip.mp4 --no-display --save-every 30 --agent-on-save
```

What is recorded for Ollama runs:

- **`run_meta.json` → `backend`:** model tag, model digest, `num_ctx`,
  `num_predict`, temperature and seed. It also notes that
  `prompt_eval_count` includes cached tokens.
- **`run_meta.json` → `model_calls`:** per call, prompt, cached and output
  tokens, `done_reason`, latency, and a truncation flag.
- **`run_meta.json` → `invalid_model_runs`:** every run judged truncated or
  failed.
- **`run_meta.json` → `backend.agent_options`:** the agent and image options above.
- **`*_agent_log.json`:** `stopped_by` (`finished`, `max_steps`, `no_tool_calls`,
  `context_budget`, `no_progress` or `context_overflow`), `options`, `turns` (tokens per
  turn), `images_dropped`, `crops_per_node`, `repeated_lookups`, the annotations, and under `backend` the same
  settings and per-call records, plus `validity`: `valid`, `invalid: truncated`,
  `invalid: context overflow` or `invalid: failed (...)`. A warning is printed for
  invalid runs.
- **Context overflow** (Ollama refusing an oversized request with HTTP 400) ends the
  run with `stopped_by: "context_overflow"`. The partial log and the annotations
  written so far are kept.
- **Other failed agent runs** (any other model error) still write no `*_agent.json`
  or `*_agent_log.json`. They appear only in `run_meta.json`.

Run command (Anthropic):

```powershell
$env:ANTHROPIC_API_KEY = "..."   # set in the shell, never in a file
.\.venv\Scripts\python webcam_scene_graph.py --vlm-model claude-sonnet-5-5
```

### Known limits of the Ollama back-end (tested with Ollama 0.40.14)

- **Too-long prompts are handled in two ways:**
  - **Text-only prompts are cut silently** to exactly `num_ctx/2 + 2` tokens.
    The reply looks normal (`done_reason` stays `"stop"`), and only Ollama's
    `server.log` says `truncating input prompt`. The prototype flags a call as
    truncated if `prompt_eval_count` equals that cut size, if it reaches
    `num_ctx`, or if the prompt shrinks while the conversation grows (and no
    images were dropped).
  - **In the agent run (with images), Ollama refused the request** with HTTP
    400: `request (9193 tokens) exceeds the available context size (8192
    tokens)`. `server.log` shows that Ollama's truncation step did not run for
    that request and its `llama-server` runner rejected it; why the step
    skipped it is not confirmed. Such runs are marked
    `invalid: context overflow`.
- **Images cost many tokens.** From `server.log`, a 640×480 frame cost 1036
  tokens and each 512 px crop 1056–1058. Measured on 2026-11-06: crops at 256,
  512 and 1024 px all cost 1058 tokens, which fits the runner's
  `--image-min-tokens 1024` floor. Smaller crops therefore do not save tokens;
  fewer images do (`--drop-old-images` cut prompt tokens by 34% in one run). The agent's first prompt is about
  2550 tokens; about 1128 of them are the system prompt, tool schemas and
  chat template.
- **The agent can repeat itself.** With `qwen3-vl:2b-instruct` it kept
  cropping the same object and never wrote an annotation before the context
  ran out (`docs/claude-code-notes.md`). With the crop limit, image dropping,
  vocabulary in the prompt and the progress reminder it now writes, but in the
  2026-11-06 runs every `qwen3-vl:2b-instruct` write was rejected for a
  missing `confidence`. It also sent `finish` in the same turn, so it never
  saw the rejections. `qwen3-vl:4b-instruct` wrote accepted annotations but
  ran about 2.7× slower (60% of it on the CPU).
- **The validator checks the form of the evidence, not what it shows.** An
  asserted spill hazard on a capped bottle passed because it cited existing
  crops and a geometry call.
- **`v` output can be cut** at `--max-tokens` (1024). Such a call is still
  labelled `valid`; only `output_cut` in the call record shows it.
- **Thinking builds:** `qwen3-vl:2b` and `:4b` think even with
  `think: false`. Use the `-instruct` builds.
- **The seed is reproducible only with the same prompt-cache state.** Same
  seed and temperature 0.2 gave identical output for identical requests when
  the cache state matched; a cold first call differed.

On an RTX 3050 (4 GB), `ollama ps` showed this CPU/GPU split:

| model / num_ctx | 4096 | 8192 |
|---|---|---|
| `qwen3-vl:2b` | 18% / 82% | 32% / 68% |
| `qwen3-vl:4b` | 54% / 46% | 60% / 40% |
| `qwen3-vl:2b-instruct` | not measured | 34% / 66% |
| `qwen3-vl:4b-instruct` | not measured | 60% / 40% |
