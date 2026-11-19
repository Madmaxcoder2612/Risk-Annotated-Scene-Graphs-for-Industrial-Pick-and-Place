# Claude Code notes

## 2026-11-05: Local Ollama back-end for the VLM features and the risk agent

**Changed**
- **New `ollama_backend.py`.** It converts Anthropic-style messages to Ollama
  `/api/chat` and converts the replies back.
  - text, base64 images, tool_use, and tool_result (sent as `role: "tool"` with `tool_name`)
  - Ollama's tool call ids are kept; an id is generated when one is missing.
  - Token counts come from `prompt_eval_count` and `eval_count`.
  - Uses only the Python standard library (urllib), so no new dependency.
- **`webcam_scene_graph.py`:**
  - New options `--backend anthropic|ollama` (default anthropic),
    `--ollama-host` and `--num-ctx`.
  - `--vlm-model` is the model name for either back-end.
  - `_vlm_text` and `run_agent_step` use the chosen back-end.
  - Nothing else changed.
- **`risk_agent.py`:** unchanged. The agent already accepts `call_model`.
- **New `test_ollama_backend.py`:** 15 tests with a fake transport. They cover
  conversion in both directions, the client request body, the truncation
  heuristic, a full agent loop through the adapter, and the `_vlm_text`
  switch.
- **New files:** `README.md`, `docs/ai-use.md`, `docs/claude-code-notes.md`
  (none of them existed before).
- **pytest** installed into `.venv`; it was missing there.
- **Paths:** the task named `prototypes/`, but the scripts are in `Code Work/`,
  so the work was done there.
  - `test_risk_agent.py` adds `../prototypes` to `sys.path`. It still passes
    because pytest also puts the test's own folder on the path.

**API checked, not assumed (Ollama 0.40.14)**

| What | Result |
|---|---|
| Tool calls in the reply | `message.tool_calls[].id` (for example `call_ez3ordky`) and `.function.name`; `.function.arguments` is a dict, not a string |
| Tool results | `role: "tool"` with `tool_name`, as in the Ollama API docs; `tool_call_id` is not documented, so it is not sent |
| Images on tool messages | Reach the model with both models: a test crop showing a red box with "42" was read correctly. Crops stay on the tool message, so no extra user message is needed |
| `prompt_eval_count` | Includes the cached prefix (1270 total, 1269 cached on a repeated call) |
| Prompt longer than `num_ctx` | Cut silently to about num_ctx/2 (`limit=2050` at 4096). The reply is a normal 200 with `done_reason: "length"`, and only `server.log` shows `truncating input prompt` |
| `num_predict` | Caps output, including thinking tokens |
| `think: false` | Not honoured by these builds (both use the `qwen3-vl-thinking` renderer) |

**Passed**
- `pytest`: 23 passed (15 new, 8 existing). One real bug was found and fixed:
  a tool call with no `arguments` became `{"_raw_arguments": null}` instead
  of `{}`.
- **Ollama end to end:**
  - A simple native tool call came back correctly formatted (2B).
  - Images worked in user and tool messages (2B, 4B).
  - One single `v` call returned valid JSON (2B, 4096, trial run, 823 output tokens).

**Failed**

Smoke test on `out/graph_001496.jpg`: a bottle held in a hand, with graph
nodes `bottle_20` and `hand_24`. `v` and `a` were run through the same
functions the keys call, 3× `v` and 2× `a` per setting. The agent ran through
the real `run_agent_step`.

| model / num_ctx | agent prompt tokens | truncated (server.log) | tool calls in agent | validator rejections | `v` JSON malformed | latency `v` / agent turn | CPU/GPU (`ollama ps`) |
|---|---|---|---|---|---|---|---|
| 2b / 4096 | 2555 | no | 0 of 2 runs (no tool_calls) | n/a (nothing written) | 3/3 (all empty) | 28–35 s / 23–26 s | 18% / 82% |
| 2b / 8192 | 2555 | no | 0 of 2 | n/a | 3/3 (2 empty, 1 cut off) | 44–53 s / 43 s | 32% / 68% |
| 4b / 4096 | 2555 | no | 0 of 2 | n/a | 3/3 (2 empty, 1 cut off) | 194–207 s / 171–174 s | 54% / 46% |
| 4b / 8192 | 2555 | no | 0 of 1 | n/a | not measured | not measured / 184 s | 60% / 40% |

- **The cause is the same in all 9 matrix `v` calls and all 7 matrix agent
  runs, plus the trial agent run.** The model spent the entire output budget
  on thinking:
  - `done_reason: "length"`
  - 5–8k characters of thinking
  - empty text, or JSON cut off partway
  - `max_tokens`, which maps to `num_predict`, is 2000 for `v` and 1500 per
    agent turn.
  - This is a budget problem, not a tool-format problem, so the prompt-based
    JSON tool-call fallback is not needed for now.
- **"Malformed" here means empty or cut off,** not badly formed JSON.
- **The trial `v` call that worked** (2B, 4096) labelled colours and object
  names as attributes (for example `attribute: "bottle"`), which is weak content.

**Not tested**
- **4b / 8192 is incomplete:** Claude Code stopped the background matrix run
  for low system memory. Only one agent run finished; `v` and the second
  agent run did not. It was not restarted.
- **Larger output budget not run:** a diagnostic with a larger `num_predict`
  (harness only, no code change) to see whether tool calls appear after
  thinking was not run because of that memory stop.
- **The `c` key** (assertion check) on Ollama.
- **Live camera with `--backend ollama`**, and the headless CLI with
  `--agent-on-save`. The smoke test called the same functions directly.
- **Multi-turn agent behaviour:** crops in context, validator loop, whether
  truncation happens after a few crops at 4096.
- **The Anthropic back-end after this change:** no API key is available. Its
  code path is unchanged apart from `call_model=None` in the default case.

**Open (needs a decision)**
1. **Output budget for Ollama.** Raise it, or make it an option (for example
   `--max-tokens`)? Thinking alone used 1500–2000+ tokens. A larger budget
   also needs `--num-ctx 8192`: the agent prompt is about 2550 tokens, and
   each crop adds about 1000 or more.
2. **Non-thinking models.** Try a model build without forced thinking,
   for example an instruct variant, if one is available for qwen3-vl in
   Ollama. That would be faster and keep the budget for the answer.
3. **Prompt-based JSON tool-call fallback.** Only worth it if native tool calls
   still fail once the budget problem is solved. Native tool calls worked in
   the probe.
4. **Machine recommendation:** `qwen3-vl:2b` with `--num-ctx 8192` and a larger
   output budget. 4b is 4–7× slower here (more than half of it runs on CPU).
   2b/4096 is the fastest, but leaves too little room for thinking plus
   crops. This is a recommendation for the next test, not a measured result.

## 2026-11-05 (later): Instruct builds, --max-tokens, truncation flag, run records

**Changed**
- **New models pulled into `D:\OllamaModels`:** `qwen3-vl:2b-instruct`
  (digest `afc393e3…fb573e`) and `qwen3-vl:4b-instruct` (`84f7ed80…5bc318`).
  - Both tags were checked in the registry first (HTTP 200).
  - `ollama show`: capabilities `completion, vision, tools`, no `thinking`;
    renderer `qwen3-vl-instruct`.
- **`webcam_scene_graph.py`:**
  - New options `--max-tokens` (default 1024, sent as `num_predict`),
    `--temperature` (default 0.2) and `--seed` (default 42). All three apply
    to Ollama only; the Anthropic calls keep their fixed 2000/1500.
  - `run_meta.json` gets a `backend` section, `model_calls` and
    `invalid_model_runs`.
  - `*_agent_log.json` gets a `backend` section with settings, per-call
    records and `validity`.
- **`ollama_backend.py`:**
  - `judge_truncation()` decides whether a prompt was truncated.
  - Every call is recorded: prompt, cached and output tokens, `done_reason`,
    latency, truncated or not.
  - `settings()` lists the run settings, and `model_digest()` reads the
    digest from `/api/tags`.
  - Temperature and seed are sent as Ollama options.
- **Truncation detection:** a call counts as truncated if any of these holds:
  - `prompt_eval_count == num_ctx // 2 + 2` (Ollama's cut size)
  - `prompt_eval_count >= num_ctx`
  - the prompt shrank while the conversation grew

  A truncated run is labelled `invalid: truncated` in the agent log and in
  `run_meta.json`, and a warning is printed. A run that fails (for example
  the HTTP 400 below) is labelled `invalid: failed (...)`.
- **`test_ollama_backend.py`:** 8 new tests, 31 in total with
  `test_risk_agent.py`.

**Correction to the earlier entry**
- **`done_reason` is not a truncation signal.** The earlier entry said a cut
  prompt returns `done_reason: "length"`. That was a side effect of that
  particular test.
  - Re-checked on `qwen3-vl:2b-instruct`: cut prompts returned
    `done_reason: "stop"`.
  - The cut size was exactly `num_ctx/2 + 2` at both 4096 (2050) and 8192
    (4098), confirmed in `server.log`.
  - A 6016-token prompt at 8192 was not cut.
  - `done_reason: "length"` is now recorded separately as `output_cut`.
- **Ollama does not always truncate silently.** In the agent run below it
  refused the request with HTTP 400 instead. Not verified: whether this is
  because the request contains images.

**Verified**

| What | Result |
|---|---|
| Seed and temperature as options | Accepted. Same seed and temperature 0.2 gave identical output 3/3 when the prompt-cache state matched (21 tokens cached each time). A cold first call with the same seed gave different output, so reproducibility depends on the cache state |
| Cached tokens | `prompt_eval_count` includes the cached prefix. Recorded per call as `cached_tokens` next to `prompt_tokens` |
| Instruct build tool calls | A simple tool call came back in the right format (`id`, dict arguments) in 32 tokens, with no thinking |

**Passed**
- `pytest`: 31 passed.

**Smoke test**

Run on `out/graph_001496.jpg` with the same functions the keys call.
Settings: `max_tokens` 1024, temperature 0.2, seed 42. Free RAM was checked
before every model call.

| model / num_ctx | call | output tokens | latency | tool calls in right format | validator rejections | empty / cut off / malformed JSON | truncation flag (and server.log) | CPU/GPU |
|---|---|---|---|---|---|---|---|---|
| **2b-instruct / 8192** | `v` ×3 | 229, 229, 229 | 8.3 s, 6.8 s, 6.9 s | n/a | n/a | 0 / 0 / 0 (valid JSON 3/3, identical) | no (no log lines) | 34% / 66% |
| **2b-instruct / 8192** | agent run 1 | 49, 57, 72, 57, 72 in turns 1–5 | 5–11 s per turn, 37.6 s in total | 12/12 calls: id, dict arguments, known tool | 0 (no `write_annotation` ever called) | n/a | no (no log lines); the run **failed** at turn 6 with HTTP 400 `request (9193 tokens) exceeds the available context size (8192 tokens)` | 34% / 66% |
| 2b-instruct / 8192 | agent run 2 | not run: free RAM was 3.0 GB, below the 3.0 GB guard | | | | | | |
| 4b-instruct / 4096 | `v` ×1, agent ×1 | not run: free RAM stayed at 3.0 GB, and the model needs about 4 GB | | | | | | |
| *thinking build* `qwen3-vl:2b` / 4096 | (earlier entry) | 2000 / 1500 (limit) | 28–35 s / 23–26 s | 0 tool calls | n/a | 3/3 empty | no | 18% / 82% |
| *thinking build* `qwen3-vl:2b` / 8192 | (earlier entry) | 2000 / 1500 (limit) | 44–53 s / 43 s | 0 | n/a | 2 empty, 1 cut off | no | 32% / 68% |
| *thinking build* `qwen3-vl:4b` / 4096 | (earlier entry) | 2000 / 1500 (limit) | 194–207 s / 171–174 s | 0 | n/a | 2 empty, 1 cut off | no | 54% / 46% |
| *thinking build* `qwen3-vl:4b` / 8192 | (earlier entry) | 1500 (limit) | 184 s per agent turn | 0 | n/a | not measured | no | 60% / 40% |

**Agent run 1, read from its call records**
- **Prompt growth:** the prompt grew 2553 → 4890 → 5252 → 7753 → 8115 tokens.
- **Repetition:** it cropped `bottle_20` 5 times and `hand_24` once, and called
  `get_geometry` and `lookup_knowledge` repeatedly.
- **No annotations:** it never called `write_annotation` or `finish`.
- **So it asserted nothing:** there is nothing to compare with the crops, and
  nothing for the validator to reject.
- **No files:** because the run failed, `run_agent_step` wrote no
  `_agent.json` or `_agent_log.json`. That is existing behaviour. The run's
  call records are in the smoke-test report, and the run is in `CALLS`.

**`v` content check (2b-instruct)**

All three replies were identical. Compared with the frame:
- **Supported by the image:** "green plastic bottle with a matching cap",
  held by a hand.
- **Not supported by the image:**
  - A hazard `bottle_20 has bottle` with the evidence "could pose a risk of
    tipping", at confidence 0.95. Nothing in the image shows a tipping risk,
    and `has` is not a hazard relation.
  - The facts misuse `attribute: "part"` for the object name.
- **No validator covers `v`:** the single-call path, unlike the agent, has no
  validator to catch this.

**Not tested**
- **Agent run 2 (2b-instruct / 8192), and the whole `qwen3-vl:4b-instruct`
  run:** free RAM was 3.0 GB, so the runs stopped as requested. 4b-instruct's
  vision and tool support were confirmed only through `ollama show`.
- **2b-instruct at 4096.**
- **The `c` key on Ollama,** the live camera, and the CLI end to end with a
  real model. `run()` with `run_meta.json` was tested only with a fake model.
- **Whether the HTTP 400 comes from the images** or from something else. A
  text-only request at 8192 was truncated silently instead.
- **A truncated agent run with real images:** none happened. The truncation
  flag was tested with recorded numbers only.
- **The Anthropic back-end:** still no API key.

**Open (needs a decision)**
1. **Agent repetition and context growth.** Each crop costs about 1,200
   tokens and the 2B model repeats crops, so 8192 fills in about 5 turns.
   Options:
   - smaller crops (`min_side` 512 → 256)
   - drop old crop images from the history
   - a tool-call budget per node
   - a larger `--num-ctx`, if RAM allows
2. **Failed agent runs write no log file.** Should `run_agent_step` write a
   partial `_agent_log.json` when the model call fails? At the moment the
   failure is only in `run_meta.json`.
3. **No validator for `v`.** Its output has no validator; the tipping hazard
   above was asserted without visual support.
4. **Memory.** About 3 GB of RAM was free while the 2B model was loaded.
   Closing other programs would let the 4b-instruct run and agent run 2 go
   ahead.
5. **Recommendation for this machine:** `qwen3-vl:2b-instruct` with
   `--num-ctx 8192`. It is about 4–7× faster than the thinking build, gives
   valid JSON, and returns correctly formatted tool calls. Comparing it with
   4b-instruct is still open (not measured).

## 2026-11-06: HTTP 400 diagnosis, overflow handling, agent context options

**Stopped for memory before any model run.** Free RAM was 4.2 GB of 15.3 GB
with no model loaded. Loading `qwen3-vl:2b-instruct` at 8192 used about
3.6 GB in the last run (6.6 → 3.0 GB free), which would leave well under the
3.0 GB guard. So steps 3 (token measurement), 4 (measured part), 5 and 6 were
**not run**. No new model results exist; nothing below is a new model result
unless it says it comes from server.log.

**1. HTTP 400 in agent turn 6 (from `server.log`, 2026-11-05 22:34:10)**
- Confirmed from the log:
  - The request went to Ollama's `llama-server` runner, started with `-c 8192
    --context-shift --keep 4 --image-min-tokens 1024`.
  - The runner tokenised the request to 9193 tokens and answered
    `send_error: ... request (9193 tokens) exceeds the available context size
    (8192 tokens)`; Ollama returned 400 (`[GIN] ... | 400 | 440.17ms`).
  - Ollama's own truncation step (`llama_server.go:344 "truncating input
    prompt"`) did **not** run for this request. It did run for the earlier
    text-only probes (for example `limit=4098 prompt=12016` at 8192).
  - The previous turn ended at 8186 tokens (prompt 8115 + 71 output); turn 6
    added the turn-5 tool results, about 1007 more tokens.
- **Not confirmed:** why the truncation step skipped this request. The log has
  no token count from that step for it. Images in the request are one
  possibility, but I could not confirm it.

**What the log shows about image cost** (task 719, the first agent turn,
2553 tokens):
- tokens 0–1127: system prompt, tool schemas and chat template (the frame
  starts at index 1128);
- 1036 tokens: the 640×480 frame (image batches 512 + 512 + 12);
- about 389 tokens: the user text (graph and candidates).
- The crops in turns 2 and 4 cost 1056–1058 tokens each. They were 512×281
  and 512×256 px.
- A 512×281 image at Qwen-VL's usual 32 px per token would be well under
  1000 tokens, and the runner flag `--image-min-tokens 1024` suggests each
  image is scaled up to at least 1024 tokens. **This is an inference from
  the flag and these numbers, not a measured sweep.** If it holds,
  `--crop-max-side` below 512 will not save tokens, and only fewer images
  (drop old images, crop limit) will.

**Memory observation (from the log, not measured per process):** after the
agent run, `llama-server` reported `cache state: 4 prompts, 2166.847 MiB
(limits: 8192.000 MiB ...)`. That is its prompt cache; the 8192 MiB limit
matches llama.cpp's default host-RAM prompt cache. If it is in host RAM, it
explains part of the RAM drop during runs. Unloading the model (`ollama stop
qwen3-vl:2b-instruct`) should free it; not tested.

**2. Context overflow handling (done, tested with fakes)**
- `ollama_backend.ContextOverflow` is raised for HTTP 400 with "exceeds the
  available context size". The call is recorded with `context_overflow: true`,
  the token count and the context size.
- `risk_agent.run_agent` ends with `stopped_by: "context_overflow"` and
  returns the partial log and all annotations written so far. Other model
  errors still raise as before.
- `webcam_scene_graph.py` writes `_agent.json` and `_agent_log.json` for such
  runs, labels them `invalid: context overflow` (agent log `backend.validity`,
  `run_meta.json` `model_calls` and `invalid_model_runs`) and prints a
  warning.

**3. New options (Ollama back-end only; Anthropic keeps the old behaviour)**

| Option | Default | Effect |
|---|---|---|
| `--crop-max-side` | 512 | longest crop side sent to the model (small crops are still scaled up to it) |
| `--frame-max-side` | 640 | longest frame side, for `a`, `v` and `c` |
| `--max-crops-per-node` | 2 | further `crop_view` calls on that node return an error telling the agent to write the annotation now, or `unknown` |
| `--drop-old-images` / `--no-drop-old-images` | on | after each turn, crop images in earlier tool results become `[crop image of bottle_20 removed from history, call_id c1]` |
| `--budget-remind` | 0.70 | when prompt + output of a turn reach this share of `num_ctx`, a reminder to annotate and finish is added |
| `--budget-stop` | 0.85 | at this share the turn's tool calls are executed, then the run stops with `stopped_by: "context_budget"` |

- All of them are logged in `run_meta.json` (`backend.agent_options`) and in
  the agent log (`options`, plus `frame_sent_px`, `turns` with tokens per
  turn, `images_dropped` and `crops_per_node`).
- **Design choices to review:**
  - The camera frame is kept in the history; only crops are dropped. The
    model therefore sees each crop in exactly one turn and has to annotate
    from it then.
  - Dropping images changes the prompt prefix, so Ollama's prompt cache is
    reused only up to the first dropped image. The latency cost is not
    measured.
  - The budget uses the last turn's tokens. One turn can still add two or
    more crops after the stop check, so an overflow is still possible; that
    case is now handled (item 2).
- **Tokens per crop at each size: not measured** (RAM stop).

**4. System prompt and tool schemas:** about 1128 tokens together with the
chat template (from the log, see above): 44% of the 2553-token first prompt
and 14% of 8192. The split between the system prompt and the tool schemas is
not measured. Trimming looks worthwhile only after the image cost is
settled: images are 1036 + about 1057 per crop, while the prompt and schemas
are about 1128 once per turn (and mostly cached). I did not change them.

**Tests:** 39 pass (8 new):
- crop limit per node
- crop and frame size
- old images dropped while the frame is kept
- budget reminder and stop, and no budget without `num_ctx`
- overflow keeps the partial run
- client overflow record
- overflow agent log through `webcam_scene_graph.py`
- other HTTP errors still `invalid: failed`
- CLI options reach the settings

**Not tested**
- Every model run asked for: tokens per crop at each size, the system/tools
  split, 3 agent runs and 3 `v` calls with seeds 1, 2, 3, and the assertion
  check on the `v` output (all blocked by RAM).
- Whether the new options stop the agent from repeating crops or overflowing
  with the real model.
- Whether the verifier flags the "risk of tipping" claim. One note for when
  it runs: `annotation_to_assertions` turns a hazard into a statement about
  the relation only (`subject predicate object`); the hazard type and harm
  text, where "risk of tipping" sits, are not checked. The verifier is also
  the same 2B model.
- Why Ollama's truncation step skipped the 9193-token request.
- Whether the prompt cache is in host RAM.
- The Anthropic back-end (no API key).

## 2026-11-06 (later): Image tokens, seeded runs, effect of dropping images

**Setup.**
- `qwen3-vl:2b-instruct` (digest `afc393e3…`), num_ctx 8192, temperature 0.2,
  num_predict 1024, on `out/graph_001496.jpg`.
- Agent defaults: crops 512 px, frame 640 px, 2 crops per node, old images
  dropped, budget 0.70 / 0.85.
- Before each run free RAM had to be at least 5 GB with the model unloaded,
  and `ollama stop qwen3-vl:2b-instruct` ran after each run. So **every run
  started cold**, and its first call includes model loading.
- Driver and raw results (scratchpad, not in the project):
  `runs3.py`, `out3/report.json`, `out3/a_seed*_agent_log.json`,
  `out3/v_seed*.json`.

**RAM.**
- Free before each run: 5.29–5.91 GB, so all 8 runs ran.
- Lowest free during a run: 3.56 GB (image-token run) and 4.05–4.42 GB (agent
  runs).
- `ollama ps`: 2.8 GB, 34% / 66% CPU/GPU.

**Image tokens: the floor is confirmed for these sizes.** One call each; the
text-only prompt was 15 tokens.

| Image | prompt_eval_count | Image tokens (minus 15) | server.log image batches |
|---|---|---|---|
| crop 256 (141×256) | 1073 | 1058 | 512 + 512 + 32 = 1056 |
| crop 512 (282×512) | 1073 | 1058 | 512 + 512 + 32 = 1056 |
| crop 1024 (563×1024) | 1073 | 1058 | 512 + 512 + 32 = 1056 |
| frame 640×480 | 1053 | 1038 | 512 + 512 + 12 = 1036 |

- Every image cost the same, about 1056 image tokens, between 256 and 1024 px.
  This fits the runner's `--image-min-tokens 1024`.
- So `--crop-max-side` does not change the token cost on this setup. Only
  the number of images does.
- Not tested: images larger than 1024 px, and whether a larger image ever
  costs more than the floor.

**System prompt and tool schemas** (text-only calls):
- user "x": 9 tokens
- system prompt + "x": 315 tokens, so the system prompt is about 306
- system prompt + tool schemas + "x": 1133 tokens, so the tool schemas are
  about 818

The schemas are the larger part, 25–27% of a turn's prompt when old images
are dropped. They sit in the cached prefix, though, so trimming them saves
context more than time. I did not change them.

**`v`, seeds 1, 2 and 3** (1 call each, cold start):

| Seed | Prompt | Output | Latency | JSON | Truncated |
|---|---|---|---|---|---|
| 1 | 1274 | 232 | 11.6 s | valid | no |
| 2 | 1274 | 224 | 11.2 s | valid | no |
| 3 | 1274 | 231 | 11.3 s | valid | no |

- The latency includes loading the model. Warm calls took about 7 s on
  2026-11-05.
- The outputs differ by seed:
  - Seed 1: hazard `bottle_20 has hand_24`, type "potential hazard",
    severity S1, confidence 0.95. Evidence: "could pose a risk of the bottle
    being dropped or causing injury if the hand is not properly secured".
  - Seeds 2 and 3: hazard `bottle_20 has bottle`, with type and severity
    `unknown`.
  - All three: facts `bottle_20 part=bottle` and `hand_24 part=hand`.
- Checked against the frame: the facts' evidence ("green plastic bottle with
  a matching cap, held in a person's hand") matches what the image shows. The
  seed-1 hazard is speculation. Nothing in the image shows the hand losing
  grip or an injury path, and it is still asserted at 0.95.
- The earlier "risk of tipping" text (seed 42) was not reproduced.

**Agent, seeds 1, 2 and 3 (default, old images dropped) and seed 1 with
`--no-drop-old-images`:**

| Run | Tool calls | Crops per node | Annotations | Rejections | stopped_by | Latency | Tool-call format |
|---|---|---|---|---|---|---|---|
| seed 1 | 13 | bottle_20: 1, hand_24: 1 | 0 | 0 | max_steps (12) | 43.8 s | 13/13 |
| seed 2 | 13 | bottle_20: 1, hand_24: 1 | 0 | 0 | max_steps (12) | 43.9 s | 13/13 |
| seed 3 | 13 | bottle_20: 1, hand_24: 1 | 0 | 0 | max_steps (12) | 44.2 s | 13/13 |
| seed 1, no drop | 13 | bottle_20: 1, hand_24: 1 | 0 | 0 | max_steps (12) | 45.9 s | 13/13 |

What all four runs did:
- **The same path every time:**
  1. turn 1: crop the bottle and the hand
  2. turn 2: `get_geometry(bottle_20, hand_24)`
  3. turns 3–12: one `lookup_knowledge` per turn (bottle, spill,
     chemical_corrosive, chemical_toxic, chemical_flammable, then repeats of
     hazard, hazard_type, hazard_pictogram, GHS or vocabulary)
- **The seed changed only the later lookup terms.**
- **The new behaviour is different from 2026-11-05.** The agent no longer
  re-crops (the crop limit was never reached), but it now loops on
  `lookup_knowledge` and never calls `write_annotation` or `finish`.
- **No context budget was reached in the default runs:** the largest prompt
  was 3879 tokens, and the reminder threshold is 5734. The no-drop run got
  the reminder only at turn 12 (5769 + 25 tokens), its last turn, so the
  reminder's effect is untested.
- **No truncation and no overflow in server.log** for any run.

Tokens per turn (prompt / output):

| Turn | seed 1 | seed 2 | seed 3 | seed 1, no drop |
|---|---|---|---|---|
| 1 | 2553 / 49 | 2553 / 49 | 2553 / 49 | 2553 / 49 |
| 2 | 4890 / 35 | 4890 / 35 | 4890 / 35 | 4890 / 35 |
| 3 | 3003 / 22 | 3003 / 22 | 3003 / 22 | 5087 / 22 |
| 4 | 3176 / 22 | 3176 / 22 | 3176 / 22 | 5260 / 22 |
| 5 | 3236 / 24 | 3236 / 24 | 3236 / 24 | 5320 / 24 |
| 6 | 3300 / 23 | 3300 / 23 | 3300 / 23 | 5384 / 23 |
| 7 | 3362 / 23 | 3362 / 23 | 3362 / 23 | 5446 / 23 |
| 8 | 3424 / 22 | 3424 / 22 | 3424 / 22 | 5508 / 22 |
| 9 | 3484 / 23 | 3484 / 25 | 3484 / 22 | 5568 / 25 |
| 10 | 3547 / 22 | 3551 / 24 | 3545 / 25 | 5635 / 25 |
| 11 | 3608 / 23 | 3616 / 22 | 3612 / 22 | 5702 / 25 |
| 12 | 3671 / 22 | 3677 / 25 | 3879 / 25 | 5769 / 25 |

**Effect of dropping images** (seed 1, same tool calls in both runs):
- **Prompt:** from turn 3 on, about 2085–2100 tokens smaller per turn, which
  is the two crops minus their placeholders. Total prompt tokens 41,254 vs
  62,122 (−34%).
- **Latency:** 43.8 s vs 45.9 s (−5%). Turns 3–12 took 2.8–3.0 s with
  dropping and 3.0–3.3 s without. Turns 1–2 are identical in both.
- **Cache:** at turn 3 with dropping, 2608 of 3003 tokens came from the cache
  (the prefix up to the first dropped image). The re-evaluation cost no
  visible time.
- **What dropping changes:** it makes room in the context, but it did not
  change what the agent did here.

**Did the agent assert anything the crops do not show?** No. All four runs
wrote no annotations, no text and no `finish` summary. The saved crops (in
`out3/a_seed*_crops`) were not needed to judge an assertion, because there
was none.

**Bug found and fixed: false truncation flag.**
- All three default runs were labelled `invalid: truncated` by rule 3 ("the
  prompt shrank while the conversation grew") at turn 3. The prompt shrank
  because two crops were dropped on purpose: 4890 + 35 + about 190 tokens
  of geometry result, minus 2 × 1058, is about 3003.
- server.log has no "truncating input prompt" line for these runs.
- Fix in `ollama_backend.py`: rule 3 now applies only if the request has at
  least as many images as the previous one. New test
  `test_shrink_from_dropped_images_is_not_truncation`; 40 tests pass.
- Re-judging the recorded numbers with the fixed rule flags no call in any
  of the 4 runs.
- The agent log files from these runs still say `invalid: truncated`. I left
  them unchanged; treat the label as wrong for these 3 runs.

**Step 6 (replaced): what the assertion check could verify in the `v` output.**
`annotation_to_assertions` produced these statements (same for all seeds,
except the hazard object):

| `v` content | Statement the check would ask | Checkable? |
|---|---|---|
| fact `bottle_20 part=bottle` | "The bottle's part is bottle." | only object presence, and the wording is broken because `part` is misused |
| fact `hand_24 part=hand` | "The hand's part is hand." | the same |
| hazard relation (seed 1) | "The bottle is has the hand." | the visible relation in principle, but `has` is not a known predicate, so the sentence is ungrammatical |
| hazard relation (seeds 2, 3) | "The bottle is has the bottle." | no; the subject relates to itself |
| hazard_type, severity, confidence | none | no |
| evidence text ("risk of being dropped", earlier "risk of tipping") | none | no |

- **What it can check:** object presence and visible relations, and only when
  the `v` output uses a known predicate and the attributes correctly.
- **What it cannot check:** hazard type, harm, severity, confidence and the
  evidence text, which is where the speculative claims sit.
- The graph check (key `c`) checks the detector graph instead (2 nodes, plus
  `near` and `left_of`), not the `v` output.

**Not tested**
- Images larger than 1024 px; whether the floor depends on the aspect ratio
  beyond the 4 shapes measured.
- Warm (non-cold) latency for these runs; every run started cold by design.
- The crop limit with the real model: the agent never cropped a node twice.
- The budget reminder's effect: it fired only on the last turn of the no-drop
  run.
- `stopped_by` "context_budget" and "context_overflow" with the real model:
  neither happened.
- 4b-instruct, 2b-instruct at 4096, the `c` key, the live camera, and the
  Anthropic back-end.

**Open (needs a decision)**
1. **The agent loops on `lookup_knowledge`.** Options:
   - a per-tool call limit, like the crop limit
   - a reminder after N turns without `write_annotation`
   - returning the vocabulary in the first prompt so lookups are not needed
   - a lower `--agent-steps`
2. **`v` asserts hazards with no visual support at confidence 0.95,** and the
   check cannot catch it (see the table above).
3. **Trimming the tool schemas** (about 818 tokens): worth it only if
   context, not latency, is the limit.

> **Correction (added 2026-11-06, see the next entry):** the
> `invalid: truncated` label on the three default agent runs above (seeds 1,
> 2, 3) is a **false alarm**. The prompt shrank because old crop images were
> dropped on purpose; server.log shows no truncation. The rule was fixed, and
> the three agent log files now carry `validity_note` and
> `validity_corrected: "valid"` (the original label is kept).

## 2026-11-06 (evening): Vocabulary in the prompt, repeat lookups, no-progress stop

**Changes**
1. **Vocabulary in the system prompt.**
   - The hazard types, predicates, fact attributes, enumerated values and the
     S0–S4 scale are now in the system prompt (`risk_agent.vocabulary_text()`).
   - `lookup_knowledge` answers only object hazard knowledge and GHS
     pictograms. For `vocabulary` or `severity` it says they are in the system
     prompt.
   - `PROMPT_VERSION` is now `risk-agent-0.2`.
2. **Repeated `lookup_knowledge` for the same term** (case and spaces
   ignored) returns the earlier answer plus `repeat_of` and a note telling the
   agent to write its annotation now. The log entry gets `repeat_of` and
   `new_information: false`, and the result counts `repeated_lookups`.
3. **No-progress rule** (`--progress-remind 4`, `--progress-stop 8`, Ollama
   only; off for Anthropic):
   - If there is no `write_annotation` call after turn 4, a reminder lists the
     inspected nodes with their call_ids and asks for `write_annotation`
     (status `unknown` if unsure), then `finish`.
   - With still none after turn 8, the run stops with `stopped_by:
     "no_progress"`.
   - **Choice to review:** a rejected write counts as progress, so a run
     stuck fixing rejected writes is not stopped by this rule.
4. **`v` and `c` use the same vocabulary.**
   - `VLM_PROMPT` (now `vlm-0.2`) includes the same vocabulary block.
   - The verify prompt names the six relation phrases and says relations are
     2D.
   - `annotation_to_assertions` skips hazards whose predicate is not in the
     vocabulary (such as `has`) or that relate a node to itself.
   - The `v` JSON now stores `prompt_version` and the back-end settings.
5. **`--temperature`** was already there (default 0.2, recorded in
   `run_meta.json` → `backend.temperature`), so nothing changed.
6. **False-alarm note:** see the correction above.
- **Tests:** 44 pass (new: repeated lookup, vocabulary in the prompt,
  no-progress reminder and stop, no-progress off after a write and by
  default, shared vocabulary in `v`/`c`).

**Prompt cost** (text-only calls, 2b-instruct):

| | Before | Now |
|---|---|---|
| system prompt | 306 | 532 |
| tool schemas | 818 | 822 |
| first agent prompt (with frame) | 2553 | 2783 |
| `v` prompt (with frame) | 1274 | 1498 |

**Runs.**
- Setup: `out/graph_001496.jpg`, num_ctx 8192, num_predict 1024, crops
  512, frame 640, 2 crops per node, old images dropped, budget 0.70 / 0.85,
  progress 4 / 8.
- RAM: free before each run 5.96–6.40 GB, so all 11 runs ran. Lowest during
  a run: 4.48 GB (2B) and 3.32 GB (4B).
- `ollama stop` ran after each run, so every run started cold.
- server.log has no truncation or overflow lines, and every run is labelled
  `valid`.
- Raw data: scratchpad `out4/report.json` and `out4/a_*_agent_log.json`.

| Run | Tool calls | Crops per node | Annotations accepted / rejected | Why rejected | Repeat lookups | stopped_by | Latency |
|---|---|---|---|---|---|---|---|
| 2B, T 0.2, seed 1 | 12 | bottle 2 | 0 / 5 | `status` and `confidence` missing | 1 | finished (turn 2) | 31.1 s |
| 2B, T 0.2, seed 2 | 12 | bottle 1 | 0 / 7 | `confidence` missing | 0 | finished (turn 5) | 47.5 s |
| 2B, T 0.2, seed 3 | 8 | bottle 2 | 0 / 0 | – | 2 | **no_progress** (turn 8) | 33.4 s |
| 2B, T 0.8, seed 1 | 15 | bottle 2 (+2 refused by the crop limit) | 0 / 2 | `confidence` missing | 3 | finished (turn 5) | 48.5 s |
| 2B, T 0.8, seed 2 | 12 | bottle 1 | 0 / 7 | `confidence` missing | 0 | finished (turn 5) | 46.6 s |
| 2B, T 0.8, seed 3 | 20 | bottle 2 (+2 refused by the crop limit) | 0 / 4 | `confidence` missing | 5 | finished (turn 5) | 69.3 s |
| 4B, T 0.2, seed 1 | 9 | bottle 1, hand 1 | **3 / 1** | `asserted` without evidence | 0 | finished (turn 5) | 130.3 s |

Tool-call format was correct in every call (12/12, 12/12, 8/8, 15/15, 12/12,
20/20, 9/9).

Tokens per turn (prompt / output):
- 2B T0.2 s1: 2783/82, 4309/472
- 2B T0.2 s2: 2783/25, 3955/35, 3111/22, 3284/22 (reminder), 3413/776
- 2B T0.2 s3: 2783/25, 3955/35, 3111/22, 3284/22 (reminder), 3413/33,
  4594/35, 3750/22, 3980/22 (stop)
- 2B T0.8 s1: 2783/82, 4309/82, 4851/82, 4317/82 (reminder), 4941/390
- 2B T0.8 s2: 2783/25, 3955/35, 3111/22, 3284/22 (reminder), 3413/870
- 2B T0.8 s3: 2783/82, 4309/105, 4905/104, 4595/104 (reminder), 5442/751
- 4B T0.2 s1: 2783/34, 3964/33, 4105/35, 3259/22 (reminder), 3510/545.
  4B ran at 60% / 40% CPU/GPU (4.6 GB); turn latencies were 32.1, 15.0, 8.8,
  4.7 and 69.7 s.

**What changed in the agent's behaviour**
- **The agent now writes, but the 2B model's writes are malformed.**
  - Five of six 2B runs called `write_annotation` 2–7 times, and every call
    was rejected: `confidence` was missing every time, `status` once. Both
    are required in the schema, but Ollama does not enforce it.
  - **The 2B model never saw its rejections.** It put all its writes and
    `finish` in the same turn, and `run_agent` stops when `finish` is called,
    so no fix was possible.
  - Its `finish` summaries say annotations "have been written" and list
    hazards, but none were accepted.
- **The progress reminder came right before the writes.** It fired at turn
  4 in 6 of 7 runs. In 5 of those 6 runs the agent wrote in turn 5; seed 3
  at T 0.2 ignored it and was stopped at turn 8. Seed 1 at T 0.2 wrote in
  turn 2 with no reminder.
- **The repeat-lookup note did not stop repeats.** T 0.8 seed 3 repeated
  `bottle` 5 times after getting the note.
- **The crop limit worked when reached.** It refused 2 extra crops each in
  T 0.8 seeds 1 and 3. The 2B model then kept calling `get_geometry` and
  `lookup_knowledge` in the same pattern.
- **The 2B model never cropped the hand** in these runs; it did on
  2026-11-06 with the old prompt.
- **Temperature 0.2 vs 0.8 with the same seed:**
  - Seed 2: identical through turn 4; only the turn-5 writes differed.
  - Seed 1: the same first turn, then different paths.
  - Seed 3: different from turn 1.

**Annotations against the crops** (all 4B; no 2B annotation was accepted).
I looked at both crops (`a_4b-instruct_t0.2_seed1_crops`). The bottle crop
shows a green bottle with its cap screwed on, gripped at the side by a hand;
the hand crop shows fingers wrapped around the bottle's side.
- `ra_001` bottle closure `unknown` (status unknown): **not contradicted, but
  too cautious.** The cap is clearly on, so "closed" would have been
  supported. The agent's own note says "the cap appears to be on".
- `ra_002` bottle contents `unknown` (status unknown): **supported.** No
  label is visible.
- `ra_004` **asserted** hazard `bottle_20 positioned_above hand_24`, spill,
  S1, confidence 0.8, citing crop c1, crop c2 and geometry c3. **Not
  supported by the crops:**
  - **Spill:** needs an open container. The crop shows the cap on, and the
    agent's own closure note says so.
  - **"Positioned above":** the hand holds the bottle at its side, and the
    boxes overlap vertically (bottle y 73–407, hand y 184–371). The bottle
    reaches higher but is not above the hand.
  - **Why it passed:** the validator checks only that the cited calls exist
    and are of the right kind, not whether they show the claim.
- Rejected 2B writes (not stored, but they show what the model would claim):
  bottle `closure: open` and `tip_exposed: true`, plus `pointing_toward hand`
  and `mechanical_impact` S3. The crops contradict `open`, and the others
  have no support. The validator rejected them only for missing fields, not
  for content.

**`v`, seeds 1, 2 and 3 (T 0.2, new vocabulary prompt)**

| Seed | Prompt | Output | Latency | JSON | Vocabulary use |
|---|---|---|---|---|---|
| 1 | 1498 | 253 | 11.9 s | valid | attributes `closure` and `part`, predicate `carried_over`, hazard type `unknown` |
| 2 | 1498 | 333 | 13.8 s | valid | uses `bottle` and `hand` as attribute names (not in the vocabulary); predicates `in_contact_with` and `pointing_toward` |
| 3 | 1498 | 1024 | 31.3 s | **invalid: output cut at 1024 tokens** (`done_reason: length`) | – |

- **Content:**
  - Seed 1: `closure: closed`, which the cap supports. Hazard `carried_over
    hand_24` at S0 is not shown: the bottle is held at its side.
  - Seed 2: `in_contact_with hand_24` is supported (the hand grips it).
    `bottle_20 pointing_toward bottle_20` is a self-relation.
  - Seed 3: before it was cut off, it said "the bottle is empty" with
    confidence 0.0.
- **Checkable statements after the change:**
  - Seed 1: "The bottle's closure is closed.", "The hand's part is hand.",
    "The bottle is carried over the hand."
  - Seed 2: two broken fact sentences ("The bottle's bottle is a green
    plastic bottle.") and "The bottle is in contact with the hand." The
    self-relation is now skipped.
- **Validity gap:** the `v` validity label stays `valid` when the output is
  cut (`output_cut` is recorded per call but does not affect the label).

**Not tested**
- `c` (assertion check) runs with the new prompt: no model run was made.
- 4B at 4096, a second 4B run, or 4B with temperature 0.8.
- Whether the no-progress stop works for runs with only rejected writes:
  none happened.
- The Anthropic back-end, the live camera, and frames other than
  `graph_001496.jpg`.

**Open (needs a decision)**
1. **`finish` in the same turn as rejected writes** ends the run before the
   agent can fix them. Options:
   - refuse `finish` while the last turn's writes were rejected and none
     accepted
   - or execute `finish` only if it is the only call in the turn
2. **Missing `confidence` and `status` from the 2B model.** Options:
   - a clearer error that lists the missing fields with an example call
   - or a default `confidence` for `status: unknown` (this would change the
     validator rules, so it is your call)
3. **The validator does not check content.** The 4B spill hazard passed with
   evidence that does not show an open bottle. One option is to require a
   fact (for example `closure: open`) before a spill hazard on a container.
4. **`v` output cut at 1024 tokens:** raise `--max-tokens` for `v`, or label
   cut outputs invalid.
