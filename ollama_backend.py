#!/usr/bin/env python3
"""
Local model back-end (Ollama) for webcam_scene_graph.py and risk_agent.py.

The prototype speaks Anthropic-style messages: content blocks of type text,
image, tool_use and tool_result, and a call returns
{"content": [...], "stop_reason": ..., "usage": {"input": n, "output": n}}.
This module converts that to Ollama's /api/chat format and back, so the VLM
features and the risk agent run on a local model. Nothing leaves the machine:
requests go only to --ollama-host (default http://localhost:11434).

Ollama /api/chat fields used here, checked against Ollama 0.40.14 with test
calls (and the API docs, docs/api.md in the ollama repository):
  request   model, messages, tools, stream=false, options.num_ctx,
            options.num_predict
  message   role (system|user|assistant|tool), content, images (raw base64,
            no data: prefix), tool_calls [{"function": {"name", "arguments"}}],
            tool_name (on role "tool")
  reply     message.content, message.thinking, message.tool_calls
            [{"id", "function": {"index", "name", "arguments": {...}}}],
            done_reason, prompt_eval_count, eval_count

Observed with qwen3-vl:2b and qwen3-vl:4b (not documented, so re-check for
other models or versions):
  - Images on a role "tool" message reach the model (it read a test crop), so
    crop images stay on the tool message; no extra user message is needed.
  - prompt_eval_count is the whole prompt, cached prefix included.
  - A prompt longer than num_ctx is cut silently to num_ctx // 2 + 2 tokens:
    the reply is a normal 200 (done_reason "stop"); only the server log says
    "truncating input prompt". See judge_truncation.
  - The thinking builds (qwen3-vl:2b, :4b) always think, also with
    think=false, and thinking counts against num_predict. The -instruct builds
    do not think.
  - options.seed with temperature 0.2 gave identical output for identical
    requests only when the prompt cache state was the same.
  - The agent request with images that did not fit was not cut: the runner
    (llama-server) answered HTTP 400 "request (N tokens) exceeds the available
    context size". OllamaClient raises ContextOverflow for that.
"""
import json
import re
import time
import urllib.error
import urllib.request

DEFAULT_HOST = "http://localhost:11434"


# --------------------------------------------------- Anthropic -> Ollama
def _text_of(blocks):
    """Text of a content value (string or list of blocks)."""
    if isinstance(blocks, str):
        return blocks
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def _images_of(blocks):
    if isinstance(blocks, str):
        return []
    return [b["source"]["data"] for b in blocks
            if b.get("type") == "image" and b.get("source", {}).get("type") == "base64"]


def convert_tools(tools):
    """Anthropic tool schemas -> Ollama function tools."""
    return [{"type": "function", "function": {
        "name": t["name"], "description": t.get("description", ""),
        "parameters": t.get("input_schema", {"type": "object", "properties": {}})}}
            for t in tools or []]


def to_ollama_messages(system, messages):
    """Anthropic-style messages -> Ollama messages.

    tool_result blocks become role "tool" messages (in order) with tool_name
    looked up from the matching tool_use; their images go on the same message.
    Anthropic's is_error flag has no Ollama field, so it becomes a text prefix.
    """
    out = []
    if system:
        out.append({"role": "system", "content": system})
    names = {}  # tool_use id -> tool name
    for m in messages:
        content = m["content"]
        if m["role"] == "assistant":
            msg = {"role": "assistant", "content": _text_of(content)}
            calls = []
            for b in content if isinstance(content, list) else []:
                if b.get("type") == "tool_use":
                    names[b["id"]] = b["name"]
                    calls.append({"function": {"name": b["name"], "arguments": b.get("input") or {}}})
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        rest = []
        for b in blocks:
            if b.get("type") != "tool_result":
                rest.append(b)
                continue
            inner = b.get("content", "")
            text = _text_of(inner)
            if b.get("is_error"):
                text = "tool error: " + text
            msg = {"role": "tool", "content": text,
                   "tool_name": names.get(b.get("tool_use_id"), "")}
            imgs = _images_of(inner)
            if imgs:
                msg["images"] = imgs
            out.append(msg)
        if rest:
            msg = {"role": "user", "content": _text_of(rest)}
            imgs = _images_of(rest)
            if imgs:
                msg["images"] = imgs
            out.append(msg)
    return out


# --------------------------------------------------- Ollama -> Anthropic
STOP_REASONS = {"stop": "end_turn", "length": "max_tokens"}


def from_ollama_reply(reply, id_prefix="toolu_ollama"):
    """Ollama /api/chat reply -> {"content", "stop_reason", "usage"}.

    Tool call ids come from Ollama when present, otherwise they are generated.
    Arguments given as a JSON string are decoded; if that fails the raw text is
    kept under "_raw_arguments" so the tool reports an error instead of crashing.
    The model's thinking text is not put into the history (kept in "thinking").
    """
    msg = reply.get("message") or {}
    content = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if args is None:
            args = {}
        elif isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw_arguments": args}
        if not isinstance(args, dict):
            args = {"_raw_arguments": args}
        content.append({"type": "tool_use", "id": tc.get("id") or "%s_%d" % (id_prefix, i),
                        "name": fn.get("name", ""), "input": args})
    has_tools = any(b["type"] == "tool_use" for b in content)
    done = reply.get("done_reason")
    return {"content": content,
            "stop_reason": "tool_use" if has_tools else STOP_REASONS.get(done, done),
            "usage": {"input": reply.get("prompt_eval_count", 0),
                      "output": reply.get("eval_count", 0)},
            "thinking": msg.get("thinking", "")}


# ------------------------------------------------------------- the client
def _http_post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:  # Ollama puts the reason in the body
        raise RuntimeError("ollama HTTP %d: %s" % (exc.code, exc.read().decode(errors="replace")[:300]))


def _http_get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def model_digest(model, host=DEFAULT_HOST, get=None):
    """Digest of a local model from /api/tags, or None if it cannot be read."""
    try:
        tags = (get or _http_get)(host.rstrip("/") + "/api/tags", 30)
    except Exception:  # the digest is metadata; a failed lookup must not stop a run
        return None
    for m in tags.get("models", []):
        if model in (m.get("name"), m.get("model")):
            return m.get("digest")
    return None


def judge_truncation(prompt_tokens, num_ctx, prev=None):
    """Was the prompt cut by Ollama? Returns a reason string or None.

    Ollama 0.40.14 does not flag this in the reply (done_reason stayed "stop"
    for cut prompts). Observed rule, checked against server.log at num_ctx
    4096 and 8192: a prompt longer than num_ctx is cut to exactly
    num_ctx // 2 + 2 tokens ("truncating input prompt limit=2050", 4098).
      1. prompt_eval_count == num_ctx // 2 + 2  -> cut (a real prompt of
         exactly that length would be a false positive)
      2. prompt_eval_count >= num_ctx           -> cannot fit (not observed)
      3. in a growing conversation the prompt got smaller -> cut
    prev = (messages before, prompt tokens before) for rule 3.
    """
    if not prompt_tokens:
        return None
    if prompt_tokens == num_ctx // 2 + 2:
        return "prompt_eval_count %d == num_ctx/2+2, Ollama's cut size" % prompt_tokens
    if prompt_tokens >= num_ctx:
        return "prompt_eval_count %d >= num_ctx %d" % (prompt_tokens, num_ctx)
    if prev and prev[1] and prompt_tokens < prev[1]:
        return "prompt shrank from %d to %d tokens while the conversation grew" % (prev[1], prompt_tokens)
    return None


class ContextOverflow(RuntimeError):
    """Ollama refused the request because it does not fit into num_ctx.

    Seen on Ollama 0.40.14 (llama-server runner) for an agent request with
    images: HTTP 400 {"error": "request (9193 tokens) exceeds the available
    context size (8192 tokens), try increasing it"}. risk_agent.run_agent ends
    the run with stopped_by "context_overflow" when it sees this flag.
    """
    context_overflow = True

    def __init__(self, message, prompt_tokens=None, context_size=None):
        super().__init__(message)
        self.prompt_tokens, self.context_size = prompt_tokens, context_size


_OVERFLOW = re.compile(r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)")


class OllamaClient:
    """Callable with the same signature as risk_agent._anthropic_call.

    max_tokens, temperature and seed given here are sent as Ollama options
    (num_predict, temperature, seed) and override the per-call max_tokens.
    Every call is recorded in .records (tokens, cache, latency, truncation).
    `post(url, body, timeout)` can be replaced by a fake in tests.
    """

    def __init__(self, host=DEFAULT_HOST, num_ctx=4096, timeout=900, post=None,
                 max_tokens=None, temperature=None, seed=None):
        self.host, self.num_ctx, self.timeout = host.rstrip("/"), num_ctx, timeout
        self.max_tokens, self.temperature, self.seed = max_tokens, temperature, seed
        self.post = post or _http_post
        self.calls = 0
        self.records = []
        # (number of messages, prompt tokens, number of images) of the previous call
        self._last = (0, 0, 0)

    def options(self, max_tokens):
        opts = {"num_ctx": self.num_ctx,
                "num_predict": self.max_tokens if self.max_tokens is not None else max_tokens}
        if self.temperature is not None:
            opts["temperature"] = self.temperature
        if self.seed is not None:
            opts["seed"] = self.seed
        return opts

    def chat(self, model, ollama_messages, tools=None, max_tokens=1500):
        body = {"model": model, "messages": ollama_messages, "stream": False,
                "options": self.options(max_tokens)}
        if tools:
            body["tools"] = tools
        t0 = time.time()
        try:
            return self.post(self.host + "/api/chat", body, self.timeout)
        except RuntimeError as exc:
            m = _OVERFLOW.search(str(exc))
            if not m:
                raise
            self.records.append({
                "call": self.calls, "prompt_tokens": int(m.group(1)), "cached_tokens": None,
                "output_tokens": 0, "done_reason": None, "output_cut": False,
                "truncated": False, "truncation_reason": None,
                "context_overflow": True, "context_size": int(m.group(2)),
                "error": str(exc)[:300], "latency_s": round(time.time() - t0, 2)})
            print("warning: Ollama refused the request: %d tokens > context %d; this run is invalid"
                  % (int(m.group(1)), int(m.group(2))))
            raise ContextOverflow(str(exc), int(m.group(1)), int(m.group(2)))

    @property
    def truncated(self):
        return any(r["truncated"] for r in self.records)

    @property
    def overflowed(self):
        return any(r.get("context_overflow") for r in self.records)

    def _record(self, reply, n_messages, latency_s, n_images=0):
        prompt = reply.get("prompt_eval_count", 0)
        # Rule 3 (prompt shrank) only holds if nothing was removed on purpose:
        # with --drop-old-images the request has fewer images and is legitimately
        # shorter (false positive seen on 2026-11-06 before this check).
        grew = n_messages > self._last[0] and n_images >= self._last[2]
        reason = judge_truncation(prompt, self.num_ctx, self._last[:2] if grew else None)
        self._last = (n_messages, prompt, n_images)
        rec = {"call": self.calls, "prompt_tokens": prompt,
               "cached_tokens": reply.get("prompt_eval_cached_count"),
               "output_tokens": reply.get("eval_count", 0),
               "done_reason": reply.get("done_reason"),
               "output_cut": reply.get("done_reason") == "length",
               "truncated": reason is not None, "truncation_reason": reason,
               "latency_s": round(latency_s, 2)}
        self.records.append(rec)
        if reason:
            print("warning: Ollama truncated the prompt (%s); this run is invalid" % reason)
        return rec

    def __call__(self, model, system, tools, messages, max_tokens=1500):
        self.calls += 1
        t0 = time.time()
        omsgs = to_ollama_messages(system, messages)
        reply = self.chat(model, omsgs, convert_tools(tools), max_tokens)
        rec = self._record(reply, len(messages), time.time() - t0,
                           sum(len(m.get("images", [])) for m in omsgs))
        out = from_ollama_reply(reply, "toolu_ollama_%d" % self.calls)
        out["usage"]["truncated"] = rec["truncated"]
        return out

    def vlm_text(self, image_b64, prompt, model, max_tokens=2000):
        """One image + prompt -> text (the single-call VLM path)."""
        self.calls += 1
        msgs = to_ollama_messages(None, [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
            {"type": "text", "text": prompt}]}])
        t0 = time.time()
        reply = self.chat(model, msgs, None, max_tokens)
        self._record(reply, 1, time.time() - t0)
        self.last_reply = reply
        return (reply.get("message") or {}).get("content", "")

    def settings(self, model, digest=None):
        """What to store with a result so the run can be repeated."""
        return {"backend": "ollama", "host": self.host, "model": model, "model_digest": digest,
                "num_ctx": self.num_ctx, "num_predict": self.max_tokens,
                "temperature": self.temperature, "seed": self.seed,
                "prompt_eval_count_includes_cached": True,
                "note": "prompt_eval_count includes the cached prefix (observed on Ollama 0.40.14: "
                        "1270 total with 1269 cached); cached_tokens per call is in calls[]. "
                        "Same seed gave identical output only with the same cache state."}
