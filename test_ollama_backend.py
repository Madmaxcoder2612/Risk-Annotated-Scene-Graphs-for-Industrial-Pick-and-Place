"""Message conversion tests for ollama_backend.py. No Ollama server is used:
the HTTP transport is replaced by a fake that records requests and replays
canned replies shaped like the ones Ollama 0.40.14 returned in test calls."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import ollama_backend as ob  # noqa: E402

IMG = "aGVsbG8="  # base64 placeholder, content does not matter here


def image(data=IMG):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


class FakePost:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def __call__(self, url, body, timeout):
        self.requests.append((url, json.loads(json.dumps(body))))  # must be JSON-serialisable
        return self.replies.pop(0)


def reply(content="", tool_calls=None, prompt=100, out=10, done="stop", thinking=""):
    msg = {"role": "assistant", "content": content, "thinking": thinking}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"model": "m", "message": msg, "done": True, "done_reason": done,
            "prompt_eval_count": prompt, "eval_count": out}


# ------------------------------------------------- Anthropic -> Ollama
def test_tools_are_converted_to_function_schemas():
    t = [{"name": "crop_view", "description": "d",
          "input_schema": {"type": "object", "properties": {"node_id": {"type": "string"}}}}]
    assert ob.convert_tools(t) == [{"type": "function", "function": {
        "name": "crop_view", "description": "d",
        "parameters": {"type": "object", "properties": {"node_id": {"type": "string"}}}}}]


def test_system_user_text_and_image():
    msgs = ob.to_ollama_messages("SYS", [{"role": "user", "content": [image(), {"type": "text", "text": "hi"}]}])
    assert msgs == [{"role": "system", "content": "SYS"},
                    {"role": "user", "content": "hi", "images": [IMG]}]


def test_plain_string_content():
    assert ob.to_ollama_messages(None, [{"role": "user", "content": "hello"}]) == \
        [{"role": "user", "content": "hello"}]


def test_assistant_tool_use_and_tool_results_in_order():
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "looking"},
            {"type": "tool_use", "id": "a1", "name": "crop_view", "input": {"node_id": "knife_1"}},
            {"type": "tool_use", "id": "a2", "name": "get_geometry", "input": {"node_a": "x", "node_b": "y"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a1",
             "content": [{"type": "text", "text": "{\"call_id\": \"c1\"}"}, image("CROP")]},
            {"type": "tool_result", "tool_use_id": "a2", "is_error": True,
             "content": [{"type": "text", "text": "unknown node"}]}]},
    ]
    out = ob.to_ollama_messages(None, msgs)
    assert out[1] == {"role": "assistant", "content": "looking", "tool_calls": [
        {"function": {"name": "crop_view", "arguments": {"node_id": "knife_1"}}},
        {"function": {"name": "get_geometry", "arguments": {"node_a": "x", "node_b": "y"}}}]}
    assert out[2] == {"role": "tool", "tool_name": "crop_view", "content": "{\"call_id\": \"c1\"}",
                      "images": ["CROP"]}
    assert out[3] == {"role": "tool", "tool_name": "get_geometry", "content": "tool error: unknown node"}
    assert len(out) == 4, "a user turn with only tool_results adds no empty user message"


def test_tool_results_followed_by_user_text():
    msgs = [{"role": "assistant", "content": [{"type": "tool_use", "id": "a1", "name": "finish", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a1", "content": "ok"},
                                         {"type": "text", "text": "anything else?"}]}]
    out = ob.to_ollama_messages(None, msgs)
    assert [m["role"] for m in out] == ["assistant", "tool", "user"]
    assert out[0]["content"] == "" and out[1]["content"] == "ok" and out[2]["content"] == "anything else?"


# ------------------------------------------------- Ollama -> Anthropic
def test_reply_with_tool_calls_keeps_ollama_ids_and_usage():
    r = reply(thinking="hmm", tool_calls=[
        {"id": "call_ez3ordky", "function": {"index": 0, "name": "get_geometry",
                                             "arguments": {"node_a": "knife_1", "node_b": "hand_1"}}}],
        prompt=179, out=150)
    out = ob.from_ollama_reply(r)
    assert out["content"] == [{"type": "tool_use", "id": "call_ez3ordky", "name": "get_geometry",
                               "input": {"node_a": "knife_1", "node_b": "hand_1"}}]
    assert out["stop_reason"] == "tool_use"
    assert out["usage"] == {"input": 179, "output": 150}
    assert out["thinking"] == "hmm"


def test_reply_generates_missing_ids_and_decodes_string_arguments():
    r = reply(content="ok", tool_calls=[
        {"function": {"name": "a", "arguments": "{\"x\": 1}"}},
        {"function": {"name": "b", "arguments": "not json"}},
        {"function": {"name": "c"}}])
    out = ob.from_ollama_reply(r, "toolu_t")
    assert out["content"][0] == {"type": "text", "text": "ok"}
    tus = out["content"][1:]
    assert [t["id"] for t in tus] == ["toolu_t_0", "toolu_t_1", "toolu_t_2"]
    assert tus[0]["input"] == {"x": 1}
    assert tus[1]["input"] == {"_raw_arguments": "not json"}
    assert tus[2]["input"] == {}


def test_reply_text_only_and_length_stop():
    out = ob.from_ollama_reply(reply(content="", done="length"))
    assert out["content"] == [] and out["stop_reason"] == "max_tokens"
    assert ob.from_ollama_reply(reply(content="done"))["stop_reason"] == "end_turn"


def test_reply_without_counts():
    out = ob.from_ollama_reply({"message": {"content": "x"}, "done_reason": "stop"})
    assert out["usage"] == {"input": 0, "output": 0}


# ------------------------------------------------- client and round trip
def test_client_request_body():
    fake = FakePost([reply(content="hi")])
    c = ob.OllamaClient("http://h:1/", num_ctx=8192, post=fake)
    c("qwen3-vl:2b", "SYS", [{"name": "finish", "input_schema": {"type": "object"}}],
      [{"role": "user", "content": "x"}], max_tokens=777)
    url, body = fake.requests[0]
    assert url == "http://h:1/api/chat"
    assert body["model"] == "qwen3-vl:2b" and body["stream"] is False
    assert body["options"] == {"num_ctx": 8192, "num_predict": 777}, "no overrides set"
    assert body["tools"][0]["function"]["name"] == "finish"
    assert body["messages"][0] == {"role": "system", "content": "SYS"}


def test_judge_truncation_rules():
    # observed cut sizes: 2050 at 4096, 4098 at 8192 (server.log, Ollama 0.40.14)
    assert ob.judge_truncation(2050, 4096) and ob.judge_truncation(4098, 8192)
    assert ob.judge_truncation(2555, 4096) is None
    assert ob.judge_truncation(6016, 8192) is None, "a long prompt that fits is not cut"
    assert ob.judge_truncation(4096, 4096)
    assert ob.judge_truncation(3000, 8192, prev=(3, 3500)), "shrank in a growing conversation"
    assert ob.judge_truncation(3600, 8192, prev=(3, 3500)) is None
    assert ob.judge_truncation(0, 4096) is None, "no count, no judgement"


def test_client_records_truncation_per_call():
    fake = FakePost([reply(prompt=3000), reply(prompt=2050), reply(prompt=2500)])
    c = ob.OllamaClient(post=fake)
    msgs = [{"role": "user", "content": "x"}]
    assert c("m", "", [], msgs)["usage"]["truncated"] is False
    assert not c.truncated
    msgs = msgs + [{"role": "assistant", "content": "y"}, {"role": "user", "content": "z"}]
    assert c("m", "", [], msgs)["usage"]["truncated"] is True   # cut size and shrank
    msgs = msgs + [{"role": "assistant", "content": "y"}, {"role": "user", "content": "z"}]
    assert c("m", "", [], msgs)["usage"]["truncated"] is False
    assert c.truncated, "one truncated call makes the whole run invalid"
    assert [r["truncated"] for r in c.records] == [False, True, False]


def test_shrink_from_dropped_images_is_not_truncation():
    # numbers from the 2026-11-06 seed runs: two crops dropped between turns 2 and 3
    fake = FakePost([reply(prompt=4890), reply(prompt=3003), reply(prompt=2800)])
    c = ob.OllamaClient(post=fake, num_ctx=8192)
    tr = lambda n: [{"type": "tool_result", "tool_use_id": "t", "content": [image()] * n}]
    msgs = [{"role": "user", "content": [image()]}, {"role": "assistant", "content": "y"},
            {"role": "user", "content": tr(2)}]
    c("m", "", [], msgs)
    msgs = [msgs[0], msgs[1], {"role": "user", "content": tr(0)},
            {"role": "assistant", "content": "y"}, {"role": "user", "content": "z"}]
    assert c("m", "", [], msgs)["usage"]["truncated"] is False, "fewer images: shrinking is expected"
    msgs = msgs + [{"role": "assistant", "content": "y"}, {"role": "user", "content": "z"}]
    assert c("m", "", [], msgs)["usage"]["truncated"] is True, "same images and shorter: still flagged"


def test_done_reason_length_is_output_cut_not_truncation():
    c = ob.OllamaClient(post=FakePost([reply(prompt=1276, out=1024, done="length")]))
    c.vlm_text(IMG, "p", "m")
    assert c.records[0]["output_cut"] is True and c.records[0]["truncated"] is False


def test_options_max_tokens_temperature_seed():
    fake = FakePost([reply(), reply()])
    c = ob.OllamaClient(num_ctx=8192, post=fake, max_tokens=1024, temperature=0.2, seed=42)
    c("m", "", [], [{"role": "user", "content": "x"}], max_tokens=1500)
    c.vlm_text(IMG, "p", "m", max_tokens=2000)
    for _, body in fake.requests:  # the client's max_tokens overrides the per-call value
        assert body["options"] == {"num_ctx": 8192, "num_predict": 1024, "temperature": 0.2, "seed": 42}


def test_records_keep_cached_tokens_and_settings():
    r = reply(prompt=1270)
    r["prompt_eval_cached_count"] = 1269
    c = ob.OllamaClient(post=FakePost([r]), max_tokens=1024, temperature=0.2, seed=42)
    c.vlm_text(IMG, "p", "m")
    assert c.records[0]["cached_tokens"] == 1269 and c.records[0]["prompt_tokens"] == 1270
    s = c.settings("qwen3-vl:2b-instruct", "sha256:abc")
    assert s["model_digest"] == "sha256:abc" and s["num_predict"] == 1024
    assert s["temperature"] == 0.2 and s["seed"] == 42 and s["num_ctx"] == 4096
    assert s["prompt_eval_count_includes_cached"] is True


def test_model_digest_lookup():
    tags = {"models": [{"name": "qwen3-vl:2b-instruct", "model": "qwen3-vl:2b-instruct", "digest": "afc3"}]}
    assert ob.model_digest("qwen3-vl:2b-instruct", get=lambda url, t: tags) == "afc3"
    assert ob.model_digest("other", get=lambda url, t: tags) is None

    def down(url, t):
        raise OSError("no server")
    assert ob.model_digest("x", get=down) is None


def test_vlm_text_sends_one_image():
    fake = FakePost([reply(content="{\"facts\": []}")])
    c = ob.OllamaClient(post=fake)
    assert c.vlm_text(IMG, "describe", "m") == "{\"facts\": []}"
    body = fake.requests[0][1]
    assert body["messages"] == [{"role": "user", "content": "describe", "images": [IMG]}]
    assert "tools" not in body and body["options"]["num_predict"] == 2000


def test_risk_agent_runs_through_the_adapter():
    """Full agent loop with the adapter in between: ids, tool results and crop images survive."""
    cv2 = pytest.importorskip("cv2")  # noqa: F841
    np = pytest.importorskip("numpy")
    import risk_agent as ra
    graph = {"image_size": [640, 480], "edges": [],
             "nodes": [{"id": "bottle_1", "label": "bottle", "bbox2d": [100, 100, 160, 260],
                        "observed_now": True}]}
    frame = np.full((480, 640, 3), 90, np.uint8)
    fake = FakePost([
        reply(tool_calls=[{"id": "call_1", "function": {"name": "crop_view", "arguments": {"node_id": "bottle_1"}}}]),
        reply(tool_calls=[{"function": {"name": "write_annotation", "arguments": {
            "kind": "fact", "node": "bottle_1", "attribute": "closure", "value": "closed",
            "status": "asserted", "confidence": 0.7,
            "evidence": [{"call_id": "c1", "supports": "cap on"}]}}}]),
        reply(tool_calls=[{"function": {"name": "finish", "arguments": {"summary": "done"}}}]),
    ])
    res = ra.run_agent(frame, graph, model="qwen3-vl:2b", call_model=ob.OllamaClient(post=fake))
    assert res["stopped_by"] == "finished" and len(res["annotations"]) == 1
    assert res["tokens"] == {"input": 300, "output": 30}
    second = fake.requests[1][1]["messages"]
    tool_msg = [m for m in second if m["role"] == "tool"][0]
    assert tool_msg["tool_name"] == "crop_view" and len(tool_msg["images"]) == 1
    assert second[0]["role"] == "system" and second[1]["images"], "first user turn carries the frame"


def test_webcam_vlm_text_uses_ollama_backend(monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    import webcam_scene_graph as w
    fake = FakePost([reply(content="[]")])
    monkeypatch.setattr(ob, "_http_post", fake)
    monkeypatch.setitem(w.BACKEND, "name", "ollama")
    assert w._vlm_text(np.zeros((8, 8, 3), np.uint8), "p", "qwen3-vl:2b") == "[]"
    assert fake.requests[0][0] == "http://localhost:11434/api/chat"
    assert w.make_call_model() is not None
    assert fake.requests[0][1]["options"]["num_predict"] == 1024, "--max-tokens default"
    monkeypatch.setitem(w.BACKEND, "name", "anthropic")
    assert w.make_call_model() is None
    assert w.backend_settings("claude-x") == {"backend": "anthropic", "model": "claude-x"}


def test_truncated_agent_run_is_marked_invalid(monkeypatch, tmp_path, capsys):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    import argparse
    import webcam_scene_graph as w
    graph = {"image_size": [640, 480], "edges": [],
             "nodes": [{"id": "bottle_1", "label": "bottle", "bbox2d": [100, 100, 160, 260],
                        "observed_now": True}]}
    fake = FakePost([
        reply(prompt=2555, tool_calls=[{"id": "c", "function": {"name": "crop_view",
                                                                "arguments": {"node_id": "bottle_1"}}}]),
        reply(prompt=2050, tool_calls=[{"function": {"name": "finish", "arguments": {"summary": "x"}}}]),
    ])
    monkeypatch.setattr(ob, "_http_post", fake)
    monkeypatch.setattr(w, "CALLS", [])
    for k, v in {"name": "ollama", "digest": "afc3", "num_ctx": 4096}.items():
        monkeypatch.setitem(w.BACKEND, k, v)
    stem = str(tmp_path / "g")
    w.run_agent_step(np.zeros((480, 640, 3), np.uint8), graph, stem,
                     argparse.Namespace(vlm_model="qwen3-vl:2b-instruct", agent_steps=4))
    log = json.load(open(stem + "_agent_log.json"))
    assert log["backend"]["validity"] == "invalid: truncated"
    assert log["backend"]["model_digest"] == "afc3" and log["backend"]["seed"] == 42
    assert [c["truncated"] for c in log["backend"]["calls"]] == [False, True]
    assert w.CALLS[-1]["validity"] == "invalid: truncated" and w.CALLS[-1]["stem"] == stem
    assert "invalid: truncated" in capsys.readouterr().out


def test_run_meta_lists_backend_settings_and_invalid_runs(monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    import webcam_scene_graph as w
    src = tmp_path / "frames"
    src.mkdir()
    for i in range(3):
        cv2.imwrite(str(src / ("f%d.jpg" % i)), np.full((480, 640, 3), 90, np.uint8))
    fake = FakePost([reply(prompt=2050, tool_calls=[{"function": {"name": "finish", "arguments": {}}}])])
    monkeypatch.setattr(ob, "_http_post", fake)
    monkeypatch.setattr(w, "CALLS", [])
    for k, v in {"name": "ollama", "digest": "afc3", "num_ctx": 4096}.items():
        monkeypatch.setitem(w.BACKEND, k, v)
    out = tmp_path / "out"
    args = w.parse_args(["--source", str(src), "--no-display", "--save-every", "3", "--agent-on-save",
                         "--backend", "ollama", "--vlm-model", "qwen3-vl:2b-instruct", "--out", str(out)])
    detect = lambda frame: [("bottle", (100, 100, 160, 260), 0.9)]
    w.run(args, detect)
    meta = json.load(open(out / "run_meta.json"))
    assert meta["backend"]["model"] == "qwen3-vl:2b-instruct" and meta["backend"]["model_digest"] == "afc3"
    assert meta["backend"]["num_predict"] == 1024 and meta["backend"]["temperature"] == 0.2
    assert meta["invalid_model_runs"] == [str(out / "graph_000003")]
    assert meta["model_calls"][0]["calls"][0]["truncated"] is True


OVERFLOW_400 = ('ollama HTTP 400: {"error":"request (9193 tokens) exceeds the available '
                'context size (8192 tokens), try increasing it"}')  # what Ollama 0.40.14 returned


def test_client_turns_http_400_overflow_into_context_overflow():
    def refuse(url, body, timeout):
        raise RuntimeError(OVERFLOW_400)
    c = ob.OllamaClient(post=refuse, num_ctx=8192)
    with pytest.raises(ob.ContextOverflow) as e:
        c("m", "s", [], [{"role": "user", "content": "x"}])
    assert e.value.context_overflow and e.value.prompt_tokens == 9193 and e.value.context_size == 8192
    assert c.overflowed and not c.truncated
    assert c.records[-1]["context_overflow"] and c.records[-1]["prompt_tokens"] == 9193

    def other(url, body, timeout):
        raise RuntimeError("ollama HTTP 500: boom")
    c = ob.OllamaClient(post=other)
    with pytest.raises(RuntimeError) as e:
        c("m", "s", [], [{"role": "user", "content": "x"}])
    assert not isinstance(e.value, ob.ContextOverflow) and c.records == []


def test_context_overflow_keeps_partial_agent_log(monkeypatch, tmp_path, capsys):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    import argparse
    import webcam_scene_graph as w
    graph = {"image_size": [640, 480], "edges": [],
             "nodes": [{"id": "bottle_1", "label": "bottle", "bbox2d": [100, 100, 160, 260],
                        "observed_now": True}]}
    replies = [
        reply(prompt=2000, tool_calls=[{"function": {"name": "crop_view", "arguments": {"node_id": "bottle_1"}}}]),
        reply(prompt=3300, tool_calls=[{"function": {"name": "write_annotation", "arguments": {
            "kind": "fact", "node": "bottle_1", "attribute": "closure", "value": "closed",
            "status": "asserted", "confidence": 0.6,
            "evidence": [{"call_id": "c1", "supports": "cap visible"}]}}}]),
    ]

    def post(url, body, timeout):
        if replies:
            return replies.pop(0)
        raise RuntimeError(OVERFLOW_400)
    monkeypatch.setattr(ob, "_http_post", post)
    monkeypatch.setattr(w, "CALLS", [])
    monkeypatch.setitem(w.BACKEND, "name", "ollama")
    stem = str(tmp_path / "g")
    w.run_agent_step(np.zeros((480, 640, 3), np.uint8), graph, stem,
                     argparse.Namespace(vlm_model="m", agent_steps=6))
    log = json.load(open(stem + "_agent_log.json"))
    assert log["stopped_by"] == "context_overflow" and log["backend"]["validity"] == "invalid: context overflow"
    assert [a["value"] for a in log["annotations"]] == ["closed"], "annotations written so far are kept"
    assert log["backend"]["calls"][-1]["context_overflow"] is True
    assert log["options"]["crop_max_side"] == 512 and log["options"]["max_crops_per_node"] == 2
    assert log["options"]["num_ctx"] == w.BACKEND["num_ctx"] and log["options"]["drop_old_images"] is True
    assert json.load(open(stem + "_agent.json"))["agent_run"]["stopped_by"] == "context_overflow"
    assert w.CALLS[-1]["validity"] == "invalid: context overflow" and w.CALLS[-1]["stem"] == stem
    assert "invalid: context overflow" in capsys.readouterr().out


def test_other_failed_agent_run_is_marked_invalid(monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    import argparse
    import webcam_scene_graph as w

    def broken(url, body, timeout):
        raise RuntimeError("ollama HTTP 500: model crashed")
    monkeypatch.setattr(ob, "_http_post", broken)
    monkeypatch.setattr(w, "CALLS", [])
    monkeypatch.setitem(w.BACKEND, "name", "ollama")
    graph = {"image_size": [640, 480], "edges": [], "nodes": []}
    stem = str(tmp_path / "g")
    assert w.run_agent_step(np.zeros((480, 640, 3), np.uint8), graph, stem,
                            argparse.Namespace(vlm_model="m", agent_steps=2)) == {}
    assert w.CALLS[-1]["validity"].startswith("invalid: failed (ollama HTTP 500")
    assert w.CALLS[-1]["stem"] == stem


def test_cli_agent_options_reach_backend_settings(monkeypatch):
    import webcam_scene_graph as w
    a = w.parse_args(["--backend", "ollama", "--crop-max-side", "256", "--no-drop-old-images",
                      "--max-crops-per-node", "1"])
    assert (a.crop_max_side, a.frame_max_side, a.max_crops_per_node, a.drop_old_images) == (256, 640, 1, False)
    assert (a.budget_remind, a.budget_stop) == (0.70, 0.85)
    monkeypatch.setitem(w.BACKEND, "name", "ollama")
    monkeypatch.setitem(w.BACKEND, "crop_max_side", 256)
    s = w.backend_settings("m")
    assert s["agent_options"]["crop_max_side"] == 256 and s["agent_options"]["num_ctx"] == w.BACKEND["num_ctx"]
    monkeypatch.setitem(w.BACKEND, "name", "anthropic")
    assert w.agent_options() is None, "Anthropic keeps the original agent behaviour"


def test_v_and_c_prompts_use_the_agent_vocabulary():
    import risk_agent as ra
    import webcam_scene_graph as w
    for word in ra.HAZARD_TYPES + ra.PREDICATES + ra.FACT_ATTRIBUTES + list(ra.SEVERITY):
        assert word in w.VLM_PROMPT, word
    w.VLM_PROMPT.format(objs="bottle_1: bottle")  # still a valid format string
    for p in ra.PREDICATES:
        assert w.PREDICATE_PHRASE[p] in w.VERIFY_PROMPT
    w.VERIFY_PROMPT.format(items="a1: x")
    graph = {"nodes": [{"id": "bottle_20", "label": "bottle"}, {"id": "hand_24", "label": "hand"}], "edges": []}
    ann = {"facts": [], "hazards": [
        {"subject": "bottle_20", "predicate": "has", "object": "hand_24"},          # not in vocabulary
        {"subject": "bottle_20", "predicate": "within_reach_of", "object": "bottle_20"},  # self
        {"subject": "bottle_20", "predicate": "within_reach_of", "object": "hand_24"}]}
    out = w.annotation_to_assertions(ann, graph)
    assert [a["text"] for a in out] == ["The bottle is within reach of the hand."]
