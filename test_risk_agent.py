import json
import os
import sys

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prototypes"))
import risk_agent as ra  # noqa: E402

W, H = 640, 480
GRAPH = {
    "image_size": [W, H],
    "nodes": [
        {"id": "knife_1", "label": "knife", "bbox2d": [300, 200, 380, 230], "observed_now": True},
        {"id": "hand_1", "label": "hand", "bbox2d": [420, 180, 520, 280], "observed_now": True},
        {"id": "bottle_1", "label": "bottle", "bbox2d": [100, 100, 160, 260], "observed_now": True},
        {"id": "laptop_1", "label": "laptop", "bbox2d": [60, 270, 260, 400], "observed_now": True},
        {"id": "cup_9", "label": "cup", "bbox2d": [1, 1, 10, 10], "observed_now": False},
    ],
    "edges": [
        {"predicate": "near", "source": "knife_1", "target": "hand_1"},
        {"predicate": "above", "source": "bottle_1", "target": "laptop_1"},
    ],
}
FRAME = np.full((H, W, 3), 90, np.uint8)


def session():
    return ra.AgentSession(FRAME, GRAPH, None, "t", "m")


def fact(**kw):
    d = {"kind": "fact", "node": "bottle_1", "attribute": "closure", "value": "open",
         "status": "asserted", "confidence": 0.7,
         "evidence": [{"call_id": "c1", "supports": "cap is off"}]}
    d.update(kw)
    return d


def test_rule_candidates_use_graph_and_knowledge():
    c = {x["node"]: x for x in ra.rule_candidates(GRAPH)}
    assert "knife_1" in c and "bottle_1" in c
    assert any("hand" in r for r in c["knife_1"]["reasons"])
    assert any("electronics" in r for r in c["bottle_1"]["reasons"])
    assert "cup_9" not in c, "unobserved nodes are not candidates"
    assert "hand_1" not in c and "laptop_1" not in c


def test_geometry_directions():
    g = ra.geometry_report(GRAPH, "knife_1", "hand_1")
    assert g["direction_from_a_to_b"] == "right"
    g2 = ra.geometry_report(GRAPH, "laptop_1", "bottle_1")
    assert g2["direction_from_a_to_b"] in ("up", "up-left")
    assert ra.geometry_report(GRAPH, "knife_1", "nope") is None


def test_unknown_tool_and_bad_node_are_reported_not_raised():
    s = session()
    content, err = s.execute("crop_view", {"node_id": "ghost"})
    assert err and "unknown node_id" in content[0]["text"]
    content, err = s.execute("crop_view", {"node_id": "cup_9"})
    assert err and "not visible" in content[0]["text"]
    content, err = s.execute("no_such_tool", {})
    assert err


def test_crop_returns_image_and_logs(tmp_path):
    s = ra.AgentSession(FRAME, GRAPH, str(tmp_path), "t", "m")
    content, err = s.execute("crop_view", {"node_id": "knife_1"})
    assert not err
    assert content[0]["type"] == "text" and content[1]["type"] == "image"
    assert s.log[0]["node"] == "knife_1" and os.path.exists(s.log[0]["crop_file"])


def test_validator_rules():
    s = session()
    s.execute("crop_view", {"node_id": "bottle_1"})  # c1
    ok, errs, _ = s.validate_and_normalize(fact(), "c2")
    assert ok, errs
    assert not s.validate_and_normalize(fact(value="half_open"), "c2")[0]          # bad enum
    assert not s.validate_and_normalize(fact(node="ghost"), "c2")[0]               # unknown node
    assert not s.validate_and_normalize(fact(confidence=1.5), "c2")[0]             # range
    assert not s.validate_and_normalize(fact(evidence=[]), "c2")[0]                # no evidence
    assert not s.validate_and_normalize(
        fact(evidence=[{"call_id": "c99", "supports": "x"}]), "c2")[0]             # fake call id
    assert not s.validate_and_normalize(
        fact(node="knife_1"), "c2")[0]                                              # crop is of another node
    unk = fact(status="unknown", evidence=[], value="unknown", note="too small")
    assert s.validate_and_normalize(unk, "c2")[0], "unknown needs no evidence"


def test_hazard_needs_subject_crop_and_geometry():
    s = session()
    s.execute("crop_view", {"node_id": "knife_1"})                    # c1
    haz = {"kind": "hazard_triple", "subject": "knife_1.blade", "predicate": "pointing_toward",
           "object": "hand_1", "hazard_type": "mechanical_cut", "harm": "cut to the hand",
           "severity_level": "S2", "status": "asserted", "confidence": 0.8,
           "evidence": [{"call_id": "c1", "supports": "tip faces right"}]}
    ok, errs, _ = s.validate_and_normalize(haz, "c9")
    assert not ok and any("get_geometry" in e for e in errs)
    s.execute("get_geometry", {"node_a": "knife_1", "node_b": "hand_1"})  # c2
    haz["evidence"].append({"call_id": "c2", "supports": "hand is to the right"})
    assert s.validate_and_normalize(haz, "c9")[0]
    haz["severity_level"] = "S9"
    assert not s.validate_and_normalize(haz, "c9")[0]


def scripted_model():
    """A fake model that behaves like an agent: inspects, gets rejected, fixes, finishes."""
    seen = []

    def call(model, system, tools, messages):
        n = len(seen)
        seen.append(messages)
        if n > 0:  # every tool_use of the previous turn must get a tool_result
            prev = messages[-2]["content"]
            res = messages[-1]["content"]
            assert [b["id"] for b in prev if b["type"] == "tool_use"] == \
                   [b["tool_use_id"] for b in res if b["type"] == "tool_result"]

        def tu(i, name, **inp):
            return {"type": "tool_use", "id": "toolu_%d_%d" % (n, i), "name": name, "input": inp}
        ev = lambda *ids: [{"call_id": c, "supports": "seen in crop"} for c in ids]
        turns = [
            [tu(0, "crop_view", node_id="knife_1"), tu(1, "get_geometry", node_a="knife_1", node_b="hand_1"),
             tu(2, "lookup_knowledge", term="knife")],                                             # c1 c2 c3
            [tu(0, "write_annotation", kind="fact", node="knife_1", attribute="tip_exposed", value="true",
                status="asserted", confidence=0.8, evidence=ev("c1")),                              # c4 ok
             tu(1, "write_annotation", kind="hazard_triple", subject="knife_1.blade",
                predicate="pointing_toward", object="hand_1", hazard_type="mechanical_cut",
                harm="cut to the hand", severity_level="S2", status="asserted", confidence=0.8,
                evidence=ev("c1")),                                                                 # c5 rejected
             tu(2, "write_annotation", kind="fact", node="bottle_1", attribute="closure", value="open",
                status="asserted", confidence=0.7, evidence=ev("c99")),                             # c6 rejected
             tu(3, "crop_view", node_id="bottle_1")],                                               # c7
            [tu(0, "write_annotation", kind="hazard_triple", subject="knife_1.blade",
                predicate="pointing_toward", object="hand_1", hazard_type="mechanical_cut",
                harm="cut to the hand", severity_level="S2", status="asserted", confidence=0.8,
                evidence=ev("c1", "c2")),                                                           # c8 ok
             tu(1, "write_annotation", kind="fact", node="bottle_1", attribute="closure", value="open",
                status="asserted", confidence=0.7, evidence=ev("c7")),                              # c9 ok
             tu(2, "write_annotation", kind="fact", node="bottle_1", attribute="closure", value="closed",
                status="asserted", confidence=0.6, evidence=ev("c7"))],                             # c10 conflict
            [tu(0, "finish", summary="done")],                                                      # c11
        ]
        return {"content": turns[n], "usage": {"input": 100, "output": 10}}
    return call


def test_full_agent_run_with_rejection_fix_and_conflict(tmp_path):
    res = ra.run_agent(FRAME, GRAPH, model="fake", call_model=scripted_model(), out_dir=str(tmp_path))
    assert res["stopped_by"] == "finished" and res["steps"] == 4
    assert res["tool_calls"] == 11 and [e["call_id"] for e in res["log"]][-1] == "c11"
    assert len(res["rejected"]) == 2
    assert res["tokens"] == {"input": 400, "output": 40}
    kinds = [(a["kind"], a["status"]) for a in res["annotations"]]
    assert ("hazard_triple", "asserted") in kinds
    closure = [a for a in res["annotations"] if a.get("attribute") == "closure"]
    assert len(closure) == 2 and all(a["status"] == "unknown" for a in closure), "contradiction -> unknown"
    badges = ra.risk_badges(res)
    assert badges == {"knife_1": ["EXPOSED TIP"] + ["RISK mechanical_cut S2"]} or \
           set(badges["knife_1"]) == {"EXPOSED TIP", "RISK mechanical_cut S2"}
    assert "bottle_1" not in badges, "unknown facts are not shown as risks"
    out = ra.annotated_graph(GRAPH, res)
    json.dumps(out)  # must be serialisable
    assert out["rejected_count"] == 2
    prov = [a for a in res["annotations"] if a["kind"] == "hazard_triple"][0]["provenance"]
    assert prov["cited_calls"] == ["c1", "c2"] and prov["prompt_version"] == ra.PROMPT_VERSION


def test_loop_stops_on_max_steps_and_on_plain_text():
    def forever(model, system, tools, messages):
        return {"content": [{"type": "tool_use", "id": "t%d" % len(messages), "name": "lookup_knowledge",
                             "input": {"term": "GHS05"}}], "usage": {"input": 1, "output": 1}}
    r = ra.run_agent(FRAME, GRAPH, call_model=forever, max_steps=3)
    assert r["stopped_by"] == "max_steps" and r["tool_calls"] == 3

    def chatty(model, system, tools, messages):
        return {"content": [{"type": "text", "text": "nothing to report"}], "usage": {}}
    r = ra.run_agent(FRAME, GRAPH, call_model=chatty)
    assert r["stopped_by"] == "no_tool_calls" and r["annotations"] == []


def test_knowledge_lookup_covers_ghs_and_objects_vocabulary_is_in_prompt():
    s = session()
    c, _ = s.execute("lookup_knowledge", {"term": "GHS05"})
    assert "corrosion" in c[0]["text"]
    c, _ = s.execute("lookup_knowledge", {"term": "skull and crossbones"})
    assert "GHS06" in c[0]["text"]
    c, _ = s.execute("lookup_knowledge", {"term": "knife"})
    assert "sharp_tool" in c[0]["text"]
    c, _ = s.execute("lookup_knowledge", {"term": "vocabulary"})
    assert "system prompt" in c[0]["text"] and "mechanical_cut" not in c[0]["text"]
    c, _ = s.execute("lookup_knowledge", {"term": "banana"})
    assert "no entry" in c[0]["text"]
    for word in ra.HAZARD_TYPES + ra.PREDICATES + ra.FACT_ATTRIBUTES + list(ra.SEVERITY) + ["toward_camera"]:
        assert word in ra.SYSTEM_PROMPT, word
    assert "'vocabulary'" not in ra.SYSTEM_PROMPT


def test_repeated_lookup_returns_earlier_answer_with_note():
    s = session()
    first, _ = s.execute("lookup_knowledge", {"term": "bottle"})
    again, err = s.execute("lookup_knowledge", {"term": " Bottle "})
    assert not err
    a, b = json.loads(first[0]["text"]), json.loads(again[0]["text"])
    assert b["repeat_of"] == "c1" and b["call_id"] == "c2" and "write_annotation" in b["note"]
    assert {k: v for k, v in b.items() if k not in ("repeat_of", "call_id", "note")} == \
        {k: v for k, v in a.items() if k != "call_id"}, "same answer as before"
    assert s.log[0]["new_information"] is True
    assert s.log[1]["repeat_of"] == "c1" and s.log[1]["new_information"] is False, "logged, marked"
    assert "repeat lookup" in s.log[1]["output_summary"]


# ------------------------------------------- options for small local models
def tool(name, i, **inp):
    return {"type": "tool_use", "id": "t%d" % i, "name": name, "input": inp}


def test_crop_limit_per_node_tells_agent_to_annotate():
    s = ra.AgentSession(FRAME, GRAPH, None, "t", "m", max_crops_per_node=2)
    assert not s.execute("crop_view", {"node_id": "bottle_1"})[1]
    assert not s.execute("crop_view", {"node_id": "bottle_1"})[1]
    content, err = s.execute("crop_view", {"node_id": "bottle_1"})
    assert err and "crop limit reached" in content[0]["text"] and "c1, c2" in content[0]["text"]
    assert "unknown" in content[0]["text"]
    assert not s.execute("crop_view", {"node_id": "knife_1"})[1], "limit is per node"
    assert s.crops_of == {"bottle_1": ["c1", "c2"], "knife_1": ["c4"]}


def test_crop_and_frame_size_options():
    s = ra.AgentSession(FRAME, GRAPH, None, "t", "m", crop_max_side=256)
    s.execute("crop_view", {"node_id": "laptop_1"})  # 300x195 native -> at most 256
    assert max(s.log[0]["crop_sent_px"]) == 256
    seen = []

    def once(model, system, tools, messages):
        seen.append(messages[0]["content"][0]["source"]["data"])
        return {"content": [tool("finish", 1, summary="x")], "usage": {}}
    big = np.full((960, 1280, 3), 90, np.uint8)
    r = ra.run_agent(big, GRAPH, call_model=once, options={"frame_max_side": 640})
    assert r["options"]["frame_sent_px"] == [640, 480]
    assert cv2.imdecode(np.frombuffer(__import__("base64").b64decode(seen[0]), np.uint8), 1).shape[:2] == (480, 640)


def test_old_crop_images_are_dropped_from_history_but_frame_is_kept():
    snapshots = []

    def model(model, system, tools, messages):
        snapshots.append(json.loads(json.dumps(messages)))
        n = len(snapshots)
        if n == 1:
            return {"content": [tool("crop_view", 1, node_id="bottle_1")], "usage": {}}
        if n == 2:
            return {"content": [tool("crop_view", 2, node_id="knife_1")], "usage": {}}
        return {"content": [tool("finish", 3, summary="x")], "usage": {}}
    r = ra.run_agent(FRAME, GRAPH, call_model=model, options={"drop_old_images": True})
    assert r["stopped_by"] == "finished" and r["images_dropped"] == 1
    imgs = lambda blocks: [b for b in blocks if b["type"] == "image"]
    second, third = snapshots[1], snapshots[2]
    assert len(imgs(second[2]["content"][0]["content"])) == 1, "a new crop is seen once"
    old = third[2]["content"][0]["content"]
    assert imgs(old) == [] and "crop image of bottle_1 removed from history, call_id c1" in old[1]["text"]
    assert len(imgs(third[4]["content"][0]["content"])) == 1, "the newest crop is kept"
    assert imgs(third[0]["content"]), "the camera frame stays"
    r = ra.run_agent(FRAME, GRAPH, call_model=lambda *a: {"content": [tool("finish", 1)], "usage": {}})
    assert r["images_dropped"] == 0 and r["options"]["drop_old_images"] is False, "off by default"


def test_context_budget_reminder_and_stop():
    seen = []

    def model(model, system, tools, messages):
        seen.append(messages[-1]["content"])
        prompt = {1: 1000, 2: 5800, 3: 7000}[len(seen)]  # num_ctx 8192: 70% = 5734, 85% = 6963
        return {"content": [tool("lookup_knowledge", len(seen), term="GHS05")],
                "usage": {"input": prompt, "output": 50}}
    r = ra.run_agent(FRAME, GRAPH, call_model=model, options={"num_ctx": 8192})
    assert r["stopped_by"] == "context_budget" and r["steps"] == 3 and r["tool_calls"] == 3
    assert "budget" not in r["turns"][0]
    assert r["turns"][1]["budget"] == "reminder" and r["turns"][2]["budget"] == "stop"
    reminder = seen[2][-1]
    assert reminder["type"] == "text" and "5850 of 8192" in reminder["text"] and "finish" in reminder["text"]
    big = lambda *a: {"content": [tool("lookup_knowledge", 1, term="GHS05")],
                      "usage": {"input": 8000, "output": 5}}
    r = ra.run_agent(FRAME, GRAPH, call_model=big, max_steps=2)
    assert r["stopped_by"] == "max_steps", "no budget without num_ctx (Anthropic default)"


def test_context_overflow_keeps_partial_run():
    class Overflow(RuntimeError):
        context_overflow = True
    calls = []

    def model(model, system, tools, messages):
        calls.append(1)
        if len(calls) == 1:
            return {"content": [tool("crop_view", 1, node_id="bottle_1")], "usage": {"input": 10, "output": 1}}
        if len(calls) == 2:
            return {"content": [tool("write_annotation", 2, kind="fact", node="bottle_1",
                                     attribute="closure", value="unknown", status="unknown",
                                     confidence=0.3)], "usage": {"input": 20, "output": 1}}
        raise Overflow("request (9193 tokens) exceeds the available context size (8192 tokens)")
    r = ra.run_agent(FRAME, GRAPH, call_model=model)
    assert r["stopped_by"] == "context_overflow" and "9193" in r["error"]
    assert len(r["annotations"]) == 1 and r["tool_calls"] == 2 and r["steps"] == 3
    assert r["turns"][-1] == {"step": 3, "error": r["error"]}
    assert ra.annotated_graph(GRAPH, r)["agent_run"]["stopped_by"] == "context_overflow"

    def other(*a):
        raise RuntimeError("ollama HTTP 500: boom")
    with pytest.raises(RuntimeError):
        ra.run_agent(FRAME, GRAPH, call_model=other)


def test_no_progress_reminder_after_4_turns_and_stop_after_8():
    seen = []

    def model(model, system, tools, messages):
        seen.append(messages[-1]["content"])
        n = len(seen)
        if n == 1:
            return {"content": [tool("crop_view", 1, node_id="bottle_1")], "usage": {}}
        if n == 2:
            return {"content": [tool("get_geometry", 2, node_a="bottle_1", node_b="laptop_1")], "usage": {}}
        return {"content": [tool("lookup_knowledge", n, term="GHS0%d" % (n % 9 + 1))], "usage": {}}
    r = ra.run_agent(FRAME, GRAPH, call_model=model, max_steps=12,
                     options={"progress_remind": 4, "progress_stop": 8})
    assert r["stopped_by"] == "no_progress" and r["steps"] == 8 and r["tool_calls"] == 8
    assert [t.get("progress") for t in r["turns"]] == [None, None, None, "reminder", None, None, None, "stop"]
    reminder = seen[4][-1]
    assert reminder["type"] == "text" and "after 4 turns" in reminder["text"]
    assert "bottle_1 (crop c1)" in reminder["text"] and "geometry bottle_1/laptop_1 (c2)" in reminder["text"]
    assert "write_annotation" in reminder["text"] and "unknown" in reminder["text"] and "finish" in reminder["text"]
    assert sum(1 for c in seen for b in c if b.get("type") == "text"
               and "No annotation written" in b["text"]) == 1, "reminder only once"


def test_no_progress_is_off_when_a_write_was_made_or_by_default():
    def model(model, system, tools, messages):
        n = len([m for m in messages if m["role"] == "assistant"]) + 1
        if n == 1:
            return {"content": [tool("write_annotation", 1, kind="fact", node="bottle_1",
                                     attribute="closure", value="open", status="asserted",
                                     confidence=0.5)], "usage": {}}  # rejected: no evidence
        return {"content": [tool("lookup_knowledge", n, term="GHS0%d" % (n % 9 + 1))], "usage": {}}
    r = ra.run_agent(FRAME, GRAPH, call_model=model, max_steps=10,
                     options={"progress_remind": 4, "progress_stop": 8})
    assert r["stopped_by"] == "max_steps" and len(r["rejected"]) == 1, "a rejected write counts as progress"
    assert not any(t.get("progress") for t in r["turns"])
    r = ra.run_agent(FRAME, GRAPH, max_steps=10,
                     call_model=lambda *a: {"content": [tool("lookup_knowledge", 1, term="GHS05")], "usage": {}})
    assert r["stopped_by"] == "max_steps" and r["repeated_lookups"] == 9, "off without options (Anthropic)"
