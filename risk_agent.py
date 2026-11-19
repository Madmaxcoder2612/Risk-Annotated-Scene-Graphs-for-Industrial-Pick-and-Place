#!/usr/bin/env python3
"""
Agentic risk annotation for the webcam scene graph prototype (2D).

An LLM/VLM agent receives the frame and the scene graph, decides which objects
to inspect, calls tools, and writes risk annotations. A validator rejects
annotations without valid evidence, and the agent can fix and resubmit them.

Tools the agent can call
  crop_view(node_id)           zoomed crop of one object (the agent looks at it)
  get_geometry(node_a, node_b) distance, direction and graph relations (2D only)
  lookup_knowledge(term)       hazard knowledge: object classes, GHS pictograms,
                               allowed vocabulary, severity scale
  write_annotation(...)        write a fact or a hazard triple with evidence
  finish(summary)              end the run

Everything is logged (tool calls, inputs, outputs, latency, tokens, crops).

Limits: a webcam has no depth, so directions are image-space only. The agent
should say "unknown" when a crop does not show the fact. Hazard labels are
advisory research output, not a safety function.

Used by webcam_scene_graph.py (key 'a'), or from code:
    result = run_agent(frame_bgr, graph, model="claude-sonnet-5-5")
"""
import base64
import json
import math
import os
import time

PROMPT_VERSION = "risk-agent-0.2"  # 0.2: vocabulary and severity scale in the system prompt
RISK_SCHEMA = "risk-0.1-2d"
SEVERITY_SCALE = "thesis-S-0.1"

# ------------------------------------------------------------- vocabularies
HAZARD_TYPES = ["mechanical_cut", "mechanical_puncture", "mechanical_impact",
                "chemical_corrosive", "chemical_toxic", "chemical_flammable",
                "chemical_irritant", "spill", "equipment_damage"]
PREDICATES = ["pointing_toward", "open_toward", "positioned_above", "carried_over",
              "within_reach_of", "in_contact_with"]
SEVERITY = {
    "S0": "negligible: no injury or damage",
    "S1": "minor: first aid or trivial damage",
    "S2": "serious but reversible: medical treatment, repairable damage",
    "S3": "severe or irreversible injury or major damage",
    "S4": "life-threatening",
}
FACT_ATTRIBUTES = ["closure", "label_text", "pictograms", "part", "tip_direction",
                   "tip_exposed", "contents"]
ENUMS = {
    "closure": {"closed", "open", "missing", "unknown"},
    "tip_direction": {"left", "right", "up", "down", "toward_camera",
                      "away_from_camera", "unknown"},
    "tip_exposed": {"true", "false", "unknown"},
}
STATUSES = {"asserted", "unknown"}

GHS = {
    "GHS01": ("exploding bomb", "explosives, self-reactive substances"),
    "GHS02": ("flame", "flammable gases, liquids, solids"),
    "GHS03": ("flame over circle", "oxidizers"),
    "GHS04": ("gas cylinder", "gases under pressure"),
    "GHS05": ("corrosion", "skin burns, eye damage, corrosive to metals"),
    "GHS06": ("skull and crossbones", "acute toxicity (can be fatal)"),
    "GHS07": ("exclamation mark", "irritant, harmful, skin sensitiser"),
    "GHS08": ("health hazard", "serious long-term health hazards"),
    "GHS09": ("environment", "hazardous to the aquatic environment"),
}

# label -> knowledge entry. Aliases map several detector labels to one entry.
KNOWLEDGE = {
    "sharp_tool": {
        "tags": ["sharp", "pointed"],
        "hazards": ["mechanical_cut", "mechanical_puncture"],
        "inspect": ["Is a blade or sharp tip visible and exposed (not covered)?",
                    "Which way does the tip point: left, right, up, down, toward or away "
                    "from the camera?",
                    "Is the handle or the blade end closer to the nearest hand?"],
        "notes": "The risk depends on the part: blade or tip toward a person is a hazard, "
                 "handle toward a person is normally not.",
    },
    "pointed_object": {
        "tags": ["pointed"],
        "hazards": ["mechanical_puncture"],
        "inspect": ["Is the tip exposed (no cap)?", "Which way does the tip point?"],
        "notes": "Pens and pencils are low severity; needles and screwdrivers can be higher.",
    },
    "container": {
        "tags": ["container"],
        "hazards": ["spill", "chemical_corrosive", "chemical_toxic", "chemical_flammable"],
        "inspect": ["Is it closed, open, or is the cap or cork missing?",
                    "Does it seem to contain a liquid?",
                    "Is there a label with a hazard pictogram or text?",
                    "Is it above or next to electronics or a hand?"],
        "notes": "An open container can spill. Severity depends on the contents: check the "
                 "label for GHS pictograms. Without a readable label, contents are unknown.",
    },
    "electronics": {
        "tags": ["electronics"],
        "hazards": [],
        "inspect": [],
        "notes": "Not a hazard source itself. Liquids above or next to electronics can cause "
                 "equipment damage (hazard_type equipment_damage, usually S0 to S2).",
    },
    "exposure_target": {
        "tags": ["exposure_target"],
        "hazards": [],
        "inspect": [],
        "notes": "A hand or person is where harm can occur (hazard zone).",
    },
}
ALIASES = {
    "knife": "sharp_tool", "scissors": "sharp_tool", "cutter": "sharp_tool",
    "screwdriver": "sharp_tool", "blade": "sharp_tool",
    "pen": "pointed_object", "pencil": "pointed_object", "needle": "pointed_object",
    "bottle": "container", "vial": "container", "cup": "container",
    "glass": "container", "bowl": "container", "can": "container",
    "laptop": "electronics", "keyboard": "electronics", "phone": "electronics",
    "mouse": "electronics", "monitor": "electronics",
    "hand": "exposure_target", "person": "exposure_target", "finger": "exposure_target",
}


def kb_entry(label, kb=None):
    kb = kb or KNOWLEDGE
    key = ALIASES.get(str(label).lower(), str(label).lower())
    return key, kb.get(key)


# ----------------------------------------------------------- rule candidates
def rule_candidates(graph, kb=None):
    """Deterministic pre-selection from graph + knowledge. These are hints for
    the agent, not annotations: nothing here is asserted as a risk."""
    nodes = {n["id"]: n for n in graph["nodes"] if n.get("observed_now", True)}
    entry_of = {i: kb_entry(n["label"], kb) for i, n in nodes.items()}
    out = {}

    def add(node_id, reason):
        key, entry = entry_of[node_id]
        c = out.setdefault(node_id, {"node": node_id, "label": nodes[node_id]["label"],
                                      "class": key, "reasons": [],
                                      "questions": list((entry or {}).get("inspect", []))})
        c["reasons"].append(reason)

    for i, (key, entry) in entry_of.items():
        if entry and entry["hazards"]:
            add(i, "object class '%s' can be hazardous" % key)
    for e in graph["edges"]:
        a, b = e["source"], e["target"]
        if a not in nodes or b not in nodes:
            continue
        ka, kb_ = entry_of[a][0], entry_of[b][0]
        for x, kx, y, ky in ((a, ka, b, kb_), (b, kb_, a, ka)):
            if ky == "exposure_target" and kx in ("sharp_tool", "pointed_object", "container"):
                add(x, "%s is %s the hand/person %s" % (x, e["predicate"], y))
            if ky == "electronics" and kx == "container":
                add(x, "container %s is %s electronics %s" % (x, e["predicate"], y))
    return sorted(out.values(), key=lambda c: -len(c["reasons"]))


# ----------------------------------------------------------------- geometry
def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _gap(a, b):
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def _compass(dx, dy):
    """Image coordinates (y points down) -> 8-way direction word."""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "same_position"
    ang = math.degrees(math.atan2(-dy, dx)) % 360  # 0=right, 90=up
    words = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
    return words[int(((ang + 22.5) % 360) // 45)]


def geometry_report(graph, node_a, node_b):
    boxes = {n["id"]: n["bbox2d"] for n in graph["nodes"]}
    if node_a not in boxes or node_b not in boxes:
        return None
    a, b = boxes[node_a], boxes[node_b]
    w, h = graph.get("image_size") or (max(a[2], b[2]), max(a[3], b[3]))
    diag = math.hypot(w, h)
    (ax, ay), (bx, by) = _center(a), _center(b)
    directed = ["%s %s %s" % (e["source"], e["predicate"], e["target"]) for e in graph["edges"]
                if {e["source"], e["target"]} == {node_a, node_b}]
    return {
        "a": node_a, "b": node_b,
        "gap_px": round(_gap(a, b), 1),
        "gap_relative_to_image_diagonal": round(_gap(a, b) / diag, 3),
        "direction_from_a_to_b": _compass(bx - ax, by - ay),
        "angle_deg_0_is_right_90_is_up": round(math.degrees(math.atan2(-(by - ay), bx - ax)) % 360, 1),
        "graph_relations": directed,
        "size_ratio_a_to_b": round(((a[2] - a[0]) * (a[3] - a[1])) /
                                   max(1.0, (b[2] - b[0]) * (b[3] - b[1])), 2),
        "note": "2D image-space only; there is no depth, so 'near' can mean 'in line of sight'.",
    }


# -------------------------------------------------------------- tool schemas
TOOLS = [
    {"name": "crop_view",
     "description": "Get a zoomed crop of one detected object so you can look at details "
                    "(cap, tip, label, pictogram). Returns the image and a call_id you can cite "
                    "as evidence.",
     "input_schema": {"type": "object", "properties": {
         "node_id": {"type": "string"},
         "padding": {"type": "number", "description": "extra margin around the box, 0 to 1 (default 0.25)"}},
         "required": ["node_id"]}},
    {"name": "get_geometry",
     "description": "Distance, direction and graph relations between two nodes (2D image space). "
                    "Returns a call_id you can cite as evidence.",
     "input_schema": {"type": "object", "properties": {
         "node_a": {"type": "string"}, "node_b": {"type": "string"}},
         "required": ["node_a", "node_b"]}},
    {"name": "lookup_knowledge",
     "description": "Look up hazard knowledge for an object label (knife, bottle) or a GHS "
                    "pictogram code or name (GHS05, corrosion). The allowed vocabulary and the "
                    "severity scale are in the system prompt, not here.",
     "input_schema": {"type": "object", "properties": {"term": {"type": "string"}},
                      "required": ["term"]}},
    {"name": "write_annotation",
     "description": "Write one annotation. kind='fact' needs node, attribute, value. "
                    "kind='hazard_triple' needs subject, predicate, object, hazard_type, harm, "
                    "severity_level. status='asserted' needs evidence: a list of "
                    "{call_id, supports} citing earlier crop_view / get_geometry calls. A hazard "
                    "triple needs a crop_view of the subject and, if the object is a graph node, a "
                    "get_geometry call on both. If you cannot verify, use status='unknown' and "
                    "explain in note. The tool may reject the annotation; fix it and resubmit.",
     "input_schema": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["fact", "hazard_triple"]},
         "node": {"type": "string"}, "attribute": {"type": "string"}, "value": {"type": "string"},
         "subject": {"type": "string", "description": "node id, optionally with part: knife_1.blade"},
         "predicate": {"type": "string"}, "object": {"type": "string"},
         "hazard_type": {"type": "string"}, "harm": {"type": "string"},
         "severity_level": {"type": "string"},
         "status": {"type": "string", "enum": ["asserted", "unknown"]},
         "confidence": {"type": "number"}, "note": {"type": "string"},
         "evidence": {"type": "array", "items": {"type": "object", "properties": {
             "call_id": {"type": "string"}, "supports": {"type": "string"}},
             "required": ["call_id", "supports"]}}},
         "required": ["kind", "status", "confidence"]}},
    {"name": "finish",
     "description": "End the run when all candidates are handled.",
     "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}},
                      "required": ["summary"]}},
]


def vocabulary_text():
    """Allowed vocabulary and severity scale as prompt text (agent, v and c use the same)."""
    sev = "\n".join("- %s %s" % (k, v) for k, v in SEVERITY.items())
    enums = "; ".join("%s = %s" % (k, " | ".join(sorted(v))) for k, v in ENUMS.items())
    return ("Allowed vocabulary (use exactly these values; 'unknown' and 'other:<text>' are "
            "always allowed):\n"
            "- hazard_type: %s\n"
            "- predicate (hazard relation, 2D image space): %s\n"
            "- fact attribute: %s\n"
            "- enumerated values: %s\n"
            "Severity scale %s (a thesis proposal, not an ISO scale):\n%s"
            % (", ".join(HAZARD_TYPES), ", ".join(PREDICATES), ", ".join(FACT_ATTRIBUTES),
               enums, SEVERITY_SCALE, sev))


SYSTEM_PROMPT = """You are a risk annotation agent for a robot work cell study. You get a camera
frame and a scene graph of detected objects. Find risk-relevant details (for example an open
bottle, an exposed pointed tip, a hazard pictogram) and the relations that turn them into
hazards (for example a tip pointing toward a hand, an open bottle above a laptop).

Work like this: pick the objects worth inspecting (start from the suggested candidates, but you
may skip or add), call crop_view to look closely, use get_geometry for relations between
objects, use lookup_knowledge for object hazard knowledge and GHS pictograms, then write
annotations with write_annotation, citing the call_ids that support them, and finally call
finish.

Rules:
- Only state what the images or tool outputs support. If a crop is too small, blurry or
  ambiguous, write status='unknown' with a note. A missing annotation is better than a wrong one.
- This is a webcam without depth: directions are image-space only. Lower your confidence for
  anything that depends on depth.
- Use only the vocabulary and the S0 to S4 severity scale below.
- Do not invent objects or text. Quote label text only if you can read it.
- Be economical: at most one or two tool calls per candidate where possible.
- If write_annotation rejects something, read the errors, gather the missing evidence, fix it.

""" + vocabulary_text()


# ------------------------------------------------------------- tool session
def _jpeg_b64(img, max_side=1024, min_side=0):
    import cv2
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
    elif min_side and max(h, w) < min_side:
        scale = min(4.0, min_side / float(max(h, w)))
    if scale != 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("could not encode image")
    return base64.standard_b64encode(buf.tobytes()).decode(), img


def _image_block(b64):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


class AgentSession:
    """Executes tool calls, keeps the log, validates and stores annotations."""

    def __init__(self, frame, graph, out_dir=None, run_id="run", model="", kb=None,
                 crop_max_side=1024, max_crops_per_node=None):
        self.frame, self.graph, self.out_dir = frame, graph, out_dir
        self.run_id, self.model, self.kb = run_id, model, kb
        self.crop_max_side, self.max_crops_per_node = crop_max_side, max_crops_per_node
        self.log, self.log_by_id = [], {}
        self.annotations, self.rejected = [], []
        self.finished, self.summary = False, ""
        self._n_calls = 0
        self.nodes = {n["id"]: n for n in graph["nodes"]}
        self.crops_of = {}  # node id -> call_ids of successful crops
        self.lookups = {}   # lower-case term -> (first call_id, answer)

    # ---- dispatch
    def execute(self, name, args):
        """Returns (content_blocks, is_error)."""
        self._n_calls += 1
        call_id = "c%d" % self._n_calls
        t0 = time.time()
        entry = {"call_id": call_id, "tool": name, "input": args, "ts": t0}
        self.log.append(entry)
        self.log_by_id[call_id] = entry
        try:
            handler = getattr(self, "_tool_" + name, None)
            if handler is None:
                raise ValueError("unknown tool '%s'" % name)
            content, is_error, summary = handler(call_id, args or {})
        except Exception as exc:  # report to the agent instead of crashing the run
            content = [{"type": "text", "text": "error: %s" % exc}]
            is_error, summary = True, "error: %s" % exc
        entry["is_error"] = is_error
        entry["output_summary"] = summary
        entry["latency_ms"] = round((time.time() - t0) * 1000.0, 1)
        return content, is_error

    # ---- tools
    def _tool_crop_view(self, call_id, a):
        node = self.nodes.get(a.get("node_id"))
        if node is None:
            raise ValueError("unknown node_id '%s'. Known: %s" % (a.get("node_id"), sorted(self.nodes)))
        if not node.get("observed_now", True):
            raise ValueError("%s is not visible in the current frame" % node["id"])
        done = self.crops_of.get(node["id"], [])
        if self.max_crops_per_node is not None and len(done) >= self.max_crops_per_node:
            raise ValueError(
                "crop limit reached: %s already has %d crop(s) (%s). Do not crop it again. Write its "
                "annotation now with write_annotation citing these call_ids, or with status 'unknown' "
                "if the crops do not show it." % (node["id"], len(done), ", ".join(done)))
        pad = float(a.get("padding", 0.25))
        pad = min(max(pad, 0.0), 1.0)
        x1, y1, x2, y2 = node["bbox2d"]
        h, w = self.frame.shape[:2]
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        xa, ya = int(max(0, x1 - px)), int(max(0, y1 - py))
        xb, yb = int(min(w, x2 + px)), int(min(h, y2 + py))
        crop = self.frame[ya:yb, xa:xb]
        if crop.size == 0:
            raise ValueError("empty crop for %s" % node["id"])
        b64, resized = _jpeg_b64(crop, max_side=self.crop_max_side,
                                 min_side=min(512, self.crop_max_side))
        fname = None
        if self.out_dir:
            import cv2
            os.makedirs(self.out_dir, exist_ok=True)
            fname = os.path.join(self.out_dir, "%s_%s_%s.jpg" % (self.run_id, call_id, node["id"]))
            cv2.imwrite(fname, resized)
        self.log_by_id[call_id].update({"node": node["id"], "crop_file": fname,
                                        "crop_box": [xa, ya, xb, yb],
                                        "crop_native_px": [xb - xa, yb - ya],
                                        "crop_sent_px": [resized.shape[1], resized.shape[0]]})
        self.crops_of.setdefault(node["id"], []).append(call_id)
        text = json.dumps({"call_id": call_id, "node": node["id"], "label": node["label"],
                           "crop_box": [xa, ya, xb, yb],
                           "native_size_px": [xb - xa, yb - ya],
                           "note": "small native size means fine details may be unreadable"})
        return ([{"type": "text", "text": text}, _image_block(b64)], False,
                "crop of %s (%dx%d px native)" % (node["id"], xb - xa, yb - ya))

    def _tool_get_geometry(self, call_id, a):
        rep = geometry_report(self.graph, a.get("node_a"), a.get("node_b"))
        if rep is None:
            raise ValueError("unknown node in (%s, %s). Known: %s"
                             % (a.get("node_a"), a.get("node_b"), sorted(self.nodes)))
        rep["call_id"] = call_id
        self.log_by_id[call_id].update({"nodes": [a["node_a"], a["node_b"]]})
        return [{"type": "text", "text": json.dumps(rep)}], False, \
            "%s -> %s: %s" % (a["node_a"], a["node_b"], rep["direction_from_a_to_b"])

    def _tool_lookup_knowledge(self, call_id, a):
        term = str(a.get("term", "")).strip()
        low = term.lower()
        log_entry = self.log_by_id[call_id]
        if low in self.lookups:  # same term again: same answer, no new information
            first, res = self.lookups[low]
            res = dict(res, call_id=call_id, repeat_of=first, note=(
                "You already looked up '%s' in %s; this is the same answer. Do not look it up "
                "again. Write your annotation now with write_annotation (status 'unknown' if "
                "unsure), then call finish." % (term, first)))
            log_entry.update(repeat_of=first, new_information=False)
            return [{"type": "text", "text": json.dumps(res)}], False, "repeat lookup %s (as %s)" % (term, first)
        if low in ("vocabulary", "severity"):
            res = {"term": term, "result": "not here: the allowed vocabulary and the severity "
                                           "scale are in the system prompt"}
        elif term.upper() in GHS:
            n, m = GHS[term.upper()]
            res = {"code": term.upper(), "name": n, "meaning": m}
        else:
            hit = [(c, GHS[c]) for c in GHS if low == GHS[c][0]]
            if hit:
                res = {"code": hit[0][0], "name": hit[0][1][0], "meaning": hit[0][1][1]}
            else:
                key, entry = kb_entry(term, self.kb)
                res = ({"term": term, "class": key, **entry} if entry
                       else {"term": term, "result": "no entry; treat as unknown"})
        self.lookups[low] = (call_id, dict(res))
        log_entry["new_information"] = True
        res["call_id"] = call_id
        return [{"type": "text", "text": json.dumps(res)}], False, "lookup %s" % term

    def _tool_write_annotation(self, call_id, a):
        ok, errors, ann = self.validate_and_normalize(a, call_id)
        if not ok:
            self.rejected.append({"call_id": call_id, "input": a, "errors": errors})
            return ([{"type": "text", "text": json.dumps({"accepted": False, "errors": errors})}],
                    True, "rejected: %s" % "; ".join(errors))
        notes = self._store(ann)
        return ([{"type": "text", "text": json.dumps(
            {"accepted": True, "annotation_id": ann["annotation_id"], "notes": notes})}],
                False, "accepted %s" % ann["annotation_id"])

    def _tool_finish(self, call_id, a):
        self.finished, self.summary = True, str(a.get("summary", ""))
        return [{"type": "text", "text": "ok"}], False, "finish"

    # ---- validation
    def _base(self, ref):
        return str(ref).split(".")[0]

    def _check_evidence(self, ann, errors):
        ev = ann.get("evidence") or []
        if not ev:
            errors.append("status 'asserted' needs evidence: a list of {call_id, supports}")
            return
        crops, geoms = [], []
        for item in ev:
            cid, sup = str(item.get("call_id", "")), str(item.get("supports", "")).strip()
            entry = self.log_by_id.get(cid)
            if entry is None or entry["tool"] not in ("crop_view", "get_geometry") \
                    or entry.get("is_error"):
                errors.append("evidence call_id '%s' is not a successful crop_view/get_geometry call" % cid)
                continue
            if not sup:
                errors.append("evidence %s needs a non-empty 'supports' text" % cid)
            (crops if entry["tool"] == "crop_view" else geoms).append(entry)
        if ann["kind"] == "fact":
            need = ann["node"]
            if not any(c.get("node") == need for c in crops):
                errors.append("a fact about %s needs evidence from crop_view of %s" % (need, need))
        else:
            subj = self._base(ann["triple"]["subject"])
            obj = self._base(ann["triple"]["object"])
            if not any(c.get("node") == subj for c in crops):
                errors.append("a hazard triple needs evidence from crop_view of the subject %s" % subj)
            if obj in self.nodes and not any(set(g.get("nodes", [])) == {subj, obj} for g in geoms):
                errors.append("the object %s is a graph node: cite a get_geometry call on (%s, %s)"
                              % (obj, subj, obj))

    @staticmethod
    def _allowed(value, allowed):
        v = str(value)
        return v == "unknown" or v.startswith("other:") or v in allowed

    def validate_and_normalize(self, a, call_id):
        errors = []
        kind, status = a.get("kind"), a.get("status")
        if kind not in ("fact", "hazard_triple"):
            errors.append("kind must be 'fact' or 'hazard_triple'")
        if status not in STATUSES:
            errors.append("status must be 'asserted' or 'unknown'")
        conf = a.get("confidence")
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not 0.0 <= conf <= 1.0:
            errors.append("confidence must be a number between 0 and 1")
        if errors:
            return False, errors, None
        ann = {"kind": kind, "status": status, "confidence": {"value": float(conf), "basis": "vlm_verbalised"},
               "evidence": [{"call_id": str(e.get("call_id")), "supports": str(e.get("supports", ""))}
                            for e in (a.get("evidence") or []) if isinstance(e, dict)],
               "note": a.get("note", "")}
        if kind == "fact":
            node, attr, val = a.get("node"), a.get("attribute"), a.get("value")
            if node not in self.nodes:
                errors.append("node '%s' is not in the graph. Known: %s" % (node, sorted(self.nodes)))
            if not self._allowed(attr, FACT_ATTRIBUTES):
                errors.append("attribute '%s' not allowed. Use one of %s or other:<text>" % (attr, FACT_ATTRIBUTES))
            if val in (None, ""):
                errors.append("value is required (use 'unknown' if you cannot tell)")
            elif attr in ENUMS and not self._allowed(val, ENUMS[attr]):
                errors.append("value '%s' not allowed for %s. Use one of %s" % (val, attr, sorted(ENUMS[attr])))
            ann.update({"node": node, "attribute": attr, "value": val})
        else:
            subj, pred, obj = a.get("subject"), a.get("predicate"), a.get("object")
            if not subj or self._base(subj) not in self.nodes:
                errors.append("subject '%s' must be a graph node id (optionally node.part)" % subj)
            if not obj:
                errors.append("object is required (a graph node id or a zone name)")
            if not self._allowed(pred, PREDICATES):
                errors.append("predicate '%s' not allowed. Use one of %s" % (pred, PREDICATES))
            if not self._allowed(a.get("hazard_type"), HAZARD_TYPES):
                errors.append("hazard_type '%s' not allowed. Use one of %s" % (a.get("hazard_type"), HAZARD_TYPES))
            sev = a.get("severity_level")
            if not (sev == "unknown" or sev in SEVERITY):
                errors.append("severity_level must be one of %s or unknown" % sorted(SEVERITY))
            if status == "asserted" and not str(a.get("harm", "")).strip():
                errors.append("harm (a short description of the possible injury or damage) is required")
            ann.update({"triple": {"subject": subj, "predicate": pred, "object": obj},
                        "hazard": {"type": a.get("hazard_type"), "harm": a.get("harm", ""),
                                   "severity": {"level": sev, "scale": SEVERITY_SCALE},
                                   "probability": {"class": "unknown"}}})
        if status == "asserted" and not errors:
            self._check_evidence(ann, errors)
        if errors:
            return False, errors, None
        ann["provenance"] = {"agent_run": self.run_id, "model": self.model,
                             "prompt_version": PROMPT_VERSION,
                             "cited_calls": [e["call_id"] for e in ann["evidence"]],
                             "written_in_call": call_id}
        return True, [], ann

    def _store(self, ann):
        """Add the annotation, handling duplicates and contradictions."""
        notes = []
        if ann["kind"] == "fact":
            key = lambda x: (x["kind"], x.get("node"), x.get("attribute"))
        else:
            key = lambda x: (x["kind"], x.get("triple", {}).get("subject"),
                             x.get("triple", {}).get("predicate"), x.get("triple", {}).get("object"))
        for old in list(self.annotations):
            if key(old) != key(ann):
                continue
            if ann["kind"] == "fact" and old["status"] == "asserted" and ann["status"] == "asserted" \
                    and old["value"] != ann["value"]:
                # contradictory observations -> both become unknown
                old["status"], old["note"] = "unknown", "conflict with a later observation: %r vs %r" % (old["value"], ann["value"])
                ann["status"], ann["note"] = "unknown", "conflict with an earlier observation: %r" % old["value"]
                notes.append("contradiction with %s: both set to unknown" % old["annotation_id"])
            else:
                self.annotations.remove(old)
                notes.append("replaced %s" % old["annotation_id"])
        ann["annotation_id"] = "ra_%03d" % (len(self.annotations) + len(self.rejected) + 1)
        self.annotations.append(ann)
        return notes


# ------------------------------------------------------------------ the loop
def _anthropic_call(model, system, tools, messages, max_tokens=1500):
    import anthropic
    global _CLIENT
    try:
        client = _CLIENT
    except NameError:
        client = _CLIENT = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    resp = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                  tools=tools, messages=messages)
    content = []
    for b in resp.content:
        if b.type == "text":
            content.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return {"content": content, "stop_reason": resp.stop_reason,
            "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}}


def build_user_prompt(graph, candidates):
    nodes = [{"id": n["id"], "label": n["label"], "bbox2d": n["bbox2d"],
              "observed_now": n.get("observed_now", True)} for n in graph["nodes"]]
    return ("Scene graph (2D, image size %s):\nnodes: %s\nedges: %s\n\n"
            "Suggested candidates from simple rules (verify before using, may be empty):\n%s\n\n"
            "Annotate the risk-relevant facts and hazards you can support, then call finish."
            % (graph.get("image_size"), json.dumps(nodes), json.dumps(graph["edges"]),
               json.dumps(candidates) if candidates else "none"))


# Defaults keep the original behaviour (used with the Anthropic back-end).
# webcam_scene_graph.py passes smaller images and the limits below for Ollama.
DEFAULT_OPTIONS = {"crop_max_side": 1024, "frame_max_side": 1024, "max_crops_per_node": None,
                   "drop_old_images": False, "num_ctx": None,
                   "budget_remind": 0.70, "budget_stop": 0.85,
                   "progress_remind": None, "progress_stop": None}

BUDGET_REMINDER = ("Context budget: %d of %d tokens used. Stop inspecting. Write your annotations "
                   "now (status 'unknown' where the crops do not show the fact) and then call finish.")
PROGRESS_REMINDER = ("No annotation written after %d turns. Inspected so far: %s. Now call "
                     "write_annotation for these nodes, citing the call_ids above (status "
                     "'unknown' if you are unsure), and then call finish.")


def _inspected(session):
    """Short text of what the agent has looked at, for the progress reminder."""
    parts = ["%s (crop %s)" % (n, ", ".join(c)) for n, c in session.crops_of.items()]
    for e in session.log:
        if e["tool"] == "get_geometry" and not e.get("is_error") and e.get("nodes"):
            parts.append("geometry %s/%s (%s)" % (e["nodes"][0], e["nodes"][1], e["call_id"]))
    return "; ".join(parts) or "nothing yet (use crop_view first)"


def _drop_old_images(messages, placeholders):
    """Replace crop images in tool results with a short text placeholder.

    Only tool results are changed; the camera frame in the first user message is
    kept, since it is the only view of the whole scene. Returns the number of
    images replaced.
    """
    n = 0
    for m in messages:
        if m["role"] != "user" or not isinstance(m["content"], list):
            continue
        for b in m["content"]:
            if b.get("type") != "tool_result" or not isinstance(b.get("content"), list):
                continue
            new = []
            for inner in b["content"]:
                if inner.get("type") == "image":
                    n += 1
                    inner = {"type": "text", "text": placeholders.get(
                        b.get("tool_use_id"), "[image removed from history]")}
                new.append(inner)
            b["content"] = new
    return n


def run_agent(frame, graph, model="claude-sonnet-5-5", max_steps=12, out_dir=None,
              call_model=None, kb=None, run_id=None, options=None):
    """Run the agent on one frame and graph. `call_model` can be replaced in tests.

    options (see DEFAULT_OPTIONS):
      crop_max_side, frame_max_side  longest image side sent to the model
      max_crops_per_node             after that, crop_view returns an error
                                     telling the agent to annotate now
      drop_old_images                after a turn, crop images in earlier tool
                                     results are replaced by a text placeholder
      num_ctx, budget_remind, budget_stop
                                     context budget: when prompt + output of a
                                     turn reach budget_remind * num_ctx a reminder
                                     is added; at budget_stop * num_ctx the run
                                     stops (stopped_by "context_budget") after
                                     executing that turn's tool calls
      progress_remind, progress_stop if no write_annotation call (accepted or
                                     rejected) was made after this many turns,
                                     a reminder listing the inspected nodes is
                                     added / the run stops with stopped_by
                                     "no_progress". None = off.
    A model call that raises an exception with `context_overflow = True` (the
    Ollama client does this for HTTP 400 "exceeds the available context size")
    ends the run with stopped_by "context_overflow"; the partial log and the
    annotations written so far are returned.
    """
    opts = dict(DEFAULT_OPTIONS, **(options or {}))
    call_model = call_model or _anthropic_call
    run_id = run_id or "run%d" % int(time.time())
    session = AgentSession(frame, graph, out_dir, run_id, model, kb,
                           crop_max_side=opts["crop_max_side"],
                           max_crops_per_node=opts["max_crops_per_node"])
    candidates = rule_candidates(graph, kb)
    b64, sent = _jpeg_b64(frame, max_side=opts["frame_max_side"])
    messages = [{"role": "user", "content": [
        _image_block(b64), {"type": "text", "text": build_user_prompt(graph, candidates)}]}]
    tokens, steps, stopped, t0 = {"input": 0, "output": 0}, 0, "max_steps", time.time()
    turns, placeholders, dropped, error = [], {}, 0, None
    num_ctx = opts["num_ctx"]
    for steps in range(1, max_steps + 1):
        try:
            resp = call_model(model, SYSTEM_PROMPT, TOOLS, messages)
        except Exception as exc:
            if not getattr(exc, "context_overflow", False):
                raise
            stopped, error = "context_overflow", str(exc)
            turns.append({"step": steps, "error": error})
            break
        usage = resp.get("usage", {})
        tokens["input"] += usage.get("input", 0)
        tokens["output"] += usage.get("output", 0)
        used = usage.get("input", 0) + usage.get("output", 0)
        messages.append({"role": "assistant", "content": resp["content"]})
        tool_uses = [b for b in resp["content"] if b["type"] == "tool_use"]
        turn = {"step": steps, "prompt_tokens": usage.get("input", 0),
                "output_tokens": usage.get("output", 0),
                "tools": [tu["name"] for tu in tool_uses]}
        turns.append(turn)
        if not tool_uses:
            stopped = "no_tool_calls"
            break
        results = []
        for tu in tool_uses:
            content, is_error = session.execute(tu["name"], tu.get("input") or {})
            entry = session.log[-1]
            if entry.get("node") and entry["tool"] == "crop_view" and not is_error:
                placeholders[tu["id"]] = "[crop image of %s removed from history, call_id %s]" % (
                    entry["node"], entry["call_id"])
            r = {"type": "tool_result", "tool_use_id": tu["id"], "content": content}
            if is_error:
                r["is_error"] = True
            results.append(r)
        if session.finished:
            messages.append({"role": "user", "content": results})
            stopped = "finished"
            break
        if num_ctx and used >= opts["budget_stop"] * num_ctx:
            messages.append({"role": "user", "content": results})
            stopped = "context_budget"
            turn["budget"] = "stop"
            break
        wrote = any(e["tool"] == "write_annotation" for e in session.log)
        if not wrote and opts["progress_stop"] and steps >= opts["progress_stop"]:
            messages.append({"role": "user", "content": results})
            stopped = "no_progress"
            turn["progress"] = "stop"
            break
        if num_ctx and used >= opts["budget_remind"] * num_ctx:
            results.append({"type": "text", "text": BUDGET_REMINDER % (used, num_ctx)})
            turn["budget"] = "reminder"
        if not wrote and opts["progress_remind"] and steps == opts["progress_remind"]:
            results.append({"type": "text", "text": PROGRESS_REMINDER % (steps, _inspected(session))})
            turn["progress"] = "reminder"
        if opts["drop_old_images"]:
            dropped += _drop_old_images(messages, placeholders)
        messages.append({"role": "user", "content": results})
    return {"run_id": run_id, "model": model, "prompt_version": PROMPT_VERSION,
            "steps": steps, "stopped_by": stopped, "error": error,
            "tool_calls": len(session.log),
            "tokens": tokens, "latency_s": round(time.time() - t0, 2),
            "summary": session.summary, "candidates": candidates,
            "annotations": session.annotations, "rejected": session.rejected,
            "options": dict(opts, frame_sent_px=[sent.shape[1], sent.shape[0]],
                            frame_kept_in_history=True),
            "turns": turns, "images_dropped": dropped,
            "crops_per_node": {k: len(v) for k, v in session.crops_of.items()},
            "repeated_lookups": sum(1 for e in session.log if e.get("repeat_of")),
            "log": session.log}


# ------------------------------------------------------------------- outputs
def annotated_graph(graph, result):
    meta = {k: result.get(k) for k in ("run_id", "model", "prompt_version", "steps", "stopped_by",
                                       "error", "tool_calls", "tokens", "latency_s", "summary",
                                       "options")}
    return {"schema_version": RISK_SCHEMA, "graph": graph,
            "annotations": result["annotations"], "rejected_count": len(result["rejected"]),
            "agent_run": meta}


def risk_badges(result):
    """node id -> short text lines for the overlay (asserted items only)."""
    out = {}
    for a in result["annotations"]:
        if a["status"] != "asserted":
            continue
        if a["kind"] == "hazard_triple":
            node = str(a["triple"]["subject"]).split(".")[0]
            out.setdefault(node, []).append("RISK %s %s" % (a["hazard"]["type"], a["hazard"]["severity"]["level"]))
        elif a["attribute"] == "closure" and a["value"] in ("open", "missing"):
            out.setdefault(a["node"], []).append("OPEN" if a["value"] == "open" else "NO CAP")
        elif a["attribute"] == "tip_exposed" and a["value"] == "true":
            out.setdefault(a["node"], []).append("EXPOSED TIP")
        elif a["attribute"] == "pictograms":
            out.setdefault(a["node"], []).append("GHS %s" % a["value"])
    return out
