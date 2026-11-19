#!/usr/bin/env python3
"""
Webcam scene graph prototype (2D) for early thesis experiments.

Pipeline
  camera frame -> open-vocabulary detector (YOLO-World)
               -> simple tracker (persistent IDs, "observed_now" flag)
               -> rule-based spatial relations from 2D boxes
               -> scene graph JSON (+ overlay window)
  optional: press 'v' to send the current frame to a VLM in ONE call
            (this is the "single VLM call without tools" baseline idea).

Limits: a laptop camera has no depth, so all relations are 2D image-space
approximations. This tests the pipeline shape, not 3D accuracy.

Input (--source): a camera index (0), a video file, or a folder of images.
Files and folders give repeatable runs: same input, same settings, same graphs.
Timestamps for files come from the frame index, not the wall clock.

Added from Acharjee et al. (2025), "Scene Graph-Guided Generative AI Framework
for Synthesizing and Evaluating Industrial Hazard Scenarios": the graph is turned
into yes/no assertions (nodes, attributes, edges) that a VLM verifies against the
image, with extra weight on hazard-critical objects. Unlike the paper, the verifier
may answer "unknown", and the score is reported together with its coverage.

Keys: q = quit, s = save graph + frame, v = VLM annotation (needs API key),
      c = check the graph with the VLM (assertion verification, needs API key),
      a = agentic risk annotation (tools + evidence + validator, see risk_agent.py)
      With --backend ollama, v/c/a use a local model instead (no API key, see
      ollama_backend.py).

Install:  pip install ultralytics opencv-python anthropic
Run:      python webcam_scene_graph.py --classes "cup,bottle,knife,hand,book"
          python webcam_scene_graph.py --source clip.mp4 --no-display --save-every 10
          python webcam_scene_graph.py --source frames_dir/ --fps 10
"""
import argparse
import base64
import glob
import json
import math
import os
import platform
import statistics
import sys
import time

from risk_agent import PREDICATES, vocabulary_text
from dataclasses import dataclass

SCHEMA_VERSION = "sg-0.1-2d"
DEFAULT_CLASSES = ["cup", "bottle", "knife", "scissors", "pen", "phone",
                   "laptop", "book", "keyboard", "mouse", "hand", "bowl"]


# ----------------------------------------------------------------- geometry
def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter_area(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def iou(a, b):
    i = inter_area(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def h_overlap(a, b):
    """Horizontal overlap relative to the narrower box (0..1)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    m = min(a[2] - a[0], b[2] - b[0])
    return max(0.0, w) / m if m > 0 else 0.0


def v_overlap(a, b):
    h = min(a[3], b[3]) - max(a[1], b[1])
    m = min(a[3] - a[1], b[3] - b[1])
    return max(0.0, h) / m if m > 0 else 0.0


def box_gap(a, b):
    """Shortest distance between two boxes (0 if they overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


# ---------------------------------------------------------------- relations
def directed_relation(a, b, img_h, tol=0.03, max_gap=0.25):
    """Return 'in', 'on' or 'above' if box a has that relation to box b."""
    tol_px = tol * img_h
    if area(a) > 0 and inter_area(a, b) / area(a) >= 0.9 and area(a) < area(b):
        return "in"
    if h_overlap(a, b) >= 0.5 and center(a)[1] < center(b)[1]:
        if abs(a[3] - b[1]) <= tol_px:
            return "on"
        gap = b[1] - a[3]
        if tol_px < gap < max_gap * img_h:
            return "above"
    return None


def compute_relations(boxes, img_w, img_h, near_ratio=0.12):
    """boxes: dict id -> (x1,y1,x2,y2). Returns a list of edge dicts.

    At most one relation family per pair: in/on/above has priority over near.
    """
    diag = math.hypot(img_w, img_h)
    ids = sorted(boxes)
    edges = []
    for i, ida in enumerate(ids):
        for idb in ids[i + 1:]:
            a, b = boxes[ida], boxes[idb]
            found = False
            for s, t, bs, bt in ((ida, idb, a, b), (idb, ida, b, a)):
                rel = directed_relation(bs, bt, img_h)
                if rel:
                    edges.append(_edge(s, t, rel, "spatial"))
                    found = True
            if found:
                continue
            if box_gap(a, b) <= near_ratio * diag:
                edges.append(_edge(ida, idb, "near", "spatial"))
                if v_overlap(a, b) >= 0.5:
                    left, right = (ida, idb) if center(a)[0] < center(b)[0] else (idb, ida)
                    edges.append(_edge(left, right, "left_of", "spatial"))
    return edges


def _edge(src, tgt, pred, etype):
    return {"type": etype, "predicate": pred, "source": src, "target": tgt,
            "derived_by": "rule:%s@0.1" % pred}


# ------------------------------------------------------------------ tracker
@dataclass
class Track:
    id: str
    label: str
    box: tuple
    conf: float
    first_seen: float
    last_seen: float
    hits: int = 1
    missed: int = 0


class Tracker:
    """Greedy same-label tracker using IoU, with a centroid fallback."""

    def __init__(self, iou_thr=0.3, max_missed=15, min_hits=3):
        self.iou_thr = iou_thr
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.tracks = []
        self.counters = {}

    def _score(self, tr, box):
        s = iou(tr.box, box)
        if s >= self.iou_thr:
            return s
        (cx1, cy1), (cx2, cy2) = center(tr.box), center(box)
        size = max(tr.box[2] - tr.box[0], tr.box[3] - tr.box[1])
        if size > 0 and math.hypot(cx1 - cx2, cy1 - cy2) < 0.5 * size:
            return 0.05 + s
        return 0.0

    def update(self, detections, t):
        """detections: list of (label, box, conf)."""
        for tr in self.tracks:
            tr.missed += 1
        used = set()
        for label, box, conf in sorted(detections, key=lambda d: -d[2]):
            best, best_s = None, 0.0
            for tr in self.tracks:
                if tr.label != label or id(tr) in used:
                    continue
                s = self._score(tr, box)
                if s > best_s:
                    best, best_s = tr, s
            if best is not None:
                used.add(id(best))
                best.box, best.conf = tuple(box), conf
                best.last_seen, best.missed = t, 0
                best.hits += 1
            else:
                n = self.counters.get(label, 0) + 1
                self.counters[label] = n
                new = Track("%s_%d" % (label.replace(" ", "_"), n), label,
                            tuple(box), conf, t, t)
                used.add(id(new))
                self.tracks.append(new)
        self.tracks = [tr for tr in self.tracks if tr.missed <= self.max_missed]

    def confirmed(self):
        return [tr for tr in self.tracks if tr.hits >= self.min_hits]


# -------------------------------------------------------------------- graph
def build_graph(tracker, img_w, img_h, t, graph_id, camera_id="cam_laptop"):
    nodes, visible = [], {}
    for tr in tracker.confirmed():
        observed = tr.missed == 0
        nodes.append({
            "id": tr.id, "type": "object", "label": tr.label,
            "bbox2d": [round(v, 1) for v in tr.box],
            "confidence": round(tr.conf, 3),
            "first_seen": tr.first_seen, "last_seen": tr.last_seen,
            "observed_now": observed,
            "sources": [{"module": "detector", "model": "yolo-world"}],
        })
        if observed:  # relations only between objects visible right now
            visible[tr.id] = tr.box
    return {
        "schema_version": SCHEMA_VERSION, "graph_id": graph_id, "stamp": t,
        "frame_id": "image_2d", "image_size": [img_w, img_h], "sensors": [{"id": camera_id, "role": "webcam"}],
        "nodes": nodes, "edges": compute_relations(visible, img_w, img_h),
    }


# ---------------------------------------------------------------------- VLM
# The v prompt uses the same vocabulary and severity scale as the risk agent's
# system prompt (risk_agent.vocabulary_text), so v is a comparable baseline.
VLM_PROMPT_VERSION = "vlm-0.2"
VLM_PROMPT = """You are annotating a scene for a robot safety study.
Detected objects (id: label): {objs}
Look at the image and return ONLY a JSON object (no markdown) with:
{{"facts": [{{"node": "<id or 'unlisted'>", "attribute": "<fact attribute>",
   "value": "<value or 'unknown'>", "confidence": 0.0-1.0, "evidence": "<what you see>"}}],
 "hazards": [{{"subject": "<id[.part]>", "predicate": "<predicate>", "object": "<id or zone>",
   "hazard_type": "<hazard_type or 'unknown'>", "severity": "<S0-S4 or 'unknown'>",
   "confidence": 0.0-1.0, "evidence": "<what you see>"}}]}}
Rules: use 'unknown' when you cannot verify a fact in the image. Do not invent
objects. Give evidence for every item. Return empty lists if nothing applies.

""" + vocabulary_text()


# Model back-end, set from the command line in main(): "anthropic" (default) or
# "ollama" (local, see ollama_backend.py). Tests can leave it as is.
# max_tokens/temperature/seed are Ollama options only; the Anthropic calls keep
# their fixed max_tokens. CALLS collects one entry per Ollama model call for
# run_meta.json.
BACKEND = {"name": "anthropic", "host": "http://localhost:11434", "num_ctx": 4096,
           "max_tokens": 1024, "temperature": 0.2, "seed": 42, "digest": None,
           # agent and image options, used with Ollama only (see risk_agent.run_agent)
           "crop_max_side": 512, "frame_max_side": 640, "max_crops_per_node": 2,
           "drop_old_images": True, "budget_remind": 0.70, "budget_stop": 0.85,
           "progress_remind": 4, "progress_stop": 8}
AGENT_OPTION_KEYS = ("crop_max_side", "frame_max_side", "max_crops_per_node", "drop_old_images",
                     "budget_remind", "budget_stop", "progress_remind", "progress_stop")
CALLS = []


def agent_options():
    """Options for risk_agent.run_agent: None (original behaviour) unless Ollama."""
    if BACKEND["name"] != "ollama":
        return None
    return dict({k: BACKEND[k] for k in AGENT_OPTION_KEYS}, num_ctx=BACKEND["num_ctx"])


def make_call_model():
    """Agent model call for the chosen back-end (None = risk_agent's Anthropic call)."""
    if BACKEND["name"] != "ollama":
        return None
    from ollama_backend import OllamaClient
    return OllamaClient(BACKEND["host"], BACKEND["num_ctx"], max_tokens=BACKEND["max_tokens"],
                        temperature=BACKEND["temperature"], seed=BACKEND["seed"])


def backend_settings(model):
    if BACKEND["name"] != "ollama":
        return {"backend": BACKEND["name"], "model": model}
    return dict(make_call_model().settings(model, BACKEND["digest"]), agent_options=agent_options())


def _log_calls(kind, client, error=None):
    """Keep the client's call records for run_meta.json; returns the validity label."""
    if client.truncated:
        valid = "invalid: truncated"
    elif client.overflowed:
        valid = "invalid: context overflow"
    elif error is not None:
        valid = "invalid: failed (%s)" % error
    else:
        valid = "valid"
    CALLS.append({"kind": kind, "validity": valid, "calls": client.records})
    return valid


def _vlm_text(frame_bgr, prompt, model):
    """One VLM call with one image. Isolated so tests can replace it."""
    import cv2
    if BACKEND["name"] == "ollama":  # same frame size as the agent gets (--frame-max-side)
        h, w = frame_bgr.shape[:2]
        side = BACKEND["frame_max_side"]
        if side and max(h, w) > side:
            s = side / float(max(h, w))
            frame_bgr = cv2.resize(frame_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("could not encode frame")
    b64 = base64.standard_b64encode(buf.tobytes()).decode()
    if BACKEND["name"] == "ollama":
        client = make_call_model()
        try:
            text = client.vlm_text(b64, prompt, model, max_tokens=2000)
        except Exception as exc:
            _log_calls("vlm", client, error=exc)
            raise
        _log_calls("vlm", client)
        return text
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=model, max_tokens=2000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": prompt}]}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def vlm_annotate(frame_bgr, graph, model="claude-sonnet-5-5"):
    objs = ", ".join("%s: %s" % (n["id"], n["label"]) for n in graph["nodes"]) or "none"
    text = _vlm_text(frame_bgr, VLM_PROMPT.format(objs=objs), model)
    return parse_json_reply(text), text


# ------------------------------------------------ assertion-based checking
PREDICATE_PHRASE = {
    "near": "is near", "on": "is on", "in": "is inside", "above": "is above",
    "left_of": "is to the left of", "right_of": "is to the right of",
    "held_by": "is held by", "grasped_by": "is grasped by",
    "touching": "is touching", "pointing_toward": "is pointing toward",
    "open_toward": "is open toward", "positioned_above": "is positioned above",
    "carried_over": "is carried over", "within_reach_of": "is within reach of",
    "in_contact_with": "is in contact with",
}

VERIFY_PROMPT = """You are a strict verifier for a robot safety study.
Check each numbered assertion against the image. Answer ONLY a JSON list, one item
per assertion: {{"id": "a1", "verdict": "yes|no|unknown", "confidence": 0.0-1.0,
"evidence": "<short, what you see>"}}.
Rules: say "yes" only if the image clearly shows it, "no" if the image clearly
contradicts it, and "unknown" if you cannot decide (occluded, too small, too
blurry). Do not assume an assertion is true because it is listed.
Relations are judged in the 2D image only (there is no depth): %s.
Assertions:
{items}""" % "; ".join("'%s'" % PREDICATE_PHRASE[p] for p in PREDICATES)


def _name(ref, labels):
    base, _, part = str(ref).partition(".")
    label = labels.get(base, base.replace("_", " "))
    if part:
        return "the %s of the %s" % (part.replace("_", " "), label)
    return "the %s" % label


def _sentence(text):
    return text[0].upper() + text[1:] + "."


def _where(box, size):
    if not size:
        return "in the image"
    cx, cy = center(box)
    col = "left" if cx < size[0] / 3 else ("right" if cx > 2 * size[0] / 3 else "")
    row = "upper" if cy < size[1] / 3 else ("lower" if cy > 2 * size[1] / 3 else "")
    place = " ".join(x for x in (row, col) if x)
    return "in the %s of the image" % place if place else "near the center of the image"


def _weight(refs, labels, key_labels, key_weight):
    for r in refs:
        if labels.get(str(r).partition(".")[0]) in key_labels:
            return key_weight
    return 1.0


def graph_to_assertions(graph, key_labels=(), key_weight=2.0):
    """Nodes and edges of the graph -> checkable yes/no statements.

    Assertions that involve a key object (for example a knife) get more weight,
    following the hazard-node weighting idea of Acharjee et al. (2025).
    """
    labels = {n["id"]: n["label"] for n in graph["nodes"]}
    size = graph.get("image_size")
    out = []

    def add(kind, text, refs, weight=None):
        w = weight if weight is not None else _weight(refs, labels, key_labels, key_weight)
        out.append({"id": "a%d" % (len(out) + 1), "kind": kind, "text": text,
                    "refs": list(refs), "weight": w})

    for n in graph["nodes"]:
        if n.get("observed_now", True):
            add("node", _sentence("there is a %s %s" % (n["label"], _where(n["bbox2d"], size))),
                [n["id"]])
    for e in graph["edges"]:
        phrase = PREDICATE_PHRASE.get(e["predicate"], "is " + e["predicate"].replace("_", " "))
        add("edge", _sentence("%s %s %s" % (_name(e["source"], labels), phrase,
                                              _name(e["target"], labels))),
            [e["source"], e["target"]])
    return out


def annotation_to_assertions(annotation, graph, key_weight=2.0):
    """VLM facts and hazard triples -> checkable statements.

    'unknown' facts are skipped (nothing to verify). Hazard statements describe
    only the visible relation; the hazard type and severity are judgements and
    are not checked. Hazard assertions get key_weight. Hazards whose predicate
    is not in the shared vocabulary (risk_agent.PREDICATES, for example 'has')
    or that relate a node to itself are skipped: they cannot be phrased as a
    checkable relation.
    """
    if not isinstance(annotation, dict) or annotation.get("parse_error"):
        return []
    labels = {n["id"]: n["label"] for n in graph["nodes"]}
    out = []
    for f in annotation.get("facts", []) or []:
        value = str(f.get("value", "unknown"))
        if value.lower() == "unknown" or not f.get("node") or not f.get("attribute"):
            continue
        text = "%s's %s is %s" % (_name(f["node"], labels),
                                   str(f["attribute"]).replace("_", " "), value)
        out.append({"id": "a%d" % (len(out) + 1), "kind": "fact",
                    "text": _sentence(text), "refs": [f["node"]], "weight": 1.0})
    for h in annotation.get("hazards", []) or []:
        if not (h.get("subject") and h.get("predicate") and h.get("object")):
            continue
        pred = str(h["predicate"])
        if pred not in PREDICATES or str(h["subject"]).split(".")[0] == str(h["object"]).split(".")[0]:
            continue
        phrase = PREDICATE_PHRASE[pred]
        text = "%s %s %s" % (_name(h["subject"], labels), phrase, _name(h["object"], labels))
        out.append({"id": "a%d" % (len(out) + 1), "kind": "hazard",
                    "text": _sentence(text), "refs": [h["subject"], h["object"]],
                    "weight": key_weight})
    return out


def compute_support(assertions, verdicts):
    """Weighted share of supported assertions, plus coverage.

    Only yes/no verdicts enter the score. 'unknown' and missing answers are
    counted separately, so a high score with low coverage is easy to spot.
    """
    by_id = {v["id"]: v for v in verdicts if isinstance(v, dict) and "id" in v}
    counts = {"yes": 0, "no": 0, "unknown": 0, "missing": 0}
    num = den = 0.0
    unsupported = []
    for a in assertions:
        verdict = str((by_id.get(a["id"]) or {}).get("verdict", "missing")).lower()
        if verdict not in ("yes", "no", "unknown"):
            verdict = "missing"
        counts[verdict] += 1
        if verdict in ("yes", "no"):
            den += a["weight"]
            num += a["weight"] if verdict == "yes" else 0.0
        if verdict == "no":
            unsupported.append(a["id"])
    answered = counts["yes"] + counts["no"]
    return {"score": (num / den) if den else None, "counts": counts,
            "coverage": (answered / len(assertions)) if assertions else None,
            "unsupported": unsupported}


def verify_assertions(frame_bgr, assertions, model="claude-sonnet-5-5"):
    items = "\n".join("%s: %s" % (a["id"], a["text"]) for a in assertions)
    raw = _vlm_text(frame_bgr, VERIFY_PROMPT.format(items=items), model)
    parsed = parse_json_reply(raw)
    verdicts = parsed if isinstance(parsed, list) else []
    merged = []
    by_id = {v["id"]: v for v in verdicts if isinstance(v, dict) and "id" in v}
    for a in assertions:
        v = by_id.get(a["id"], {})
        merged.append(dict(a, verdict=v.get("verdict", "missing"),
                           confidence=v.get("confidence"), evidence=v.get("evidence")))
    return {"model": model, "parse_error": not isinstance(parsed, list),
            "summary": compute_support(assertions, verdicts),
            "assertions": merged, "raw": raw}


def parse_json_reply(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw": text}


# ------------------------------------------------------------- frame source
class FrameSource:
    """Camera index, video file or image folder -> (ok, frame, t_seconds)."""

    IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

    def __init__(self, source, fps=30.0):
        import cv2
        self.cv2 = cv2
        self.live = False
        self.images, self.cap, self.idx = None, None, 0
        self.fps = fps
        s = str(source)
        if s.isdigit():
            self.cap, self.live = cv2.VideoCapture(int(s)), True
        elif os.path.isdir(s):
            files = []
            for pat in self.IMG_EXT:
                files += glob.glob(os.path.join(s, pat))
            self.images = sorted(files)
            if not self.images:
                raise SystemExit("no images found in %s" % s)
        elif os.path.isfile(s):
            self.cap = cv2.VideoCapture(s)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        else:
            raise SystemExit("source not found: %s" % s)
        if self.cap is not None and not self.cap.isOpened():
            raise SystemExit("cannot open source: %s" % s)

    def read(self):
        if self.images is not None:
            if self.idx >= len(self.images):
                return False, None, 0.0
            frame = self.cv2.imread(self.images[self.idx])
            t = self.idx / self.fps
        else:
            ok, frame = self.cap.read()
            if not ok:
                return False, None, 0.0
            t = time.time() if self.live else self.idx / self.fps
        self.idx += 1
        return frame is not None, frame, t

    def release(self):
        if self.cap is not None:
            self.cap.release()


# ----------------------------------------------------------------- run loop
def process_frame(frame, t, detect, tracker, graph_no):
    """One step: detect -> track -> graph. Returns (graph, latency_ms)."""
    t0 = time.time()
    dets = detect(frame)
    h, w = frame.shape[:2]
    tracker.update(dets, t)
    graph = build_graph(tracker, w, h, t, "graph-%06d" % graph_no)
    return graph, (time.time() - t0) * 1000.0


def draw(frame, graph, ms, badges=None):
    import cv2
    badges = badges or {}
    view = frame.copy()
    boxes = {n["id"]: n["bbox2d"] for n in graph["nodes"]}
    for n in graph["nodes"]:
        x1, y1, x2, y2 = map(int, n["bbox2d"])
        color = (0, 200, 0) if n["observed_now"] else (0, 140, 255)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        cv2.putText(view, "%s %.2f" % (n["id"], n["confidence"]), (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for k, line in enumerate(badges.get(n["id"], [])):  # agent risk badges (red)
            cv2.putText(view, line, (x1, min(view.shape[0] - 5, y2 + 16 + 16 * k)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    for e in graph["edges"]:
        if e["predicate"] == "left_of":
            continue
        ca, cb = center(boxes[e["source"]]), center(boxes[e["target"]])
        cv2.line(view, tuple(map(int, ca)), tuple(map(int, cb)), (255, 200, 0), 1)
        mid = (int((ca[0] + cb[0]) / 2), int((ca[1] + cb[1]) / 2))
        cv2.putText(view, e["predicate"], mid, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 200, 0), 1)
    cv2.putText(view, "%d nodes, %d edges, %.0f ms | q s v c a"
                % (len(graph["nodes"]), len(graph["edges"]), ms),
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return view


def save_graph(out, name, graph, frame):
    import cv2
    stem = os.path.join(out, name)
    with open(stem + ".json", "w") as f:
        json.dump(graph, f, indent=2)
    cv2.imwrite(stem + ".jpg", frame)
    return stem


def _key_labels(args):
    return {x.strip() for x in args.key_labels.split(",") if x.strip()}


def _print_summary(tag, result):
    sm = result["summary"]
    score = "n/a" if sm["score"] is None else "%.2f" % sm["score"]
    cov = "n/a" if sm["coverage"] is None else "%.2f" % sm["coverage"]
    print("%s: score %s, coverage %s, %s" % (tag, score, cov, sm["counts"]))


def run_check(frame, graph, stem, args):
    assertions = graph_to_assertions(graph, _key_labels(args), args.key_weight)
    if not assertions:
        print("nothing to check")
        return None
    try:
        result = verify_assertions(frame, assertions, args.vlm_model)
    except Exception as exc:
        print("check failed:", exc)
        return None
    with open(stem + "_check.json", "w") as f:
        json.dump(result, f, indent=2)
    _print_summary("graph check", result)
    return result


def run_check_annotation(frame, graph, annotation, stem, args):
    assertions = annotation_to_assertions(annotation, graph, args.key_weight)
    if not assertions:
        print("no checkable facts or hazards in the annotation")
        return None
    try:
        result = verify_assertions(frame, assertions, args.vlm_model)
    except Exception as exc:
        print("annotation check failed:", exc)
        return None
    with open(stem + "_vlm_check.json", "w") as f:
        json.dump(result, f, indent=2)
    _print_summary("annotation check", result)
    return result


def run_agent_step(frame, graph, stem, args):
    """Run the risk agent on this frame and graph; returns badge dict or {}."""
    client = make_call_model()
    try:
        from risk_agent import run_agent, annotated_graph, risk_badges
        result = run_agent(frame, graph, model=args.vlm_model, max_steps=args.agent_steps,
                           out_dir=stem + "_crops", call_model=client, options=agent_options())
    except Exception as exc:  # keep the live loop running
        print("agent failed:", exc)
        if client is not None:
            _log_calls("agent", client, error=exc)
            CALLS[-1]["stem"] = stem
        return {}
    backend = backend_settings(args.vlm_model)
    if client is not None:
        backend["validity"] = _log_calls("agent", client)
        backend["calls"] = client.records
        CALLS[-1]["stem"] = stem
        if backend["validity"] != "valid":
            print("warning: agent run %s is %s" % (result["run_id"], backend["validity"]))
    with open(stem + "_agent.json", "w") as f:
        json.dump(annotated_graph(graph, result), f, indent=2)
    with open(stem + "_agent_log.json", "w") as f:
        json.dump(dict({k: result.get(k) for k in (
            "run_id", "model", "prompt_version", "steps", "stopped_by", "error", "options", "turns",
            "images_dropped", "crops_per_node", "repeated_lookups", "candidates", "annotations", "rejected", "log")},
            backend=backend), f, indent=2)
    print("agent: %d annotations (%d rejected writes), %d tool calls, %d steps, %.1f s, "
          "tokens in/out %d/%d, stopped by %s" % (
              len(result["annotations"]), len(result["rejected"]), result["tool_calls"],
              result["steps"], result["latency_s"], result["tokens"]["input"],
              result["tokens"]["output"], result["stopped_by"]))
    for a in result["annotations"]:
        desc = (a["triple"] if a["kind"] == "hazard_triple" else (a["node"], a["attribute"], a["value"]))
        print("  %s [%s] %s" % (a["annotation_id"], a["status"], desc))
    return risk_badges(result)


def run(args, detect, run_info=None):
    """Main loop, separated from the detector so it can be tested."""
    import cv2
    src = FrameSource(args.source, args.fps)
    os.makedirs(args.out, exist_ok=True)
    save_every = args.save_every
    if args.no_display and not save_every:
        save_every = 30
    tracker, n_saved, graph_no, latencies = Tracker(), 0, 0, []
    badges = {}
    while True:
        ok, frame, t = src.read()
        if not ok or (args.max_frames and graph_no >= args.max_frames):
            break
        graph_no += 1
        graph, ms = process_frame(frame, t, detect, tracker, graph_no)
        latencies.append(ms)

        if save_every and graph_no % save_every == 0:
            n_saved += 1
            stem = save_graph(args.out, "graph_%06d" % graph_no, graph, frame)
            if args.check_on_save:
                run_check(frame, graph, stem, args)
            if args.agent_on_save:
                run_agent_step(frame, graph, stem, args)

        if args.no_display:
            continue
        cv2.imshow("scene graph", draw(frame, graph, ms, badges))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("s"), ord("v"), ord("c"), ord("a")):
            n_saved += 1
            stem = save_graph(args.out, "graph_%06d" % graph_no, graph, frame)
            print("saved", stem + ".json")
            if key == ord("a"):
                badges = run_agent_step(frame, graph, stem, args)
            if key == ord("c") or (key == ord("s") and args.check_on_save):
                run_check(frame, graph, stem, args)
            if key == ord("v"):
                try:
                    parsed, raw = vlm_annotate(frame, graph, args.vlm_model)
                    with open(stem + "_vlm.json", "w") as f:
                        json.dump({"model": args.vlm_model, "prompt_version": VLM_PROMPT_VERSION,
                                   "backend": backend_settings(args.vlm_model), "result": parsed,
                                   "raw": raw}, f, indent=2)
                    print("saved", stem + "_vlm.json")
                    if args.check_annotations:
                        run_check_annotation(frame, graph, parsed, stem, args)
                except Exception as exc:  # keep the live loop running
                    print("VLM call failed:", exc)
    src.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    meta = {
        "script_schema": SCHEMA_VERSION,
        "source": str(args.source), "frames_processed": graph_no,
        "graphs_saved": n_saved, "args": vars(args),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "latency_ms": ({"mean": statistics.mean(latencies),
                        "median": statistics.median(latencies),
                        "max": max(latencies)} if latencies else None),
        "backend": backend_settings(args.vlm_model),
    }
    if CALLS:
        meta["model_calls"] = CALLS
        meta["invalid_model_runs"] = [c.get("stem", c["kind"]) for c in CALLS if c["validity"] != "valid"]
    meta.update(run_info or {})
    with open(os.path.join(args.out, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("done: %d frames, metadata in %s" % (graph_no, os.path.join(args.out, "run_meta.json")))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0",
                    help="camera index, video file or image folder (default 0)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="frame rate assumed for image folders (video files use their own)")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                    help="comma-separated open vocabulary")
    ap.add_argument("--weights", default="yolov8s-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-display", action="store_true",
                    help="process the whole input without a window")
    ap.add_argument("--save-every", type=int, default=0,
                    help="save a graph every N frames (headless default: 30)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames")
    ap.add_argument("--vlm-model", default="claude-sonnet-5-5",
                    help="model name for the chosen back-end (e.g. qwen3-vl:2b for ollama)")
    ap.add_argument("--backend", choices=["anthropic", "ollama"], default="anthropic",
                    help="anthropic (API key) or ollama (local, no frames leave the machine)")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--num-ctx", type=int, default=4096,
                    help="Ollama context window in tokens (longer prompts are cut silently)")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="Ollama only: output limit per call (num_predict)")
    ap.add_argument("--temperature", type=float, default=0.2, help="Ollama only")
    ap.add_argument("--seed", type=int, default=42, help="Ollama only")
    ap.add_argument("--crop-max-side", type=int, default=512,
                    help="Ollama only: longest side of agent crops in px")
    ap.add_argument("--frame-max-side", type=int, default=640,
                    help="Ollama only: longest side of the frame sent to the model in px")
    ap.add_argument("--max-crops-per-node", type=int, default=2,
                    help="Ollama only: after N crops of a node, crop_view tells the agent to annotate")
    ap.add_argument("--drop-old-images", action=argparse.BooleanOptionalAction, default=True,
                    help="Ollama only: replace crop images from earlier turns with a text placeholder")
    ap.add_argument("--budget-remind", type=float, default=0.70,
                    help="Ollama only: share of num_ctx at which the agent is told to annotate and finish")
    ap.add_argument("--budget-stop", type=float, default=0.85,
                    help="Ollama only: share of num_ctx at which the agent run stops (context_budget)")
    ap.add_argument("--progress-remind", type=int, default=4,
                    help="Ollama only: remind the agent to annotate if it wrote nothing after N turns")
    ap.add_argument("--progress-stop", type=int, default=8,
                    help="Ollama only: stop the agent (no_progress) if it wrote nothing after N turns")
    ap.add_argument("--key-labels", default="",
                    help="comma-separated labels that get extra weight in checks (e.g. knife)")
    ap.add_argument("--key-weight", type=float, default=2.0)
    ap.add_argument("--check-on-save", action="store_true",
                    help="run the VLM assertion check whenever a graph is saved")
    ap.add_argument("--agent-steps", type=int, default=12,
                    help="maximum model turns for the risk agent ('a' key)")
    ap.add_argument("--agent-on-save", action="store_true",
                    help="run the risk agent whenever a graph is saved")
    ap.add_argument("--check-annotations", action="store_true",
                    help="after 'v', also verify the VLM's own facts and hazards")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    BACKEND.update(name=args.backend, host=args.ollama_host, num_ctx=args.num_ctx,
                   max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed,
                   crop_max_side=args.crop_max_side, frame_max_side=args.frame_max_side,
                   max_crops_per_node=args.max_crops_per_node, drop_old_images=args.drop_old_images,
                   budget_remind=args.budget_remind, budget_stop=args.budget_stop,
                   progress_remind=args.progress_remind, progress_stop=args.progress_stop)
    if args.backend == "ollama":
        from ollama_backend import model_digest
        BACKEND["digest"] = model_digest(args.vlm_model, args.ollama_host)
    from ultralytics import YOLOWorld
    import ultralytics

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    model = YOLOWorld(args.weights)
    model.set_classes(classes)

    def detect(frame):
        res = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append((res.names[int(b.cls[0])], (x1, y1, x2, y2), float(b.conf[0])))
        return dets

    run(args, detect, {"classes": classes, "weights": args.weights,
                       "ultralytics": ultralytics.__version__})


if __name__ == "__main__":
    main()
