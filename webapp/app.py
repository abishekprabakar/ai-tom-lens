"""Flask viewer for tomlens results.

    python -m webapp.app            # http://127.0.0.1:5000

Reads results/<model>/results.json and viewer.json; any model added by
`python -m tomlens.run` shows up on the next page load.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache

from flask import Flask, abort, jsonify, render_template, request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("TOMLENS_RESULTS", os.path.join(ROOT, "results"))

app = Flask(__name__)


def _models():
    if not os.path.isdir(RESULTS):
        return []
    return sorted(d for d in os.listdir(RESULTS)
                  if os.path.isfile(os.path.join(RESULTS, d, "results.json")))


@lru_cache(maxsize=32)
def _load(model, kind):
    path = os.path.join(RESULTS, model, f"{kind}.json")
    if model not in _models() or not os.path.isfile(path):
        abort(404, f"no {kind} for {model}")
    with open(path) as f:
        return json.load(f)


def softmax_at(logits, T, logsumexp=None, full_vocab=False):
    """Softmax of the given logits at temperature T.

    If the stored logits are only the top-k of a larger vocabulary, the
    result is renormalised over those k (flagged in the response). At T = 1
    the stored full-vocabulary logsumexp gives exact probabilities instead.
    """
    T = max(float(T), 1e-3)
    if logsumexp is not None and abs(T - 1.0) < 1e-9 and not full_vocab:
        return [math.exp(l - logsumexp) for l in logits], True
    z = [l / T for l in logits]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e], full_vocab


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/models")
def models():
    out = []
    for m in _models():
        r = _load(m, "results")
        out.append({"name": m, "n_layers": r["n_layers"], "d_model": r["d_model"], "n_prompts": r["n_prompts"],
                    "test_acc": r["behaviour"].get("test|all", {}).get("acc")})
    return jsonify(out)


@app.get("/api/results/<model>")
def results(model):
    return jsonify(_load(model, "results"))


@app.get("/api/mds/<model>")
def mds(model):
    v = _load(model, "viewer")["mds"]
    layer = int(request.args.get("layer", v and len(v["cloud"]) - 1))
    if not 0 <= layer < len(v["cloud"]):
        abort(400, "layer out of range")
    return jsonify({"layer": layer, "n_layers": len(v["cloud"]) - 1, "points": v["points"],
                    "cloud": v["cloud"][layer], "cloud_stress": v["cloud_stress"][layer],
                    "centroid_keys": v["centroid_keys"], "centroids": v["centroids"][layer],
                    "centroid_stress": v["centroid_stress"][layer]})


@app.get("/api/showcase/<model>")
def showcase_index(model):
    sc = _load(model, "viewer")["showcase"]
    return jsonify([{k: s[k] for k in ("qid", "sid", "group", "variant", "skeleton", "qtype", "order",
                                       "answer", "foil", "false_belief", "question")} for s in sc])


@app.get("/api/token/<model>/<path:qid>")
def token(model, qid):
    sc = {s["qid"]: s for s in _load(model, "viewer")["showcase"]}
    if qid not in sc:
        abort(404, "prompt not in this model's showcase")
    s = sc[qid]
    T = float(request.args.get("T", 1.0))
    k = int(request.args.get("k", 10))
    full = len(s["layers"][0]["logits"]) >= s["vocab_size"]
    layers = []
    for L in s["layers"]:
        ids = L["ids"]
        extra = [lg for tid, lg in zip(s["track_ids"], L["tracked"]) if tid not in ids]
        p, exact = softmax_at(L["logits"] + extra, T, L.get("logsumexp"), full)
        by_id = dict(zip(ids, p))
        ex = iter(p[len(ids):])
        tracked = [by_id[tid] if tid in by_id else next(ex) for tid in s["track_ids"]]
        p = p[:len(ids)]
        layers.append({
            "top": [{"token": t, "p": round(pp, 5), "logit": lg}
                    for t, pp, lg in list(zip(L["tokens"], p, L["logits"]))[:k]],
            "answer_p": round(tracked[0], 5), "foil_p": round(tracked[1], 5),
            "answer_logit": L["tracked"][0], "foil_logit": L["tracked"][1],
            "exact": exact,
        })
    return jsonify({"qid": qid, "T": T, "story": s["story"], "question": s["question"],
                    "answer": s["answer"], "foil": s["foil"], "false_belief": s["false_belief"],
                    "qtype": s["qtype"], "variant": s["variant"], "tokens": s["tokens"],
                    "surprisal_bits": s["surprisal_bits"], "layers": layers,
                    "renormalised_over_topk": not full})


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
