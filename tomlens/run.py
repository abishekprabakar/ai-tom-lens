"""Extract representations from a model and run the full analysis.

    # the from-scratch models trained by `python -m tomlens.train`
    python -m tomlens.run --tiny gpt llama gru

    # real open-weight LLMs (run on a machine that can reach huggingface.co)
    python -m tomlens.run --hf Qwen/Qwen3-0.6B deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
                               HuggingFaceTB/SmolLM2-360M-Instruct --max-groups 150

Each model gets results/<name>/results.json (behaviour, triplet alignment,
learned-metric selectivity, logit-lens curves) and results/<name>/viewer.json
(MDS projections and token-level showcase prompts for the web tool).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np

from . import alignment as al
from .generate import questions
from .train import build_corpus, ROOT

QTYPES = ("reality", "belief_leaver", "belief_actor", "nested_actor_about_leaver", "nested_leaver_about_actor")


def collect(backend, stories, log=print):
    items, H, margins, lens = [], [], [], []
    t0 = time.time()
    qs = list(questions(stories))
    for i, (s, q) in enumerate(qs):
        h, m, ln = backend.score(q.prompt, q.answer, q.foil)
        H.append(h.astype(np.float32)); margins.append(m); lens.append(ln)
        items.append(dict(index=i, qid=q.qid, sid=s.sid, group=s.group, split=s.split,
                          skeleton=s.skeleton, variant=s.variant, qtype=q.qtype, order=q.order,
                          obj=s.obj, false_belief=q.false_belief))
        if (i + 1) % 500 == 0:
            log(f"  [{backend.name}] {i + 1}/{len(qs)} prompts ({time.time() - t0:.0f}s)")
    return items, np.stack(H), np.array(margins), np.stack(lens)


def behaviour(items, margins):
    out = defaultdict(lambda: [0, 0])
    for it, m in zip(items, margins):
        for key in (f"{it['split']}|all", f"{it['split']}|{it['qtype']}|{'FB' if it['false_belief'] else 'TB'}",
                    f"{it['split']}|order{it['order']}"):
            out[key][0] += int(m > 0)
            out[key][1] += 1
    return {k: {"acc": a / n, "n": n} for k, (a, n) in sorted(out.items())}


def alignment_curves(items, H, seed=0, steps=300, log=print):
    tr_items = [it for it in items if it["split"] == "train" and it["qtype"] in al.LABELLED_QTYPES]
    te_items = [it for it in items if it["split"] == "test" and it["qtype"] in al.LABELLED_QTYPES]
    groups = [it["group"] for it in items]
    ctrl = al.control_labels(items, seed=seed)
    for it, c in zip(items, ctrl):
        it["control"] = c
    res = {"layers": list(range(H.shape[1])), "raw": {}, "learned": [], "control": [],
           "raw_ci": {}, "n_test_triplets": {}}
    tri_te = {qt: al.build_triplets([i for i in te_items if i["qtype"] == qt], seed=seed) for qt in al.LABELLED_QTYPES}
    tri_te["pooled"] = np.concatenate(list(tri_te.values()))
    tri_te["binding"] = al.build_binding_triplets([i for i in items if i["split"] == "test"], seed=seed)
    tri_tr = al.build_triplets(tr_items, seed=seed + 1)
    ctl_tr = al.build_triplets(tr_items, label_key="control", seed=seed + 2)
    ctl_te = al.build_triplets(te_items, label_key="control", seed=seed + 3)
    for qt, t in tri_te.items():
        res["n_test_triplets"][qt] = int(len(t))
    tr_idx = np.array([it["index"] for it in tr_items])
    for l in range(H.shape[1]):
        X = H[:, l, :].astype(np.float64)
        mean = X[tr_idx].mean(0)
        Xc = X - mean
        for qt, t in tri_te.items():
            res["raw"].setdefault(qt, []).append(al.triplet_accuracy(Xc, t))
            res["raw_ci"].setdefault(qt, []).append(al.bootstrap_ci(Xc, t, groups, seed=seed))
        # learned metric: fit on train triplets, score held-out test triplets
        _, Z = al.standardize(X[tr_idx], X)
        W, _ = al.train_metric(Z, tri_tr, steps=steps, seed=seed)
        res["learned"].append(al.triplet_accuracy(Z @ W.T, tri_te["pooled"], dist=al.euclid))
        Wc, _ = al.train_metric(Z, ctl_tr, steps=steps, seed=seed)
        res["control"].append(al.triplet_accuracy(Z @ Wc.T, ctl_te, dist=al.euclid))
        log(f"  layer {l:2d}  raw pooled {res['raw']['pooled'][-1]:.3f}  learned {res['learned'][-1]:.3f}"
            f"  control {res['control'][-1]:.3f}")
    res["selectivity"] = [a - c for a, c in zip(res["learned"], res["control"])]
    return res


def lens_curves(items, lens):
    """Mean logit-lens margin (answer - foil) per layer, by test qtype x belief."""
    out = defaultdict(list)
    for it, ln in zip(items, lens):
        if it["split"] != "test":
            continue
        out[f"{it['qtype']}|{'FB' if it['false_belief'] else 'TB'}"].append(ln[:, 0] - ln[:, 1])
    return {k: np.mean(v, 0).round(4).tolist() for k, v in sorted(out.items())}


def mds_views(items, H, n_groups=60, seed=0):
    rng = random.Random(seed)
    test_groups = sorted({it["group"] for it in items if it["split"] == "test"})
    pick = set(rng.sample(test_groups, min(n_groups, len(test_groups))))
    sel = [it for it in items if it["group"] in pick and it["qtype"] in ("belief_leaver", "nested_actor_about_leaver")]
    idx = np.array([it["index"] for it in sel])
    cen_keys = [(qt, v) for qt in QTYPES for v in ("away", "return", "watch")]
    test = [it for it in items if it["split"] == "test"]
    views = {"points": [{k: it[k] for k in ("qid", "qtype", "variant", "skeleton", "false_belief")} for it in sel],
             "cloud": [], "cloud_stress": [], "centroid_keys": [list(k) for k in cen_keys],
             "centroids": [], "centroid_stress": []}
    for l in range(H.shape[1]):
        X = H[:, l, :].astype(np.float64)
        Xc = X - X[[it["index"] for it in test]].mean(0)
        Y, st = al.classical_mds(al.pairwise_cosine(Xc[idx]))
        Y /= (np.abs(Y).max() + 1e-12)
        views["cloud"].append(Y.round(4).tolist()); views["cloud_stress"].append(round(st, 4))
        C = np.stack([Xc[[it["index"] for it in test if (it["qtype"], it["variant"]) == k]].mean(0) for k in cen_keys])
        Yc, stc = al.classical_mds(al.pairwise_cosine(C))
        Yc /= (np.abs(Yc).max() + 1e-12)
        views["centroids"].append(Yc.round(4).tolist()); views["centroid_stress"].append(round(stc, 4))
    return views


def showcase(backend, stories, n_groups=6, topk=50, seed=0):
    rng = random.Random(seed)
    test_groups = sorted({s.group for s in stories if s.split == "test"})
    by_sk = defaultdict(list)
    for g in test_groups:
        sk = next(s.skeleton for s in stories if s.group == g)
        by_sk[sk].append(g)
    pick = []
    for sk in sorted(by_sk):                        # balance the skeletons
        pick += rng.sample(by_sk[sk], min(len(by_sk[sk]), max(1, n_groups // len(by_sk))))
    out = []
    for s in stories:
        if s.group not in pick:
            continue
        for q in s.questions:
            a_id = backend.continuation_ids(q.prompt, q.answer)[0]
            f_id = backend.continuation_ids(q.prompt, q.foil)[0]
            v = backend.token_view(q.prompt, topk=topk, track=(a_id, f_id))
            v.update(track_ids=[a_id, f_id], qid=q.qid, sid=s.sid, group=s.group, skeleton=s.skeleton, variant=s.variant,
                     qtype=q.qtype, order=q.order, answer=q.answer, foil=q.foil,
                     false_belief=q.false_belief, story=s.text,
                     question=q.prompt.split("\n")[1])
            out.append(v)
    return out


def analyse(backend, stories, outdir, log=print, metric_steps=300, showcase_groups=6, topk=50):
    os.makedirs(outdir, exist_ok=True)
    items, H, margins, lens = collect(backend, stories, log)
    log(f"  [{backend.name}] hidden states {H.shape}")
    results = {
        "model": backend.name, "n_layers": int(H.shape[1] - 1), "d_model": int(H.shape[2]),
        "n_prompts": len(items),
        "behaviour": behaviour(items, margins),
        "alignment": alignment_curves(items, H, steps=metric_steps, log=log),
        "lens": lens_curves(items, lens),
    }
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    viewer = {"model": backend.name, "n_layers": results["n_layers"], "mds": mds_views(items, H),
              "showcase": showcase(backend, stories, showcase_groups, topk)}
    with open(os.path.join(outdir, "viewer.json"), "w") as f:
        json.dump(viewer, f, separators=(",", ":"))
    return results


def subsample(stories, max_groups, seed=0):
    if not max_groups:
        return stories
    rng = random.Random(seed)
    groups = sorted({s.group for s in stories})
    keep = set(rng.sample(groups, min(max_groups, len(groups))))
    return [s for s in stories if s.group in keep]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiny", nargs="*", default=[], help="from-scratch archs in models/<arch>.pt")
    ap.add_argument("--hf", nargs="*", default=[], help="Hugging Face model ids")
    ap.add_argument("--max-groups", type=int, default=0, help="subsample story groups (x3 stories, x5 prompts)")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: best available)")
    ap.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"])
    a = ap.parse_args(argv)
    stories, ood = build_corpus()
    corpus = subsample(stories, a.max_groups) + subsample(ood, a.max_groups // 6 if a.max_groups else 0)
    from .extract import TinyBackend, HFBackend
    for arch in a.tiny:
        be = TinyBackend(os.path.join(ROOT, "models", f"{arch}.pt"), name=f"tiny-{arch}")
        print(f"== {be.name}")
        analyse(be, corpus, os.path.join(a.results, be.name), topk=200)
    for mid in a.hf:
        import torch
        be = HFBackend(mid, device=a.device, dtype=getattr(torch, a.dtype) if a.dtype else None)
        print(f"== {be.name}")
        analyse(be, corpus, os.path.join(a.results, be.name), topk=40)


if __name__ == "__main__":
    main()
