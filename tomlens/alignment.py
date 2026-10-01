"""Triplet alignment, triplet-loss metric learning, and classical MDS.

Triplet alignment
-----------------
For a question type (say "where does the leaver think the object is?"), a
triplet is (anchor, positive, negative):

  anchor    a prompt
  positive  a prompt from a *different* story with the same belief status
            (false vs true belief) -- different names, object, containers,
            wording, often a different skeleton
  negative  the anchor's own minimal-pair twin: the same story with one
            sentence changed, which flips the belief status

The representation is aligned with the epistemic structure on that triplet
if d(anchor, positive) < d(anchor, negative). The negative shares ~95% of
its tokens with the anchor, so surface similarity pulls the other way:
triplet accuracy above 0.5 means belief state beats wording; below 0.5
means wording wins.

Metric learning
---------------
A linear map W (k x d) trained with the triplet loss
    max(0, margin + |W(a-p)|^2 - |W(a-n)|^2)
on training-split triplets and scored on held-out test triplets. It asks
whether the belief distinction is *linearly recoverable*, even when it is
not the dominant direction. To keep that honest, the same probe is trained
on a control task (Hewitt & Liang 2019): labels that are a random function
of (object, slot variant), which a probe can only fit by memorising surface
features. Selectivity = real accuracy - control accuracy.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

LABELLED_QTYPES = ("belief_leaver", "nested_actor_about_leaver", "nested_leaver_about_actor")


# ------------------------------------------------------------- triplets ----

def build_triplets(items: Sequence[dict], label_key="false_belief", n_per_anchor=4, seed=0):
    """items: dicts with index, group, qtype, skeleton and the label key.

    Returns an int array (n, 3) of item indices: anchor, positive, negative.
    """
    rng = random.Random(seed)
    by_group = defaultdict(list)
    by_label = defaultdict(list)
    for it in items:
        by_group[(it["group"], it["qtype"])].append(it)
        by_label[(it["qtype"], it[label_key])].append(it)
    out = []
    for a in items:
        twins = [t for t in by_group[(a["group"], a["qtype"])] if t[label_key] != a[label_key]]
        pool = by_label[(a["qtype"], a[label_key])]
        if not twins or len(pool) < 2:
            continue
        for _ in range(n_per_anchor):
            p = rng.choice(pool)
            while p["group"] == a["group"]:
                p = rng.choice(pool)
            out.append((a["index"], p["index"], rng.choice(twins)["index"]))
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def build_binding_triplets(items, n_per_anchor=4, seed=0):
    """Name-binding diagnostic. In an `away` story the leaver's and the
    actor's first-order beliefs differ, and the two questions differ only in
    the name. Anchor: one of them; negative: the other question about the
    same story; positive: the same question type about a different story.
    A model that never uses the name in the question puts the anchor and
    negative on top of each other and scores <= 0.5."""
    rng = random.Random(seed)
    sel = [it for it in items if it["variant"] == "away" and it["qtype"] in ("belief_leaver", "belief_actor")]
    by_sid = defaultdict(list)
    by_qt = defaultdict(list)
    for it in sel:
        by_sid[it["sid"]].append(it)
        by_qt[it["qtype"]].append(it)
    out = []
    for a in sel:
        neg = [t for t in by_sid[a["sid"]] if t["qtype"] != a["qtype"]]
        pool = [p for p in by_qt[a["qtype"]] if p["group"] != a["group"]]
        if not neg or not pool:
            continue
        for _ in range(n_per_anchor):
            out.append((a["index"], rng.choice(pool)["index"], neg[0]["index"]))
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def center(X, mean=None):
    mean = X.mean(0) if mean is None else mean
    return X - mean, mean


def cosine_dist(A, B):
    A = A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=-1, keepdims=True) + 1e-12)
    return 1.0 - (A * B).sum(-1)


def triplet_accuracy(X, triplets, dist=cosine_dist):
    """Fraction of triplets with d(a,p) < d(a,n). Ties count half."""
    if len(triplets) == 0:
        return float("nan")
    a, p, n = X[triplets[:, 0]], X[triplets[:, 1]], X[triplets[:, 2]]
    dp, dn = dist(a, p), dist(a, n)
    return float(((dp < dn) + 0.5 * (dp == dn)).mean())


def bootstrap_ci(X, triplets, groups, n_boot=200, seed=0, dist=cosine_dist):
    """95% CI resampling *story groups* (triplets sharing an anchor group are
    not independent, so resampling triplets would understate the width)."""
    rng = np.random.default_rng(seed)
    a, p, n = X[triplets[:, 0]], X[triplets[:, 1]], X[triplets[:, 2]]
    dp, dn = dist(a, p), dist(a, n)
    hit = (dp < dn) + 0.5 * (dp == dn)                  # ties count half, as in triplet_accuracy
    g = np.asarray(groups)[triplets[:, 0]]
    ug, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=hit, minlength=len(ug))
    c = np.bincount(inv, minlength=len(ug)).astype(float)
    stats = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        stats.append((w * s).sum() / (w * c).sum())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ------------------------------------------------------ metric learning ----

def triplet_loss_and_grad(W, A, P, N, margin=1.0):
    """Mean hinge triplet loss on squared Euclidean distances after x -> Wx.

    dL/dW = (2/n) sum_active [ W(a-p)(a-p)^T - W(a-n)(a-n)^T ]
    """
    dap, dan = A - P, A - N
    zp, zn = dap @ W.T, dan @ W.T
    sp, sn = (zp ** 2).sum(1), (zn ** 2).sum(1)
    h = margin + sp - sn
    act = h > 0
    loss = float(np.where(act, h, 0.0).mean())
    m = act.astype(float)[:, None] / len(A)
    grad = 2.0 * ((zp * m).T @ dap - (zn * m).T @ dan)
    return loss, grad


def train_metric(X, triplets, k=16, steps=300, lr=0.01, margin=1.0, batch=512, seed=0, l2=1e-4):
    """Adam on the triplet loss. X should already be standardised."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W = rng.normal(0, 1 / np.sqrt(d), (k, d))
    m1, m2 = np.zeros_like(W), np.zeros_like(W)
    b1, b2, eps = 0.9, 0.999, 1e-8
    losses = []
    for t in range(1, steps + 1):
        idx = rng.integers(0, len(triplets), min(batch, len(triplets)))
        tr = triplets[idx]
        loss, g = triplet_loss_and_grad(W, X[tr[:, 0]], X[tr[:, 1]], X[tr[:, 2]], margin)
        g = g + 2 * l2 * W
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b2 * m2 + (1 - b2) * g * g
        W -= lr * (m1 / (1 - b1 ** t)) / (np.sqrt(m2 / (1 - b2 ** t)) + eps)
        losses.append(loss)
    return W, losses


def euclid(A, B):
    return ((A - B) ** 2).sum(-1)


def standardize(Xtr, Xte, max_dim=128):
    """z-score with train statistics, then PCA (fit on train) to at most
    max_dim dimensions -- keeps 1-2k-wide LLM states tractable."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    if Ztr.shape[1] > max_dim:
        _, _, Vt = np.linalg.svd(Ztr - Ztr.mean(0), full_matrices=False)
        V = Vt[:max_dim].T
        Ztr, Zte = Ztr @ V, Zte @ V
        s = Ztr.std(0) + 1e-6
        Ztr, Zte = Ztr / s, Zte / s
    return Ztr, Zte


def control_labels(items, seed=0):
    """Hewitt-Liang control task: a random binary label per (object, variant)."""
    rng = random.Random(seed)
    table = {}
    out = []
    for it in items:
        key = (it["obj"], it["variant"])
        if key not in table:
            table[key] = rng.random() < 0.5
        out.append(table[key])
    return out


# ------------------------------------------------------------------ MDS ----

def classical_mds(D, k=2):
    """Torgerson MDS: double-centre -D^2/2, take the top-k eigenpairs.

    Returns coordinates (n, k) and Kruskal stress-1 of the embedding.
    """
    D = np.asarray(D, float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1][:k]
    w, V = np.clip(w[order], 0, None), V[:, order]
    Y = V * np.sqrt(w)
    Dy = np.sqrt(((Y[:, None] - Y[None]) ** 2).sum(-1))
    iu = np.triu_indices(n, 1)
    stress = float(np.sqrt(((D[iu] - Dy[iu]) ** 2).sum() / max((D[iu] ** 2).sum(), 1e-12)))
    # Fix the sign of each axis so plots don't flip between layers.
    for j in range(Y.shape[1]):
        if Y[np.argmax(np.abs(Y[:, j])), j] < 0:
            Y[:, j] *= -1
    return Y, stress


def pairwise_cosine(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    D = 1.0 - Xn @ Xn.T
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0, None)
