"""Static figures and a markdown summary table from results/*/results.json.

    python -m tomlens.report            # writes figures/*.png and prints tables
"""
from __future__ import annotations

import json
import os

import numpy as np

from .train import ROOT

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TB, FB = "#2a78d6", "#eb6834"
QT_LABEL = {
    "reality": "Reality (0th)", "belief_leaver": "Leaver's belief (1st)", "belief_actor": "Actor's belief (1st)",
    "nested_actor_about_leaver": "Actor about leaver (2nd)", "nested_leaver_about_actor": "Leaver about actor (2nd)",
    "pooled": "All belief questions", "binding": "Name binding (diagnostic)",
}


def load_all(results_dir):
    out = {}
    for m in sorted(os.listdir(results_dir)):
        p = os.path.join(results_dir, m, "results.json")
        if os.path.isfile(p):
            with open(p) as f:
                out[m] = json.load(f)
            with open(os.path.join(results_dir, m, "viewer.json")) as f:
                out[m]["_viewer"] = json.load(f)
    return out


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c9c7c0"); ax.spines["bottom"].set_color("#c9c7c0")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.grid(axis="y", color="#ebe9e4", lw=0.8)
    ax.set_axisbelow(True)


def figures(res, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    models = list(res)

    # 1. triplet alignment by depth, one panel per question type
    qts = ["pooled", "belief_leaver", "nested_actor_about_leaver", "nested_leaver_about_actor", "binding"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), dpi=130, sharey=True)
    for ax, qt in zip(axes, qts):
        for i, m in enumerate(models):
            a = res[m]["alignment"]; n = len(a["layers"]) - 1
            x = np.arange(n + 1) / n
            y = np.array(a["raw"][qt]); ci = np.array(a["raw_ci"][qt])
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=COLORS[i % 3], alpha=0.15, lw=0)
            ax.plot(x, y, "-o", color=COLORS[i % 3], ms=3, lw=2, label=m)
        ax.axhline(0.5, color="#7a7974", ls=(0, (3, 4)), lw=1)
        ax.set_title(QT_LABEL[qt], fontsize=10, color="#0b0b0b")
        ax.set_xlabel("relative depth", fontsize=9, color="#52514e")
        _style(ax)
    axes[0].set_ylabel("triplet accuracy (raw cosine)", fontsize=9, color="#52514e")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "alignment_by_depth.png")); plt.close(fig)

    # 2. learned metric vs control
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=130)
    for i, m in enumerate(models):
        a = res[m]["alignment"]; n = len(a["layers"]) - 1; x = np.arange(n + 1) / n
        ax.plot(x, a["learned"], "-o", color=COLORS[i % 3], ms=3, lw=2, label=f"{m} learned")
        ax.plot(x, a["control"], "--", color=COLORS[i % 3], lw=1.5, alpha=0.7, label=f"{m} control")
    ax.axhline(0.5, color="#7a7974", ls=(0, (3, 4)), lw=1)
    ax.set_xlabel("relative depth", fontsize=9, color="#52514e")
    ax.set_ylabel("held-out triplet accuracy", fontsize=9, color="#52514e")
    ax.set_title("Triplet-loss linear metric vs control task", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2); _style(ax)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "metric_vs_control.png")); plt.close(fig)

    # 3. behaviour
    keys = [k for k in res[models[0]]["behaviour"] if k.startswith("test|") and k.count("|") == 2]
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=130)
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        ys = [res[m]["behaviour"][k]["acc"] for k in keys]
        ax.bar(np.arange(len(keys)) + i * w - 0.4 + w / 2, ys, w * 0.9, color=COLORS[i % 3], label=m)
    ax.axhline(0.5, color="#7a7974", ls=(0, (3, 4)), lw=1)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([f"{QT_LABEL[k.split('|')[1]]}\n{'false' if k.endswith('FB') else 'true'} belief" for k in keys],
                       fontsize=7)
    ax.set_ylim(0, 1.02); ax.set_ylabel("forced-choice accuracy", fontsize=9, color="#52514e")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right"); _style(ax)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "behaviour.png")); plt.close(fig)

    # 4. MDS cognitive maps: models x {embeddings, middle, last}
    fig, axes = plt.subplots(len(models), 3, figsize=(10, 3.3 * len(models)), dpi=130, squeeze=False)
    for r, m in enumerate(models):
        v = res[m]["_viewer"]["mds"]; L = len(v["cloud"]) - 1
        pts = v["points"]
        for c, layer in enumerate([0, L // 2, L]):
            ax = axes[r][c]
            Y = np.array(v["cloud"][layer])
            for qt, marker in (("belief_leaver", "o"), ("nested_actor_about_leaver", "^")):
                for fb, col in ((False, TB), (True, FB)):
                    sel = [i for i, p in enumerate(pts) if p["qtype"] == qt and p["false_belief"] == fb]
                    ax.scatter(Y[sel, 0], Y[sel, 1], s=14, marker=marker, color=col, edgecolor="white", lw=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#e2e0da")
            ax.set_title(f"{m} · layer {layer}/{L} · stress {v['cloud_stress'][layer]:.2f}", fontsize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=TB, label="true belief"),
               Line2D([], [], marker="o", ls="", color=FB, label="false belief"),
               Line2D([], [], marker="o", ls="", color="#7a7974", label="leaver's belief"),
               Line2D([], [], marker="^", ls="", color="#7a7974", label="actor about leaver")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(outdir, "cognitive_maps.png")); plt.close(fig)


def tables(res):
    models = list(res)
    lines = ["| Question | Belief | " + " | ".join(models) + " |", "|---|---|" + "---:|" * len(models)]
    keys = [k for k in res[models[0]]["behaviour"] if k.startswith("test|") and k.count("|") == 2]
    for k in keys:
        _, qt, fb = k.split("|")
        lines.append(f"| {QT_LABEL[qt]} | {'false' if fb == 'FB' else 'true'} | " +
                     " | ".join(f"{100 * res[m]['behaviour'][k]['acc']:.1f}%" for m in models) + " |")
    lines.append("| **All (test)** | | " + " | ".join(f"**{100 * res[m]['behaviour']['test|all']['acc']:.1f}%**" for m in models) + " |")
    lines.append("| **All (OOD, longer stories)** | | " + " | ".join(
        f"{100 * res[m]['behaviour'].get('ood|all', {'acc': float('nan')})['acc']:.1f}%" for m in models) + " |")
    out = "\n".join(lines) + "\n\n"
    # Layer 0 is the embedding of the last token, which is the same word ("the")
    # in every prompt, so every distance ties there; best layers skip it.
    rows = ["| Model | Layers | Raw alignment, best layer >= 1 | Raw, last layer | Learned metric, best | Control at that layer | Selectivity |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for m in models:
        a = res[m]["alignment"]; raw = a["raw"]["pooled"]
        b = 1 + int(np.argmax(raw[1:])); bl = 1 + int(np.argmax(a["learned"][1:]))
        rows.append(f"| {m} | {len(raw) - 1} | {raw[b]:.3f} (layer {b}) | {raw[-1]:.3f} | {a['learned'][bl]:.3f} (layer {bl}) | "
                    f"{a['control'][bl]:.3f} | {a['selectivity'][bl]:+.3f} |")
    return out + "\n".join(rows)


def main():
    res = load_all(os.path.join(ROOT, "results"))
    if not res:
        raise SystemExit("no results/ yet: run python -m tomlens.run first")
    figures(res, os.path.join(ROOT, "figures"))
    print(tables(res))


if __name__ == "__main__":
    main()
