#!/usr/bin/env python3
"""Generate every figure in the paper, from frozen artifacts.

Two families.

**System figures** (appendix): projector scaling, HumanEval, joint system, judge.
These illustrate the logit-correction system, which the bound demotes to a
secondary interface; they are generated here so the appendix stays reproducible.

**Thesis figures** (main text), added in round 164.  Before this, all six
figures belonged to the system and none illustrated the paper's actual claim:

* ``fig_window_bound.svg``     -- the window arithmetic and the bound itself
* ``fig_content_null.svg``     -- the three-corpus experiment, both regimes
* ``fig_readout_collapse.svg`` -- the read-out collapse (round 164)

Numbers are read from committed artifacts and never typed in:

* window arithmetic  -- ``tests/golden/qwen38_flash_next_text_config.json``
* content experiment -- ``outputs/round162-0.8B/analysis.json`` (saturated),
                        ``outputs/round162-0.8B-nosft600/analysis.json`` (unsaturated)
* read-out collapse  -- ``outputs/round164/vector-analysis.json``

All output is SVG with ``svg.fonttype = "path"``: vector at any zoom, and no
font dependency at compile time.

Usage::

    python scripts/make_paper_figures.py                 # everything
    python scripts/make_paper_figures.py --only thesis   # figures 1-3 only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]

INK = "#1c1c1c"
MUTED = "#6b7280"
GRID = "#e4e4e7"
KEY = "#c2410c"        # tokens the row id itself reads
CONV = "#1d4ed8"       # extra positions the short convolution reaches
SAT = "#94a3b8"
UNSAT = "#0f766e"
ACCENT = "#b91c1c"
BLUE = "#4C72B0"
RED = "#C44E52"
GREEN = "#55A868"


def house_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "svg.fonttype": "path",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def _load(path: str) -> dict:
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name)
    plt.close(fig)
    print(f"[figs] wrote {out / name}")


# ---------------------------------------------------------------------------
# Thesis figure 1 -- the window arithmetic and the bound
# ---------------------------------------------------------------------------
def figure_window_and_bound(cfg: dict, out: Path) -> None:
    kernel = cfg["ple_conv_kernel_size"]
    ngram = cfg["ngram_size"]
    conv_reach = (kernel - 1) * ngram        # 9
    window = conv_reach + ngram              # 12

    fig = plt.figure(figsize=(7.0, 2.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.16)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-2.6, 14.4)
    ax.set_ylim(-1.5, 3.7)
    ax.axis("off")
    ax.set_title("(a)  the addressing window is not the key order", loc="left", pad=5)

    bw, bh = 0.86, 0.62

    def strip(y, n, colors):
        for i in range(n):
            x = 11 - i
            ax.add_patch(FancyBboxPatch(
                (x - bw / 2, y), bw, bh,
                boxstyle="round,pad=0.012,rounding_size=0.10",
                linewidth=0.6, edgecolor=INK, facecolor=colors[i]))
            ax.text(x, y + bh / 2, "$t$" + ("" if i == 0 else f"$-{i}$"),
                    ha="center", va="center", fontsize=6.6,
                    color="white" if colors[i] in (KEY, CONV) else INK)

    y1 = 2.0
    colors = [CONV] * window
    colors[:ngram] = [KEY] * ngram
    strip(y1, window, colors)
    ax.annotate("", xy=(11 - window + 1 - bw / 2, y1 + bh + 0.22),
                xytext=(11 + bw / 2, y1 + bh + 0.22),
                arrowprops=dict(arrowstyle="-", color=CONV, lw=1.0))
    ax.text(5.5, y1 + bh + 0.30,
            f"short conv reaches {conv_reach} more  (kernel {kernel}, dilation {ngram})",
            ha="center", va="bottom", fontsize=6.8, color=CONV)
    ax.annotate("", xy=(11 - ngram + 1 - bw / 2, y1 - 0.24),
                xytext=(11 + bw / 2, y1 - 0.24),
                arrowprops=dict(arrowstyle="-", color=KEY, lw=1.0))
    ax.text(10.1, y1 - 0.34, f"row id reads {ngram}", ha="center", va="top",
            fontsize=6.8, color=KEY)
    ax.text(-0.9, y1 + bh / 2, "Engram\nQwen3.8", ha="right", va="center",
            fontsize=7.0, color=INK, linespacing=1.15)
    ax.text(5.5, y1 + bh + 1.02, f"window = {window} tokens", ha="center",
            va="center", fontsize=8.4, color=INK, fontweight="bold")

    y2 = 0.35
    strip(y2, 4, [KEY] * 4)
    ax.text(-0.9, y2 + bh / 2, "DeepSeek\nV4.1-Flash", ha="right", va="center",
            fontsize=7.0, color=INK, linespacing=1.15)
    ax.text(5.5, y2 - 0.50, "convolution removed  →  window = 4 tokens",
            ha="center", va="center", fontsize=7.4, color=MUTED)
    ax.text(-2.5, -1.28, f"an order-{ngram} key alone would suggest {ngram}.",
            ha="left", va="center", fontsize=7.0, color=MUTED, style="italic")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-0.1, 10.2)
    ax2.axis("off")
    ax2.set_title("(b)  why the reader cannot escape it", loc="left", pad=5)

    def node(x, y, w, h, text, fc, ec=INK, tc=INK):
        ax2.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                     boxstyle="round,pad=0.02,rounding_size=0.45",
                                     linewidth=0.7, edgecolor=ec, facecolor=fc))
        ax2.text(x, y, text, ha="center", va="center", fontsize=7.2, color=tc)

    def arrow(a, b):
        ax2.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8,
                                      linewidth=0.8, color=INK, shrinkA=1, shrinkB=1))

    node(2.0, 8.3, 3.4, 1.5, "$w_t$  window", GRID)
    node(7.6, 8.3, 3.4, 1.5, "$a_t$  row id", GRID)
    node(7.6, 5.4, 3.4, 1.5, "$e_t$  rows", "#fde68a")
    node(1.9, 2.3, 3.4, 1.5, "$h_t$  hidden", GRID)
    node(6.3, 2.3, 2.9, 1.5, "$c_t$", "#bbf7d0", ec=MUTED)

    arrow((3.7, 8.3), (5.9, 8.3))
    ax2.text(4.8, 8.72, "deterministic", ha="center", va="bottom", fontsize=6.4, color=MUTED)
    arrow((7.6, 7.55), (7.6, 6.15))
    # The gate is an edge property, not a pipeline stage: drawing it as a node
    # made it collide with e_t and implied an ordering the maths does not have.
    arrow((3.6, 2.3), (4.85, 2.3))
    ax2.text(4.22, 2.68, "gate", ha="center", va="bottom", fontsize=6.4, color=MUTED)
    ax2.text(4.22, 2.06, "query", ha="center", va="top", fontsize=6.4, color=MUTED)
    arrow((7.6, 4.65), (6.9, 3.05))
    ax2.text(9.35, 3.75, "row\ncontent", ha="center", va="center",
             fontsize=6.2, color=MUTED, linespacing=1.15)
    ax2.text(5.0, 0.78,
             r"$I(\mathrm{future};\,e_t \mid h_t)\ \leq\ I(\mathrm{future};\,w_t \mid h_t)$",
             ha="center", va="center", fontsize=8.4, color=ACCENT)
    ax2.text(5.0, 0.02, "depth, width, linearity and budget do not appear",
             ha="center", va="center", fontsize=6.6, color=MUTED, style="italic")

    _save(fig, out, "fig_window_bound.svg")


# ---------------------------------------------------------------------------
# Thesis figure 2 -- does memory content reach the output?
# ---------------------------------------------------------------------------
DESCRIPTORS = [
    ("tv_label_dist", "answer\nlabel"),
    ("tv_prefix_dist", "leading\nsurface"),
    ("tv_joint_dist", "joint"),
    ("tv_first_token_dist", "first\ntoken"),
]


def figure_content_null(sat: dict, uns: dict, out: Path) -> None:
    fig = plt.figure(figsize=(7.0, 2.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(DESCRIPTORS))
    w = 0.36
    sat_v, sat_f, uns_v, uns_f = [], [], [], []
    for key, _ in DESCRIPTORS:
        s = sat["verdict"]["co_primary"][key]
        u = uns["verdict"]["co_primary"][key]
        sat_v.append(s["trained_123_spread"])
        sat_f.append(s.get("split_half_noise_floor", 0.0) or 0.0)
        uns_v.append(u["trained_123_spread"])
        uns_f.append(u.get("split_half_noise_floor", 0.0) or 0.0)
    sat_v, sat_f = np.array(sat_v), np.array(sat_f)
    uns_v, uns_f = np.array(uns_v), np.array(uns_f)

    ax.bar(x - w / 2, sat_v, w, color=SAT, label="saturated  (n=1500)", zorder=2)
    ax.bar(x + w / 2, uns_v, w, color=UNSAT, label="unsaturated  (n=600)", zorder=2)
    ax.scatter(x - w / 2, sat_f, marker="_", s=95, color=INK, zorder=4, linewidths=1.2)
    ax.scatter(x + w / 2, uns_f, marker="_", s=95, color=INK, zorder=4, linewidths=1.2)

    for xi, v in zip(x + w / 2, uns_v):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom",
                fontsize=6.2, color=UNSAT)
    for xi, v in zip(x - w / 2, sat_v):
        ax.text(xi, 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6.2, color=SAT)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in DESCRIPTORS])
    ax.set_ylabel("cross-corpus total variation")
    ax.set_ylim(0, 0.62)
    ax.set_title("(a)  three corpora, four descriptors", loc="left", pad=5)
    ax.legend(frameon=False, loc="upper left", handlelength=1.0, borderpad=0.15,
              handletextpad=0.5)
    ax.text(0.995, 0.955, "—  split-half floor", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.4, color=INK)

    ax2 = fig.add_subplot(gs[0, 1])
    keys = [("code>wiki:code", "code markers"),
            ("stem>wiki:math", "math markers"),
            ("wiki>code:prose", "prose markers")]
    ys = np.arange(len(keys))[::-1]
    h = 0.30
    for (k, lab), y in zip(keys, ys):
        s, u = sat["directional"][k], uns["directional"][k]
        ax2.barh(y + h / 2 + 0.02, s["mean_diff"], h, color=SAT, zorder=2)
        ax2.barh(y - h / 2 - 0.02, u["mean_diff"], h, color=UNSAT, zorder=2)
        for val, yy, col, p in ((s["mean_diff"], y + h / 2 + 0.02, SAT, s["one_sided_p"]),
                                (u["mean_diff"], y - h / 2 - 0.02, UNSAT, u["one_sided_p"])):
            ptxt = "n.s." if not (p < 0.05) else (f"$p$={p:.0e}" if p < 1e-3 else f"$p$={p:.3f}")
            ax2.text(val + 0.02, yy, ptxt, va="center", ha="left", fontsize=6.0, color=col)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([lab for _, lab in keys])
    ax2.axvline(0, color=INK, lw=0.6)
    ax2.set_xlim(-0.04, 1.30)
    ax2.set_xticks([0, 0.2, 0.4, 0.6])
    ax2.set_xlabel("paired difference in marker rate")
    ax2.set_title("(b)  pre-registered lexical predictions", loc="left", pad=5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SAT),
               plt.Rectangle((0, 0), 1, 1, color=UNSAT)]
    ax2.legend(handles, ["saturated", "unsaturated"], frameon=False, loc="upper right",
               handlelength=1.0, borderpad=0.15, handletextpad=0.5)

    _save(fig, out, "fig_content_null.svg")


# ---------------------------------------------------------------------------
# Thesis figure 3 -- the read-out collapses the window to a constant
# ---------------------------------------------------------------------------
def figure_readout_collapse(va: dict, out: Path) -> None:
    e = va["exploratory"]
    fig = plt.figure(figsize=(7.0, 2.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.35], wspace=0.42)

    # (a) effective dimensionality
    ax = fig.add_subplot(gs[0, 0])
    rand = e["PR_h_after_random_linear_map"]
    vals = [e["PR_h"], float(np.mean(rand)), e["PR_c"]]
    errs = [0.0, float(np.std(rand)), 0.0]
    cols = [GRID, GRID, ACCENT]
    ax.bar(range(3), vals, 0.62, color=cols, zorder=2,
           yerr=errs, error_kw=dict(lw=0.7, capsize=2, ecolor=INK))
    ax.axhline(1.0, color=INK, ls=":", lw=0.7, zorder=1)
    ax.text(2.45, 0.93, "rank 1", fontsize=6.0, color=MUTED, ha="right", va="top")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.07 + errs[i], f"{v:.3f}", ha="center", va="bottom",
                fontsize=6.5, color=ACCENT if i == 2 else INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["$h_t$", "$h_t$\n+random\nmap", "$c_t$"], fontsize=6.4)
    ax.set_ylabel("participation ratio")
    ax.set_ylim(0, 1.85)
    ax.set_title("(a)  effective dimensionality", loc="left", pad=5)

    # (b) leading-direction share
    ax2 = fig.add_subplot(gs[0, 1])
    tc, th = e["top_eig_share_c"], e["top_eig_share_h"]
    ax2.bar(0, th, 0.55, color=GRID, zorder=2)
    ax2.bar(0, 1 - th, 0.55, bottom=th, color="#f4f4f5", zorder=2)
    ax2.bar(1, tc, 0.55, color=ACCENT, zorder=2)
    ax2.bar(1, 1 - tc, 0.55, bottom=tc, color="#fee2e2", zorder=2)
    ax2.text(0, th / 2, f"{th * 100:.1f}%", ha="center", va="center",
             fontsize=6.4, color=INK)
    ax2.text(1, tc / 2, f"{tc * 100:.2f}%", ha="center", va="center",
             fontsize=6.4, color="white")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["$h_t$", "$c_t$"], fontsize=7)
    ax2.set_ylim(0, 1.14)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_ylabel("share of variance")
    ax2.set_title("(b)  leading direction", loc="left", pad=5)

    # (c) what the rows contribute and what the prompt costs
    ax3 = fig.add_subplot(gs[0, 2])
    cos_ws = e["cos_real_vs_shuffled_rows"]["mean"]
    cross = e["cross_template_cosine_boolq_vs_short"]
    within = float(np.mean(list(e["within_task_cosine_c"].values())))
    labels = ["permute the\nretrieved rows",
              "change the\nprompt template",
              "change the item\n(same template)"]
    degs = [np.degrees(np.arccos(min(1.0, v))) for v in (cos_ws, cross, within)]
    cols2 = [ACCENT, "#0891b2", MUTED]
    ax3.barh(range(3), degs, 0.5, color=cols2, zorder=2)
    for i, d in enumerate(degs):
        ax3.text(d + 0.12, i, f"{d:.1f}°", va="center", ha="left",
                 fontsize=6.4, color=cols2[i])
    ax3.set_yticks(range(3))
    ax3.set_yticklabels(labels, fontsize=6.3)
    ax3.invert_yaxis()
    ax3.set_xlabel("angle between injected vectors")
    ax3.set_xlim(0, 12.5)
    ax3.grid(axis="x", color=GRID, lw=0.5, zorder=1)
    ax3.set_axisbelow(True)
    ax3.set_title("(c)  what moves $c_t$", loc="left", pad=5)
    ax3.text(0.98, 0.06, "≈2°: the whole item", transform=ax3.transAxes,
             ha="right", va="bottom", fontsize=6.0, color=MUTED)

    _save(fig, out, "fig_readout_collapse.svg")


# ---------------------------------------------------------------------------
# System figures (appendix) -- kept reproducible, restyled
# ---------------------------------------------------------------------------
def figure_system_figures(out: Path) -> None:
    # Table 2 of the paper records five seeds for the 10k condition, but per-seed
    # JSONs survive for only three of them (0, 1, 2).  We read the JSONs where
    # they exist and *cross-check* them against the recorded table rather than
    # trusting either alone: if a retained artifact disagrees with the published
    # row, this raises instead of quietly changing the figure.  Seeds without an
    # artifact fall back to the recorded row, and the discrepancy is reported in
    # the paper's limitations.
    recorded = {
        0: (3.148, 3.051, 2.930),
        1: (3.328, 3.346, 3.114),
        2: (3.451, 3.426, 3.002),
        3: (3.238, 3.066, 2.848),
        4: (3.197, 3.041, 2.913),
    }
    seeds, deltas, sources = [], [], []
    for s, (b_rec, f_rec, p_rec) in recorded.items():
        path = REPO / f"outputs/ple-projector-10k-seed{s}.json"
        if path.exists():
            pe = json.loads(path.read_text())["paired_eval"]
            got = (round(pe["base_nll"], 3), round(pe["fixed_nll"], 3),
                   round(pe["proj_nll"], 3))
            if got != (b_rec, f_rec, p_rec):
                raise SystemExit(
                    f"seed {s}: retained artifact {got} disagrees with the "
                    f"published table row {(b_rec, f_rec, p_rec)}; resolve before "
                    f"regenerating figures")
            deltas.append(pe["delta_proj_vs_fixed_nll"])
            sources.append("artifact")
        else:
            deltas.append(f_rec - p_rec)
            sources.append("table")
        seeds.append(s)
    print(f"[figs] 10k per-seed provenance: "
          f"{sum(1 for x in sources if x == 'artifact')} artifact, "
          f"{sum(1 for x in sources if x == 'table')} table-only")

    fig, ax = plt.subplots(figsize=(3.35, 2.1))
    colors = [BLUE if x == "artifact" else "#93b4d8" for x in sources]
    ax.bar([str(s) for s in seeds], deltas, 0.6, color=colors, zorder=2)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xlabel("seed")
    ax.set_ylabel("projector − fixed NLL")
    ax.set_title("10k local continuation, five seeds", loc="left", pad=5)
    _save(fig, out, "fig_10k_improvement.svg")

    d = _load("outputs/humaneval-real-20-fast.json")["summary"]
    labels = ["base", "BM25+PLE"]
    pass1 = [d["base"]["pass@1"], d["bm25_ple"]["pass@1"]]
    rep = [d["base"]["mean_repetition_rate"], d["bm25_ple"]["mean_repetition_rate"]]
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.75))
    for ax_, vals, title in ((axes[0], pass1, "pass@1"), (axes[1], rep, "repetition")):
        ax_.bar(labels, vals, 0.6, color=[GREEN, RED], zorder=2)
        ax_.set_title(title, loc="left", pad=4, fontsize=7.5)
        ax_.tick_params(labelsize=6.4)
    fig.subplots_adjust(wspace=0.45)
    _save(fig, out, "fig_humaneval.svg")

    rows = [_load(f"outputs/ms-purified-mora-s{s}.json")["combos"] for s in range(3)]
    combos = list(rows[0].keys())
    tasks = ["knowledge", "arithmetic", "code-output"]
    means = {c: {t: sum(r[c]["summary"][t]["answer_logprob"] for r in rows) / 3
                 for t in tasks} for c in combos}
    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    x = np.arange(len(combos))
    width = 0.26
    for i, t in enumerate(tasks):
        ax.bar(x + i * width, [means[c][t] for c in combos], width,
               label=t, color=[BLUE, RED, GREEN][i], zorder=2)
    ax.set_xticks(x + width)
    ax.set_xticklabels(combos, rotation=35, ha="right", fontsize=6.2)
    ax.set_ylabel("mean answer log-prob")
    ax.legend(frameon=False, fontsize=6.2, handlelength=0.9, borderpad=0.15)
    ax.set_title("Joint system, three seeds", loc="left", pad=5)
    _save(fig, out, "fig_joint_system.svg")

    he = _load("outputs/llm-judge-humaneval.json")
    tq = _load("outputs/llm-judge-triviaqa20.json")
    base_he = [r["judge_score"] for r in he["rows"] if r["condition"] == "base"]
    ple_he = [r["judge_score"] for r in he["rows"] if r["condition"] == "bm25_ple"]
    tq_scores = [r["judge_score"] for r in tq["rows"]]
    scores = [np.mean(base_he), np.mean(ple_he), np.mean(tq_scores)]
    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    ax.bar(["HE\nbase", "HE\nBM25+PLE", "TQA\nbase"], scores, 0.6,
           color=[BLUE, RED, GREEN], zorder=2)
    ax.set_ylim(0, max(1.0, max(scores) * 1.25))
    ax.set_ylabel("mean judge score (0–5)")
    ax.set_title("LLM-as-judge", loc="left", pad=5)
    _save(fig, out, "fig_judge.svg")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="paper/figures")
    ap.add_argument("--only", choices=["all", "thesis", "system"], default="all")
    args = ap.parse_args(argv)
    out = Path(args.outdir)
    house_style()

    if args.only in ("all", "thesis"):
        figure_window_and_bound(
            _load("tests/golden/qwen38_flash_next_text_config.json"), out)
        figure_content_null(
            _load("outputs/round162-0.8B/analysis.json"),
            _load("outputs/round162-0.8B-nosft600/analysis.json"), out)
        figure_readout_collapse(_load("outputs/round164/vector-analysis.json"), out)
    if args.only in ("all", "system"):
        figure_system_figures(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
