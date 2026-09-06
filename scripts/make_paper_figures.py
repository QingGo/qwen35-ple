#!/usr/bin/env python3
"""Generate paper figures from experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    out = Path("figures")
    out.mkdir(parents=True, exist_ok=True)

    # Figure 1: 10k projector vs fixed improvements.
    seeds = [0, 1, 2]
    deltas = []
    for s in seeds:
        d = _load(f"outputs/ple-projector-10k-seed{s}.json")
        p = d["paired_eval"]
        deltas.append(p["delta_proj_vs_fixed_nll"])
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([str(s) for s in seeds], deltas, color="#4C72B0")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Projector vs fixed NLL improvement")
    ax.set_title("10k Data: Learned Projector Improvement")
    fig.tight_layout()
    fig.savefig(out / "fig_10k_improvement.png", dpi=200)
    plt.close(fig)

    # Figure 2: HumanEval pass@1 and repetition.
    d = _load("outputs/humaneval-real-20-fast.json")["summary"]
    labels = ["Base", "BM25+PLE"]
    pass1 = [d["base"]["pass@1"], d["bm25_ple"]["pass@1"]]
    rep = [d["base"]["mean_repetition_rate"], d["bm25_ple"]["mean_repetition_rate"]]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes[0].bar(labels, pass1, color=["#55A868", "#C44E52"])
    axes[0].set_ylim(0, max(0.2, max(pass1) * 1.2))
    axes[0].set_title("HumanEval pass@1")
    axes[1].bar(labels, rep, color=["#55A868", "#C44E52"])
    axes[1].set_title("Mean repetition rate")
    fig.tight_layout()
    fig.savefig(out / "fig_humaneval.png", dpi=200)
    plt.close(fig)

    # Figure 3: Joint system table aggregated.
    rows = []
    for s in range(3):
        rows.append(_load(f"outputs/ms-purified-mora-s{s}.json")["combos"])
    combos = list(rows[0].keys())
    tasks = ["knowledge", "arithmetic", "code-output"]
    means = {c: {t: sum(r[c]["summary"][t]["answer_logprob"] for r in rows) / 3 for t in tasks} for c in combos}
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = range(len(combos))
    width = 0.25
    for i, t in enumerate(tasks):
        vals = [means[c][t] for c in combos]
        ax.bar([xi + i * width for xi in x], vals, width=width, label=t)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(combos, rotation=30, ha="right")
    ax.set_ylabel("Mean answer logprob")
    ax.legend()
    ax.set_title("Joint System: Answer Log-Probability (3 seeds)")
    fig.tight_layout()
    fig.savefig(out / "fig_joint_system.png", dpi=200)
    plt.close(fig)

    # Figure 4: LLM judge scores.
    he = _load("outputs/llm-judge-humaneval.json")
    tq = _load("outputs/llm-judge-triviaqa20.json")
    base_he = [r["judge_score"] for r in he["rows"] if r["condition"] == "base"]
    ple_he = [r["judge_score"] for r in he["rows"] if r["condition"] == "bm25_ple"]
    tq_scores = [r["judge_score"] for r in tq["rows"]]
    fig, ax = plt.subplots(figsize=(5, 3))
    scores = [sum(base_he) / len(base_he), sum(ple_he) / len(ple_he), sum(tq_scores) / len(tq_scores)]
    labels = ["HE base", "HE BM25+PLE", "TQA base"]
    ax.bar(labels, scores, color=["#4C72B0", "#C44E52", "#55A868"])
    ax.set_ylim(0, max(1.0, max(scores) * 1.2))
    ax.set_ylabel("Mean LLM judge score (0-5)")
    ax.set_title("LLM-as-Judge Scores")
    fig.tight_layout()
    fig.savefig(out / "fig_judge.png", dpi=200)
    plt.close(fig)

    print(f"Wrote figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
