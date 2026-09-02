#!/usr/bin/env python3
"""Plot article-ready figures from exported Phase 0 metrics CSVs.

Requires ``matplotlib`` and the CSVs produced by ``export_phase0_metrics.py``.

Usage:

    python scripts/plot_phase0_metrics.py \
      --input-dir outputs/paper-metrics \
      --output-dir outputs/paper-figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

MODES = ["no-reader", "control", "real"]
MODE_COLORS = {
    "no-reader": "#1f77b4",
    "control": "#ff7f0e",
    "real": "#2ca02c",
}
TASKS = ["trivia_em", "nq_em", "boolq_em"]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/paper-metrics")
    parser.add_argument("--output-dir", default="outputs/paper-figures")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_csv(in_dir / "train_loss.csv")
    val = load_csv(in_dir / "val_curve.csv")
    summary = load_csv(in_dir / "summary.csv")

    if not summary.empty:
        # 1. QA EM grouped bar by mix/mode.
        mixes = sorted(summary["mix"].unique())
        fig, ax = plt.subplots(figsize=(10, 5))
        width = 0.25
        x = range(len(mixes))
        for i, mode in enumerate(MODES):
            vals = []
            for mix in mixes:
                row = summary[(summary["mix"] == mix) & (summary["mode"] == mode)]
                vals.append(row.iloc[0]["qa_em"] * 100 if not row.empty and pd.notna(row.iloc[0]["qa_em"]) else 0)
            ax.bar([xi + (i - 1) * width for xi in x], vals, width, label=mode, color=MODE_COLORS[mode])
        ax.set_xticks(list(x))
        ax.set_xticklabels(mixes)
        ax.set_ylabel("QA EM (%)")
        ax.set_title("Phase 0 QA EM by mix and mode")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "qa_em_by_mix.png", dpi=150)
        plt.close(fig)

        # 2. Per-task EM grouped bars for real only.
        fig, ax = plt.subplots(figsize=(10, 5))
        width = 0.22
        for i, task in enumerate(TASKS):
            vals = []
            for mix in mixes:
                row = summary[(summary["mix"] == mix) & (summary["mode"] == "real")]
                v = row.iloc[0][task] if not row.empty and pd.notna(row.iloc[0][task]) else None
                vals.append(v * 100 if v is not None and pd.notna(v) else 0)
            ax.bar([xi + (i - 1) * width for xi in x], vals, width, label=task)
        ax.set_xticks(list(x))
        ax.set_xticklabels(mixes)
        ax.set_ylabel("EM (%)")
        ax.set_title("Real arm per-task EM")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "real_task_em.png", dpi=150)
        plt.close(fig)

        # 3. Val loss bar grouped by mix/mode.
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, mode in enumerate(MODES):
            vals = []
            for mix in mixes:
                row = summary[(summary["mix"] == mix) & (summary["mode"] == mode)]
                v = row.iloc[0]["val_loss"] if not row.empty and pd.notna(row.iloc[0]["val_loss"]) else 0
                vals.append(v)
            ax.bar([xi + (i - 1) * width for xi in x], vals, width, label=mode, color=MODE_COLORS[mode])
        ax.set_xticks(list(x))
        ax.set_xticklabels(mixes)
        ax.set_ylabel("val loss")
        ax.set_title("Phase 0 validation loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "val_loss_by_mix.png", dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {out_dir / 'qa_em_by_mix.png'}")
        print(f"[plot] wrote {out_dir / 'real_task_em.png'}")
        print(f"[plot] wrote {out_dir / 'val_loss_by_mix.png'}")

    if not train.empty:
        mixes = sorted(train["mix"].unique())
        n = len(mixes)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, mix in zip(axes, mixes):
            for mode in ["real", "control"]:
                sub = train[(train["mix"] == mix) & (train["mode"] == mode)]
                if sub.empty:
                    continue
                grouped = sub.groupby("step")["train_loss"].mean().reset_index()
                ax.plot(grouped["step"], grouped["train_loss"], label=mode, color=MODE_COLORS[mode], linewidth=1)
            ax.set_title(mix)
            ax.set_xlabel("step")
            ax.set_ylabel("train loss")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "train_loss_curves.png", dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {out_dir / 'train_loss_curves.png'}")

    if not val.empty:
        mixes = sorted(val["mix"].unique())
        n = len(mixes)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, mix in zip(axes, mixes):
            for mode in ["real", "control"]:
                sub = val[(val["mix"] == mix) & (val["mode"] == mode)]
                if sub.empty:
                    continue
                ax.plot(sub["step"], sub["val_loss"], label=mode, color=MODE_COLORS[mode], linewidth=1)
            ax.set_title(mix)
            ax.set_xlabel("step")
            ax.set_ylabel("val loss")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "val_loss_curves.png", dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {out_dir / 'val_loss_curves.png'}")

    print("[plot] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
