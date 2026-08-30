#!/usr/bin/env python3
"""Real-PLE knowledge probe: can frozen e_t features classify semantic categories?

This is the cheapest scientific check before training anything.  It uses the
real PLE table to build per-segment mean ``e_t`` features and trains a simple
ridge/linear classifier.  If PLE features contain useful world-knowledge-like
signal, even a linear probe should beat random-label control.

Usage:
    PYTHONPATH=src:../EngramDB/python:../engram-peft/src \\
    python scripts/run_ple_knowledge_probe.py \\
        --rows-dir "/Volumes/My Passport/qwen38-rows" \\
        --tokenizer data/models/Qwen3.5-0.8B \\
        --output outputs/ple-probe.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qwen35_ple.real_ple import (
    fetch_e_t,
    resolve_ple_weight_scale,
    rowids_from_tokens,
    tokenize_texts_with_offsets,
)

LABELED_CORPUS: dict[str, list[str]] = {
    "capital": [
        "The capital of France is Paris.",
        "The capital of Japan is Tokyo.",
        "The capital of Italy is Rome.",
        "The capital of Spain is Madrid.",
        "The capital of Germany is Berlin.",
        "The capital of China is Beijing.",
    ],
    "planet": [
        "Jupiter is the largest planet.",
        "Mars is the red planet.",
        "Venus is the hottest planet.",
        "Saturn has beautiful rings.",
        "Mercury is closest to the Sun.",
        "Neptune is a cold blue planet.",
    ],
    "element": [
        "Gold has chemical symbol Au.",
        "Oxygen has chemical symbol O.",
        "Iron has chemical symbol Fe.",
        "Helium has chemical symbol He.",
        "Carbon has chemical symbol C.",
        "Silver has chemical symbol Ag.",
    ],
    "arithmetic": [
        "Two plus two equals four.",
        "Three plus five equals eight.",
        "Six times seven equals forty-two.",
        "Ten minus three equals seven.",
        "Nine divided by three equals three.",
        "Twelve plus fifteen equals twenty-seven.",
    ],
    "animal": [
        "A cat says meow.",
        "A dog says woof.",
        "An elephant is very large.",
        "A whale lives in the ocean.",
        "An eagle flies high in the sky.",
        "A rabbit can jump quickly.",
    ],
    "color": [
        "The sky is blue.",
        "Grass is green.",
        "Blood is red.",
        "Lemons are yellow.",
        "Coal is black.",
        "Snow is white.",
    ],
}


def _segment_features(
    e_t: np.ndarray, offsets: list[tuple[int, int]]
) -> np.ndarray:
    feats = []
    for start, end in offsets:
        seg = e_t[start:end]
        if len(seg) == 0:
            continue
        feats.append(seg.mean(axis=0))
    return np.asarray(feats, dtype=np.float32)


def _zscore(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mean) / std, mean, std


def _ridge_fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    lam: float = 1.0,
) -> np.ndarray:
    eye = np.eye(x_train.shape[1]) * lam
    y_onehot = np.eye(n_classes)[y_train]
    w = np.linalg.solve(x_train.T @ x_train + eye, x_train.T @ y_onehot)
    return w


def _ridge_acc(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    pred = np.argmax(x @ w, axis=1)
    return float((pred == y).mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows-dir",
        default="/Volumes/My Passport/qwen38-rows",
    )
    parser.add_argument(
        "--tokenizer",
        default="data/models/Qwen3.5-0.8B",
    )
    parser.add_argument("--output", default="outputs/ple-probe.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model-dir",
        default="/Volumes/My Passport/qwen38-ple",
        help="Qwen3.8-Flash-Next checkpoint dir (reads FP8 weight_scale)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="explicit FP8 weight_scale; overrides discovery",
    )
    args = parser.parse_args()

    texts: list[str] = []
    labels: list[int] = []
    categories = sorted(LABELED_CORPUS)
    for cat_id, cat in enumerate(categories):
        for text in LABELED_CORPUS[cat]:
            texts.append(text)
            labels.append(cat_id)
    labels_arr = np.asarray(labels, dtype=np.int64)

    print(f"[probe] labeled segments={len(texts)} categories={len(categories)}")
    tokens, offsets = tokenize_texts_with_offsets(args.tokenizer, texts)
    print(f"[probe] tokens={len(tokens)}")
    rowids = rowids_from_tokens(tokens)
    print("[probe] fetching real FP8 e_t ...")
    t0 = time.time()
    scale = resolve_ple_weight_scale(model_dir=args.model_dir, scale=args.scale)
    e_t = fetch_e_t(args.rows_dir, rowids, scale=scale)
    print(f"[probe] fetched e_t shape={e_t.shape} in {time.time() - t0:.2f}s")

    feats = _segment_features(e_t, offsets)
    x, mean, std = _zscore(feats)
    n = len(x)
    n_train = int(n * 0.7)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    x_train = x[train_idx]
    y_train = labels_arr[train_idx]
    x_test = x[test_idx]
    y_test = labels_arr[test_idx]

    n_classes = len(categories)
    w = _ridge_fit(x_train, y_train, n_classes, lam=1.0)
    train_acc = _ridge_acc(x_train, y_train, w)
    test_acc = _ridge_acc(x_test, y_test, w)

    # Shuffled-label control: destroys label-feature correspondence.
    control_accs: list[float] = []
    for seed in range(3):
        r = np.random.default_rng(seed)
        shuffled = r.permutation(y_train)
        wc = _ridge_fit(x_train, shuffled, n_classes, lam=1.0)
        control_accs.append(_ridge_acc(x_test, y_test, wc))

    result = {
        "categories": categories,
        "num_segments": n,
        "num_tokens": len(tokens),
        "feature_dim": int(x.shape[1]),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "random_baseline": 1.0 / n_classes,
        "shuffled_label_controls": control_accs,
        "metadata": {
            "rows_dir": args.rows_dir,
            "tokenizer": args.tokenizer,
            "e_t_mean": float(e_t.mean()),
            "e_t_std": float(e_t.std()),
            "weight_scale": scale,
            "finite": bool(np.isfinite(e_t).all()),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[probe] saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
