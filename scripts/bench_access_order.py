#!/usr/bin/env python3
"""Access-order scheduling A/B benchmark.

Compares the existing token-order Store-P read path against
``access_order=True`` (sorted physical slot reads + scatter back).

Usage:

    PYTHONPATH=src:/path/to/EngramDB/python \
    python scripts/bench_access_order.py \
        --view /tmp/corpus.view \
        --slot-indices-npy /tmp/corpus.slot_indices.npy \
        --tokens 100000 --seq-len 128 --step 128 --reps 3 \
        --csv /tmp/access-order.csv

Outputs a CSV with per-mode median fetch seconds and a threshold check:

* ACCESS_ORDER_BENCH_OK  if sorted_time <= max_ratio * naive_time
* ACCESS_ORDER_BENCH_FAIL otherwise
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

from qwen35_ple.live_store import LiveETDataset, LiveETViewStore
from qwen35_ple.real_ple import resolve_ple_weight_scale


def _run_once(
    view_store: LiveETViewStore,
    tokens: np.ndarray,
    seq_len: int,
    step: int,
    *,
    access_order: bool,
) -> float:
    if access_order:
        view_store.access_order = True
    else:
        view_store.access_order = False
    ds = LiveETDataset(
        tokens,
        view_store,
        seq_len=seq_len,
        step=step,
    )
    t0 = time.perf_counter()
    total_fetch = 0.0
    for batch in ds:
        total_fetch += batch.fetch_seconds
    wall = time.perf_counter() - t0
    # Return wall time; fetch time is also recorded separately when needed.
    _ = total_fetch
    return wall


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", required=True)
    parser.add_argument("--slot-indices-npy", required=True)
    parser.add_argument("--tokens-npy", default=None)
    parser.add_argument("--tokens", type=int, default=100_000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--model-dir", default="/Volumes/My Passport/qwen38-ple")
    parser.add_argument("--max-ratio", type=float, default=1.5)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    if args.tokens_npy:
        tokens = np.load(args.tokens_npy).astype(np.int64)
        tokens = tokens[: args.tokens]
    else:
        tokens = np.arange(args.tokens, dtype=np.int64)

    slot_indices = np.load(args.slot_indices_npy).astype(np.int64)
    if len(slot_indices) < len(tokens):
        raise ValueError(
            f"slot_indices length {len(slot_indices)} < tokens {len(tokens)}"
        )
    slot_indices = slot_indices[: len(tokens)]

    scale = resolve_ple_weight_scale(model_dir=args.model_dir, scale=args.scale)
    import engramdb

    view = engramdb.View(args.view)
    store = LiveETViewStore(
        view,
        slot_indices,
        scale,
        num_heads=16,
        head_dim=160,
        embedding_dim=2560,
        view_path=args.view,
    )
    step = args.step or args.seq_len

    rows: list[dict[str, float | int]] = []
    for rep in range(args.reps):
        naive = _run_once(store, tokens, args.seq_len, step, access_order=False)
        ordered = _run_once(store, tokens, args.seq_len, step, access_order=True)
        rows.append(
            {
                "rep": rep,
                "tokens": len(tokens),
                "seq_len": args.seq_len,
                "step": step,
                "naive_s": naive,
                "ordered_s": ordered,
                "ratio": ordered / naive if naive > 0 else float("nan"),
            }
        )
        print(
            f"[bench:{rep}] naive={naive:.4f}s ordered={ordered:.4f}s "
            f"ratio={rows[-1]['ratio']:.3f}"
        )

    median_naive = statistics.median(r["naive_s"] for r in rows)
    median_ordered = statistics.median(r["ordered_s"] for r in rows)
    ratio = median_ordered / median_naive if median_naive > 0 else float("nan")
    print(f"[summary] median naive={median_naive:.4f}s ordered={median_ordered:.4f}s ratio={ratio:.3f}")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[csv] wrote {path}")

    if ratio <= args.max_ratio:
        print("ACCESS_ORDER_BENCH_OK")
        return 0
    print(f"ACCESS_ORDER_BENCH_FAIL: ratio {ratio:.3f} > max_ratio {args.max_ratio}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
