#!/usr/bin/env python3
"""Lazy per-window live e_t benchmark (Track B/C).

This is the disk-first 1M benchmark harness: instead of fetching the full e_t
array at once, it iterates a LiveETDataset window by window and records the
per-batch fetch time.  It never materializes more than one window in memory.

Store-I mode:
    --rows-dir ... --tokens N [--tokens-npy file]

Store-P mode (raw or access-order slots):
    --rows-dir ... --view /path/to/view.bin --slot-indices-npy file
    (if no slot-indices file, sequential 0..N-1 slots are used for raw I/O A/B)

Outputs CSV of per-window timings and prints summary percentiles.

Usage:
    PYTHONPATH=src:/path/to/EngramDB/python \\
    python scripts/bench_lazy_windows.py \\
        --rows-dir "/Volumes/My Passport/qwen38-rows" \\
        --tokens 100000 --seq-len 128 --step 128 --csv /tmp/lazy-100k.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from qwen35_ple.live_store import LiveETDataset, LiveETStore, LiveETViewStore
from qwen35_ple.real_ple import resolve_ple_weight_scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-dir", default="/Volumes/My Passport/qwen38-rows")
    parser.add_argument("--view", default=None)
    parser.add_argument("--slot-indices-npy", default=None)
    parser.add_argument("--synthetic", action="store_true", help="run on a synthetic Store-P stub")
    parser.add_argument("--synthetic-tokens", type=int, default=1000)
    parser.add_argument("--slot-index", default=None)
    parser.add_argument("--tokens-npy", default=None)
    parser.add_argument("--tokens", type=int, default=100_000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--access-order", action="store_true", help="schedule reads by physical slot order")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shards", type=int, default=128)
    parser.add_argument("--rows-per-shard", type=int, default=2_500_012)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--model-dir", default="/Volumes/My Passport/qwen38-ple")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--max-batch-s", type=float, default=None)
    args = parser.parse_args()

    if args.tokens_npy:
        tokens = np.load(args.tokens_npy).astype(np.int64)
        if len(tokens) > args.tokens:
            tokens = tokens[: args.tokens]
    else:
        n = args.synthetic_tokens if args.synthetic else args.tokens
        tokens = np.arange(n, dtype=np.int64)
    print(
        f"[lazy] tokens={len(tokens)} seq_len={args.seq_len} "
        f"step={args.step or args.seq_len} view={args.view or '-'}"
    )

    scale = resolve_ple_weight_scale(model_dir=args.model_dir, scale=args.scale)

    store = None
    if not args.synthetic:
        import engramdb

        store = engramdb.Store(
            args.rows_dir,
            shards=args.shards,
            rows_per_shard=args.rows_per_shard,
            width=args.width,
        )

    live_store = None
    view_store = None
    dataset = None
    try:
        if args.synthetic:
            rec_len = 16 * args.width
            rng = np.random.default_rng(args.seed)
            slot_indices = rng.permutation(len(tokens)).astype(np.int64)
            view_store = LiveETViewStore(
                None,
                slot_indices,
                scale,
                num_heads=16,
                head_dim=args.width,
                embedding_dim=rec_len,
                access_order=args.access_order,
            )

            def fake_fetch(positions: np.ndarray) -> np.ndarray:
                positions = np.asarray(positions, dtype=np.int64)
                return np.stack(
                    [np.full(rec_len, float(p), dtype=np.float32) for p in positions]
                )

            view_store._fetch = fake_fetch  # type: ignore[method-assign]
            dataset = LiveETDataset(
                tokens,
                view_store,
                seq_len=args.seq_len,
                step=args.step,
                control=args.control,
                seed=args.seed,
                max_windows=args.max_batches,
                access_order=args.access_order,
            )
        elif args.view is not None:
            import engramdb

            view = engramdb.View(args.view)
            if args.slot_index:
                from qwen35_ple.slot_index import SlotIndex

                slot_index = SlotIndex.load(args.slot_index)
                rowids = np.asarray(engramdb.rowids_for_seq(tokens.tolist()), dtype=np.int64)
                slot_indices = slot_index.to_slots(rowids)
            elif args.slot_indices_npy:
                slot_indices = np.load(args.slot_indices_npy).astype(np.int64)
                if len(slot_indices) < len(tokens):
                    raise ValueError(
                        f"slot_indices length {len(slot_indices)} < tokens {len(tokens)}"
                    )
                slot_indices = slot_indices[: len(tokens)]
            else:
                slot_indices = np.arange(len(tokens), dtype=np.int64)
            view_store = LiveETViewStore(
                view,
                slot_indices,
                scale,
                num_heads=16,
                head_dim=args.width,
                embedding_dim=16 * args.width,
                view_path=args.view,
                access_order=args.access_order,
            )
            dataset = LiveETDataset(
                tokens,
                view_store,
                seq_len=args.seq_len,
                step=args.step,
                control=args.control,
                seed=args.seed,
                max_windows=args.max_batches,
                access_order=args.access_order,
            )
        else:
            import engramdb

            print("[lazy] building rowids ...")
            t0 = time.perf_counter()
            rowids = engramdb.rowids_for_seq(tokens.tolist())
            print(
                f"[lazy] rowids={len(rowids)}x{len(rowids[0])} "
                f"rowid_s={time.perf_counter() - t0:.3f}s"
            )
            live_store = LiveETStore(
                store,
                np.asarray(rowids, dtype=np.int64),
                scale,
                store_path=args.rows_dir,
                shards=args.shards,
                rows_per_shard=args.rows_per_shard,
                width=args.width,
                num_heads=16,
                head_dim=args.width,
                embedding_dim=16 * args.width,
            )
            dataset = LiveETDataset(
                tokens,
                live_store,
                seq_len=args.seq_len,
                step=args.step,
                control=args.control,
                seed=args.seed,
                max_windows=args.max_batches,
            )

        if args.workers and args.workers > 0:
            from torch.utils.data import DataLoader

            iterator = DataLoader(
                dataset,
                batch_size=None,
                num_workers=args.workers,
            )
        else:
            iterator = dataset

        rows: list[dict[str, Any]] = []
        t_start = time.perf_counter()
        for batch in iterator:
            rows.append(
                {
                    "batch": len(rows),
                    "start": batch.start,
                    "length": batch.length,
                    "rows": batch.rows,
                    "fetch_seconds": batch.fetch_seconds,
                }
            )
        wall = time.perf_counter() - t_start

        if not rows:
            print("[lazy] no windows produced")
            return 0

        times = [r["fetch_seconds"] for r in rows]
        sorted_times = sorted(times)
        summary = {
            "mode": "synthetic" if args.synthetic else ("view" if args.view is not None else "store"),
            "workers": args.workers,
            "tokens": len(tokens),
            "windows": len(rows),
            "wall_seconds": wall,
            "fetch_total_seconds": float(sum(times)),
            "fetch_mean_s": float(statistics.mean(times)),
            "fetch_p50_s": float(statistics.median(times)),
            "fetch_p90_s": float(sorted_times[min(len(times) - 1, int(len(times) * 0.9))]),
            "fetch_p99_s": float(sorted_times[min(len(times) - 1, int(len(times) * 0.99))]),
            "fetch_max_s": float(max(times)),
        }
        print("[summary]")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        if args.csv:
            Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[csv] wrote {args.csv}")

        if args.max_batch_s is not None and max(times) > args.max_batch_s:
            print(
                f"[threshold] FAIL max_batch {max(times):.3f}s > "
                f"{args.max_batch_s:.3f}s"
            )
            print("LAZY_WINDOWS_BENCH_FAIL")
            return 1
        print("LAZY_WINDOWS_BENCH_OK")
        return 0
    finally:
        if view_store is not None:
            view_store.close()
        if live_store is not None:
            live_store.close()
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
