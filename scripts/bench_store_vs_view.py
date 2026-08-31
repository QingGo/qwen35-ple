#!/usr/bin/env python3
"""Reproducible Store-I vs Store-P live-read benchmark.

This is the Track-B harness skeleton: it measures the actual Python-facing read
paths used by LiveET datasets on the same token set.

Store-I path:
    ``LiveETStore.get`` -> ``engramdb.fetch_e_t_tensor`` -> Store.fetch

Store-P path (when ``--view`` / ``--slot-indices-npy`` are supplied):
    ``LiveETViewStore.get`` -> ``View.read_records`` -> torch dequant

Usage:

    PYTHONPATH=src:/path/to/EngramDB/python \\
    python scripts/bench_store_vs_view.py \\
        --rows-dir "/Volumes/My Passport/qwen38-rows" \\
        --tokens 20000 --reps 3 --csv /tmp/store-vs-view.csv

Store-P example:

    python scripts/bench_store_vs_view.py \\
        --rows-dir "/Volumes/My Passport/qwen38-rows" \\
        --view /tmp/qwen.view.bin \\
        --slot-indices-npy /tmp/qwen.slot_indices.npy \\
        --tokens 20000 --reps 3

Prints ``STORE_VS_VIEW_BENCH_OK`` when all optional thresholds pass.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from typing import Any

import numpy as np

from qwen35_ple.live_store import LiveETStore, LiveETViewStore
from qwen35_ple.real_ple import resolve_ple_weight_scale


def _open_labels() -> list[str]:
    return [
        "rep",
        "tokens",
        "store_fetch_s",
        "live_store_get_s",
        "fetch_tensor_s",
        "view_fetch_s",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-dir", default="/Volumes/My Passport/qwen38-rows")
    parser.add_argument("--view", default=None)
    parser.add_argument("--slot-indices-npy", default=None)
    parser.add_argument("--tokens", type=int, default=20_000)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--start-token", type=int, default=100)
    parser.add_argument("--shards", type=int, default=128)
    parser.add_argument("--rows-per-shard", type=int, default=2_500_012)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--model-dir", default="/Volumes/My Passport/qwen38-ple")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--max-store-s", type=float, default=None)
    parser.add_argument("--max-tensor-s", type=float, default=None)
    parser.add_argument("--max-view-s", type=float, default=None)
    args = parser.parse_args()

    if args.view is not None and args.slot_indices_npy is None:
        raise SystemExit("--view requires --slot-indices-npy")

    scale = resolve_ple_weight_scale(model_dir=args.model_dir, scale=args.scale)
    tokens = list(range(args.start_token, args.start_token + args.tokens))

    import engramdb

    print(
        f"[bench] tokens={len(tokens)} scale={scale:.6g} "
        f"store={args.rows_dir} view={args.view or '-'}"
    )
    t0 = time.perf_counter()
    rowids = engramdb.rowids_for_seq(tokens)
    flat = [r for row in rowids for r in row]
    print(f"[bench] rowids={len(rowids)}x{len(rowids[0])} rowid_s={time.perf_counter() - t0:.3f}s")

    store = engramdb.Store(
        args.rows_dir,
        shards=args.shards,
        rows_per_shard=args.rows_per_shard,
        width=args.width,
    )
    live = LiveETStore(
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
    view_store = None
    view = None
    slot_indices: np.ndarray | None = None
    try:
        # Warm-up.
        for i in range(args.warmup):
            live.get(np.arange(min(1000, len(live)), dtype=np.int64))
            print(f"[bench] warmup {i + 1}/{args.warmup} done")

        if args.view is not None:
            view = engramdb.View(args.view)
            slot_indices = np.load(args.slot_indices_npy).astype(np.int64)
            if len(slot_indices) < len(live):
                raise ValueError(
                    f"slot_indices length {len(slot_indices)} < tokens {len(live)}"
                )
            view_store = LiveETViewStore(
                view,
                slot_indices[: len(live)],
                scale,
                num_heads=16,
                head_dim=args.width,
                embedding_dim=16 * args.width,
            )
            for i in range(args.warmup):
                view_store.get(0, min(1000, len(view_store)))
                print(f"[bench] view warmup {i + 1}/{args.warmup} done")

        rows: list[dict[str, Any]] = []
        for rep in range(args.reps):
            t = time.perf_counter()
            _raw = store.fetch(flat)
            store_fetch_s = time.perf_counter() - t

            t = time.perf_counter()
            live.get(np.arange(len(live), dtype=np.int64))
            live_store_get_s = time.perf_counter() - t

            t = time.perf_counter()
            _arr = engramdb.fetch_e_t_tensor(
                store,
                flat,
                scale=scale,
                num_heads=16,
                head_dim=args.width,
            )
            fetch_tensor_s = time.perf_counter() - t

            view_fetch_s = float("nan")
            if view_store is not None:
                t = time.perf_counter()
                view_store.get(0, len(view_store))
                view_fetch_s = time.perf_counter() - t

            row = {
                "rep": rep,
                "tokens": len(live),
                "store_fetch_s": store_fetch_s,
                "live_store_get_s": live_store_get_s,
                "fetch_tensor_s": fetch_tensor_s,
                "view_fetch_s": view_fetch_s,
            }
            rows.append(row)
            print(
                f"[bench:{rep}] store={store_fetch_s:.3f}s "
                f"live_get={live_store_get_s:.3f}s "
                f"tensor={fetch_tensor_s:.3f}s "
                f"view={view_fetch_s:.3f}s"
            )

        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_open_labels())
                writer.writeheader()
                writer.writerows(rows)
            print(f"[csv] wrote {args.csv}")

        med = {
            "store_fetch_s": statistics.median(r["store_fetch_s"] for r in rows),
            "live_store_get_s": statistics.median(r["live_store_get_s"] for r in rows),
            "fetch_tensor_s": statistics.median(r["fetch_tensor_s"] for r in rows),
        }
        if view_store is not None:
            med["view_fetch_s"] = statistics.median(r["view_fetch_s"] for r in rows)
        print(
            f"[summary] median store={med['store_fetch_s']:.3f}s "
            f"live_get={med['live_store_get_s']:.3f}s "
            f"tensor={med['fetch_tensor_s']:.3f}s "
            f"view={med.get('view_fetch_s', float('nan')):.3f}s"
        )

        failures: list[str] = []
        for key, limit in (
            ("store_fetch_s", args.max_store_s),
            ("fetch_tensor_s", args.max_tensor_s),
            ("view_fetch_s", args.max_view_s),
        ):
            if limit is not None and key in med and med[key] > limit:
                failures.append(f"{key} {med[key]:.3f}s > {limit:.3f}s")

        if failures:
            for f in failures:
                print(f"[threshold] FAIL: {f}")
            print("STORE_VS_VIEW_BENCH_FAIL")
            return 1
        print("STORE_VS_VIEW_BENCH_OK")
        return 0
    finally:
        if view_store is not None:
            view_store.close()
        live.close()


if __name__ == "__main__":
    raise SystemExit(main())
