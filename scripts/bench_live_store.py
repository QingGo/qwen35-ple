#!/usr/bin/env python3
"""Reproducible live-store PLE read benchmark for EngramDB v0.2.9+.

Measures the actual paths used by qwen35-ple live training/precompute:

* ``Store.fetch`` raw bytes
* ``PleDiskGather.fetch`` (now also a direct Store.fetch fast path)
* ``engramdb.fetch_e_t_tensor`` (Store.fetch + torch conversion)
* ``fetch_e_t_tensor(dedup=True)`` (optional dedup path)

This is deliberately not a full model benchmark.  It is the first
Track-0 harness: fixed token count, repeated warm calls, CSV output.

Usage:

    PYTHONPATH=src:../EngramDB/python \\
    python scripts/bench_live_store.py \\
        --rows-dir "/Volumes/My Passport/qwen38-rows" \\
        --tokens 20000 --reps 3 --csv /tmp/live-store-bench.csv

Output:

    LIVE_STORE_BENCH_OK
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from typing import Any

import engramdb
from engramdb.vllm import PleDiskGather, fetch_e_t_tensor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-dir", default="/Volumes/My Passport/qwen38-rows")
    parser.add_argument("--tokens", type=int, default=20_000)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--start-token", type=int, default=100)
    parser.add_argument("--shards", type=int, default=128)
    parser.add_argument("--rows-per-shard", type=int, default=2_500_012)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--max-store-s", type=float, default=None)
    parser.add_argument("--max-tensor-s", type=float, default=None)
    parser.add_argument("--max-tensor-dedup-s", type=float, default=None)
    args = parser.parse_args()

    store = engramdb.Store(
        args.rows_dir,
        shards=args.shards,
        rows_per_shard=args.rows_per_shard,
        width=args.width,
    )
    try:
        tokens = list(range(args.start_token, args.start_token + args.tokens))
        t0 = time.perf_counter()
        rowids = engramdb.rowids_for_seq(tokens)
        flat = [r for row in rowids for r in row]
        rowid_s = time.perf_counter() - t0

        print(
            f"[bench] tokens={args.tokens} flat_rows={len(flat)} "
            f"rowid_s={rowid_s:.3f}s"
        )

        # Warm-up: force first-call / page-cache effects out of measured reps.
        for i in range(args.warmup):
            fetch_e_t_tensor(
                store,
                flat,
                scale=args.scale,
                num_heads=16,
                head_dim=160,
            )
            print(f"[bench] warmup {i + 1}/{args.warmup} done")

        rows: list[dict[str, Any]] = []
        gather = PleDiskGather(store, row_bytes=args.width)
        for rep in range(args.reps):
            t = time.perf_counter()
            raw = store.fetch(flat)
            store_fetch_s = time.perf_counter() - t

            t = time.perf_counter()
            raw2 = gather.fetch(flat)
            ple_gather_s = time.perf_counter() - t
            assert raw == raw2

            t = time.perf_counter()
            arr = fetch_e_t_tensor(
                store,
                flat,
                scale=args.scale,
                num_heads=16,
                head_dim=160,
            )
            fetch_tensor_s = time.perf_counter() - t

            t = time.perf_counter()
            arr_dedup = fetch_e_t_tensor(
                store,
                flat,
                scale=args.scale,
                num_heads=16,
                head_dim=160,
                dedup=True,
            )
            fetch_tensor_dedup_s = time.perf_counter() - t
            assert arr.shape == arr_dedup.shape

            row = {
                "rep": rep,
                "tokens": args.tokens,
                "flat_rows": len(flat),
                "store_fetch_s": store_fetch_s,
                "ple_gather_s": ple_gather_s,
                "fetch_tensor_s": fetch_tensor_s,
                "fetch_tensor_dedup_s": fetch_tensor_dedup_s,
            }
            rows.append(row)
            print(
                f"[bench:{rep}] store={store_fetch_s:.3f}s "
                f"gather={ple_gather_s:.3f}s "
                f"tensor={fetch_tensor_s:.3f}s "
                f"tensor_dedup={fetch_tensor_dedup_s:.3f}s"
            )

        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[csv] wrote {args.csv}")

        med = {
            "store_fetch_s": statistics.median(r["store_fetch_s"] for r in rows),
            "ple_gather_s": statistics.median(r["ple_gather_s"] for r in rows),
            "fetch_tensor_s": statistics.median(r["fetch_tensor_s"] for r in rows),
            "fetch_tensor_dedup_s": statistics.median(
                r["fetch_tensor_dedup_s"] for r in rows
            ),
        }
        print(
            f"[summary] median store={med['store_fetch_s']:.3f}s "
            f"gather={med['ple_gather_s']:.3f}s "
            f"tensor={med['fetch_tensor_s']:.3f}s "
            f"tensor_dedup={med['fetch_tensor_dedup_s']:.3f}s"
        )

        failures: list[str] = []
        if args.max_store_s is not None and med["store_fetch_s"] > args.max_store_s:
            failures.append(
                f"store_fetch {med['store_fetch_s']:.3f}s > {args.max_store_s:.3f}s"
            )
        if args.max_tensor_s is not None and med["fetch_tensor_s"] > args.max_tensor_s:
            failures.append(
                f"fetch_tensor {med['fetch_tensor_s']:.3f}s > {args.max_tensor_s:.3f}s"
            )
        if (
            args.max_tensor_dedup_s is not None
            and med["fetch_tensor_dedup_s"] > args.max_tensor_dedup_s
        ):
            failures.append(
                "fetch_tensor_dedup "
                f"{med['fetch_tensor_dedup_s']:.3f}s > {args.max_tensor_dedup_s:.3f}s"
            )

        if failures:
            for f in failures:
                print(f"[threshold] FAIL: {f}")
            print("LIVE_STORE_BENCH_FAIL")
            return 1

        print("LIVE_STORE_BENCH_OK")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
