#!/usr/bin/env python3
"""Smoke test for the reusable disk-first LiveETDataset data flow.

This script is the Track-A entry point: three lines connect any experiment to
the live-store stream without materializing a full ``e_t`` array.

Usage:

    PYTHONPATH=src:/path/to/EngramDB/python \\
    python scripts/run_live_et_dataset_smoke.py \\
        --rows-dir "/Volumes/My Passport/qwen38-rows" \\
        --tokens-npy /tmp/tokens.npy \\
        --model-dir "/Volumes/My Passport/qwen38-ple" \\
        --seq-len 128 --max-batches 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qwen35_ple.live_store import LiveETDataset, LiveETStore
from qwen35_ple.real_ple import resolve_ple_weight_scale, rowids_from_tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows-dir", default="/Volumes/My Passport/qwen38-rows"
    )
    parser.add_argument("--tokens-npy", default=None)
    parser.add_argument("--model-dir", default="/Volumes/My Passport/qwen38-ple")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.tokens_npy:
        tokens = np.load(args.tokens_npy).astype(np.int64)
    else:
        tokens = np.arange(1000, dtype=np.int64)
    print(f"[live-et] tokens={len(tokens)}")

    scale = resolve_ple_weight_scale(
        model_dir=args.model_dir, scale=args.scale
    )
    print("[live-et] building rowids ...")
    t0 = time.perf_counter()
    rowids = rowids_from_tokens(tokens)
    print(f"[live-et] rowids={rowids.shape} scale={scale:.6g} "
          f"rowid_s={time.perf_counter() - t0:.2f}s")

    import engramdb

    store = engramdb.Store(
        args.rows_dir,
        shards=128,
        rows_per_shard=2_500_012,
        width=160,
    )
    live = LiveETStore(
        store,
        rowids,
        scale,
        store_path=args.rows_dir,
        shards=128,
        rows_per_shard=2_500_012,
        width=160,
    )
    try:
        dataset = LiveETDataset(
            tokens,
            live,
            seq_len=args.seq_len,
            step=args.step,
            control=args.control,
            seed=args.seed,
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

        batches = []
        t0 = time.perf_counter()
        for i, batch in enumerate(iterator):
            if args.max_batches and i >= args.max_batches:
                break
            batches.append(
                {
                    "start": batch.start,
                    "length": batch.length,
                    "tokens_shape": list(batch.tokens.shape),
                    "e_t_shape": list(batch.e_t.shape),
                    "fetch_seconds": batch.fetch_seconds,
                    "rows": batch.rows,
                }
            )
        elapsed = time.perf_counter() - t0

        result = {
            "config": {
                "rows_dir": args.rows_dir,
                "tokens": len(tokens),
                "seq_len": args.seq_len,
                "step": args.step or args.seq_len,
                "control": args.control,
                "seed": args.seed,
                "workers": args.workers,
            },
            "batch_count": len(batches),
            "elapsed_seconds": elapsed,
            "fetch_stats": live.stats.as_dict(),
            "batches": batches,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.output:
            Path(args.output).write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"[live-et] wrote {args.output}")
        print("LIVE_ET_DATASET_SMOKE_OK")
        return 0
    finally:
        live.close()


if __name__ == "__main__":
    raise SystemExit(main())
