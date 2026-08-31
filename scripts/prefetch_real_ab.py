#!/usr/bin/env python3
"""Low-resource prefetch A/B smoke against the real EngramDB Store.

This does not load a full model.  It measures the wall time of:

* synchronous DiskPle forward (disk fetch on the critical path);
* prefetch + simulated earlier-layer compute + forward (disk fetch overlaps with
  the simulated compute window).

Usage:

    PYTHONPATH=src:../EngramDB/python \\
    python scripts/prefetch_real_ab.py \\
        --checkpoint "/Volumes/My Passport/qwen38-ple" \\
        --store "/Volumes/My Passport/qwen38-rows" \\
        --tokens 64 --compute-ms 20
"""

from __future__ import annotations

import argparse
import time

import torch

import engramdb
from engramdb.ple_adapter import (
    DiskPleNGramEmbedding,
    PLE_EOS,
    head_offsets,
    head_vocab_sizes,
    padded_vocab_size,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/Volumes/My Passport/qwen38-ple")
    parser.add_argument("--store", default="/Volumes/My Passport/qwen38-rows")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--compute-ms", type=float, default=20.0)
    parser.add_argument("--shards", type=int, default=128)
    parser.add_argument("--rows-per-shard", type=int, default=2_500_012)
    parser.add_argument("--width", type=int, default=160)
    args = parser.parse_args()

    info = engramdb.discover_ple(args.checkpoint)
    if info is None:
        raise SystemExit("no PLE metadata")
    multipliers = info.get("layer_multipliers") or info.get("rowid_multipliers")
    sizes = head_vocab_sizes(heads=16)
    offsets = head_offsets(sizes)
    scale = float(info.get("weight_scale") or 1.0)

    token_ids = [100 + i for i in range(args.tokens)]
    if args.tokens > 128:
        token_ids[128] = PLE_EOS
    tokens = torch.tensor([token_ids], dtype=torch.long)

    def make_disk():
        return DiskPleNGramEmbedding(
            store=engramdb.Store(
                args.store,
                shards=args.shards,
                rows_per_shard=args.rows_per_shard,
                width=args.width,
            ),
            num_embeddings=padded_vocab_size(sizes),
            embedding_dim=int(info["ple_embed_dim"]),
            num_heads=16,
            layer_multipliers=multipliers,
            scale=scale,
            dtype=torch.float8_e4m3fn,
            cache_size=0,
            eos=PLE_EOS,
            prime_sizes=sizes,
            offsets=offsets,
            ngram_size=3,
            heads_per_ngram=8,
        )

    # Sync path.
    disk_sync = make_disk()
    try:
        t0 = time.perf_counter()
        _ = disk_sync(tokens, None)
        sync_s = time.perf_counter() - t0
        sync_stats = disk_sync.table.get_stats()
    finally:
        disk_sync.store.close()

    # Prefetch path + simulated earlier-layer compute.
    disk_pre = make_disk()
    try:
        t0 = time.perf_counter()
        disk_pre.prefetch(tokens)
        time.sleep(args.compute_ms / 1000.0)
        _ = disk_pre(tokens, None)
        prefetch_s = time.perf_counter() - t0
        pre_stats = disk_pre.table.get_stats()
    finally:
        disk_pre.store.close()

    print(f"[sync]     total={sync_s * 1000:.3f}ms fetch_s={sync_stats['fetch_s'] * 1000:.3f}ms")
    print(
        f"[prefetch] total={prefetch_s * 1000:.3f}ms "
        f"fetch_s={pre_stats['fetch_s'] * 1000:.3f}ms "
        f"wait_s={pre_stats['prefetch_wait_s'] * 1000:.3f}ms "
        f"issued={int(pre_stats['prefetch_issued'])}"
    )
    print(
        f"[delta]    sync - prefetch = {(sync_s - prefetch_s) * 1000:.3f}ms "
        f"(compute window {args.compute_ms:.1f}ms)"
    )
    print("PREFETCH_AB_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
