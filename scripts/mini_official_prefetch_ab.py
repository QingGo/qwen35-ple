#!/usr/bin/env python3
"""Mini official-model prefetch A/B on the real EngramDB Store.

Low-resource bridge between ``prefetch_real_ab.py`` and a full Qwen4Exp model.
Uses the frozen official ``Qwen4ExpTextPLELayer`` as the real PLE layer,
replaces its giant n-gram embedding with ``DiskPleNGramEmbedding``, and wraps it
in a small model with real dense blocks before and after the PLE layer.

A/B:
  sync     -- no model-level prefetch, PLE disk reads are on the critical path
  prefetch -- install_disk_ple_prefetch_hook, PLE reads overlap with pre-PLE compute

Output is human-readable plus optional CSV, and records whether prefetch had
already completed by the time the PLE layer was entered.

Usage:
    PYTHONPATH=src:../EngramDB/python \\
    python scripts/mini_official_prefetch_ab.py \\
        --checkpoint "/Volumes/My Passport/qwen38-ple" \\
        --store "/Volumes/My Passport/qwen38-rows" \\
        --tokens 64 --pre-layers 12 --hidden 256

Result:
    MINI_OFFICIAL_PREFETCH_AB_OK
"""

from __future__ import annotations

import argparse
import csv
import time
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

import engramdb
from engramdb.official_loader import (
    install_disk_ple_in_official_model,
    install_disk_ple_prefetch_hook,
    patch_official_ngram_embedding_for_disk_load,
)
from engramdb.ple_adapter import PLE_EOS
from qwen35_ple.official_ple_snapshot import (
    Qwen4ExpTextNGramEmbedding,
    Qwen4ExpTextPLELayer,
)


class DenseBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class MiniQwenPleModel(nn.Module):
    def __init__(
        self,
        config: SimpleNamespace,
        hidden_size: int,
        pre_layers: int,
        post_layers: int,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(int(config.vocab_size), hidden_size)
        self.pre = nn.Sequential(*[DenseBlock(hidden_size) for _ in range(pre_layers)])
        with patch_official_ngram_embedding_for_disk_load(
            embedding_class=Qwen4ExpTextNGramEmbedding
        ):
            self.ple = Qwen4ExpTextPLELayer(config, layer_idx=1, ple_layer_index=0)
        self.post = nn.Sequential(*[DenseBlock(hidden_size) for _ in range(post_layers)])
        self.last_timings: dict[str, float] = {}
        self.prefetch_pending_at_ple: int = -1
        self.prefetch_ready_at_ple: bool = False
        self.prefetch_done_at_ple_s: float | None = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        t_start = time.perf_counter()
        h = self.embed(input_ids)
        h = self.pre(h)

        t_before_ple = time.perf_counter()
        disk = self.ple.ple_embedding
        pending = [f for f in disk.table._pending if not f.done()]
        self.prefetch_pending_at_ple = len(pending)
        self.prefetch_ready_at_ple = len(pending) == 0
        fut = getattr(disk, "_last_prefetch_future", None)
        if fut is not None:
            self.prefetch_done_at_ple_s = (
                time.perf_counter() - t_start if fut.done() else None
            )

        h = self.ple(h, input_ids, None)
        t_after_ple = time.perf_counter()
        h = self.post(h)
        t_end = time.perf_counter()

        self.last_timings = {
            "earlier_s": t_before_ple - t_start,
            "ple_s": t_after_ple - t_before_ple,
            "post_s": t_end - t_after_ple,
            "total_s": t_end - t_start,
        }
        return h


def make_config(ple_embed_dim: int, hidden_size: int, vocab_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=8,
        vocab_size=vocab_size,
        ngram_vocab_size_base=20_000_000,
        seed=0,
        eos_token_id=PLE_EOS,
        make_ngram_vocab_size_divisible_by=128,
        hidden_size=hidden_size,
        hc_count=1,
        ple_embed_dim=ple_embed_dim,
        ple_conv_kernel_size=4,
        rms_norm_eps=1e-5,
    )


def run_ab(args: argparse.Namespace) -> list[dict[str, Any]]:
    info = engramdb.discover_ple(args.checkpoint)
    if info is None:
        raise SystemExit(f"no PLE metadata found in {args.checkpoint}")

    ple_embed_dim = int(info["ple_embed_dim"])
    hidden_size = int(args.hidden)
    vocab_size = max(int(args.vocab), int(args.tokens) + 100)
    config = make_config(ple_embed_dim, hidden_size, vocab_size)

    rows: list[dict[str, Any]] = []
    for rep in range(args.reps):
        start = 100 + rep * args.tokens * 7
        token_ids = [start + i for i in range(args.tokens)]
        tokens = torch.tensor([token_ids], dtype=torch.long)

        for mode in ("sync", "prefetch"):
            store = engramdb.Store(
                args.store,
                shards=128,
                rows_per_shard=2_500_012,
                width=160,
            )
            model = None
            disk = None
            try:
                model = MiniQwenPleModel(config, hidden_size, args.pre_layers, args.post_layers)
                replaced = install_disk_ple_in_official_model(
                    model,
                    store,
                    info=info,
                    scale=float(info.get("weight_scale") or 1.0),
                    cache_size=0,
                    prefetch=False,
                )
                assert len(replaced) == 1, replaced
                disk = model.ple.ple_embedding
                disk.reset_history()
                disk.table.reset_stats()
                if mode == "prefetch":
                    install_disk_ple_prefetch_hook(model)

                model.eval()
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model(tokens)
                wall_s = time.perf_counter() - t0
                stats = disk.table.get_stats()

                row = {
                    "mode": mode,
                    "rep": rep,
                    "tokens": args.tokens,
                    "wall_ms": wall_s * 1000.0,
                    "earlier_ms": model.last_timings["earlier_s"] * 1000.0,
                    "ple_ms": model.last_timings["ple_s"] * 1000.0,
                    "post_ms": model.last_timings["post_s"] * 1000.0,
                    "prefetch_issued": int(stats["prefetch_issued"]),
                    "prefetch_wait_ms": stats["prefetch_wait_s"] * 1000.0,
                    "fetch_ms": stats["fetch_s"] * 1000.0,
                    "convert_ms": stats["convert_s"] * 1000.0,
                    "calls": int(stats["calls"]),
                    "hits": int(stats["hits"]),
                    "misses": int(stats["misses"]),
                    "prefetch_ready_at_ple": int(model.prefetch_ready_at_ple),
                    "prefetch_pending_at_ple": int(model.prefetch_pending_at_ple),
                }
                rows.append(row)
                print(
                    f"[{mode}:{rep}] wall={row['wall_ms']:.3f}ms "
                    f"earlier={row['earlier_ms']:.3f}ms ple={row['ple_ms']:.3f}ms "
                    f"issued={row['prefetch_issued']} wait={row['prefetch_wait_ms']:.3f}ms "
                    f"fetch={row['fetch_ms']:.3f}ms ready={row['prefetch_ready_at_ple']}"
                )
            finally:
                if disk is not None:
                    disk.close()
                store.close()

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/Volumes/My Passport/qwen38-ple")
    parser.add_argument("--store", default="/Volumes/My Passport/qwen38-rows")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--pre-layers", type=int, default=12)
    parser.add_argument("--post-layers", type=int, default=2)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    rows = run_ab(args)
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[csv] wrote {args.csv}")

    print("MINI_OFFICIAL_PREFETCH_AB_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
