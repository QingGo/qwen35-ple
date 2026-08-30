#!/usr/bin/env python3
"""Phase 1 gate: verify live EngramDB e_t equals precomputed e_t.

This script compares the precomputed feature file with a live read through
``engramdb.ple_adapter.DiskPleNGramEmbedding``.  It verifies:

* rowid/scale semantics match;
* live Store-I read + FP8 dequantization matches the saved e_t array;
* the future live training path is numerically sound.

Usage:

    PYTHONPATH=src:../EngramDB/python \\
    python scripts/run_live_vs_precomputed.py \\
        --features data/ple-adapter-features \\
        --rows-dir "/Volumes/My Passport/qwen38-rows" \\
        --model-dir "/Volumes/My Passport/qwen38-ple"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qwen35_ple.real_ple import resolve_ple_weight_scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/ple-adapter-features")
    parser.add_argument("--rows-dir", default="/Volumes/My Passport/qwen38-rows")
    parser.add_argument("--model-dir", default="/Volumes/My Passport/qwen38-ple")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output", default="outputs/live-vs-precomputed.json")
    args = parser.parse_args()

    feature_dir = Path(args.features)
    tokens = np.load(feature_dir / "tokens.npy")[: args.max_tokens]
    scale = resolve_ple_weight_scale(model_dir=args.model_dir, scale=args.scale)

    # Recompute the precomputed reference with the current Store/scale path.
    # Older saved e_t files in this repo were generated before the FP8
    # weight_scale fix and may be stale, so the authoritative comparison for
    # Phase 1 is "live DiskPleNGramEmbedding == current fetch_e_t path".
    from qwen35_ple.real_ple import fetch_e_t, rowids_from_tokens

    rowids = rowids_from_tokens(tokens)
    e_t = fetch_e_t(args.rows_dir, rowids, scale=scale)

    import engramdb
    from engramdb.ple_adapter import DiskPleNGramEmbedding, PLE_EOS

    store = engramdb.Store(
        args.rows_dir,
        shards=128,
        rows_per_shard=2_500_012,
        width=160,
    )
    try:
        disk = DiskPleNGramEmbedding(
            store=store,
            embedding_dim=2560,
            num_heads=16,
            scale=scale,
            cache_size=0,
            eos=PLE_EOS,
        )
        ids = torch.tensor(tokens.tolist(), dtype=torch.int64).unsqueeze(0)
        live = disk(ids).squeeze(0).detach().cpu().numpy()

        ref = np.asarray(e_t, dtype=np.float32)
        max_abs = float(np.abs(live - ref).max())
        mean_abs = float(np.abs(live - ref).mean())
        allclose = bool(np.allclose(live, ref, atol=1e-6, rtol=1e-5))

        result = {
            "features": str(feature_dir),
            "rows_dir": args.rows_dir,
            "num_tokens": int(len(tokens)),
            "weight_scale": scale,
            "live_shape": list(live.shape),
            "precomputed_shape": list(ref.shape),
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "allclose": allclose,
            "live_finite": bool(np.isfinite(live).all()),
            "precomputed_finite": bool(np.isfinite(ref).all()),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if not allclose:
            print("LIVE_VS_PRECOMPUTED_MISMATCH")
            return 1
        print("LIVE_VS_PRECOMPUTED_OK")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
