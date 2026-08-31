#!/usr/bin/env python3
"""Verify a fallback raw-shard PLE reader against precomputed features.

This is useful on machines where the EngramDB Python bindings are not built.
It reads the Store-I shard files directly (160-byte FP8 rows) and compares the
resulting e_t with an existing ``e_t.npy`` produced by the official EngramDB
path.

Usage (WSL):
    cd /home/zeng/qwen35-ple
    .venv/bin/python scripts/verify_raw_shard_reader.py \
        --rows-dir /home/zeng/qwen38-rows \
        --features data/ple-books-160k
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from qwen35_ple.ple_hash import real_spec


def open_shards(rows_dir: str):
    spec = real_spec()
    mmaps = []
    for i in range(spec.shards):
        path = Path(rows_dir) / f"shard_{i:03d}.bin"
        mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(spec.rows_per_shard, 160))
        mmaps.append(mm)
    return mmaps, spec


def fetch_e_t_raw(rowids: np.ndarray, mmaps, spec, scale: float, chunk: int = 200_000) -> np.ndarray:
    flat = rowids.reshape(-1).astype(np.int64)
    out = np.empty((len(flat), 160), dtype=np.uint8)
    for start in range(0, len(flat), chunk):
        end = min(start + chunk, len(flat))
        ids = flat[start:end]
        shard = ids // spec.rows_per_shard
        off = ids % spec.rows_per_shard
        for j, (s, o) in enumerate(zip(shard.tolist(), off.tolist())):
            out[start + j] = mmaps[s][o]
    # Convert uint8 bytes to FP8 E4M3 and dequantize.
    t = torch.from_numpy(out).view(torch.float8_e4m3fn).float().reshape(-1)
    return (t.numpy() * scale).reshape(len(rowids), 16, 160).reshape(len(rowids), 2560)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-dir", default="/home/zeng/qwen38-rows")
    parser.add_argument("--features", default="data/ple-books-160k")
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    feat = Path(args.features)
    tokens = np.load(feat / "tokens.npy")
    expected = np.load(feat / "e_t.npy")
    if args.max_tokens is not None:
        tokens = tokens[: args.max_tokens]
        expected = expected[: args.max_tokens]

    spec = real_spec()
    rows = spec.rowids_for_seq(tokens.tolist())
    rowids = np.asarray(rows, dtype=np.int64)
    print(f"tokens={len(tokens)} rows={rowids.size}")

    import json
    meta = json.loads((feat / "meta.json").read_text())
    scale = float(meta.get("weight_scale", 0.00019931793212890625))
    print(f"scale={scale}")

    mmaps, _ = open_shards(args.rows_dir)
    got = fetch_e_t_raw(rowids, mmaps, spec, scale)
    diff = float(np.abs(got - expected).max())
    print(f"max_abs_diff={diff}")
    print("allclose:", bool(np.allclose(got, expected, atol=1e-6, rtol=1e-5)))
    return 0 if diff < 1e-5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
