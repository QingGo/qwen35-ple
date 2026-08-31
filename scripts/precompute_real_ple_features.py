#!/usr/bin/env python3
"""Precompute real PLE e_t features for a small corpus using EngramDB Store-I.

This is the first step of the "frozen PLE as database" plan:

1. Tokenize a small corpus with the Qwen3.5 tokenizer.
2. Generate official PLE_QWEN_V1 rowids (one 16-head row per token).
3. Read the real FP8 rows through EngramDB ``Store``.
4. Convert F8_E4M3 to float32 and assemble ``e_t = [T, 2560]``.
5. Write ``tokens.npy``, ``e_t.npy``, ``keys.npy`` and ``meta.json``.

Usage:
    PYTHONPATH=src:../EngramDB/python \
    python scripts/precompute_real_ple_features.py \
        --rows-dir "/Volumes/My Passport/qwen38-rows" \
        --tokenizer data/models/Qwen3.5-0.8B \
        --text "The capital of France is Paris. ..." \
        --output data/ple-features
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qwen35_ple.ple_hash import real_spec
from qwen35_ple.real_ple import resolve_ple_weight_scale

DEFAULT_TEXT = (
    "The capital of France is Paris. "
    "The largest planet in the Solar System is Jupiter. "
    "The chemical symbol for gold is Au. "
    "Twelve plus fifteen equals twenty-seven. "
    "Alice has three apples and Bob gives her two more apples. "
    "The library is on the third floor."
)


def _tokenize(tokenizer_path: str, texts: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True
    )
    ids: list[int] = []
    for i, text in enumerate(texts):
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        if i + 1 < len(texts):
            ids.append(tokenizer.eos_token_id)
    return np.asarray(ids, dtype=np.int64)


def _rowids_from_tokens(tokens: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    # Prefer EngramDB's native/fast rowid implementation when available.
    from qwen35_ple.real_ple import rowids_from_tokens as fast_rowids

    arr = fast_rowids(tokens)
    rows = arr.tolist()
    return arr, rows


def _fetch_fp8(rows_dir: str, flat_rowids: np.ndarray, scale: float = 1.0) -> np.ndarray:
    import engramdb

    store = engramdb.Store(
        rows_dir,
        shards=real_spec().shards,
        rows_per_shard=real_spec().rows_per_shard,
        width=160,  # real row width
    )
    try:
        # Fast path: one Store.fetch + torch.frombuffer; no Python per-row bytes
        # slicing / join.  This is the path measured at ~0.4-0.8s vs 16.9s for
        # the old PleDiskGather bytes-expansion path on 320k rows.
        arr = engramdb.fetch_e_t_tensor(
            store,
            flat_rowids.tolist(),
            scale=scale,
            num_heads=16,
            head_dim=160,
            dtype=torch.float8_e4m3fn,
            out_dtype=torch.float32,
        )
        return arr.reshape(-1).numpy()
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows-dir",
        default="/Volumes/My Passport/qwen38-rows",
        help="real Store-I rows directory",
    )
    parser.add_argument(
        "--tokenizer",
        default="data/models/Qwen3.5-0.8B",
        help="tokenizer/model dir with the same vocab as the PLE table",
    )
    parser.add_argument("--text", default=None, help="single text to precompute")
    parser.add_argument(
        "--corpus", default=None, help="text file; one line per segment"
    )
    parser.add_argument(
        "--tokens-npy", default=None, help="pre-tokenized tokens.npy (skips tokenizer)"
    )
    parser.add_argument(
        "--output", default="data/ple-features", help="output directory"
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--model-dir",
        default="/Volumes/My Passport/qwen38-ple",
        help="Qwen3.8-Flash-Next checkpoint dir (used to read FP8 weight_scale)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="explicit FP8 weight_scale; overrides discovery",
    )
    args = parser.parse_args()

    if args.tokens_npy:
        tokens = np.load(args.tokens_npy).astype(np.int64)
        if args.max_tokens is not None and len(tokens) > args.max_tokens:
            tokens = tokens[: args.max_tokens]
        print(f"[precompute] loaded tokens from {args.tokens_npy}: {len(tokens)}")
    else:
        if args.corpus:
            texts = [
                line.strip()
                for line in Path(args.corpus).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif args.text:
            texts = [args.text]
        else:
            texts = [DEFAULT_TEXT]

        print(f"[precompute] tokenizing {len(texts)} segments ...")
        tokens = _tokenize(args.tokenizer, texts)
        if args.max_tokens is not None and len(tokens) > args.max_tokens:
            print(f"[precompute] truncating to max_tokens={args.max_tokens}")
            tokens = tokens[: args.max_tokens]
        print(f"[precompute] tokens={len(tokens)}")

    print("[precompute] generating official PLE rowids ...")
    rowids, _rows = _rowids_from_tokens(tokens)
    flat = rowids.reshape(-1)
    print(f"[precompute] rowids={flat.shape[0]}")

    scale = resolve_ple_weight_scale(model_dir=args.model_dir, scale=args.scale)
    print(f"[precompute] weight_scale={scale:.10g}")

    print("[precompute] fetching real FP8 rows from EngramDB ...")
    t0 = time.time()
    fp8_vectors = _fetch_fp8(args.rows_dir, flat, scale=scale)
    elapsed = time.time() - t0
    print(f"[precompute] fetch+dequant {elapsed:.2f}s")

    e_t = fp8_vectors.reshape(len(tokens), 16, 160).reshape(len(tokens), 2560)
    print(
        f"[precompute] e_t shape={e_t.shape} "
        f"finite={bool(np.isfinite(e_t).all())} "
        f"mean={float(e_t.mean()):.4f} std={float(e_t.std()):.4f}"
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "tokens.npy", tokens)
    np.save(out_dir / "e_t.npy", e_t)
    np.save(out_dir / "keys.npy", rowids)

    meta = {
        "rows_dir": args.rows_dir,
        "tokenizer": args.tokenizer,
        "num_tokens": len(tokens),
        "e_t_dim": 2560,
        "heads": 16,
        "head_dim": 160,
        "dtype": "float32",
        "fetch_seconds": float(elapsed),
        "corpus_segments": len(texts) if args.tokens_npy is None else 1,
        "weight_scale": scale,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[precompute] saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
