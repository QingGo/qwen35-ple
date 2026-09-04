#!/usr/bin/env python3
"""Build the P1 exact longest-match PLE n-gram bank from precomputed features.

Input is one or more ``data/ple-*`` directories produced by
``scripts/precompute_real_ple_features.py`` (containing ``tokens.npy`` and
``e_t.npy``).  The script writes:

* ``--output``: real bank;
* ``--control-output``: control bank with the same exact keys but shuffled PLE
  values (key->value association is destroyed).

Usage::

    python scripts/build_exact_ple_bank.py \
        --feature-dir data/ple-books-160k \
        --feature-dir data/ple-adapter-features-20k \
        --output data/exact-ple-bank.npz \
        --control-output data/exact-ple-bank-control.npz \
        --max-order 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qwen35_ple.memory.bank import ExactNgramBank


def _load_feature_dir(path: Path, max_tokens: int | None) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.load(path / "tokens.npy", mmap_mode="r").astype(np.int64)
    e_t = np.load(path / "e_t.npy", mmap_mode="r").astype(np.float32)
    if max_tokens is not None and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        e_t = e_t[:max_tokens]
    return np.asarray(tokens), np.asarray(e_t)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir",
        action="append",
        default=[],
        help="precomputed PLE feature directory; may be repeated",
    )
    parser.add_argument("--output", default="data/exact-ple-bank.npz")
    parser.add_argument("--control-output", default="data/exact-ple-bank-control.npz")
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--min-order", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--control-seed", type=int, default=0)
    args = parser.parse_args()

    if not args.feature_dir:
        parser.error("at least one --feature-dir is required")

    all_tokens: list[np.ndarray] = []
    all_e_t: list[np.ndarray] = []
    total_tokens = 0
    t0 = time.time()
    for raw_dir in args.feature_dir:
        path = Path(raw_dir)
        tokens, e_t = _load_feature_dir(path, args.max_tokens)
        print(
            f"[build-exact-bank] loading {path}: tokens={len(tokens)} e_t={e_t.shape}",
            flush=True,
        )
        all_tokens.append(tokens)
        all_e_t.append(e_t)
        total_tokens += len(tokens)

    if not all_tokens:
        parser.error("no bank built")

    tokens = np.concatenate(all_tokens, axis=0)
    e_t = np.concatenate(all_e_t, axis=0)
    bank = ExactNgramBank.from_arrays(
        tokens,
        e_t,
        min_order=args.min_order,
        max_order=args.max_order,
    )

    print(f"[build-exact-bank] built in {time.time()-t0:.1f}s: {bank.stats()}", flush=True)
    bank.save(args.output)
    print(f"[build-exact-bank] wrote real bank: {args.output}", flush=True)

    control = bank.shuffled(seed=args.control_seed)
    control.save(args.control_output)
    print(f"[build-exact-bank] wrote control bank: {args.control_output}", flush=True)

    meta = {
        "feature_dirs": args.feature_dir,
        "max_tokens": args.max_tokens,
        "max_order": args.max_order,
        "min_order": args.min_order,
        "control_seed": args.control_seed,
        "total_tokens": total_tokens,
        "real": bank.stats(),
        "control": control.stats(),
    }
    meta_path = Path(args.output).with_suffix(".json")
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[build-exact-bank] wrote meta: {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
