#!/usr/bin/env python
"""Round 169: the rows a B1 evaluation actually needs, for one domain.

The trained bank is saved restricted to the trigrams that occur at the aligned
EVAL positions, because the full bank is 6.8 GB and this subset is ~1.2 GB.  That
subset is exactly what this script computes:

    for each aligned position t:  trigram = codes[t - 2]
    keep = sorted(unique(snapshot index of that trigram))    # only where it hit

The ``t - 2`` offset is load-bearing and is the same convention the snapshot and
the trainer use: ``codes[i]`` describes the trigram ENDING at token ``i + 2``, so
the injection at stream position ``t`` is addressed by ``codes[t - 2]``.

A trigram that is not in the snapshot cannot be trained (it had no row to start
from) and is dropped -- the evaluator then leaves the frozen value in place at
those positions, which is what makes the frozen initialisation a valid control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from round169_row_snapshot import trigram_codes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot-codes", required=True, help="trigram-codes.npy from the snapshot")
    ap.add_argument("--tokens", required=True, help="eval stream .npy")
    ap.add_argument("--positions", required=True, help="aligned positions into that stream")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta-out", default="")
    args = ap.parse_args()

    codes_uniq = np.load(args.snapshot_codes)
    tokens = np.load(args.tokens).astype(np.int64).reshape(-1)
    T = int(tokens.size)
    positions = np.load(args.positions).astype(np.int64)
    positions = positions[(positions >= 2) & (positions + 1 < T)]

    codes = trigram_codes(tokens)
    idx = positions - 2
    tri = codes[idx]
    j = np.searchsorted(codes_uniq, tri)
    np.clip(j, 0, codes_uniq.size - 1, out=j)
    hit = codes_uniq[j] == tri
    keep = np.sort(np.unique(j[hit])).astype(np.int64)
    if keep.size == 0:
        raise SystemExit("no aligned position has a snapshot row: wrong stream or codes")

    np.save(args.out, keep)
    meta = {
        "snapshot_codes": str(args.snapshot_codes),
        "tokens": str(args.tokens),
        "positions": str(args.positions),
        "n_positions": int(positions.size),
        "n_positions_with_row": int(hit.sum()),
        "n_kept_rows": int(keep.size),
        "kept_bytes_fp32": int(keep.size * 2560 * 4),
    }
    if args.meta_out:
        Path(args.meta_out).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(
        f"[keep-index] {meta['n_positions_with_row']:,}/{meta['n_positions']:,} positions have a "
        f"snapshot row -> {meta['n_kept_rows']:,} distinct rows ({keep.size * 2560 * 4 / 1e9:.2f} GB fp32)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
