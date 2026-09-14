#!/usr/bin/env python
"""Round 169 Stage B1, step 1: snapshot the frozen row bank for a training stream.

Produces the ``E0`` of the pre-registration -- the frozen initialisation that is
also B1's control arm:

    E0[i] = e_t for the i-th distinct trigram of the training stream

Everything downstream (training, evaluation) indexes rows by TRIGRAM, not by the
16 per-head row ids, because ``e_t`` is exactly the concatenation of those 16
rows.  Collapsing the 16 ids into one trigram key is what makes the embedding a
single ``[n_trigrams, 2560]`` tensor instead of an unaddressable 3.2e8-row table.

The trigram is encoded as one int64, ``t0*V^2 + t1*V + t2`` with a FROZEN ``V``.
V = 248320 (the tokenizer's vocab size) is used rather than the observed maximum
so that the encoding cannot depend on which stream was encoded first -- a
stream-dependent V would silently make train and eval codes incomparable.

CPU + row-table I/O only.  No GPU, no torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

# ``qwen35_ple.real_ple`` imports torch at module scope, and this script's pure
# part (the trigram encoding) must stay importable without torch so it can be
# unit-tested in CI.  The row-table imports therefore happen inside main(),
# matching ``make_fetcher`` in scripts/probe_table_next_token.py.

#: Frozen trigram radix.  Must satisfy ``V > max(token id)``; the largest token
#: id seen anywhere in this project is 248069, so the tokenizer's 248320 is safe
#: and, unlike ``max(tokens)+1``, does not depend on the stream being encoded.
TRIGRAM_VOCAB = 248_320
#: 248320^3 = 1.53e16, comfortably inside int64 (9.22e18).
assert TRIGRAM_VOCAB**3 < 2**63


def log(msg: str) -> None:
    print(f"[r169-snapshot] {msg}", flush=True)


def trigram_codes(tokens: np.ndarray, vocab: int = TRIGRAM_VOCAB) -> np.ndarray:
    """Encode every achievable (t-2, t-1, t) trigram of ``tokens`` as one int64.

    Position ``i`` of the result describes the trigram ending at token ``i``, so
    the result has length ``len(tokens) - 2`` and position ``t`` of the token
    stream corresponds to index ``t - 2``.  The row bank's own heads use the same
    window (8 bigram heads on (t, t-1) and 8 trigram heads on (t, t-1, t-2)), so
    a trigram key is a superset of what the 16 heads can address -- which is
    exactly the <=3-token bound, made explicit.
    """
    t = np.asarray(tokens, dtype=np.int64).reshape(-1)
    if t.size < 3:
        return np.empty(0, dtype=np.int64)
    v = np.int64(vocab)
    return (t[:-2] * v + t[1:-1]) * v + t[2:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-tokens", required=True)
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--scale", type=float, default=0.00019931793212890625)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk", type=int, default=50_000, help="rows per fetch call")
    ap.add_argument(
        "--bound-check", type=int, default=20_000,
        help="trigrams whose row ids are re-derived from a second representative "
             "(0 = skip); this is the <=3-token bound, checked rather than assumed",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from qwen35_ple.real_ple import fetch_e_t, rowids_from_tokens

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tokens = np.load(args.train_tokens).astype(np.int64).reshape(-1)
    log(f"training stream: {tokens.size:,} tokens from {args.train_tokens}")

    codes = trigram_codes(tokens)
    uniq, inv, counts = np.unique(codes, return_inverse=True, return_counts=True)
    n_tri = int(uniq.size)
    log(f"distinct trigrams: {n_tri:,} (of {codes.size:,} positions)")

    # Row ids must be computed ONCE over the contiguous stream and then indexed.
    # Calling ``rowids_from_tokens`` on a gathered subsequence is silently wrong:
    # the row id at position i is a function of tokens[i-2..i], so gathering
    # destroys exactly the context that defines it.  (The bound check below is
    # what caught this on the first run -- it reported 19,996/20,000
    # "disagreements", which was the gathering, not a violation.)
    log("computing official row ids over the whole stream ...")
    all_rowids = rowids_from_tokens(tokens)
    if all_rowids.shape != (tokens.size, 16):
        raise SystemExit(f"unexpected rowid shape {all_rowids.shape}, wanted {(tokens.size, 16)}")

    # One representative position per distinct trigram.  Any representative will
    # do IF the row ids really are a function of the trigram alone -- which the
    # next block checks on real data rather than assuming.
    rep = np.zeros(n_tri, dtype=np.int64)
    rep[inv] = np.arange(codes.size, dtype=np.int64)  # last occurrence per trigram
    rowids = all_rowids[rep + 2]

    alt_codes, alt_rep = np.unique(codes, return_index=True)  # first occurrence
    if not np.array_equal(alt_codes, uniq):
        raise SystemExit("internal error: the two unique() passes disagree")
    n_check = min(args.bound_check, n_tri)
    disagree = None
    if n_check > 0:
        pick = np.linspace(0, n_tri - 1, n_check).astype(np.int64)
        alt_rows = all_rowids[alt_rep[pick] + 2]
        # A representative is only usable if it is genuinely that trigram.
        if not np.array_equal(codes[rep[pick]], uniq[pick]):
            raise SystemExit("internal error: representative does not reproduce its trigram")
        disagree = int((alt_rows != rowids[pick]).any(axis=1).sum())
        log(f"<=3-token bound check: {n_check:,} trigrams, {disagree} disagreed")
        if disagree:
            raise SystemExit(
                f"the <=3-token bound FAILED on {disagree} trigrams: two positions with "
                f"the same trigram produced different row ids. Every bound in this "
                f"project assumes that cannot happen."
            )

    log(f"fetching e_t for {n_tri:,} trigrams (chunk={args.chunk:,}) ...")
    E = np.empty((n_tri, 2560), dtype=np.float32)
    for s in range(0, n_tri, args.chunk):
        e = min(n_tri, s + args.chunk)
        E[s:e] = fetch_e_t(args.rows_dir, rowids[s:e], scale=args.scale)
        if s % (args.chunk * 5) == 0 or e == n_tri:
            log(f"  {e:,}/{n_tri:,}  ({time.time() - t0:.0f}s)")
    if not np.isfinite(E).all():
        raise SystemExit("non-finite values in the fetched snapshot")
    zero = int((~E.any(axis=1)).sum())
    if zero:
        log(f"WARNING: {zero:,} all-zero e_t rows in the snapshot")

    np.save(out_dir / "E0.npy", E)
    np.save(out_dir / "trigram-codes.npy", uniq)
    np.save(out_dir / "trigram-train-count.npy", counts.astype(np.int64))

    meta = {
        "train_tokens": str(args.train_tokens),
        "n_tokens": int(tokens.size),
        "n_trigram_positions": int(codes.size),
        "n_trigrams": n_tri,
        "embedding_shape": [n_tri, 2560],
        "embedding_bytes": int(E.nbytes),
        "trigram_vocab": TRIGRAM_VOCAB,
        "rows_dir": str(args.rows_dir),
        "scale": float(args.scale),
        "all_zero_rows": zero,
        "bound_check_trigrams": int(n_check),
        "bound_check_disagreements": None if disagree is None else int(disagree),
        "train_count_max": int(counts.max()),
        "train_count_singletons": int((counts == 1).sum()),
        "codes_sha256": hashlib.sha256(uniq.tobytes()).hexdigest(),
        "e0_sha256": hashlib.sha256(E.tobytes()).hexdigest(),
        "elapsed_seconds": time.time() - t0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    log(f"wrote {out_dir}/E0.npy ({E.nbytes / 1e9:.2f} GB), trigram-codes.npy, meta.json")
    log(
        f"train trigram counts: max={counts.max():,} mean={counts.mean():.2f} "
        f"singletons={(counts == 1).sum():,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
