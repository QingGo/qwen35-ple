#!/usr/bin/env python3
"""Do the control rows actually differ from the real rows at evaluation time?

A negative result of the form ``real - control = 0`` is only informative if the
two arms inject different content.  The injection counters added in round 161
prove injection *happens*, but they only record contribution NORMS, and norms are
the wrong test: shuffled rows are drawn from the same table, so their norms are
statistically indistinguishable from the real ones (measured: 0.1489 vs 0.1457,
a 2% difference) while the vectors could still be entirely different.

This compares the VECTORS.  For a set of token sequences it fetches the PLE rows
twice -- once as stored, once with the same time-permutation the control arm
applies -- and reports the relative difference and cosine similarity per
position.  A control arm whose rows were a no-op would show a relative difference
near zero, and every real-vs-control conclusion drawn from it would be void.

Usage:
    python scripts/audit_injection_content.py \
        --rows-dir /path/to/qwen38-rows --tokens-npy /path/to/tokens.npy \
        --tokenizer /path/to/Qwen3.5-0.8B --num-sequences 16 --seq-len 64
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-dir", required=True)
    parser.add_argument("--tokens-npy", required=True)
    parser.add_argument("--scale", type=float, default=0.00019931793212890625)
    parser.add_argument("--num-sequences", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import engramdb

    from qwen35_ple.real_ple import real_spec, rowids_from_tokens

    spec = real_spec()
    store = engramdb.Store(
        args.rows_dir,
        shards=spec.shards,
        rows_per_shard=spec.rows_per_shard,
        width=160,
    )
    tokens = np.load(args.tokens_npy).astype(np.int64)
    need = args.num_sequences * args.seq_len
    if tokens.size < need:
        raise SystemExit(f"need {need} tokens, {args.tokens_npy} has {tokens.size}")

    rel_diffs: list[float] = []
    cosines: list[float] = []
    identical = 0
    total = 0

    for s in range(args.num_sequences):
        seq = tokens[s * args.seq_len : (s + 1) * args.seq_len]
        rowids = rowids_from_tokens(seq)
        arr = engramdb.fetch_e_t_tensor(
            store,
            rowids.reshape(-1).tolist(),
            scale=args.scale,
            num_heads=16,
            head_dim=160,
            dtype=None,
            out_dtype=None,
        ).reshape(len(seq), 2560).numpy()

        # The same manipulation run_phase0 applies for the control arm.
        rng = np.random.default_rng(args.seed * 1000 + s)
        shuffled = arr[rng.permutation(len(arr))]

        for i in range(len(arr)):
            real_row, ctrl_row = arr[i], shuffled[i]
            denom = float(np.linalg.norm(real_row))
            total += 1
            if np.array_equal(real_row, ctrl_row):
                identical += 1
            if denom:
                rel_diffs.append(float(np.linalg.norm(real_row - ctrl_row)) / denom)
                cos = float(
                    np.dot(real_row, ctrl_row)
                    / (np.linalg.norm(real_row) * np.linalg.norm(ctrl_row) + 1e-12)
                )
                cosines.append(cos)

    if not rel_diffs:
        raise SystemExit("no positions compared; every row had zero norm")

    rel = np.asarray(rel_diffs)
    cos = np.asarray(cosines)
    print(f"compared {total} positions from {args.num_sequences} sequences")
    print(f"rows identical after the control permutation : {identical}/{total}")
    print(f"relative difference ||real-control||/||real|| : mean {rel.mean():.4f}  median {np.median(rel):.4f}")
    print(f"cosine(real, control)                         : mean {cos.mean():.4f}  median {np.median(cos):.4f}")
    print()
    # The verdict is on whether the two arms can possibly differ, not on taste.
    if identical == total or rel.mean() < 0.05 or cos.mean() > 0.99:
        print("VERDICT: VOID -- the control rows are effectively the real rows, so a")
        print("         real-vs-control difference of zero would be guaranteed by")
        print("         construction rather than measured.")
        return 1
    print("VERDICT: OK -- the control permutation produces materially different content,")
    print("         so a real-vs-control comparison is a real comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
