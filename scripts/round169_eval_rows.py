#!/usr/bin/env python
"""Round 169 Stage B1, step 3: the paired frozen-vs-trained read-out.

Implements section 3 of
``docs/round-169-stageB1-row-retraining-preregistration.md``:

    Delta = L_frozen - L_trained   on held-out positions whose trigram WAS seen
                                   during row training

and section 4's pre-registered, falsifiable prediction that the gain peaks at
MID frequency rather than rising monotonically.

Three arms are measured, all on the same positions and with the same frozen
backbone and reader:

* ``none``     -- no reader installed.  This is the reference Stage 1.5c already
                  published as ``bb_final`` (the pure backbone), so measuring it
                  again here is what connects B1's numbers to 1.5c's.
* ``frozen``   -- the frozen rows, addressed exactly as the production path does.
* ``trained``  -- identical, except that at positions whose trigram was trained the
                  row vector is replaced by the trained one.  Positions whose
                  trigram was never seen keep the frozen value, so the two injected
                  arms differ ONLY where training could have had an effect.

That last property is the whole design: the frozen initialisation is its own
control, so no real/control pairing is needed.

A consistency check is included and reported: the snapshot's ``E0`` row for a
trigram must equal what the shard table returns for a position carrying that
trigram.  If it does not, the snapshot is not the frozen bank and every number
below is meaningless.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from round169_row_snapshot import trigram_codes


def log(msg: str) -> None:
    print(f"[r169-eval] {msg}", flush=True)


def resolve_snapshot_index(codes_uniq: np.ndarray, codes: np.ndarray):
    """Map each trigram code to its snapshot row index, and say whether it hit."""
    j = np.searchsorted(codes_uniq, codes)
    np.clip(j, 0, codes_uniq.size - 1, out=j)
    return j, codes_uniq[j] == codes


def paired_stats(delta: np.ndarray) -> dict:
    """Mean / SE / t of a paired difference."""
    n = int(delta.size)
    if n == 0:
        return {"n": 0, "mean": None, "se": None, "t": None}
    mean = float(delta.mean())
    if n < 2:
        return {"n": n, "mean": mean, "se": None, "t": None}
    se = float(delta.std(ddof=1) / np.sqrt(n))
    return {"n": n, "mean": mean, "se": se, "t": (mean / se if se > 0 else None)}


def per_position_record(score, rowid, delta, lf, lt, ln, ctx=None, snap=None, code=None) -> dict:
    """The per-position record written next to the primary JSON.

    A band mean cannot answer the question the band means raised.  At lr=1e-4 the
    1-9 bands come out worse while the >=10 bands come out better, and the ~150k
    scored positions whose trigram context has a TRAIN count of zero come out
    worse than either -- yet their own row can still have been moved, because a
    position reaches a row through the hash of its trigram and not through its
    own identity.  That is the interference question, and it needs one row per
    scored position plus every key the join could be made on:

    ``rowid``  the row id in the shard table (``rowids_from_tokens``);
    ``snap``   the index into ``E0``/the trained bank -- the embedding row that
               training actually moved, which is NOT the same key as ``rowid``;
    ``code``   the trigram hash whose bucket decided whether the position was
               addressed at all;
    ``ctx``    the train count of the position's own trigram context, so a
               position can be split into "its own trigram was trained" vs
               "it inherited a row it never voted for".

    Every field is parallel to ``score``: index ``i`` of any array describes
    stream position ``score[i]``.
    """
    rec = {
        "score": np.asarray(score, dtype=np.int64),
        "rowid": np.asarray(rowid, dtype=np.int64),
        "delta": np.asarray(delta, dtype=np.float32),
        "nll_frozen": np.asarray(lf, dtype=np.float32),
        "nll_trained": np.asarray(lt, dtype=np.float32),
        "nll_none": np.asarray(ln, dtype=np.float32),
    }
    if ctx is not None:
        rec["context_count"] = np.asarray(ctx, dtype=np.int64)
    if snap is not None:
        rec["snapshot_index"] = np.asarray(snap, dtype=np.int64)
    if code is not None:
        rec["trigram_code"] = np.asarray(code, dtype=np.int64)
    n = rec["delta"].size
    for key, arr in rec.items():
        if arr.size != n:
            raise ValueError(
                f"per-position record field {key!r} has {arr.size} entries, expected {n}"
            )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot-dir", required=True)
    ap.add_argument("--trained-bank", required=True)
    ap.add_argument("--keep-index", required=True)
    ap.add_argument("--tokens", required=True, help="eval stream .npy")
    ap.add_argument("--positions", required=True, help="aligned positions .npy")
    ap.add_argument("--counts-npz", default="", help="1.5c counts npz, for frequency bands")
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--reader-kind", choices=("official", "saved"), default="saved")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.00019931793212890625)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=16,
                    help="leading positions per chunk dropped from scoring (causal context)")
    ap.add_argument("--consistency-sample", type=int, default=256)
    ap.add_argument("--max-chunks", type=int, default=0, help="smoke only; 0 = all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone-dtype", default="float32")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from qwen35_ple.reader import OfficialSourceQwenReader, install_reader_hook
    from qwen35_ple.real_ple import (
        fetch_e_t,
        resolve_ple_weight_scale,
        rowids_from_tokens,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import run_phase0 as p0

    t0 = time.time()
    snap = Path(args.snapshot_dir)
    E0 = np.load(snap / "E0.npy", mmap_mode="r")
    codes_uniq = np.load(snap / "trigram-codes.npy")
    keep_index = np.load(args.keep_index)
    trained = np.load(args.trained_bank)
    if trained.shape[0] != keep_index.size:
        raise SystemExit(
            f"trained bank has {trained.shape[0]} rows but keep-index has {keep_index.size}"
        )

    tokens = np.load(args.tokens).astype(np.int64).reshape(-1)
    T = int(tokens.size)
    positions = np.load(args.positions).astype(np.int64)
    positions = positions[(positions >= 2) & (positions + 1 < T)]
    log(f"eval stream {T:,} tokens; {positions.size:,} aligned positions")

    codes = trigram_codes(tokens)
    j_all, hit_all = resolve_snapshot_index(codes_uniq, codes)
    seen_pos = np.flatnonzero(hit_all) + 2            # stream positions with a snapshot row
    unseen_pos = np.setdiff1d(np.arange(T), seen_pos, assume_unique=False)
    log(f"positions with a snapshot row: {seen_pos.size:,}; without: {unseen_pos.size:,}")

    all_rowids = rowids_from_tokens(tokens)
    E = np.zeros((T, 2560), dtype=np.float32)
    E[seen_pos] = np.asarray(E0[j_all[hit_all]], dtype=np.float32)
    if unseen_pos.size:
        log(f"fetching {unseen_pos.size:,} rows the snapshot does not cover ...")
        for s in range(0, unseen_pos.size, 50_000):
            e = min(unseen_pos.size, s + 50_000)
            E[unseen_pos[s:e]] = fetch_e_t(args.rows_dir, all_rowids[unseen_pos[s:e]], scale=args.scale)
    log(f"e_t matrix built ({E.nbytes / 1e9:.1f} GB) in {time.time() - t0:.0f}s")

    # Consistency: the snapshot's row for a trigram must equal the shard table's
    # row at a position carrying that trigram.  Without this, "frozen" is a claim.
    consistency = {"n": 0, "max_abs_diff": None}
    if args.consistency_sample > 0:
        pick = np.linspace(0, seen_pos.size - 1, min(args.consistency_sample, seen_pos.size)).astype(np.int64)
        probe = seen_pos[pick]
        fetched = fetch_e_t(args.rows_dir, all_rowids[probe], scale=args.scale)
        diff = float(np.abs(fetched - E[probe]).max())
        consistency = {"n": int(probe.size), "max_abs_diff": diff}
        log(f"snapshot-vs-shard consistency: max|diff| = {diff:.3e} over {probe.size} positions")
        if not np.isfinite(diff) or diff > 1e-3:
            raise SystemExit(
                "the snapshot bank does NOT reproduce the shard table for the same "
                "trigram -- E0 is not the frozen bank and B1 cannot be evaluated"
            )

    # ---- model + reader --------------------------------------------------- #
    scale = resolve_ple_weight_scale(model_dir=args.model, scale=args.scale)
    _tokenizer, model = p0._load_model(args.model, args.device, args.backbone_dtype)
    for prm in model.parameters():
        prm.requires_grad_(False)
    model.eval()
    if args.reader_kind == "official":
        reader = OfficialSourceQwenReader.from_official_checkpoint(
            args.reader, d_target=model.config.hidden_size, freeze_source=True,
        )
    else:
        reader, _x = p0.load_reader_with_extra(Path(args.reader), device=args.device)
    reader = reader.to(args.device).eval()
    for prm in reader.parameters():
        prm.requires_grad_(False)

    E_t = torch.from_numpy(E)

    def per_position_nll(use_reader: bool) -> np.ndarray:
        """Next-token NLL at every stream position, injected on contiguous chunks."""
        out = np.full(T, np.nan, dtype=np.float64)
        handle = install_reader_hook(model, args.layer, reader) if use_reader else None
        try:
            starts = list(range(0, T - 1, args.chunk))
            if args.max_chunks:
                starts = starts[: args.max_chunks]
            with torch.no_grad():
                for s in starts:
                    e = min(T, s + args.chunk)
                    win = torch.from_numpy(tokens[s:e]).unsqueeze(0).to(args.device)
                    if use_reader:
                        model._current_ple_e_t = E_t[s:e].unsqueeze(0).to(args.device)
                    logits = model(input_ids=win).logits[0, :-1].float()
                    tgt = win[0, 1:]
                    nll = F.cross_entropy(logits, tgt, reduction="none")
                    lo = s + (args.warmup if s > 0 else 0)
                    out[lo : s + nll.numel()] = nll[lo - s :].double().cpu().numpy()
                    del logits, nll
        finally:
            if handle is not None:
                handle.remove()
                model._current_ple_e_t = None
        return out

    log("arm none: pure backbone (this should reproduce 1.5c's bb_final) ...")
    nll_none = per_position_nll(False)
    log("arm frozen: frozen rows injected ...")
    nll_frozen = per_position_nll(True)
    log(f"  ({time.time() - t0:.0f}s)")

    # ---- trained arm: substitute ONLY where training could have acted ----- #
    # Resolve each stream position's snapshot index to a row of the kept bank.
    kk = np.searchsorted(keep_index, j_all)
    np.clip(kk, 0, keep_index.size - 1, out=kk)
    in_keep = hit_all & (keep_index[kk] == j_all)
    sub_pos = np.flatnonzero(in_keep) + 2
    E[sub_pos] = trained[kk[in_keep]]
    E_t = torch.from_numpy(E)
    log(f"arm trained: substituted {sub_pos.size:,} positions that have a trained row")
    nll_trained = per_position_nll(True)

    # ---- the paired quantity, on seen positions only ---------------------- #
    score = positions[np.isin(positions, sub_pos)]
    if score.size == 0:
        raise SystemExit("no scored position has a trained row")
    lf, lt, ln = nll_frozen[score], nll_trained[score], nll_none[score]
    ok = np.isfinite(lf) & np.isfinite(lt) & np.isfinite(ln)
    score, lf, lt, ln = score[ok], lf[ok], lt[ok], ln[ok]
    delta = lf - lt
    result: dict = {
        "n_scored_positions": int(score.size),
        "n_stream_positions": T,
        "n_positions_with_trained_row": int(sub_pos.size),
        "delta": paired_stats(delta),
        "delta_shuffled_placeholder": None,
        "mean_nll": {
            "none_pure_backbone": float(ln.mean()),
            "frozen_rows": float(lf.mean()),
            "trained_rows": float(lt.mean()),
        },
        "injection_effect_vs_pure_backbone": {
            "frozen_minus_none": paired_stats(lf - ln),
            "trained_minus_none": paired_stats(lt - ln),
        },
        "consistency": consistency,
        "config": {
            "trained_bank": str(args.trained_bank), "keep_index": str(args.keep_index),
            "tokens": str(args.tokens), "positions": str(args.positions),
            "reader": str(args.reader), "reader_kind": args.reader_kind,
            "layer": args.layer, "chunk": args.chunk, "warmup": args.warmup,
            "scale": scale,
        },
        "elapsed_seconds": time.time() - t0,
    }
    log(f"Delta = {delta.mean():+.5f} nats  (frozen {lf.mean():.5f} -> trained {lt.mean():.5f})")
    log(f"pure backbone reference: {ln.mean():.5f} nats")

    # Write the PRIMARY result before the optional analysis.  A missing or
    # mis-pathed --counts-npz once destroyed a complete 3-arm evaluation (the
    # three forward passes had all finished) because the exception fired before
    # the JSON was written.  The cheap, optional part must never be able to lose
    # the expensive part.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    flush()
    log(f"wrote {out_path} (primary result)")

    # ---- section 4: the frequency-stratified prediction ------------------- #
    ctx = None
    if args.counts_npz:
        try:
            c = np.load(args.counts_npz)
        except OSError as exc:
            result["by_context_frequency_error"] = f"{type(exc).__name__}: {exc}"
            flush()
            log(f"WARNING: could not read --counts-npz ({exc}); primary result kept")
            return 0
        ctx_all = np.asarray(c["context_counts"], dtype=np.int64)
        # ``context_counts`` is indexed by STREAM POSITION: its length is
        # ``n_eval_tokens``, i.e. the same length as ``cnt_nll``.  So the band of a
        # scored position is a direct lookup -- no searchsorted, no alignment.
        if ctx_all.size != T:
            result["by_context_frequency_error"] = (
                f"context_counts length {ctx_all.size} != eval stream {T}"
            )
            flush()
            log("WARNING: counts npz is not this stream; primary result kept")
            return 0
        ctx = ctx_all[score]
        bands = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 49), (50, 199), (200, 10**9)]
        by_band = {}
        for lo, hi in bands:
            m = (ctx >= lo) & (ctx <= hi)
            if not m.any():
                continue
            by_band[f"[{lo},{hi if hi < 10**9 else '+'}]"] = {
                **paired_stats(delta[m]),
                "mean_nll_frozen": float(lf[m].mean()),
                "mean_nll_trained": float(lt[m].mean()),
            }
        result["by_context_frequency"] = by_band
        result["n_with_seen_context"] = int((ctx >= 1).sum())
        flush()
        log("by context frequency: " + ", ".join(
            f"{k}: n={v['n']} d={v['mean']:+.5f}" for k, v in by_band.items()
        ))

    # ---- section 5: the per-position record ------------------------------- #
    # Written after the primary result and wrapped, so an extra artifact can
    # never cost the thing that took the GPU time.
    try:
        rec = per_position_record(
            score, all_rowids[score], delta, lf, lt, ln, ctx,
            snap=j_all[score], code=codes[score],
        )
        rec_path = out_path.parent / f"{out_path.stem}.deltas.npz"
        np.savez_compressed(rec_path, **rec)
        result["per_position_record"] = {
            "path": rec_path.name,
            "fields": sorted(rec),
            "n": int(score.size),
        }
        flush()
        log(f"wrote {rec_path.name} ({score.size:,} positions, fields: {', '.join(sorted(rec))})")
    except Exception as exc:  # noqa: BLE001 - an extra artifact must never lose the result
        result["per_position_record_error"] = f"{type(exc).__name__}: {exc}"
        flush()
        log(f"WARNING: could not write the per-position record ({exc}); primary result kept")

    flush()
    log(f"final write {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
