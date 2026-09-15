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


def trigram_index(score: np.ndarray, n_trigrams: int) -> np.ndarray:
    """Map scored stream positions to indices into the per-trigram arrays.

    ``codes[i]`` describes the trigram ENDING at ``i + 2`` -- that is where
    ``seen_pos = flatnonzero(hit_all) + 2`` comes from -- so the trigram ending
    at stream position ``t`` lives at ``codes[t - 2]``.  ``j_all`` is parallel to
    ``codes`` and takes the same offset, while ``context_counts`` and
    ``all_rowids`` are indexed by the stream position directly and do NOT.

    Indexing by ``t`` is off by two and fails only for the last few positions.
    The first version of the per-position record did exactly that, and because
    the record is written inside a try/except (so that it can never cost the
    primary result) the failure showed up as an empty ``.deltas.npz`` on every
    arm rather than as a crash: ``IndexError: index 1152889 is out of bounds for
    axis 0 with size 1152889``.  The bounds check below turns that class of
    mistake into a message that names the offset.
    """
    idx = np.asarray(score, dtype=np.int64) - 2
    if idx.size and (idx.min() < 0 or idx.max() >= n_trigrams):
        raise ValueError(
            f"scored positions {idx.min() + 2}..{idx.max() + 2} map to trigram "
            f"indices {idx.min()}..{idx.max()}, outside [0, {n_trigrams}); the "
            "per-trigram arrays carry the t-2 offset and the per-position ones do not"
        )
    return idx


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


# Backbone-surprisal bands for the top-1 reading.  Same spirit as the frequency
# bands: report the tail separately, because an aggregate hides whether anything
# happened where the injection is supposed to act.
SURPRISAL_EDGES = [0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf]


def paired_accuracy(surprisal, hit_a, hit_b, edges=None, name_a="a", name_b="b") -> list[dict]:
    """Top-1 accuracy in backbone-surprisal bands, tested PAIRED.

    Every round in this project has reported mean negative log-likelihood, which
    is a log quantity: it is dominated by positions where the model was very
    wrong, so a large NLL gain can coexist with the model emitting exactly the
    same token.  Top-1 is the first non-log currency here, and it is reported per
    surprisal band because the injection's NLL gain is concentrated in the tail --
    if that gain is real it must show up there, and if it does not, the NLL
    accounting is measuring something that never reaches a decision.

    The arms are paired on the same positions, so the test is on the per-position
    difference ``hit_b - hit_a``, not on two independent proportions.
    """
    surprisal = np.asarray(surprisal, dtype=np.float64)
    hit_a = np.asarray(hit_a, dtype=np.float64)
    hit_b = np.asarray(hit_b, dtype=np.float64)
    if not (surprisal.shape == hit_a.shape == hit_b.shape):
        raise ValueError("surprisal and both hit arrays must be parallel")
    edges = list(SURPRISAL_EDGES if edges is None else edges)
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (surprisal >= lo) & (surprisal < hi)
        n = int(m.sum())
        entry = {
            "band": f"[{lo:g},{'+' if np.isinf(hi) else f'{hi:g}'})",
            "n": n,
            "acc_a": float(hit_a[m].mean()) if n else None,
            "acc_b": float(hit_b[m].mean()) if n else None,
            "delta": None, "se": None, "t": None,
        }
        if n >= 2:
            diff = hit_b[m] - hit_a[m]
            se = float(diff.std(ddof=1) / np.sqrt(n))
            entry["delta"] = float(diff.mean())
            entry["se"] = se
            entry["t"] = (entry["delta"] / se) if se > 0 else None
        out.append(entry)
    return out


def top1_summary(surprisal, hits: dict, edges=None) -> dict:
    """Overall and per-band top-1 accuracy for named arms, with paired deltas.

    ``hits`` maps an arm name to its per-position 0/1 correctness.  The first arm
    is the reference; every other arm is compared to it, paired.
    """
    names = list(hits)
    if len(names) < 2:
        raise ValueError("top1_summary needs at least two arms to pair")
    ref = names[0]
    per_band = {nm: paired_accuracy(surprisal, hits[ref], hits[nm], edges) for nm in names[1:]}
    rows = []
    for i, entry in enumerate(per_band[names[1]]):
        row = {"band": entry["band"], "n": entry["n"], ref: entry["acc_a"]}
        for nm in names[1:]:
            row[nm] = per_band[nm][i]["acc_b"]
            row[f"{nm}_minus_{ref}"] = per_band[nm][i]["delta"]
            row[f"{nm}_t"] = per_band[nm][i]["t"]
        rows.append(row)
    return {
        "reference": ref,
        "overall": {nm: float(np.asarray(hits[nm], dtype=np.float64).mean()) for nm in names},
        "bands": rows,
    }


def fill_unseen(E: np.ndarray, unseen_pos: np.ndarray, mode: str, *, seed: int = 0) -> None:
    """Replace, in place, the rows injected at unseen-trigram positions.

    The seen block is never touched, so every trained-row result is unaffected by
    this ablation.

    ``zero``   is any row content needed at all, or would nothing do?
    ``shuffle`` permutes the very rows being replaced among themselves.  That
               preserves their marginal distribution EXACTLY while destroying the
               trigram-to-row correspondence, so real-above-shuffle is content and
               shuffle-below-zero would mean a wrong row is worse than none.
    ``mean``   replaces every unseen row with their average: one constant,
               content-free vector.
    ``real``   leaves the pretrained shard rows alone -- the default, and the arm
               every earlier result was produced with.

    A fill that silently fails to apply yields an arm identical to ``real``, which
    reads as "content does not matter" -- a false negative in the direction that
    kills the finding.  So the modes are exact where they can be, and the caller
    verifies a sample afterwards.
    """
    if mode == "real":
        return
    if mode not in ("zero", "mean", "shuffle"):
        raise ValueError(f"unknown unseen fill {mode!r}")
    unseen_pos = np.asarray(unseen_pos)
    if unseen_pos.size == 0:
        return
    if mode == "zero":
        E[unseen_pos] = 0.0
        return
    if mode == "mean":
        acc = np.zeros(E.shape[1], dtype=np.float64)
        cnt = 0
        for s in range(0, unseen_pos.size, 100_000):
            e = min(unseen_pos.size, s + 100_000)
            block = np.asarray(E[unseen_pos[s:e]], dtype=np.float64)
            acc += block.sum(axis=0)
            cnt += block.shape[0]
        E[unseen_pos] = (acc / max(cnt, 1)).astype(E.dtype)
        return
    block = np.array(E[unseen_pos])
    perm = np.random.default_rng(seed).permutation(unseen_pos.size)
    for s in range(0, unseen_pos.size, 50_000):
        e = min(unseen_pos.size, s + 50_000)
        E[unseen_pos[s:e]] = block[perm[s:e]]


def per_position_record(score, delta, lf, lt, ln, ctx=None, snap=None, code=None,
                        hits=None, extra=None) -> dict:
    """The per-position record written next to the primary JSON.

    A band mean cannot answer the question the band means raised.  At lr=1e-4 the
    1-9 bands come out worse while the >=10 bands come out better, and the ~150k
    scored positions whose context count is zero come out worse than either.
    Whether that ordering survives a correct frequency variable can only be asked
    of the per-position record, so it carries every key the join could need:

    ``snapshot_index``
        the index into ``E0``/the trained bank -- the embedding row training
        actually moved.  Joined against ``trigram-train-count.npy`` it gives the
        EXACT training count of the trigram that was injected, which is the
        variable the published band table should have used.
    ``trigram_code``
        the injection code of that trigram, so the band can be recomputed or the
        trigram decoded without re-deriving the stream.
    ``context_count``
        the round-168 margin variable, kept for a side-by-side with the published
        table.  It is one token behind the row it describes; see
        docs/round-169-band-variable-off-by-one.md.
    ``rowid`` is deliberately NOT here.  The graft addresses 16 heads per
    position and ``fetch_e_t`` concatenates them into the 2560-wide vector, so
    "the row id" is not a single value -- ``rowids_from_tokens`` returns [T, 16]
    and storing it would add 57 MB per arm for a key with no scalar meaning.
    ``snapshot_index`` identifies the row that training moved, which is the row
    this record is about.

    Every field is parallel to ``score``: index ``i`` of any array describes
    stream position ``score[i]``.
    """
    rec = {
        "score": np.asarray(score, dtype=np.int64),
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
    for arm_name, arr in (hits or {}).items():
        rec[f"hit_{arm_name}"] = np.asarray(arr, dtype=np.uint8)
    # `extra` exists so a second artifact can carry more per-position quantities
    # through the SAME shape check.  Merging before the check is the point: an
    # unvalidated extra array is how a (n, 16) row-id matrix slipped through once.
    for name, arr in (extra or {}).items():
        if name in rec:
            # An extra that shadows `score` or `delta` would corrupt the record
            # silently -- the shape check cannot see it, because the shape is
            # right.  Only the name gives it away.
            raise ValueError(f"extra field {name!r} would overwrite a core record field")
        rec[name] = np.asarray(arr)
    n = rec["score"].size
    for key, arr in rec.items():
        # Require EXACTLY one value per position.  A (n, 16) array has the right
        # leading dimension, and accepting that is how ``all_rowids[score]``
        # slipped through the first version -- it was only caught later by a size
        # comparison that reported the confusing "expected 665, got 10640".
        # Naming the actual shape is the diagnostic.
        if arr.shape != (n,):
            raise ValueError(
                f"per-position record field {key!r} has shape {arr.shape}, expected ({n},): "
                "every field is one value per scored position"
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
    ap.add_argument("--unseen-fill-seed", type=int, default=0)
    ap.add_argument("--unseen-fill", choices=("real", "zero", "mean", "shuffle"),
                    default="real",
                    help="what to inject at positions whose trigram has no snapshot row. "
                         "'real' (default) is the pretrained shard row and reproduces every "
                         "earlier result; the others are the content ablation.")
    ap.add_argument("--gate-record", type=Path, default=None,
                    help="also write a second record over ALL aligned positions, carrying "
                         "the confidence features and has_train_row. The primary record is "
                         "unchanged, so every number published before this flag existed "
                         "stays comparable.")
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
    # ---- the unseen-regime ablation --------------------------------------- #
    # Phase 8 found the injection still gains +0.041296 nats on the 690,303
    # positions whose trigram never appeared in training, where the row comes from
    # the PRETRAINED shard table.  That number has two readings and they mean very
    # different things for a paper:
    #
    #   (a) the row's CONTENT is doing work -- real information about this trigram,
    #       generalising beyond the fine-tuning stream;
    #   (b) the row is a generic perturbation -- any vector through the reader
    #       shifts the logits favourably, and the specific trigram is irrelevant.
    #
    # `zero` separates (b) from "nothing at all": if the gain survives on a zero
    # row, the row was never the source.  `shuffle` permutes the very rows being
    # injected among themselves, which preserves their marginal distribution
    # exactly while destroying the trigram-to-row correspondence -- so real above
    # shuffle is content, and shuffle below zero would mean a WRONG row is worse
    # than none.  `mean` replaces every unseen row with their average.
    #
    # Only the unseen block is touched; the seen positions keep exactly the values
    # they had, so the trained-row results are unaffected by this flag.
    if args.unseen_fill != "real" and unseen_pos.size:
        sample = np.array(E[unseen_pos[:1024]])
        before = float(np.abs(sample).mean())
        fill_unseen(E, unseen_pos, args.unseen_fill, seed=args.unseen_fill_seed)
        after_sample = np.asarray(E[unseen_pos[:1024]])
        after = float(np.abs(after_sample).mean())
        if args.unseen_fill == "zero" and float(np.abs(after_sample).max()) != 0.0:
            raise SystemExit("the zero fill did not take effect")
        if args.unseen_fill == "shuffle" and np.array_equal(sample, after_sample):
            raise SystemExit("the shuffle left the unseen block unchanged")
        log(f"unseen fill '{args.unseen_fill}': {unseen_pos.size:,} positions "
            f"(sample |e| {before:.3e} -> {after:.3e})")
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
        """Next-token NLL and top-1 correctness at every stream position.

        Two currencies per position, because they answer different questions.
        NLL is the quantity every round so far has reported, and it is a LOG
        quantity: mean negative log-likelihood is dominated by positions where
        the model was very wrong, so a large NLL gain can coexist with no change
        in what the model would actually have emitted.  Top-1 is the first
        reading in this project that is not a log-loss, and the injection's gain
        is concentrated exactly where NLL is largest -- so the two can disagree
        in either direction and the disagreement is the informative part.
        """
        out = np.full(T, np.nan, dtype=np.float64)
        hit = np.zeros(T, dtype=np.uint8)
        # Confidence features, for a gate that must run BEFORE the answer is known.
        # `nll` is the surprisal of the realised token and is therefore an oracle:
        # you cannot compute it without already knowing what happened.  These four
        # come from the model's own output distribution and are available at the
        # moment the decision has to be made:
        #   ent       predictive entropy, -sum p log p, in nats
        #   maxp      top-1 probability
        #   logmargin log p(1st) - log p(2nd), in nats
        #   top1      the argmax id, so two arms can be compared for agreement
        feats = {
            k: np.full(T, np.nan, dtype=np.float32)
            for k in ("ent", "maxp", "logmargin")
        }
        feats["top1"] = np.full(T, -1, dtype=np.int32)
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
                    logp = F.log_softmax(logits, dim=-1)
                    del logits
                    nll = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
                    lo = s + (args.warmup if s > 0 else 0)
                    n = nll.numel()
                    out[lo : s + n] = nll[lo - s :].double().cpu().numpy()
                    correct = (logp.argmax(dim=-1) == tgt).to(torch.uint8)
                    hit[lo : s + n] = correct[lo - s :].cpu().numpy()
                    if args.gate_record:
                        two = logp.topk(2, dim=-1)
                        p = logp.exp()
                        ent = -(p * logp).sum(dim=-1)
                        del p
                        sl = slice(lo - s, n)
                        feats["ent"][lo : s + n] = ent[sl].cpu().numpy()
                        feats["maxp"][lo : s + n] = two.values[sl, 0].exp().cpu().numpy()
                        feats["logmargin"][lo : s + n] = (two.values[sl, 0] - two.values[sl, 1]).cpu().numpy()
                        feats["top1"][lo : s + n] = two.indices[sl, 0].to(torch.int32).cpu().numpy()
                        del two, ent
                    del logp, nll, correct, tgt
        finally:
            if handle is not None:
                handle.remove()
                model._current_ple_e_t = None
        return out, hit, feats

    log("arm none: pure backbone (this should reproduce 1.5c's bb_final) ...")
    nll_none, hit_none, fe_none = per_position_nll(False)
    log("arm frozen: frozen rows injected ...")
    nll_frozen, hit_frozen, fe_frozen = per_position_nll(True)
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
    nll_trained, hit_trained, fe_trained = per_position_nll(True)

    # ---- the paired quantity, on seen positions only ---------------------- #
    score = positions[np.isin(positions, sub_pos)]
    if score.size == 0:
        raise SystemExit("no scored position has a trained row")
    lf, lt, ln = nll_frozen[score], nll_trained[score], nll_none[score]
    hf, ht, hn = hit_frozen[score], hit_trained[score], hit_none[score]
    ok = np.isfinite(lf) & np.isfinite(lt) & np.isfinite(ln)
    score, lf, lt, ln = score[ok], lf[ok], lt[ok], ln[ok]
    hf, ht, hn = hf[ok], ht[ok], hn[ok]
    delta = lf - lt

    # The regime every earlier eval could not see: positions whose trigram has no
    # E0 row, so no row training could move.  They are injected anyway (with a
    # pretrained shard row) and they feed the autoregressive state, so their
    # effect is real and was simply never scored.  Reported unconditionally and
    # cheaply, so it lands next to the headline instead of in a side artifact.
    unseen_all = np.zeros(T, dtype=bool)
    unseen_all[unseen_pos] = True
    upos = positions[unseen_all[positions] & np.isfinite(nll_none[positions])
                    & np.isfinite(nll_frozen[positions])]
    unseen_report = None
    if upos.size:
        ug = nll_none[upos] - nll_frozen[upos]
        unseen_report = {
            "n": int(upos.size),
            "mean_nll_none": float(nll_none[upos].mean()),
            "mean_nll_frozen": float(nll_frozen[upos].mean()),
            "injection_gain": float(ug.mean()),
            "frac_positive": float((ug > 0).mean()),
        }

    result: dict = {
        "n_scored_positions": int(score.size),
        "n_stream_positions": T,
        "n_positions_with_trained_row": int(sub_pos.size),
        "unseen_regime": unseen_report,
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
            "unseen_fill": args.unseen_fill,
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
    ctx_all = None
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

    # ---- section 4b: top-1, the first non-log currency --------------------- #
    # Cheap (the argmax is already computed) and it is the reading that decides
    # whether the NLL accounting above reaches a decision at all.  Written before
    # the per-position record, for the same reason the primary result is written
    # before both.
    try:
        result["top1"] = top1_summary(ln, {"none": hn, "frozen": hf, "trained": ht})
        flush()
        log("top-1 accuracy: " + ", ".join(
            f"{k}={v:.4f}" for k, v in result["top1"]["overall"].items()
        ))
        for row in result["top1"]["bands"]:
            if row["n"]:
                log(f"  surprisal {row['band']:<12} n={row['n']:>7,} "
                    f"none={row['none']:.4f} frozen={row['frozen']:.4f} "
                    f"delta={row['frozen_minus_none']:+.4f} t={row['frozen_t']:+.2f}")
    except Exception as exc:  # noqa: BLE001 - optional reading, must not cost the rest
        result["top1_error"] = f"{type(exc).__name__}: {exc}"
        flush()
        log(f"WARNING: could not compute top-1 ({exc})")

    # ---- section 5: the per-position record ------------------------------- #
    # Written after the primary result and wrapped, so an extra artifact can
    # never cost the thing that took the GPU time.
    try:
        tri = trigram_index(score, codes.size)
        rec = per_position_record(
            score, delta, lf, lt, ln, ctx,
            snap=j_all[tri], code=codes[tri],
            hits={"none": hn, "frozen": hf, "trained": ht},
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

    # ---- section 6: the gate record, over ALL aligned positions ----------- #
    # Two questions need a record the primary one cannot give.
    #
    # 1. The gate.  Whether to trust the memory is decided BEFORE the token is
    #    known, so the deciding features must be the model's own confidences, not
    #    the surprisal of what actually happened.  They are recorded here for
    #    every arm, so a gate can be trained and scored offline at zero GPU cost.
    #
    # 2. The regime this project has never measured.  `score` is restricted to
    #    positions whose trigram has an E0 row, i.e. that appeared in training --
    #    39% of the stream.  The other 61% still receive a row fetched from the
    #    shard table, so the injection acts there, but nothing was ever scored
    #    there.  Recording all aligned positions makes the unseen-trigram
    #    injection effect readable for the first time, at no extra forward pass.
    #
    # The primary record is deliberately NOT widened: every number published
    # before this flag existed has to stay comparable.
    if args.gate_record:
        try:
            finite = np.isfinite(nll_none) & np.isfinite(nll_frozen)
            allpos = positions[finite[positions]]
            tri = trigram_index(allpos, codes.size)
            has_row = np.isin(allpos, sub_pos).astype(np.uint8)
            grec = {
                "score": allpos,
                "nll_none": nll_none[allpos].astype(np.float32),
                "nll_frozen": nll_frozen[allpos].astype(np.float32),
                "nll_trained": nll_trained[allpos].astype(np.float32),
                "has_train_row": has_row,
                "has_snapshot_row": hit_all[tri].astype(np.uint8),
                "snapshot_index": np.where(hit_all[tri], j_all[tri], -1),
                "trigram_code": codes[tri],
            }
            if ctx_all is not None:
                grec["context_count"] = ctx_all[allpos]
            for tag, fe in (("none", fe_none), ("frozen", fe_frozen), ("trained", fe_trained)):
                for k in ("ent", "maxp", "logmargin", "top1"):
                    grec[f"{k}_{tag}"] = fe[k][allpos]
            for key, arr in grec.items():
                if arr.shape != (allpos.size,):
                    raise ValueError(f"gate record field {key!r} has shape {arr.shape}")
            gpath = Path(args.gate_record)
            gpath.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(gpath, **grec)
            n_norow = int((has_row == 0).sum())
            result["gate_record"] = {
                "path": gpath.name,
                "fields": sorted(grec),
                "n": int(allpos.size),
                "n_without_train_row": n_norow,
            }
            flush()
            log(f"wrote {gpath.name} ({allpos.size:,} aligned positions, "
                f"{n_norow:,} of them with no trainable row)")
        except Exception as exc:  # noqa: BLE001 - an extra artifact must never lose the result
            result["gate_record_error"] = f"{type(exc).__name__}: {exc}"
            flush()
            log(f"WARNING: could not write the gate record ({exc}); primary result kept")

    flush()
    log(f"final write {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
