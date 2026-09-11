#!/usr/bin/env python3
"""Count-based n-gram LM reference frame for the PLE graft (Round 158).

Why this script exists
----------------------
Every round so far compared "a hashed n-gram memory table is grafted" against
"no table".  The honest control -- spending the *same storage* on a plain
count-based n-gram model -- was never built (see
``docs/round-153-goal-tech-debt-and-development-plan.md`` and
``docs/round-157-why-the-graft-is-a-prior-not-a-knowledge-channel.md``).
This script produces that reference frame: explicit count-based n-gram language
models (unigram / bigram / trigram / 4-gram) with interpolated (Modified)
Kneser-Ney smoothing, evaluated on held-out token positions, together with a
*measured* storage account so the frozen 48 GiB PLE table can be placed on the
same accuracy-vs-bytes axis.

Environment constraint
----------------------
The AutoDL container is capped at **2 GiB RSS and 0.5 CPU** (cgroup
memory.max / cpu.max -- ``nproc`` and ``free`` report host values and lie).
So this implementation is deliberately frugal and fully vectorised:

* no Python dict of n-grams anywhere -- counting is ``np.lexsort`` over integer
  windows, lookup is ``np.searchsorted`` plus a vectorised ragged binary search;
* every scoring pass is numpy over the whole held-out stream (no Python loop
  over positions);
* accuracy is accumulated in position chunks so the (chunk x candidates) dense
  score matrix stays a few tens of MB;
* peak RSS (``VmHWM``) is reported in the JSON; wrap the run in
  ``flock /tmp/qwen35_heavy.lock`` when siblings share the box.

Model
-----
Interpolated Kneser-Ney with per-count absolute discounts.  For a model whose
highest order is ``M``:

    p_1(w)     = disc_1(c_1(w)) + Lambda_1 / V_uni
    p_k(w|h)   = disc_k(c_k(h w)) + gamma_k(h) * p_{k-1}(w | h[1:])      2 <= k <= M
    disc_k(c)  = max(c - D_k(min(c, 3)), 0) / Z_k(h)
    gamma_k(h) = 1 - sum_w disc_k(c_k(h w))              (= backoff mass)

where ``c_k`` is the **raw** count of the k-gram when k == M (the highest
order) and the **continuation count** N1+(. u) = number of distinct left
extensions when k < M (the usual Kneser-Ney lower-order redefinition);
``Z_k(h)`` is the total of those counts for context h; and ``Lambda_1`` is the
discounted unigram mass spread uniformly over a fixed vocabulary of size
``V_uni``.  The uniform floor is what gives tokens never seen in training a
finite probability; it is called out explicitly because OOV positions dominate
inter-model NLL differences.

``--smoothing`` selects the discounts:

* ``mkn`` (default) -- Modified Kneser-Ney, Chen & Goodman (1999):
  Y = n1/(n1+2 n2), D1 = 1 - 2Y n2/n1, D2 = 2 - 3Y n3/n2, D3 = 3 - 4Y n4/n3,
  with D(c>=3) = D3.  Discounting singletons hard is what stops higher orders
  from over-trusting count-1 contexts.
* ``kn``  -- single absolute discount D = n1/(n1+2 n2) for all counts.
* ``addk`` -- interpolated add-k: disc = c/(Z + k V_uni), gamma = k V_uni/(Z + k V_uni)
  (handled by the same code path with a zero discount table and a shifted Z).

If the three-discount estimator would leave some context with zero backoff mass
(possible when the count-of-counts degenerates), the level falls back to the
single discount, then to a generic half count; the fallback is logged and
recorded in the JSON.  Unseen contexts back off by dropping the oldest token;
the scorer walks levels down in one pass, so a position that misses the 4-gram
context still gets the full trigram/bigram/unigram mixture.

Evaluation
----------
* Every model is scored on exactly the same held-out positions: indices
  ``i >= max(orders) - 1`` of the evaluation stream.  That count is
  ``n_scored_positions_per_model``.
* ``NLL`` = mean over those positions of ``-ln p(target)`` (natural log),
  ``ppl`` = ``exp(NLL)``, ``bits/token`` = ``NLL / ln 2``.  No candidate
  restriction, no subsampling.
* Accuracy uses argmax, so it needs the candidate distribution; it runs on a
  deterministic random subsample (``--acc-positions``) and is reported for two
  candidate sets: ``top{K}`` (K most frequent *training* tokens, the set a
  sibling experiment uses) and ``full_vocab``.  ``full_vocab`` is taken to be
  the *training* vocabulary: tokens never seen in training all score exactly the
  uniform floor, which is strictly below the score of any seen token, so the
  argmax/top-5 ranking is identical to the one over all ``V_uni`` tokens (the
  full 248k-token argmax would need a 2 GB score matrix).  For the restricted
  set, positions whose target is outside the candidate set are counted in
  ``n_targets_outside_candidates`` and EXCLUDED from that accuracy; coverage is
  reported so the exclusion is visible.

Storage accounting
------------------
Each level is packed the way a serving table would be: a CSR layout of
``contexts -> (next_token, count)`` sorted lexicographically, i.e. exactly

    ctx_sorted  (C,)     int64    distinct (k-1)-gram contexts, exact mixed-radix
    starts      (C+1,)   int64    row offsets into the value arrays
    next_tokens (N_k,)   int32    distinct next tokens, sorted per context
    counts      (N_k,)   uint32   stored count of each (context, next)

No hashing (context keys up to 3 tokens x 18 bits fit exactly in int64), no
compression, no Python objects.  ``packed_bytes_total`` is the measured
``nbytes`` of exactly those arrays.  The per-context ``zeff``/``discsum`` arrays
that scoring keeps resident are reported separately as ``scoring_side_bytes``:
they are recomputable from the table and are not part of the shipped bytes.

Projection onto the frozen PLE table (default 48 GiB): (a) a linear "constant
bytes per training token" projection and (b) a fit of ``bytes(N) ~ N**b`` over
corpus prefixes (Heaps-like).  Both extrapolate far outside the fitted range;
see ``assumptions`` in the JSON.  Accuracy is never extrapolated.

Usage
-----
    flock /tmp/qwen35_heavy.lock python scripts/bench_ngram_reference.py \
        --train-npy data/phase1/PURE_WIKI/tokens.npy \
        --eval-npy  data/phase1/wikitext-heldout/tokens.npy \
        --orders 1,2,3,4 --smoothing mkn --uniform-vocab 248047 \
        --candidate-from data/phase1/PURE_WIKI/tokens.npy \
        --acc-positions 20000 --acc-candidates 5000 --acc-positions-full 2000 \
        --scaling-prefixes 0.25,0.5,1.0 \
        --output outputs/ngram-reference-PURE_WIKI-heldout.json \
        --markdown outputs/ngram-reference-PURE_WIKI-heldout.md

Use ``--holdout-tail N`` for the in-stream split (train on ``tokens[:-N]``,
evaluate on ``tokens[-N:]``) instead of ``--eval-npy``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

GIB = 1 << 30
COUNT_CAP = 3      # discounts are bucketed as count 1, 2, >=3
COUNT_CLIP = 5     # count-of-counts tracked exactly up to 4, then clipped
LOG_MIN = math.log(1e-300)


def log(msg: str) -> None:
    print("[ngram-ref] {}".format(msg), flush=True)


def peak_rss_bytes() -> int:
    """Measured high-water RSS of this process (VmHWM on Linux), in bytes."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import resource
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB, macOS reports bytes
        return rss * 1024 if sys.platform.startswith("linux") else rss
    except Exception:  # noqa: BLE001
        return -1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_tokens(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        arr = np.loadtxt(path, dtype=np.int64)
    arr = np.asarray(arr).reshape(-1)
    if not np.issubdtype(arr.dtype, np.integer):
        raise SystemExit("token array must be integer typed, got {}".format(arr.dtype))
    if arr.size and int(arr.max()) >= 2 ** 31:
        raise SystemExit("token ids must fit in int32")
    return arr.astype(np.int32, copy=False)


# --------------------------------------------------------------------------
# counting (sort based, no Python dicts)
# --------------------------------------------------------------------------
def unique_rows(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lexicographically sort rows of ``a``; return (unique_rows, counts).

    Column 0 is the primary key, which is what makes context = ``row[:-1]``
    contiguous in the CSR packing below.
    """
    n, k = a.shape
    if n == 0:
        return np.zeros((0, k), dtype=np.int32), np.zeros((0,), dtype=np.int64)
    if k == 1:
        flat = np.ascontiguousarray(a[:, 0])
        order = np.argsort(flat, kind="stable")
        sa = flat[order]
        new = np.empty(n, dtype=bool)
        new[0] = True
        new[1:] = sa[1:] != sa[:-1]
        idx = np.flatnonzero(new)
        counts = np.diff(np.append(idx, n)).astype(np.int64)
        return sa[idx].reshape(-1, 1).astype(np.int32), counts
    order = np.lexsort(a.T[::-1])
    sa = a[order]
    del order
    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = np.any(sa[1:] != sa[:-1], axis=1)
    idx = np.flatnonzero(new)
    del new
    counts = np.diff(np.append(idx, n)).astype(np.int64)
    return sa[idx].astype(np.int32), counts


def pack_contexts(rows: np.ndarray, vocab: int) -> np.ndarray:
    """Pack each row of ``rows`` into one exact int64 (mixed radix, base vocab).

    The most recent token is the least significant digit, so for a k-gram the
    context key is ``key // vocab`` and the joint key would be ``key * vocab + w``.
    """
    n, k = rows.shape
    if k == 0:
        return np.zeros(n, dtype=np.int64)
    bits = max(1, int(vocab - 1).bit_length())
    if k * bits > 62:
        raise SystemExit("context of {} tokens does not fit in int64 for V={}".format(
            k, vocab))
    acc = np.zeros(n, dtype=np.int64)
    for j in range(k):
        acc *= vocab
        acc += rows[:, j].astype(np.int64)
    return acc


def context_keys_at(stream: np.ndarray, positions: np.ndarray, ctx_len: int,
                    vocab: int) -> np.ndarray:
    """Packed key of the ``ctx_len`` tokens preceding each position."""
    if ctx_len == 0:
        return np.zeros(positions.shape[0], dtype=np.int64)
    acc = np.zeros(positions.shape[0], dtype=np.int64)
    base = positions - ctx_len
    for j in range(ctx_len):
        acc *= vocab
        acc += stream[base + j].astype(np.int64)
    return acc


class LevelTable:
    """Packed CSR table for one n-gram order (k = number of tokens)."""

    __slots__ = ("k", "ctx_sorted", "starts", "next_tokens", "counts",
                 "count_of_counts", "zeff", "discsum", "sub_table", "D",
                 "nbytes", "scoring_side_bytes", "norm_dev")

    def __init__(self, k, ctx_sorted, starts, next_tokens, counts):
        self.k = k
        self.ctx_sorted = ctx_sorted
        self.starts = starts
        self.next_tokens = next_tokens
        self.counts = counts
        self.count_of_counts = None
        self.zeff = None
        self.discsum = None
        self.sub_table = None
        self.D = None
        self.nbytes = int(ctx_sorted.nbytes + starts.nbytes
                          + next_tokens.nbytes + counts.nbytes)
        self.scoring_side_bytes = 0
        self.norm_dev = None

    @property
    def n_contexts(self) -> int:
        return int(self.ctx_sorted.shape[0])

    @property
    def n_entries(self) -> int:
        return int(self.next_tokens.shape[0])

    def storage_breakdown(self) -> Dict[str, object]:
        return {
            "order": self.k,
            "distinct_contexts": self.n_contexts,
            "distinct_ngrams": self.n_entries,
            "ctx_sorted_dtype": str(self.ctx_sorted.dtype),
            "starts_dtype": str(self.starts.dtype),
            "next_tokens_dtype": str(self.next_tokens.dtype),
            "counts_dtype": str(self.counts.dtype),
            "bytes_ctx_sorted": int(self.ctx_sorted.nbytes),
            "bytes_starts": int(self.starts.nbytes),
            "bytes_next_tokens": int(self.next_tokens.nbytes),
            "bytes_counts": int(self.counts.nbytes),
            "bytes_total": int(self.nbytes),
            "row_normalization_deviation": self.norm_dev,
        }


def build_level(rows: np.ndarray, counts: np.ndarray, vocab: int,
                key_mode: str = "exact", mults: Optional[np.ndarray] = None) -> LevelTable:
    """Build a packed CSR level from unique n-grams.

    ``key_mode`` decides how a context is stored:

    * ``exact`` -- mixed-radix packed int64 (only possible for <= 3 context
      tokens at this vocabulary);
    * ``hash``  -- 64-bit rolling hash of the context, so the per-entry key cost
      is 8 bytes at *any* context length.  This is what an Engram-style table
      does (row id = hash of the n-gram) and it is what makes a matched-bytes
      comparison across window lengths meaningful: an exact 16-token context key
      would cost 60 bytes instead of 8 and would penalise long windows twice.
      Collision probability at ~1e6 contexts is < 1e-7.
    """
    n, k = rows.shape
    if k == 1:
        ctx_sorted = np.zeros(1, dtype=np.int64)
        starts = np.array([0, n], dtype=np.int64)
        next_tokens = rows[:, 0].astype(np.int32)
    else:
        ctx_len = k - 1
        if key_mode == "hash" or ctx_len > 3:
            if mults is None:
                mults = default_mults()
            ctx_keys = ngram_hash_rows(np.ascontiguousarray(rows[:, :-1]),
                                       vocab, mults)
        else:
            ctx_keys = pack_contexts(rows[:, :-1], vocab)
        nxt = rows[:, -1].astype(np.int32)
        # sort by (context key, next token) so equal contexts stay contiguous
        order = np.lexsort((nxt, ctx_keys))
        ctx_keys = ctx_keys[order]
        nxt = nxt[order]
        counts = counts[order]
        del order
        new = np.empty(n, dtype=bool)
        new[0] = True
        new[1:] = ctx_keys[1:] != ctx_keys[:-1]
        row_start = np.flatnonzero(new)
        del new
        ctx_sorted = ctx_keys[row_start]
        starts = np.append(row_start, n).astype(np.int64)
        next_tokens = nxt
        del ctx_keys, row_start
    if n and int(counts.max()) < 2 ** 32:
        counts_small = counts.astype(np.uint32)
    else:
        counts_small = counts.astype(np.uint64)
    return LevelTable(k, ctx_sorted, starts, next_tokens, counts_small)


def prune_level(lvl: LevelTable, min_count: int, smoothing: str, addk: float,
                vocab: int, notes: List[str], protect: bool = False) -> LevelTable:
    """Return a copy keeping only entries stored >= ``min_count`` times.

    This is the pruning rule for the matched-bytes comparison: corpus statistics
    are *not* recomputed (a real pruned table does not recount the corpus
    either), so a surviving context keeps the continuation count it had in the
    full stream; rows that fall below the threshold are simply absent and the
    scorer backs off exactly as for an unseen context.  Rows that lose every
    entry are dropped, so bytes shrink monotonically with the threshold.
    """
    if min_count <= 1 or protect:
        return lvl
    counts = lvl.counts
    entry_keep = counts.astype(np.int64) >= min_count
    if not entry_keep.any():
        empty = LevelTable(lvl.k, lvl.ctx_sorted[:0], np.zeros(1, dtype=np.int64),
                           lvl.next_tokens[:0], np.zeros(0, dtype=np.uint32))
        finalize_level(empty, smoothing, addk, vocab, notes)
        return empty
    row_len = np.diff(lvl.starts)
    kept_per_row = np.add.reduceat(entry_keep.astype(np.int64), lvl.starts[:-1])
    row_keep = kept_per_row > 0
    idx = np.flatnonzero(entry_keep)
    row_of_entry = np.repeat(np.arange(row_len.shape[0], dtype=np.int64), row_len)
    idx = idx[row_keep[row_of_entry[idx]]]
    new_next = np.ascontiguousarray(lvl.next_tokens[idx])
    new_counts = np.ascontiguousarray(counts[idx])
    new_ctx = np.ascontiguousarray(lvl.ctx_sorted[row_keep])
    new_starts = np.concatenate(
        ([0], np.cumsum(kept_per_row[row_keep]))).astype(np.int64)
    pruned = LevelTable(lvl.k, new_ctx, new_starts, new_next, new_counts)
    finalize_level(pruned, smoothing, addk, vocab, notes)
    del entry_keep, kept_per_row, row_keep, idx, row_of_entry, row_len
    return pruned


def discounts_from_coc(coc: Dict[str, int], smoothing: str) -> Dict[str, object]:
    """Return the discount triple and the single-discount fallback Y.

    ``kn`` uses one absolute discount D = n1/(n1+2 n2) for every count; ``mkn``
    (Modified Kneser-Ney, Chen & Goodman 1999) uses Y = n1/(n1+2 n2),
    D1 = 1 - 2Y n2/n1, D2 = 2 - 3Y n3/n2, D3 = 3 - 4Y n4/n3.
    """
    n1, n2, n3, n4 = coc["n1"], coc["n2"], coc["n3"], coc["n4"]
    if smoothing == "addk":
        return {"D": [0.0, 0.0, 0.0], "Y": 0.0}
    if n1 == 0 and n2 == 0:
        # No count-of-counts signal (every stored n-gram occurs >= 3 times):
        # both estimators would give D = 0, which leaves no smoothing mass at
        # all.  Use a generic half count instead of silently producing -inf.
        return {"D": [0.5, 0.5, 0.5], "Y": 0.5}
    Y = n1 / (n1 + 2.0 * n2) if (n1 + 2 * n2) > 0 else 0.5
    Y = min(max(Y, 1e-3), 0.99)
    if smoothing == "kn":
        return {"D": [Y, Y, Y], "Y": Y}
    D1 = 1.0 - 2.0 * Y * n2 / n1 if n1 > 0 else Y
    D2 = 2.0 - 3.0 * Y * n3 / n2 if n2 > 0 else Y
    D3 = 3.0 - 4.0 * Y * n4 / n3 if n3 > 0 else Y
    D1 = min(max(D1, 0.0), 1.0 - 1e-6)
    D2 = min(max(D2, 0.0), 2.0 - 1e-6)
    D3 = min(max(D3, 0.0), 3.0 - 1e-6)
    if D1 <= 0.0:
        D1 = Y
    return {"D": [D1, D2, D3], "Y": Y}


def finalize_level(lvl: LevelTable, smoothing: str, addk: float, vocab: int,
                   notes: List[str]) -> None:
    """Attach discounts, Z, retained mass and the per-row invariant check."""
    counts = lvl.counts.astype(np.int64)
    coc = np.bincount(np.minimum(counts, COUNT_CLIP), minlength=COUNT_CLIP + 1) \
        if counts.size else np.zeros(COUNT_CLIP + 1, dtype=np.int64)
    n_total = int(counts.shape[0])
    lvl.count_of_counts = {
        "n1": int(coc[1]), "n2": int(coc[2]), "n3": int(coc[3]), "n4": int(coc[4]),
        "n_ge3": int(n_total - coc[1] - coc[2]), "n_total": n_total,
    }
    zeff = (np.add.reduceat(counts, lvl.starts[:-1]).astype(np.float64)
            if counts.size else np.zeros((0,), dtype=np.float64))
    if smoothing == "addk":
        lvl.sub_table = np.zeros(COUNT_CAP + 1, dtype=np.float64)
        lvl.zeff = zeff + addk * vocab
        lvl.D = [None, None, None]
    else:
        info = discounts_from_coc(lvl.count_of_counts, smoothing)
        D, Y = info["D"], info["Y"]  # type: ignore
        # A context whose stored continuations all fall in a zero-discount
        # bucket would get gamma = 0, hence p = 0 for an unseen continuation and
        # an -inf NLL.  Accept the three-discount estimator only if every row
        # keeps a strictly positive backoff weight; else fall back to the single
        # discount, then to a generic half count.
        chosen = None
        for cand in (D, [Y, Y, Y], [0.5, 0.5, 0.5]):
            sub = np.array([0.0, cand[0], cand[1], cand[2]], dtype=np.float64)
            retained = np.maximum(counts - sub[np.minimum(counts, COUNT_CAP)], 0.0)
            per_ctx = np.add.reduceat(retained, lvl.starts[:-1]) if counts.size \
                else np.zeros((0,), dtype=np.float64)
            if not counts.size or not bool((per_ctx >= zeff).any()):
                chosen = (cand, retained, per_ctx)
                break
        cand, retained, per_ctx = chosen  # type: ignore
        lvl.sub_table = np.array([0.0, cand[0], cand[1], cand[2]], dtype=np.float64)
        lvl.zeff = zeff
        lvl.D = [float(x) for x in cand]
        if cand is not D:
            notes.append("order {}: discounts {} degenerate -> {}".format(
                lvl.k, [round(float(x), 4) for x in D],
                [round(float(x), 4) for x in cand]))
    if smoothing == "addk":
        retained = counts.astype(np.float64)
    # retained mass per row, recomputed independently of discsum below
    lvl.discsum = np.add.reduceat(retained, lvl.starts[:-1]) if counts.size \
        else np.zeros((0,), dtype=np.float64)
    lvl.scoring_side_bytes = int(lvl.zeff.nbytes + lvl.discsum.nbytes)
    # invariant: per row, sum_w disc_w + gamma == 1 (this is what catches a
    # mismatch between the per-entry discount and the stored retained mass)
    if counts.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            per_entry = retained / np.repeat(lvl.zeff, np.diff(lvl.starts))
            row_sum = np.add.reduceat(per_entry, lvl.starts[:-1])
            gamma = 1.0 - lvl.discsum / lvl.zeff
            worst = float(np.abs(row_sum + gamma - 1.0).max())
        lvl.norm_dev = worst
        if not (worst < 1e-9):
            notes.append("order {}: row normalization deviation {:.2e}".format(
                lvl.k, worst))
    else:
        lvl.norm_dev = 0.0
    del counts, retained


def build_counts(train: np.ndarray, orders: Sequence[int], vocab: int):
    """Raw n-gram counts for every needed order plus KN continuation counts."""
    raw: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    cont: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    max_order = max(orders)
    needed = sorted(set(list(orders) + [o + 1 for o in orders if o < max_order]))
    for k in needed:
        win = np.lib.stride_tricks.sliding_window_view(train, k)
        uq, ct = unique_rows(win)
        del win
        raw[k] = (uq, ct)
        log("  order {}: {:,} distinct n-grams (raw)".format(k, uq.shape[0]))
        gc.collect()
    for k in orders:
        if k < max_order:
            uq, ct = cont_counts(raw[k + 1][0])
            cont[k] = (uq, ct)
            log("  order {}: {:,} distinct n-grams (continuation)".format(k, uq.shape[0]))
            gc.collect()
    return raw, cont


def cont_counts(rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """KN continuation counts of k-grams from the distinct (k+1)-grams.

    A k-gram's continuation count is the number of distinct left extensions it
    has, i.e. the multiplicity of its suffix among distinct (k+1)-grams.
    """
    return unique_rows(np.ascontiguousarray(rows[:, 1:]))


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class NgramModel:
    """Interpolated KN / add-k model assembled from prebuilt level tables."""

    def __init__(self, order: int, levels: Dict[int, LevelTable], vocab: int,
                 smoothing: str, addk: float, notes: List[str],
                 lookup: str = "interpolate", key_mode: str = "exact"):
        self.order = order
        self.levels = levels
        self.vocab = vocab
        self.smoothing = smoothing
        self.addk = addk
        self.lookup = lookup
        self.key_mode = key_mode
        self.mults = default_mults()
        lvl1 = levels[1]
        zeff1 = float(lvl1.zeff[0]) if lvl1.zeff.size else 1.0
        zeff1 = zeff1 if zeff1 > 0 else 1.0
        self.floor = max(0.0, 1.0 - float(lvl1.discsum[0]) / zeff1) / vocab
        seen = lvl1.next_tokens.astype(np.int64)
        c1 = lvl1.counts.astype(np.float64)
        sub = lvl1.sub_table[np.minimum(lvl1.counts.astype(np.int64), COUNT_CAP)]
        p1 = np.full(vocab, self.floor, dtype=np.float64)
        p1[seen] = np.maximum(c1 - sub, 0.0) / zeff1 + self.floor
        self.p1_dense = p1
        self.vocab_seen = int(seen.shape[0])

    # -- lookups -----------------------------------------------------------
    def row_index(self, lvl: LevelTable, ctx_keys: np.ndarray):
        C = lvl.ctx_sorted.shape[0]
        idx = np.searchsorted(lvl.ctx_sorted, ctx_keys)
        np.clip(idx, 0, C - 1, out=idx)
        valid = lvl.ctx_sorted[idx] == ctx_keys
        return idx, valid

    def row_counts(self, lvl: LevelTable, idx: np.ndarray, valid: np.ndarray,
                   w: np.ndarray) -> np.ndarray:
        """Vectorised ragged binary search: count of token ``w`` inside each row."""
        n_tok = lvl.next_tokens.shape[0]
        s = lvl.starts[idx]
        e = lvl.starts[idx + 1]
        lo = np.where(valid, s, 0)
        hi = np.where(valid, e, 0)
        while True:
            act = hi > lo
            if not act.any():
                break
            # converged lanes (lo == hi == row end) can equal n_tok, so clamp the
            # probe index before touching next_tokens
            mid = np.minimum((lo + hi) >> 1, n_tok - 1)
            right = lvl.next_tokens[mid] < w
            hi = np.where(act & ~right, mid, hi)
            lo = np.where(act & right, mid + 1, lo)
        probe = np.minimum(lo, n_tok - 1) if n_tok else lo
        found = valid & (lo < e) & (lvl.next_tokens[probe] == w)
        return np.where(found, lvl.counts[probe].astype(np.int64), 0)

    # -- scoring -----------------------------------------------------------
    def _ctx_keys(self, stream: np.ndarray, pos: np.ndarray, ctx_len: int) -> np.ndarray:
        if self.key_mode == "hash" or ctx_len > 3:
            acc = np.zeros(pos.shape[0], dtype=np.uint64)
            base = pos - ctx_len
            chunk = 3
            mults = self.mults
            for c in range(0, ctx_len, chunk):
                ln = min(chunk, ctx_len - c)
                sub = np.zeros(pos.shape[0], dtype=np.uint64)
                for j in range(ln):
                    sub = sub * np.uint64(self.vocab) + stream[base + c + j].astype(np.uint64)
                acc ^= sub * mults[(c // chunk) % mults.shape[0]]
            return acc
        return context_keys_at(stream, pos, ctx_len, self.vocab)

    def stream_logprob(self, stream: np.ndarray, positions: Optional[np.ndarray] = None):
        """Per-position ln p(target) for ``positions`` (default: all with a full context).

        ``lookup='interpolate'`` mixes every level (round-158/159 behaviour);
        ``lookup='longest'`` uses only the longest matching stored context and
        sends that level's discounted mass to the unigram prior -- i.e. a
        variable-length retrieval memory with backoff (round-160 arm D).
        """
        M = self.order
        if positions is None:
            pos = np.arange(M - 1, stream.shape[0], dtype=np.int64)
        else:
            pos = np.asarray(positions, dtype=np.int64)
            if pos.size and int(pos.min()) < M - 1:
                raise SystemExit("stream_logprob got a position without a full context")
        w = stream[pos].astype(np.int64)
        p = self.p1_dense[w]
        unigram = self.p1_dense[w]
        longest = self.lookup == "longest"
        top_level = np.ones(pos.shape[0], dtype=np.int8)
        for k in range(2, M + 1):
            lvl = self.levels[k]
            if lvl.next_tokens.shape[0] == 0:
                continue
            ck = self._ctx_keys(stream, pos, k - 1)
            idx, valid = self.row_index(lvl, ck)
            del ck
            if valid.any():
                c = self.row_counts(lvl, idx, valid, w)
                zeff = lvl.zeff[idx]
                disc = np.where(valid, np.maximum(
                    c - lvl.sub_table[np.minimum(c, COUNT_CAP)], 0.0) / zeff, 0.0)
                gamma = np.where(valid, 1.0 - lvl.discsum[idx] / zeff, 1.0)
                if longest:
                    pk = disc + gamma * unigram
                    p = np.where(valid, pk, p)
                    del pk
                else:
                    p = disc + gamma * p
                top_level[valid] = k
                del c, zeff, disc, gamma
            del idx, valid
        n_zero = int(np.count_nonzero(p <= 0.0))
        with np.errstate(divide="ignore"):
            lp = np.log(np.maximum(p, 1e-300))
        return lp, top_level, pos, n_zero
    def accuracy(self, stream: np.ndarray, positions: np.ndarray,
                 candidates: np.ndarray, chunk: int) -> Dict[str, object]:
        """Exact top-1/top-5 over ``candidates`` on the given positions.

        ``lookup='interpolate'`` accumulates the usual mixture; ``lookup='longest'``
        keeps, for every position, the distribution of the longest matching stored
        context (its discounted sparse part plus its mass on the unigram prior),
        falling back only where no level matched.
        """
        V = self.vocab
        C = candidates.shape[0]
        cand_col = np.full(V, -1, dtype=np.int32)
        cand_col[candidates] = np.arange(C, dtype=np.int32)
        top1 = np.zeros(positions.shape[0], dtype=bool)
        top5 = np.zeros(positions.shape[0], dtype=bool)
        unigram_c = self.p1_dense[candidates].astype(np.float32)
        longest = self.lookup == "longest"
        for start in range(0, positions.shape[0], chunk):
            pos = positions[start : start + chunk]
            w = stream[pos].astype(np.int64)
            scores = np.tile(unigram_c, (pos.shape[0], 1))
            for k in range(2, self.order + 1):
                lvl = self.levels[k]
                if lvl.next_tokens.shape[0] == 0:
                    continue
                ck = self._ctx_keys(stream, pos, k - 1)
                idx, valid = self.row_index(lvl, ck)
                del ck
                if not valid.any():
                    del idx, valid
                    continue
                zeff_all = lvl.zeff[idx]
                gamma = np.where(valid, 1.0 - lvl.discsum[idx] / zeff_all, 1.0)
                s = lvl.starts[idx]
                e = lvl.starts[idx + 1]
                lens = np.where(valid, e - s, 0).astype(np.int64)
                total = int(lens.sum())
                if longest:
                    prev = scores
                    scores = np.tile(unigram_c, (pos.shape[0], 1))
                    scores *= gamma[:, None].astype(np.float32)
                else:
                    scores *= gamma[:, None].astype(np.float32)
                if total:
                    keep = lens > 0
                    pos_rep = np.repeat(np.flatnonzero(keep), lens[keep])
                    offs = np.repeat(s[keep] - np.concatenate(
                        ([0], np.cumsum(lens[keep])[:-1])), lens[keep])
                    flat = np.arange(total, dtype=np.int64) + offs
                    col = cand_col[lvl.next_tokens[flat]]
                    ok = col >= 0
                    if ok.any():
                        zeff_rep = np.repeat(zeff_all[keep], lens[keep])
                        cnt = lvl.counts[flat].astype(np.int64)
                        val = np.maximum(
                            cnt - lvl.sub_table[np.minimum(cnt, COUNT_CAP)], 0.0)
                        val = (val / zeff_rep).astype(np.float32)
                        np.add.at(scores, (pos_rep[ok], col[ok].astype(np.int64)),
                                  val[ok])
                    del pos_rep, offs, flat, col, ok
                if longest:
                    np.copyto(scores, np.where(valid[:, None], scores, prev))
                    del prev
                del idx, valid, zeff_all, gamma, s, e, lens
            tcol = cand_col[w].astype(np.int64)
            inside = tcol >= 0
            if scores.shape[1]:
                top1[start : start + pos.shape[0]] = np.argmax(scores, axis=1) == tcol
                part = np.argpartition(-scores, min(5, C) - 1, axis=1)[:, :5]
                top5[start : start + pos.shape[0]] = np.any(part == tcol[:, None], axis=1)
            del scores
            gc.collect()
        n_out = int(np.count_nonzero(cand_col[stream[positions].astype(np.int64)] < 0))
        inside = cand_col[stream[positions].astype(np.int64)] >= 0
        return {
            "_top1_mask": top1,
            "_top5_mask": top5,
            "_inside_mask": inside,
            "n_positions": int(positions.shape[0]),
            "n_candidates": int(C),
            "n_targets_outside_candidates": n_out,
            "candidate_coverage": float(inside.mean()) if positions.shape[0] else 0.0,
            "top1": float(top1[inside].mean()) if inside.any() else 0.0,
            "top5": float(top5[inside].mean()) if inside.any() else 0.0,
            "top1_all_positions": float(top1.mean()) if positions.shape[0] else 0.0,
            "top5_all_positions": float(top5.mean()) if positions.shape[0] else 0.0,
        }


def candidate_set(train: np.ndarray, k: int, vocab: int) -> np.ndarray:
    """Top-``k`` most frequent training tokens (ties broken by token id)."""
    counts = np.bincount(train.astype(np.int64), minlength=vocab)
    order = np.lexsort((np.arange(vocab), -counts))
    keep = order[:k]
    keep = keep[counts[keep] > 0]
    return np.sort(keep.astype(np.int64))


# --------------------------------------------------------------------------
# storage projection
# --------------------------------------------------------------------------
def fit_power_law(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, float]:
    lx = np.log(np.asarray(xs, dtype=np.float64))
    ly = np.log(np.asarray(ys, dtype=np.float64))
    A = np.vstack([np.ones_like(lx), lx]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return {"a": float(coef[0]), "b": float(coef[1]),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0}


def invert_power_law(fit: Dict[str, float], y_target: float) -> Optional[float]:
    if fit["b"] <= 1e-9:
        return None
    try:
        return float(math.exp((math.log(y_target) - fit["a"]) / fit["b"]))
    except OverflowError:
        return None


def saturation_note(order: int, fit_distinct: Optional[Dict[str, float]],
                    scaling: Sequence[Dict[str, object]], vocab: int) -> str:
    """Flag orders whose distinct-n-gram count cannot grow (so the fit is void).

    The unigram table is bounded by the vocabulary and the bigram table by
    V**2; when the measured growth is already flat the power-law projection is
    an artifact of the fit, not a statement about corpora.
    """
    if fit_distinct is None:
        return "no fit (fewer than two usable prefixes)"
    b = fit_distinct["b"]
    last = scaling[-1]["per_model_order"][order]["distinct_ngrams"]  # type: ignore
    if order == 1 and last >= 0.9 * vocab:
        return ("unigram table saturated ({} of {} vocabulary types): the "
                "power-law projection is meaningless, use the linear one".format(last, vocab))
    if b < 0.2:
        return ("fitted exponent b={:.3f} is near-flat: the table is saturating, "
                "the power-law projection is not meaningful".format(b))
    if b > 1.0:
        return ("fitted exponent b={:.3f} > 1 (sub-linear growth of distinct "
                "n-grams is expected); treat the projection as an upper bound".format(b))
    return "ok (b={:.3f}, assumes the power law holds far outside the fitted range)".format(b)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
WINDOW_PREDICTION = """PRE-REGISTERED PREDICTION (fixed before any number below was computed)
  * CODE: the longer window wins materially at matched bytes -- gap on the order of the k>=8 addressable-and-memorisable mass from round 159 (0.173 at k=8), and the advantage grows with training size.
  * WIKI: gap approximately zero -- matched bytes are better spent on more short-context entries than on a few long ones.
  * If WIKI also shows a large long-window advantage, or is larger than code's, the theory is wrong and should be said so plainly."""


def parse_arms(spec: str) -> List[Dict[str, object]]:
    """Parse ``4,9,17,longest17`` into arm descriptions."""
    arms = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("longest"):
            order = int(part[len("longest"):])
            arms.append({"name": "longest{}".format(order), "order": order,
                         "lookup": "longest", "key_mode": "hash"})
        elif part.startswith("exact"):
            order = int(part[len("exact"):])
            arms.append({"name": "exact{}".format(order), "order": order,
                         "lookup": "interpolate", "key_mode": "exact"})
        else:
            order = int(part)
            arms.append({"name": "k{}".format(order - 1), "order": order,
                         "lookup": "interpolate", "key_mode": "hash"})
    return arms


def budget_sweep(args, stream: np.ndarray, vocab: int, log_fn) -> Dict[str, object]:
    """Matched-bytes window comparison across training sizes (round 160).

    For every training size, every arm and every pruning threshold, the packed
    table bytes are MEASURED; quality is then scored only at the largest table
    that fits each budget (plus the unpruned table), so the comparison is
    "best model of this window that fits in B bytes".
    """
    sizes = sorted({int(x) for x in args.sweep_train_sizes.split(",") if x.strip()})
    arms = parse_arms(args.sweep_arms)
    grid = sorted({int(x) for x in args.prune_grid.split(",") if x.strip()})
    budgets = [float(x) for x in args.budget_bytes.split(",") if x.strip()]
    max_order = max(int(a["order"]) for a in arms)
    eval_start = args.eval_start if args.eval_start is not None else max(sizes)
    eval_len = args.eval_len or (stream.shape[0] - eval_start)
    if eval_start + eval_len > stream.shape[0]:
        raise SystemExit("eval block {}+{} exceeds the stream ({} tokens)".format(
            eval_start, eval_len, stream.shape[0]))
    eval_stream = np.ascontiguousarray(stream[eval_start : eval_start + eval_len])
    log_fn("budget sweep: sizes={} arms={} budgets={} eval=[{},{})".format(
        sizes, [a["name"] for a in arms], [int(b) for b in budgets], eval_start,
        eval_start + eval_len))

    # positions with a full context for the longest arm, then decontamination
    # against the LARGEST training prefix so the eval set is identical at every
    # training size (otherwise the size comparison would be confounded)
    positions_all = np.arange(max_order - 1, eval_stream.shape[0], dtype=np.int64)
    positions_unfiltered = positions_all.copy()
    decon = {"applied": False}
    if args.position_decon_against:
        ref = np.ascontiguousarray(stream[: max(sizes)])
        mults = np.random.default_rng(args.seed).integers(
            1, 2 ** 63, size=512, dtype=np.uint64) | np.uint64(1)
        drop, usable = ngram_present_mask(ref, eval_stream, positions_all,
                                          args.position_decon_ngram, vocab, mults)
        decon = {"applied": True, "reference_tokens": int(ref.shape[0]),
                 "window_tokens": int(args.position_decon_ngram),
                 "positions_with_full_window": int(usable.sum()),
                 "positions_dropped": int(drop.sum()),
                 "positions_dropped_share": (float(drop.sum()) / int(usable.sum())
                                             if usable.any() else 0.0),
                 "note": ("mask computed against the largest training prefix and "
                          "applied identically at every size")}
        positions_all = positions_all[~drop]
        log_fn("sweep decontamination: dropped {:,} positions ({:.4f}) with a {}-token "
               "window in the largest training prefix".format(
                   int(drop.sum()), decon["positions_dropped_share"],
                   args.position_decon_ngram))
    n_pos = int(positions_all.shape[0])
    if args.sweep_acc_positions and args.sweep_acc_positions < n_pos:
        rng = np.random.default_rng(args.seed)
        acc_pos = np.sort(rng.choice(positions_all, size=args.sweep_acc_positions,
                                     replace=False))
    else:
        acc_pos = positions_all
    cand_src = np.ascontiguousarray(stream[: max(sizes)])
    cand_topk = candidate_set(cand_src, args.acc_candidates, vocab)
    log_fn("sweep positions={:,} accuracy positions={:,} candidates={:,} (fixed for "
           "every arm and size)".format(n_pos, int(acc_pos.shape[0]),
                                        int(cand_topk.shape[0])))

    notes: List[str] = []
    mults = default_mults()
    per_size: Dict[str, object] = {}
    for N in sizes:
        train = np.ascontiguousarray(stream[:N])
        raw, cont = build_counts(train, list(range(1, max_order + 1)), vocab)
        full: Dict[str, LevelTable] = {}
        for k in sorted(raw):
            lvl = build_level(raw[k][0], raw[k][1], vocab, key_mode="hash", mults=mults)
            finalize_level(lvl, args.smoothing, args.addk, vocab, notes)
            full["raw{}".format(k)] = lvl
        for k in sorted(cont):
            lvl = build_level(cont[k][0], cont[k][1], vocab, key_mode="hash", mults=mults)
            finalize_level(lvl, args.smoothing, args.addk, vocab, notes)
            full["cont{}".format(k)] = lvl
        del raw, cont
        gc.collect()
        unpruned_bytes = {a["name"]: int(sum(
            full["cont{}".format(k) if k < a["order"] else "raw{}".format(k)].nbytes
            for k in range(1, int(a["order"]) + 1))) for a in arms}
        log_fn("size {:,}: unpruned bytes {}".format(N, unpruned_bytes))
        size_entry: Dict[str, object] = {"train_tokens": int(N),
                                         "unpruned_bytes": unpruned_bytes,
                                         "arms": {}, "matched": {}, "gaps": {}}
        for arm in arms:
            M = int(arm["order"])
            base_levels = {k: full["cont{}".format(k) if k < M else "raw{}".format(k)]
                           for k in range(1, M + 1)}
            # 1) measured byte frontier over the whole threshold grid (no scoring)
            frontier = []
            for tau in grid:
                if tau == 1:
                    b = int(sum(base_levels[k].nbytes for k in base_levels))
                    e = int(sum(base_levels[k].n_entries for k in base_levels))
                else:
                    b = 0
                    e = 0
                    for k in base_levels:
                        pl = prune_level(base_levels[k], tau, args.smoothing, args.addk,
                                         vocab, notes,
                                         protect=(k <= args.prune_protect_below))
                        b += pl.nbytes
                        e += pl.n_entries
                        del pl
                frontier.append({"min_count": tau, "bytes": b, "stored_ngrams": e})
                gc.collect()
            # 2) score the points that bracket each budget, plus the unpruned table
            need = {1}
            for bud in budgets:
                below = [p for p in frontier if p["bytes"] <= bud]
                above = [p for p in frontier if p["bytes"] > bud]
                if below:
                    need.add(max(below, key=lambda p: p["bytes"])["min_count"])
                if above:
                    need.add(min(above, key=lambda p: p["bytes"])["min_count"])
            scored = []
            for tau in sorted(need):
                levels = {k: (base_levels[k] if tau == 1 else
                              prune_level(base_levels[k], tau, args.smoothing, args.addk,
                                          vocab, notes,
                                          protect=(k <= args.prune_protect_below)))
                          for k in base_levels}
                model = NgramModel(M, levels, vocab, args.smoothing, args.addk, notes,
                                   lookup=str(arm["lookup"]), key_mode="hash")
                lp, _tl, pos, _nz = model.stream_logprob(eval_stream, positions_all)
                nll = float(-lp.mean())
                acc = model.accuracy(eval_stream, acc_pos, cand_topk, args.acc_chunk)
                bytes_total = int(sum(levels[k].nbytes for k in levels))
                ngrams = int(sum(levels[k].n_entries for k in levels))
                point = {
                    "min_count": tau, "bytes": bytes_total, "stored_ngrams": ngrams,
                    "nll": nll, "perplexity": float(math.exp(nll)) if nll < 700 else None,
                    "top1": acc["top1"], "top5": acc["top5"],
                    "n_positions": int(pos.shape[0]),
                }
                if (args.sweep_raw_positions and decon.get("applied")
                        and N == max(sizes)):
                    lp_r, _t, pos_r, _z = model.stream_logprob(eval_stream,
                                                               positions_unfiltered)
                    point["nll_unfiltered_positions"] = float(-lp_r.mean())
                    point["top1_unfiltered_positions"] = model.accuracy(
                        eval_stream, acc_pos, cand_topk, args.acc_chunk)["top1"]
                    point["n_positions_unfiltered"] = int(pos_r.shape[0])
                    del lp_r, pos_r
                scored.append(point)
                log_fn("  arm {} tau={} bytes={:,} ngrams={:,} NLL={:.4f} top1={:.4f}".format(
                    arm["name"], tau, bytes_total, ngrams, nll, acc["top1"]))
                del model, lp, pos
                if tau != 1:
                    del levels
                gc.collect()
            size_entry["arms"][arm["name"]] = {
                "order": M, "lookup": arm["lookup"], "frontier": frontier,
                "scored": scored}
        # 3) read each budget off the scored points by interpolating NLL on ln(bytes),
        #    always reporting the two measured points that bracket it
        for bud in budgets:
            matched = {}
            for arm in arms:
                pts = sorted(size_entry["arms"][arm["name"]]["scored"],
                             key=lambda p: p["bytes"])
                interior = None
                for a, b in zip(pts, pts[1:]):
                    if a["bytes"] <= bud <= b["bytes"]:
                        interior = (a, b)
                        break
                lo = [p for p in pts if p["bytes"] <= bud]
                if interior:
                    a, b = interior
                    la, lb = math.log(a["bytes"]), math.log(b["bytes"])
                    w = 0.0 if lb == la else (math.log(bud) - la) / (lb - la)
                    nll = a["nll"] + w * (b["nll"] - a["nll"])
                    t1 = a["top1"] + w * (b["top1"] - a["top1"])
                    matched[arm["name"]] = {
                        "bytes": float(bud), "nll": float(nll), "top1": float(t1),
                        "interpolated": True,
                        "bracket_bytes": [a["bytes"], b["bytes"]],
                        "bracket_nll": [a["nll"], b["nll"]],
                        "bracket_min_count": [a["min_count"], b["min_count"]],
                    }
                elif lo:
                    best = max(lo, key=lambda p: p["bytes"])
                    matched[arm["name"]] = {
                        "bytes": best["bytes"], "nll": best["nll"], "top1": best["top1"],
                        "min_count": best["min_count"], "interpolated": False,
                        "caveat": "no scored point reaches the budget; this arm cannot "
                                  "spend that many bytes",
                    }
            size_entry["matched"][str(int(bud))] = matched
            base = matched.get("k3")
            if base:
                gaps = {}
                for arm in arms:
                    if arm["name"] == "k3" or arm["name"] not in matched:
                        continue
                    m = matched[arm["name"]]
                    gaps["{}_minus_k3_nll".format(arm["name"])] = base["nll"] - m["nll"]
                    gaps["{}_minus_k3_top1".format(arm["name"])] = m["top1"] - base["top1"]
                    gaps["{}_bytes_vs_k3".format(arm["name"])] = m["bytes"] - base["bytes"]
                size_entry["gaps"][str(int(bud))] = gaps
        per_size[str(N)] = size_entry
        del full
        gc.collect()

    # interaction summary: gap as a function of training size
    interaction = {}
    for bud in budgets:
        key = str(int(bud))
        for arm in arms:
            if arm["name"] == "k3":
                continue
            series = []
            for N in sizes:
                g = per_size[str(N)]["gaps"].get(key, {})
                if "{}_minus_k3_nll".format(arm["name"]) in g:
                    series.append({"train_tokens": N,
                                   "nll_gap": g["{}_minus_k3_nll".format(arm["name"])],
                                   "top1_gap": g["{}_minus_k3_top1".format(arm["name"])],
                                   "bytes": per_size[str(N)]["matched"][key].get(
                                       arm["name"], {}).get("bytes"),
                                   "bytes_k3": per_size[str(N)]["matched"][key].get(
                                       "k3", {}).get("bytes")})
            interaction.setdefault(key, {})[arm["name"]] = series
    return {"prediction": WINDOW_PREDICTION, "geometry": {
        "sweep_train_sizes": sizes, "arms": [a["name"] for a in arms],
        "prune_grid": grid, "budget_bytes": [int(b) for b in budgets],
        "eval_start": int(eval_start), "eval_tokens": int(eval_len),
        "key_mode": "hash (8-byte hashed context keys at every order)",
        "pruning_rule": ("drop stored entries with count < min_count (levels up to "
                         "--prune-protect-below are exempt); corpus statistics are NOT "
                         "recomputed, missing rows back off"),
        "prune_protect_below": int(args.prune_protect_below),
        "lookup_interpolate": "arms k3/k8/k16 mix every level (interpolated MKN)",
        "lookup_longest": ("arm longest16 uses only the longest stored matching "
                           "context and sends that level's discounted mass to the "
                           "unigram prior"),
    }, "uniform_vocab": vocab, "position_decontamination": decon,
        "n_sweep_positions": n_pos, "n_sweep_accuracy_positions": int(acc_pos.shape[0]),
        "per_size": per_size, "interaction": interaction, "notes": notes}


def parse_orders(raw: str) -> List[int]:
    orders = sorted({int(p) for p in raw.split(",") if p.strip()})
    if not orders or orders[0] < 1:
        raise SystemExit("--orders must be >= 1")
    return orders


def sweep_main(args) -> int:
    """Entry point for --budget-sweep: no per-order pipeline, no big outputs."""
    t0 = time.time()
    stream_path = Path(args.train_npy)
    if not stream_path.exists():
        raise SystemExit("stream not found: {}".format(stream_path))
    stream = load_tokens(stream_path)
    sizes = sorted({int(x) for x in args.sweep_train_sizes.split(",") if x.strip()})
    if args.uniform_vocab == "auto":
        vocab = int(stream.max()) + 1
    else:
        vocab = int(args.uniform_vocab)
        if vocab <= int(stream.max()):
            raise SystemExit("--uniform-vocab must exceed the largest token id")
    log("=" * 78)
    for line in WINDOW_PREDICTION.splitlines():
        log(line)
    log("=" * 78)
    log("stream={:,} tokens, V_uni={:,}, sizes={}".format(
        int(stream.shape[0]), vocab, sizes))
    sweep = budget_sweep(args, stream, vocab, log)
    out = {
        "schema": "qwen35-ple-window-cost-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(), "python": sys.version.split()[0],
        "numpy": np.__version__, "command": " ".join(sys.argv),
        "elapsed_sec": time.time() - t0,
        "stream": str(stream_path), "stream_sha256": sha256_file(stream_path),
        "stream_tokens": int(stream.shape[0]), "uniform_vocab": vocab,
        "smoothing": args.smoothing,
        "peak_rss_bytes": peak_rss_bytes(),
        "sweep": sweep,
    }
    if args.sweep_output:
        Path(args.sweep_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.sweep_output).write_text(json.dumps(out, indent=2) + "\n",
                                           encoding="utf-8")
        log("wrote {}".format(args.sweep_output))
    if args.sweep_markdown:
        Path(args.sweep_markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.sweep_markdown).write_text(render_sweep_markdown(out),
                                             encoding="utf-8")
        log("wrote {}".format(args.sweep_markdown))
    log("done in {:.1f}s, peak RSS {:.2f} GiB".format(
        time.time() - t0, peak_rss_bytes() / GIB))
    return 0


def render_sweep_markdown(out: Dict[str, object]) -> str:
    sw = out["sweep"]
    geo = sw["geometry"]
    arms = geo["arms"]
    sizes = geo["sweep_train_sizes"]
    budgets = geo["budget_bytes"]
    lines = ["# Window cost at matched bytes ({})".format(out["schema"]), ""]
    lines.append("```")
    lines.append(sw["prediction"])
    lines.append("```")
    lines.append("")
    lines.append("stream {:,} tokens ({}) | V_uni {:,} | eval block [{}, {}) = {:,} "
                 "tokens | positions scored {:,} | accuracy positions {:,} | "
                 "peak RSS {:.2f} GiB".format(
                     out["stream_tokens"], out["stream"], out["uniform_vocab"],
                     geo["eval_start"], geo["eval_start"] + geo["eval_tokens"],
                     geo["eval_tokens"], sw["n_sweep_positions"],
                     sw["n_sweep_accuracy_positions"], out["peak_rss_bytes"] / GIB))
    lines.append("")
    lines.append("pruning rule: {}".format(geo["pruning_rule"]))
    lines.append("")
    lines.append("key encoding: {}".format(geo["key_mode"]))
    lines.append("")
    for N in sizes:
        se = sw["per_size"][str(N)]
        lines.append("## training tokens = {:,}".format(N))
        lines.append("")
        lines.append("unpruned bytes: `{}`".format(json.dumps(se["unpruned_bytes"])))
        lines.append("")
        lines.append("### frontier (measured bytes vs pruning threshold)")
        lines.append("")
        lines.append("| arm | " + " | ".join(
            "tau={}".format(p["min_count"]) for p in
            se["arms"][arms[0]]["frontier"]) + " |")
        lines.append("|---" * (len(se["arms"][arms[0]]["frontier"]) + 1) + "|")
        for a in arms:
            cells = ["{:,}".format(p["bytes"]) for p in se["arms"][a]["frontier"]]
            lines.append("| {} | {} |".format(a, " | ".join(cells)))
        lines.append("")
        lines.append("### matched bytes")
        lines.append("")
        lines.append("| budget | arm | measured bytes | tau | NLL | top-1 |")
        lines.append("|---|---|---|---|---|---|")
        for b in budgets:
            for a in arms:
                m = se["matched"].get(str(b), {}).get(a)
                if not m:
                    continue
                if m.get("interpolated"):
                    extra = "interp {:,}-{:,} (tau {}..{})".format(
                        m["bracket_bytes"][0], m["bracket_bytes"][1],
                        m["bracket_min_count"][0], m["bracket_min_count"][1])
                else:
                    extra = "tau {}".format(m.get("min_count"))
                lines.append("| {:,} | {} | {:,} | {} | {:.4f} | {:.4f} |".format(
                    b, a, int(m["bytes"]), extra, m["nll"], m["top1"]))
        lines.append("")
        lines.append("### gaps vs the 4-gram arm (positive NLL gap = longer window better)")
        lines.append("")
        lines.append("| budget | arm | NLL gap | top-1 gap | bytes vs arm k3 |")
        lines.append("|---|---|---|---|---|")
        for b in budgets:
            for a in arms:
                if a == "k3":
                    continue
                g = se["gaps"].get(str(b), {})
                key = "{}_minus_k3_nll".format(a)
                if key not in g:
                    continue
                lines.append("| {:,} | {} | {:+.4f} | {:+.4f} | {:+,} |".format(
                    b, a, g[key], g["{}_minus_k3_top1".format(a)],
                    g["{}_bytes_vs_k3".format(a)]))
        lines.append("")
    lines.append("## interaction: NLL gap vs training size (positive = longer window better)")
    lines.append("")
    for b in budgets:
        lines.append("budget {:,}".format(b))
        lines.append("")
        lines.append("| arm | " + " | ".join("{:,}".format(N) for N in sizes) + " |")
        lines.append("|---" * (len(sizes) + 1) + "|")
        for a in arms:
            if a == "k3":
                continue
            series = sw["interaction"].get(str(b), {}).get(a, [])
            by = {s["train_tokens"]: s for s in series}
            cells = []
            for N in sizes:
                s = by.get(N)
                cells.append("{:+.4f}".format(s["nll_gap"]) if s else "n/a")
            lines.append("| {} | {} |".format(a, " | ".join(cells)))
        lines.append("")
    lines.append("sweep notes: `{}`".format(json.dumps(sw["notes"])))
    lines.append("")
    lines.append("position decontamination: `{}`".format(
        json.dumps(sw["position_decontamination"])))
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-npy", required=True, help="training token stream (.npy int array)")
    ap.add_argument("--eval-npy", default=None, help="held-out token stream (.npy)")
    ap.add_argument("--holdout-tail", type=int, default=0,
                    help="instead of --eval-npy: hold out the last N tokens of --train-npy")
    ap.add_argument("--train-cap", type=int, default=0,
                    help="truncate the stream to --train-cap + --holdout-tail tokens "
                         "before splitting, so several corpora can be run with identical "
                         "train and eval sizes")
    ap.add_argument("--orders", default="1,2,3,4")
    ap.add_argument("--smoothing", choices=["mkn", "kn", "addk"], default="mkn")
    ap.add_argument("--addk", type=float, default=0.1, help="k for add-k smoothing")
    ap.add_argument("--uniform-vocab", default="auto",
                    help="'auto' or an int: size of the uniform floor vocabulary")
    ap.add_argument("--candidate-from", default=None,
                    help="token stream defining the top-K candidate set (default: train)")
    ap.add_argument("--acc-candidates", type=int, default=5000,
                    help="K for the restricted candidate set (0 = skip)")
    ap.add_argument("--acc-positions", type=int, default=20000,
                    help="positions subsampled for the top-K accuracy")
    ap.add_argument("--acc-positions-full", type=int, default=2000,
                    help="positions subsampled for the full-vocab accuracy (0 = skip)")
    ap.add_argument("--acc-chunk", type=int, default=1000,
                    help="position chunk for the dense score matrix")
    ap.add_argument("--scaling-prefixes", default="0.25,0.5,1.0",
                    help="corpus prefixes used for the bytes-vs-tokens power-law fit")
    ap.add_argument("--target-bytes", type=float, default=48 * GIB,
                    help="storage budget to project onto (default 48 GiB = PLE table)")
    ap.add_argument("--ple-row-bytes", type=int, default=160,
                    help="bytes per frozen PLE row (fp8, 160 dims)")
    ap.add_argument("--tail-analysis", action="store_true",
                    help="bucket scored positions by the TRAIN count of their trigram "
                         "context and report position/NLL share and top-1 per bucket")
    ap.add_argument("--tail-buckets", default=DEFAULT_BUCKETS,
                    help="bucket edges for --tail-analysis")
    ap.add_argument("--verbatim-ks", default="",
                    help="comma list of context lengths k; report the fraction of scored "
                         "positions whose gold continuation follows the same k-token "
                         "context in the training stream (e.g. 1,2,3,4,8,16)")
    ap.add_argument("--source-text", default=None,
                    help="source text file (e.g. corpus.txt) for tokenizer-normalised "
                         "storage: bytes per source byte and tokens per source byte")
    ap.add_argument("--position-decon-against", default=None,
                    help="token .npy of the training stream; scored positions whose "
                         "N-token window also occurs there are dropped before scoring")
    ap.add_argument("--position-decon-ngram", type=int, default=64,
                    help="window length for --position-decon-against")
    ap.add_argument("--decon-sensitivity", default="16,32,64,128",
                    help="also report the dropped-position fraction at these windows")
    ap.add_argument("--budget-sweep", action="store_true",
                    help="round-160 matched-bytes window sweep across training sizes; "
                         "prints the pre-registered prediction first and writes "
                         "--sweep-output")
    ap.add_argument("--sweep-train-sizes", default="100000,250000,500000,800000")
    ap.add_argument("--sweep-arms", default="4,9,17,longest17",
                    help="4/9/17 = interpolated MKN with context <= 3/8/16 tokens; "
                         "longestN = variable-length longest-suffix lookup")
    ap.add_argument("--prune-grid", default="1,2,4,8,16,32,64,128,256,1024")
    ap.add_argument("--prune-protect-below", type=int, default=0,
                    help="never prune levels up to this n-gram order; the byte budget "
                         "is then spent only on longer contexts (allocation sensitivity)")
    ap.add_argument("--budget-bytes", default="1048576,5242880,20971520")
    ap.add_argument("--eval-start", type=int, default=None,
                    help="first token index of the sweep eval block "
                         "(default: after the largest training size)")
    ap.add_argument("--eval-len", type=int, default=0)
    ap.add_argument("--sweep-acc-positions", type=int, default=10000)
    ap.add_argument("--sweep-raw-positions", action="store_true",
                    help="also score every chosen point on the unfiltered position "
                         "set at the largest training size (the 64-token filter "
                         "deletes exactly the repeats a long window can memorise)")
    ap.add_argument("--sweep-output", default=None, help="JSON path for the sweep")
    ap.add_argument("--sweep-markdown", default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="run the built-in invariant tests (ragged search, "
                         "normalisation, floor) and exit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None, help="JSON result path")
    ap.add_argument("--markdown", default=None, help="optional markdown table path")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.budget_sweep:
        return sweep_main(args)
    t_start = time.time()
    train_path = Path(args.train_npy)
    if not train_path.exists():
        raise SystemExit("train stream not found: {}".format(train_path))
    orders = parse_orders(args.orders)
    max_order = max(orders)

    train_full = load_tokens(train_path)
    if args.train_cap:
        want = int(args.train_cap) + int(args.holdout_tail)
        if train_full.shape[0] < want:
            raise SystemExit("--train-cap needs {:,} tokens, stream has {:,}".format(
                want, train_full.shape[0]))
        train_full = np.ascontiguousarray(train_full[:want])
        log("train stream truncated to {:,} tokens (+{:,} held out) for a common geometry"
            .format(args.train_cap, args.holdout_tail))
    if args.holdout_tail:
        if args.holdout_tail >= train_full.shape[0]:
            raise SystemExit("--holdout-tail larger than the training stream")
        train = np.ascontiguousarray(train_full[: -args.holdout_tail])
        eval_stream = np.ascontiguousarray(train_full[-args.holdout_tail :])
        split = {
            "kind": "in-stream tail",
            "train_tokens": int(train.shape[0]),
            "eval_tokens": int(eval_stream.shape[0]),
            "train_dropped_tail_tokens": int(args.holdout_tail),
            "note": "train = first {} tokens, eval = last {} tokens of {}".format(
                train.shape[0], eval_stream.shape[0], train_path),
        }
    else:
        if not args.eval_npy:
            raise SystemExit("provide --eval-npy or --holdout-tail")
        eval_path = Path(args.eval_npy)
        if not eval_path.exists():
            raise SystemExit("eval stream not found: {}".format(eval_path))
        train = np.ascontiguousarray(train_full)
        eval_stream = np.ascontiguousarray(load_tokens(eval_path))
        split = {"kind": "disjoint stream", "train_tokens": int(train.shape[0]),
                 "eval_tokens": int(eval_stream.shape[0]), "eval_stream": str(eval_path)}
    del train_full
    if args.candidate_from and args.candidate_from != "train":
        cand_src = load_tokens(Path(args.candidate_from))
        split["candidate_source"] = args.candidate_from
    else:
        cand_src = train
        split["candidate_source"] = "in-memory training stream" if args.candidate_from else "train"

    if args.uniform_vocab == "auto":
        vocab = int(max(int(train.max()), int(eval_stream.max())) + 1)
    else:
        vocab = int(args.uniform_vocab)
        if vocab <= int(train.max()):
            raise SystemExit("--uniform-vocab must exceed the largest training token id")
    log("train={:,} tokens, eval={:,} tokens, V_uni={:,}".format(
        int(train.shape[0]), int(eval_stream.shape[0]), vocab))

    prefixes = sorted({float(p) for p in args.scaling_prefixes.split(",") if p.strip()})
    positions_all = np.arange(max_order - 1, eval_stream.shape[0], dtype=np.int64)
    if positions_all.shape[0] <= 0:
        raise SystemExit("evaluation stream shorter than the largest order")

    positions_unfiltered = positions_all.copy()   # pre-decontamination robustness view

    # ---- optional per-position decontamination ---------------------------
    decon_info = {"applied": False}
    if args.position_decon_against:
        if args.position_decon_against == "train":
            ref_train = np.ascontiguousarray(train)
            ref_path = Path("<in-memory training stream>")
        else:
            ref_path = Path(args.position_decon_against)
            if not ref_path.exists():
                raise SystemExit("decontamination stream not found: {}".format(ref_path))
            ref_train = load_tokens(ref_path)
        mults = np.random.default_rng(args.seed).integers(
            1, 2 ** 63, size=512, dtype=np.uint64) | np.uint64(1)
        keep, usable = ngram_present_mask(ref_train, eval_stream, positions_all,
                                          args.position_decon_ngram, vocab, mults)
        n_usable = int(usable.sum())
        sensitivity = {}
        for n in sorted({int(x) for x in args.decon_sensitivity.split(",") if x.strip()}):
            ref = np.unique(ngram_hash_keys(ref_train, n, vocab, mults))
            keys = ngram_hash_keys(eval_stream, n, vocab, mults)
            u = positions_all >= (n - 1)
            hit = np.zeros(positions_all.shape[0], dtype=bool)
            if u.any():
                hit[u] = np.isin(keys[positions_all[u] - (n - 1)], ref)
            sensitivity[str(n)] = int((hit & u).sum())
            del ref, keys
            gc.collect()
        decon_info = {
            "applied": True,
            "reference_stream": str(ref_path),
            "reference_tokens": int(ref_train.shape[0]),
            "reference_is_in_memory_train": args.position_decon_against == "train",
            "window_tokens": int(args.position_decon_ngram),
            "scored_positions_before": int(positions_all.shape[0]),
            "positions_with_full_window": n_usable,
            "positions_dropped": int(keep.sum()),
            "positions_dropped_share_of_full_window": (
                float(keep.sum()) / n_usable if n_usable else 0.0),
            "sensitivity_dropped_by_window": sensitivity,
            "note": ("a scored position is dropped when its window (context+gold) also "
                     "occurs in the training stream; the window is deliberately longer "
                     "than the --verbatim-ks windows so the memorisation analysis below "
                     "is not zeroed by construction"),
        }
        positions_all = positions_all[~keep]
        log("position decontamination: dropped {:,} of {:,} positions whose {}-token "
            "window occurs in the training stream (sensitivity {})".format(
                int(keep.sum()), n_usable, args.position_decon_ngram, sensitivity))
        del ref_train, keep, usable
        gc.collect()

    n_positions = int(positions_all.shape[0])
    if n_positions <= 0:
        raise SystemExit("no scored positions left after decontamination")

    def subsample(n: int):
        if n and n < n_positions:
            rng = np.random.default_rng(args.seed)
            return np.sort(rng.choice(positions_all, size=n, replace=False))
        return positions_all

    acc_pos = subsample(args.acc_positions)
    if args.acc_positions_full:
        # nested: the full-vocab subsample is a prefix of the top-K subsample so
        # the two are computed on the same positions where possible
        acc_pos_full = acc_pos[: args.acc_positions_full] \
            if args.acc_positions_full <= acc_pos.shape[0] else subsample(args.acc_positions_full)
    else:
        acc_pos_full = None
    log("scoring positions={:,} (identical for all orders); top-K accuracy positions={:,}; "
        "full-vocab accuracy positions={:,}".format(
            n_positions, int(acc_pos.shape[0]),
            int(acc_pos_full.shape[0]) if acc_pos_full is not None else 0))

    cand_topk = candidate_set(cand_src, args.acc_candidates, vocab) \
        if args.acc_candidates else None
    if cand_topk is not None:
        log("candidate set: top-{} train tokens -> {:,} usable".format(
            args.acc_candidates, int(cand_topk.shape[0])))

    notes: List[str] = []
    peak_rss_build_start = peak_rss_bytes()
    if args.tail_analysis or args.verbatim_ks or args.source_text:
        log("=" * 78)
        for line in PRE_REGISTERED_RULE.splitlines():
            log(line)
        log("operationalisation (fixed before the numbers): "
            + json.dumps(PREREGISTERED_OPERATIONALISATION))
        log("=" * 78)

    # ---- full-size count tables (built once, shared by all orders) --------
    log("counting n-grams at the full training size ...")
    raw, cont = build_counts(train, orders, vocab)
    full_levels: Dict[str, LevelTable] = {}
    for k in sorted(raw):
        if k in orders:  # raw counts: the top level of the order-k model
            lvl = build_level(raw[k][0], raw[k][1], vocab)
            finalize_level(lvl, args.smoothing, args.addk, vocab, notes)
            full_levels["raw{}".format(k)] = lvl
        del raw[k]
        gc.collect()
    for k in orders:
        if k < max_order:  # continuation counts: lower levels of higher orders
            uq, ct = cont[k]
            lvl = build_level(uq, ct, vocab)
            finalize_level(lvl, args.smoothing, args.addk, vocab, notes)
            full_levels["cont{}".format(k)] = lvl
            del cont[k]
            gc.collect()
    del raw, cont
    gc.collect()
    log("count tables built; peak RSS = {:.2f} GiB".format(peak_rss_bytes() / GIB))

    # cand_full = training vocabulary: every unseen token scores exactly the
    # uniform floor, strictly below any seen token, so the ranking over this set
    # equals the ranking over all V_uni tokens.
    seen_vocab = full_levels["raw1"].next_tokens.astype(np.int64)

    # ---- prefix measurements for the scaling fit --------------------------
    log("measuring storage growth over corpus prefixes ...")
    scaling = []
    full_bytes_per_model = {}
    for M in orders:
        full_bytes_per_model[M] = {
            "bytes": int(sum(full_levels["cont{}".format(k) if k < M else "raw{}".format(k)].nbytes
                             for k in range(1, M + 1))),
            "distinct_ngrams": int(sum(
                full_levels["cont{}".format(k) if k < M else "raw{}".format(k)].n_entries
                for k in range(1, M + 1))),
        }
    for frac in prefixes:
        if frac >= 0.999:
            scaling.append({
                "fraction": 1.0, "train_tokens": int(train.shape[0]),
                "per_model_order": {M: dict(full_bytes_per_model[M]) for M in orders},
                "quality": None,  # filled in from the headline results below
                "note": "full-size tables reused (not rebuilt)",
            })
            continue
        n = min(train.shape[0], max(max_order + 1, int(round(train.shape[0] * frac))))
        sub = np.ascontiguousarray(train[:n])
        r2, c2 = build_counts(sub, orders, vocab)
        lvls: Dict[int, LevelTable] = {}
        for k in orders:
            lvls[k] = build_level(r2[k][0], r2[k][1], vocab)
            del r2[k]
        for k in orders:
            if k < max_order:
                lvls[k] = build_level(c2[k][0], c2[k][1], vocab)
                del c2[k]
        del r2, c2
        # the prefix models need discounts/Z/retained mass too
        for k in sorted(lvls):
            finalize_level(lvls[k], args.smoothing, args.addk, vocab, notes)
        gc.collect()
        per_model = {}
        for M in orders:
            tot_b = sum(lvls[k].nbytes for k in range(1, M + 1))
            tot_d = sum(lvls[k].n_entries for k in range(1, M + 1))
            per_model[M] = {"bytes": int(tot_b), "distinct_ngrams": int(tot_d),
                            "levels": {k: {"distinct_ngrams": lvls[k].n_entries,
                                           "bytes": lvls[k].nbytes}
                                       for k in range(1, M + 1)}}
        # quality at this byte budget: same eval stream, same positions, same
        # candidate sets as the headline table
        quality = {}
        for M in orders:
            m = NgramModel(M, {k: lvls[k] for k in range(1, M + 1)}, vocab,
                           args.smoothing, args.addk, notes)
            lp, _tl, pos, _nz = m.stream_logprob(eval_stream)
            q = {"nll": float(-lp.mean()), "n_positions": int(pos.shape[0])}
            q["perplexity"] = float(math.exp(q["nll"])) if q["nll"] < 700 else float("inf")
            if acc_pos_full is not None:
                a = m.accuracy(eval_stream, acc_pos_full, seen_vocab, args.acc_chunk)
                q["top1_full_vocab"] = a["top1"]
                q["top5_full_vocab"] = a["top5"]
            if cand_topk is not None:
                a = m.accuracy(eval_stream, acc_pos, cand_topk, args.acc_chunk)
                q["top1_top{}".format(args.acc_candidates)] = a["top1"]
                q["top5_top{}".format(args.acc_candidates)] = a["top5"]
            quality[str(M)] = q
            del m, lp, pos
            gc.collect()
        scaling.append({"fraction": frac, "train_tokens": int(n),
                        "per_model_order": per_model, "quality": quality})
        log("  prefix {:.3f} ({:,} tokens): order-{} bytes={:,} distinct={:,}".format(
            frac, n, max_order, per_model[max_order]["bytes"],
            per_model[max_order]["distinct_ngrams"]))
        del lvls, sub
        gc.collect()
        gc.collect()

    # ---- per-order models -------------------------------------------------
    majority = int(np.argmax(np.bincount(cand_src.astype(np.int64), minlength=vocab)))
    majority_rate = float((eval_stream[positions_all] == majority).mean())

    results = []
    for M in orders:
        levels = {k: full_levels["cont{}".format(k) if k < M else "raw{}".format(k)]
                  for k in range(1, M + 1)}
        model = NgramModel(M, levels, vocab, args.smoothing, args.addk, notes)
        packed_bytes = int(sum(levels[k].nbytes for k in levels))
        scoring_side = int(sum(levels[k].scoring_side_bytes for k in levels))
        lp, top_level, pos, n_zero = model.stream_logprob(eval_stream, positions_all)
        nll = float(-lp.mean())
        hist = {}
        for k in range(1, M + 1):
            hist[str(k)] = float((top_level == k).mean())
        seen_mask = model.p1_dense[eval_stream[pos].astype(np.int64)] > model.floor
        target_seen = float(seen_mask.mean())
        nll_oov_share = float(-lp[~seen_mask].sum() / pos.shape[0]) if (~seen_mask).any() else 0.0
        metrics = {
            "n_positions": int(pos.shape[0]),
            "nll_nats": nll,
            "perplexity": float(math.exp(nll)) if nll < 700 else float("inf"),
            "bits_per_token": nll / math.log(2.0),
            "target_in_unigram_vocab_rate": target_seen,
            "nll_nats_from_out_of_vocab_targets": nll_oov_share,
            "nll_nats_in_vocab_targets_only": (
                float(-lp[seen_mask].mean()) if seen_mask.any() else 0.0),
            "highest_context_found_rate": hist,
            "n_zero_probability_positions": n_zero,
            "discounts_per_level": {str(k): levels[k].D for k in sorted(levels)},
            "count_of_counts_per_level": {
                str(k): levels[k].count_of_counts for k in sorted(levels)},
        }
        log("order {}: NLL={:.4f} nats ppl={:.2f} ctx-hit[top]={:.3f} "
            "target-in-train-vocab={:.4f} zero-prob={}".format(
                M, nll, metrics["perplexity"], hist[str(M)], target_seen, n_zero))
        if M != max_order:
            del lp, pos
        del top_level
        gc.collect()

        acc = {}
        if acc_pos_full is not None:
            acc["full_vocab"] = model.accuracy(
                eval_stream, acc_pos_full, seen_vocab, args.acc_chunk)
        if cand_topk is not None:
            acc["top{}".format(args.acc_candidates)] = model.accuracy(
                eval_stream, acc_pos, cand_topk, args.acc_chunk)
        acc_masks = {name: {"top1": a.pop("_top1_mask"), "top5": a.pop("_top5_mask"),
                            "inside": a.pop("_inside_mask")} for name, a in acc.items()}
        metrics["accuracy"] = acc
        if M == max_order:
            analysis_cache = {"lp": lp, "pos": pos, "order": M,
                              "acc_pos": acc_pos, "acc_masks": acc_masks,
                              "model": model, "levels": levels}
            if decon_info.get("applied") and (args.tail_analysis or args.verbatim_ks):
                lp_raw, _tl, pos_raw, _nz = model.stream_logprob(
                    eval_stream, positions_unfiltered)
                analysis_cache["raw"] = {"lp": lp_raw, "pos": pos_raw}
        for name, a in acc.items():
            log("  acc[{}]: top1={:.4f} top5={:.4f} (coverage={:.4f}, outside={:,})".format(
                name, a["top1"], a["top5"], a["candidate_coverage"],
                a["n_targets_outside_candidates"]))

        storage = {
            "packed_bytes_total": packed_bytes,
            "packed_bytes_per_train_token": packed_bytes / max(1, int(train.shape[0])),
            "scoring_side_bytes_total": scoring_side,
            "per_order": [levels[k].storage_breakdown() for k in sorted(levels)],
        }
        results.append({"order": M, "metrics": metrics, "storage": storage,
                        "peak_rss_bytes_after_model": peak_rss_bytes()})
        if M != max_order:
            del model, levels
        gc.collect()

    # the full-size scaling point reuses the headline numbers
    for s_ in scaling:
        if s_.get("quality") is None:
            s_["quality"] = {str(r["order"]): {
                "nll": r["metrics"]["nll_nats"],
                "perplexity": r["metrics"]["perplexity"],
                "top1_full_vocab": r["metrics"]["accuracy"].get(
                    "full_vocab", {}).get("top1"),
                "top5_full_vocab": r["metrics"]["accuracy"].get(
                    "full_vocab", {}).get("top5"),
                "top1_top{}".format(args.acc_candidates): r["metrics"]["accuracy"].get(
                    "top{}".format(args.acc_candidates), {}).get("top1"),
            } for r in results}

    # ---- sanity checks ----------------------------------------------------
    nlls = [r["metrics"]["nll_nats"] for r in results]
    accs = [r["metrics"]["accuracy"]["full_vocab"]["top1"]
            for r in results if "full_vocab" in r["metrics"]["accuracy"]]
    monotone_nll = all(nlls[i] > nlls[i + 1] for i in range(len(nlls) - 1))
    monotone_acc = all(accs[i] <= accs[i + 1] for i in range(len(accs) - 1)) if accs else None
    worst_norm = None
    norm_devs = [l["row_normalization_deviation"] for r in results
                 for l in r["storage"]["per_order"]
                 if l.get("row_normalization_deviation") is not None]
    if norm_devs:
        worst_norm = float(max(norm_devs))
    sanity = {
        "nll_strictly_decreasing_with_order": bool(monotone_nll),
        "nll_sequence": nlls,
        "full_vocab_top1_nondecreasing_with_order": monotone_acc,
        "full_vocab_top1_sequence": accs,
        "majority_token_id": majority,
        "majority_token_rate": majority_rate,
        "max_level_normalization_deviation": worst_norm,
        "n_zero_probability_positions": sum(
            r["metrics"]["n_zero_probability_positions"] for r in results),
        "notes": notes,
        "warning": None if (monotone_nll and (monotone_acc is None or monotone_acc)) else
                   "MONOTONICITY VIOLATED -- investigate the backoff before reporting",
    }
    if sanity["warning"]:
        log("WARNING: " + sanity["warning"])
    for n in notes:
        log("note: " + n)

    # ---- 48 GiB projection -------------------------------------------------
    target = float(args.target_bytes)
    ple_rows = target / args.ple_row_bytes
    projection = {"target_bytes": target, "target_gib": target / GIB,
                  "ple_row_bytes": args.ple_row_bytes,
                  "ple_rows_equivalent": ple_rows, "per_order": []}
    for r in results:
        M = r["order"]
        b = r["storage"]["packed_bytes_total"]
        per_tok = b / max(1, int(train.shape[0]))
        pts = [(s["train_tokens"], s["per_model_order"][M]["bytes"]) for s in scaling]
        pts = [(x, y) for x, y in pts if y > 0]
        fit_bytes = fit_power_law([x for x, _ in pts], [y for _, y in pts]) \
            if len(pts) >= 2 else None
        dpts = [(s["train_tokens"], s["per_model_order"][M]["distinct_ngrams"])
                for s in scaling if s["per_model_order"][M]["distinct_ngrams"] > 0]
        fit_distinct = fit_power_law([x for x, _ in dpts], [y for _, y in dpts]) \
            if len(dpts) >= 2 else None
        projection["per_order"].append({
            "order": M,
            "stored_ngrams_at_train_size": int(sum(
                l["distinct_ngrams"] for l in r["storage"]["per_order"])),
            "packed_bytes_at_train_tokens": int(b),
            "train_tokens": int(train.shape[0]),
            "bytes_per_train_token": per_tok,
            "train_tokens_to_fill_target_linear": target / per_tok,
            "copies_of_this_table_in_target": target / b,
            "bytes_power_law_fit": fit_bytes,
            "train_tokens_to_fill_target_power_law": (
                invert_power_law(fit_bytes, target) if fit_bytes else None),
            "distinct_ngrams_power_law_fit": fit_distinct,
            "train_tokens_for_ple_row_count_power_law": (
                invert_power_law(fit_distinct, ple_rows) if fit_distinct else None),
            "saturation_note": saturation_note(M, fit_distinct, scaling, vocab),
        })

    # ---- quality vs bytes: learning curves + byte-budget reading ----------
    byte_budget = {
        "ple_bytes": target,
        "ple_row_bytes": args.ple_row_bytes,
        "measured_count_bytes_per_stored_ngram": {},
        "ngrams_storable_in_target_at_count_density": {},
        "per_order": [],
    }
    for r in results:
        M = r["order"]
        ngrams = sum(l["distinct_ngrams"] for l in r["storage"]["per_order"])
        b = r["storage"]["packed_bytes_total"]
        bpn = b / ngrams if ngrams else None
        byte_budget["measured_count_bytes_per_stored_ngram"][str(M)] = bpn
        byte_budget["ngrams_storable_in_target_at_count_density"][str(M)] = (
            target / bpn if bpn else None)
        curve = []
        for s_ in scaling:
            q = s_["quality"].get(str(M))
            if not q:
                continue
            curve.append({"train_tokens": s_["train_tokens"],
                          "bytes": s_["per_model_order"][M]["bytes"],
                          "nll": q["nll"], "perplexity": q["perplexity"],
                          "top1_full_vocab": q.get("top1_full_vocab"),
                          "nll_at_1M_tokens_measured": s_["train_tokens"] == int(train.shape[0])})
        fit = fit_learning_curve([(c["train_tokens"], c["nll"]) for c in curve])
        fit_acc = fit_learning_curve(
            [(c["train_tokens"], -c["top1_full_vocab"]) for c in curve
             if c.get("top1_full_vocab") is not None])
        if fit_acc is not None:  # same functional form, fitted to top-1 accuracy
            fit_acc = {"asymptote_top1": -fit_acc["asymptote_nll"],
                       "slope": fit_acc["slope"], "r2": fit_acc["r2"],
                       "n_points": fit_acc["n_points"]}
        byte_budget["per_order"].append({
            "order": M,
            "curve": curve,
            "learning_curve_fit": fit,
            "learning_curve_fit_top1": fit_acc,
            "quality_at_target_bytes_extrapolated": quality_at_bytes(
                curve, target, fit["asymptote_nll"] if fit else None),
            "ple_row_equivalents": target / args.ple_row_bytes,
            "measured_bytes_per_ple_row": args.ple_row_bytes,
        })

    # ---- distribution analysis (round 159) --------------------------------
    analysis: Dict[str, object] = {
        "enabled": bool(args.tail_analysis or args.verbatim_ks or args.source_text),
        "pre_registered_rule": PRE_REGISTERED_RULE,
        "pre_registered_operationalisation": PREREGISTERED_OPERATIONALISATION,
        "position_decontamination": decon_info,
    }
    if args.tail_analysis or args.verbatim_ks:
        buckets = parse_buckets(args.tail_buckets)
        am = analysis_cache
        raw3 = full_levels.get("raw3")
        if raw3 is None:
            raise SystemExit("--tail-analysis needs order 3 in --orders (raw trigram counts)")
        counts_all = trigram_context_counts(eval_stream, am["pos"], raw3, am["model"])
        bidx_all = bucket_index(counts_all, buckets)
        lp_all = am["lp"]
        nll_all = -lp_all
        total_nll = float(nll_all.sum())
        bucket_rows = []
        for b, (lo, hi, label) in enumerate(buckets):
            m = bidx_all == b
            n_b = int(m.sum())
            nll_b = float(nll_all[m].sum()) if n_b else 0.0
            bucket_rows.append({
                "bucket": label, "count_min": lo, "count_max": hi,
                "n_positions": n_b,
                "share_positions": n_b / max(1, am["pos"].shape[0]),
                "nll_sum": nll_b,
                "share_nll": nll_b / total_nll if total_nll > 0 else 0.0,
                "nll_mean": (nll_b / n_b) if n_b else None,
            })
        # top-1 per bucket on the accuracy subsample (same positions for all orders)
        counts_acc = trigram_context_counts(eval_stream, am["acc_pos"], raw3, am["model"])
        bidx_acc = bucket_index(counts_acc, buckets)
        masks = {}
        for name in (("top{}".format(args.acc_candidates)) if cand_topk is not None
                     else "full_vocab",):
            mk = am["acc_masks"].get(name)
            if mk is None:
                continue
            t1 = mk["top1"]
            ins = mk["inside"]
            per = []
            for b, (_lo, _hi, label) in enumerate(buckets):
                sel = (bidx_acc == b) & ins
                n_sel = int(sel.sum())
                per.append({"bucket": label, "n_scored": n_sel,
                            "top1": float(t1[sel].mean()) if n_sel else None})
            masks[name] = per
        analysis["tail_buckets"] = {
            "order_used": am["order"],
            "bucketed_by": "raw train count of the trigram context ending at position-1",
            "positions_scored": int(am["pos"].shape[0]),
            "total_nll": total_nll,
            "buckets": bucket_rows,
            "top1_by_bucket": masks,
            "top1_by_bucket_candidate_set": (
                "top{}".format(args.acc_candidates) if cand_topk is not None else "full_vocab"),
            "n_accuracy_positions": int(am["acc_pos"].shape[0]),
            "head_tail_split": {
                "tail_share_nll_count_le_10": float(sum(
                    r["share_nll"] for r in bucket_rows if r["count_min"] <= 10
                    and (r["count_max"] is None or r["count_max"] <= 10))),
                "tail_share_positions_count_le_10": float(sum(
                    r["share_positions"] for r in bucket_rows if r["count_min"] <= 10
                    and (r["count_max"] is None or r["count_max"] <= 10))),
            },
        }
        ks = [int(x) for x in args.verbatim_ks.split(",") if x.strip()]
        if ks:
            mults_v = np.random.default_rng(args.seed).integers(
                1, 2 ** 63, size=512, dtype=np.uint64) | np.uint64(1)
            verb = {}
            joint = {}
            for k in ks:
                hit, usable = ngram_present_mask(train, eval_stream, am["pos"], k + 1,
                                                 vocab, mults_v)
                n_use = int(usable.sum())
                verb[str(k)] = {
                    "n_positions_with_full_context": n_use,
                    "hit_rate": float(hit[usable].mean()) if n_use else None,
                }
                per_bucket = []
                for b, (_lo, _hi, label) in enumerate(buckets):
                    sel = usable & (bidx_all == b)
                    n_sel = int(sel.sum())
                    per_bucket.append({"bucket": label, "n_positions": n_sel,
                                       "hit_rate": float(hit[sel].mean()) if n_sel else None})
                joint[str(k)] = per_bucket
                del hit, usable
                gc.collect()
            analysis["verbatim_continuation"] = {
                "definition": ("fraction of scored positions whose gold continuation "
                               "follows the same k-token context somewhere in the "
                               "TRAINING stream (exact up to 64-bit hash collisions, "
                               "p ~ 1e-7); necessary, not sufficient, for a table to help"),
                "per_k": verb,
                "per_k_by_bucket": joint,
            }
        raw = am.get("raw")
        if raw is not None:
            cnt_raw = trigram_context_counts(eval_stream, raw["pos"], raw3, am["model"])
            bidx_raw = bucket_index(cnt_raw, buckets)
            nll_raw = -raw["lp"]
            tot_raw = float(nll_raw.sum())
            rows_raw = []
            for b, (lo, hi, label) in enumerate(buckets):
                m = bidx_raw == b
                nb = int(m.sum())
                nllb = float(nll_raw[m].sum()) if nb else 0.0
                rows_raw.append({"bucket": label, "count_min": lo, "count_max": hi,
                                 "n_positions": nb,
                                 "share_positions": nb / max(1, raw["pos"].shape[0]),
                                 "nll_sum": nllb,
                                 "share_nll": nllb / tot_raw if tot_raw > 0 else 0.0,
                                 "nll_mean": (nllb / nb) if nb else None})
            no_decon = {
                "reason": ("the position filter removes exactly the spans that repeat in "
                           "the training stream, which for high-redundancy corpora is the "
                           "phenomenon under study; this variant keeps every position"),
                "positions_scored": int(raw["pos"].shape[0]),
                "nll_mean": float(nll_raw.mean()),
                "buckets": rows_raw,
                "tail_share_nll_count_le_10": float(sum(
                    r["share_nll"] for r in rows_raw if r["count_min"] <= 10
                    and (r["count_max"] is None or r["count_max"] <= 10))),
                "tail_share_positions_count_le_10": float(sum(
                    r["share_positions"] for r in rows_raw if r["count_min"] <= 10
                    and (r["count_max"] is None or r["count_max"] <= 10))),
            }
            if args.verbatim_ks:
                mults_r = np.random.default_rng(args.seed).integers(
                    1, 2 ** 63, size=512, dtype=np.uint64) | np.uint64(1)
                per_k_raw = {}
                for k in [int(x) for x in args.verbatim_ks.split(",") if x.strip()]:
                    hit, usable = ngram_present_mask(train, eval_stream, raw["pos"],
                                                     k + 1, vocab, mults_r)
                    n_use = int(usable.sum())
                    per_k_raw[str(k)] = {
                        "n_positions_with_full_context": n_use,
                        "hit_rate": float(hit[usable].mean()) if n_use else None}
                    del hit, usable
                    gc.collect()
                no_decon["verbatim_per_k"] = per_k_raw
            analysis["no_decontamination"] = no_decon

    if args.source_text:
        src = Path(args.source_text)
        if not src.exists():
            raise SystemExit("source text not found: {}".format(src))
        src_bytes = int(src.stat().st_size)
        per_order = {}
        for r in results:
            b = r["storage"]["packed_bytes_total"]
            per_order[str(r["order"])] = {
                "packed_bytes": int(b),
                "bytes_per_source_byte": b / src_bytes if src_bytes else None,
                "bytes_per_train_token": b / max(1, int(train.shape[0])),
            }
        analysis["storage_normalised"] = {
            "source_text": str(src),
            "source_bytes": src_bytes,
            "train_tokens": int(train.shape[0]),
            "tokens_per_source_byte": int(train.shape[0]) / src_bytes if src_bytes else None,
            "train_tokens_times_8_bytes": int(train.shape[0]) * 8,
            "note": ("bytes-per-stored-ngram is tokenizer dependent, so the table is "
                     "also expressed per byte of source text; corpus.txt is the decoded "
                     "text the tokens were built from (newlines normalised), not the "
                     "raw upstream file"),
            "per_order": per_order,
        }
    assumptions = [
        "packed_bytes_* are measured numpy nbytes of the packed CSR arrays "
        "(int64 exact context keys, int32 token ids, uint32 counts, int64 offsets); "
        "no compression, no hash index, no Python object overhead.",
        "scoring-side arrays (zeff, discsum per context) are reported separately as "
        "scoring_side_bytes_total; they are recomputable from the table and are not "
        "part of the shipped bytes.",
        "the linear projection assumes bytes/token stays constant, which "
        "over-estimates growth (distinct n-grams grow sublinearly), so it "
        "UNDER-estimates the corpus needed to fill the target.",
        "the power-law projection assumes bytes(N) ~ N**b fitted on a handful of "
        "prefixes inside one order of magnitude; ~1M tokens is far too little to "
        "estimate a corpus-level Heaps exponent, so those token counts are "
        "order-of-magnitude only.",
        "accuracy and NLL are measured at one training size only; nothing here "
        "justifies extrapolating them to the target storage.",
        "PLE rows are assumed to cost {} bytes each (fp8, 160 dims); {} bytes "
        "therefore equals {:,.0f} rows.".format(args.ple_row_bytes, int(target), ple_rows),
        "NLL covers the whole evaluation stream at identical positions for every "
        "order; accuracy is argmax over a candidate set on a position subsample.",
        "full_vocab accuracy ranks over the training vocabulary only; tokens never "
        "seen in training all score exactly the uniform floor, which is below any "
        "seen token's score, so the ranking is unchanged.",
        "the uniform floor is spread over V_uni={:,} tokens, so OOV targets get "
        "Lambda_1/V_uni.".format(vocab),
    ]

    out = {
        "schema": "qwen35-ple-ngram-reference-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "command": " ".join(sys.argv),
        "elapsed_sec": time.time() - t_start,
        "split": split,
        "train_path": str(train_path),
        "train_sha256": sha256_file(train_path),
        "uniform_vocab": vocab,
        "uniform_vocab_source": args.uniform_vocab,
        "smoothing": args.smoothing,
        "orders": orders,
        "max_order": max_order,
        "n_scored_positions_per_model": n_positions,
        "n_accuracy_positions": int(acc_pos.shape[0]),
        "n_accuracy_positions_full": int(acc_pos_full.shape[0]) if acc_pos_full is not None else 0,
        "acc_candidates": args.acc_candidates,
        "peak_rss_bytes": peak_rss_bytes(),
        "peak_rss_bytes_before_tables": peak_rss_build_start,
        "scaling": scaling,
        "results": results,
        "sanity": sanity,
        "projection": projection,
        "byte_budget": byte_budget,
        "analysis": analysis,
        "assumptions": assumptions,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        log("wrote {}".format(args.output))
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(render_markdown(out), encoding="utf-8")
        log("wrote {}".format(args.markdown))
    log("done in {:.1f}s, peak RSS {:.2f} GiB".format(
        time.time() - t_start, peak_rss_bytes() / GIB))
    return 0



def fit_learning_curve(points: Sequence[Tuple[float, float]]) -> Optional[Dict[str, float]]:
    """Fit NLL(N) = a + b * N**(-0.5) (the usual empirical LM learning curve).

    ``a`` is the estimated asymptote, i.e. what the order could reach with
    unlimited data of this distribution.  Three or more points are required.
    """
    pts = [(n, v) for n, v in points if n > 0]
    if len(pts) < 3:
        return None
    x = np.array([n ** -0.5 for n, _ in pts], dtype=np.float64)
    y = np.array([v for _, v in pts], dtype=np.float64)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"asymptote_nll": float(coef[0]), "slope": float(coef[1]),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0, "n_points": len(pts)}


def quality_at_bytes(curve: Sequence[Dict[str, float]], target: float,
                     asymptote: Optional[float]) -> Dict[str, object]:
    """Read the measured NLL off the bytes axis, and say how far it is extrapolated."""
    pts = [(c["bytes"], c["nll"]) for c in curve if c.get("bytes") and c.get("nll")]
    if len(pts) < 2:
        return {"available": False}
    b = np.log(np.array([p[0] for p in pts], dtype=np.float64))
    y = np.array([p[1] for p in pts], dtype=np.float64)
    A = np.vstack([np.ones_like(b), b]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    out = {
        "available": True,
        "fit": {"intercept": float(coef[0]), "slope_per_ln_byte": float(coef[1]),
                "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
                "n_points": len(pts)},
        "measured_bytes_span": [float(min(p[0] for p in pts)), float(max(p[0] for p in pts))],
        "target_bytes": float(target),
        "extrapolation_factor_in_bytes": float(target / max(p[0] for p in pts)),
        "nll_predicted_at_target": float(coef[0] + coef[1] * math.log(target)),
        "caveat": ("log-linear fit of NLL on ln(bytes) over the measured span, "
                   "extrapolated; a count table cannot actually reach this byte "
                   "budget on a 1M-token corpus -- the corpus, not the format, is "
                   "the binding constraint.  Read the asymptote column instead."),
    }
    if asymptote is not None:
        out["asymptote_nll_from_learning_curve"] = float(asymptote)
        out["nll_predicted_above_asymptote"] = (
            float(coef[0] + coef[1] * math.log(target)) - float(asymptote))
    return out



def default_mults(size: int = 512) -> np.ndarray:
    """Deterministic odd multipliers for context hashing (fixed seed).

    The table builder and the scorer must use the *same* multipliers or their
    hashes would not agree; ``self_test`` asserts that they do.
    """
    return np.random.default_rng(0).integers(
        1, 2 ** 63, size=size, dtype=np.uint64) | np.uint64(1)


def ngram_hash_rows(rows: np.ndarray, vocab: int, mults: np.ndarray) -> np.ndarray:
    """Hash each row of a 2-D token matrix as one context key (uint64).

    Must stay bit-identical to ``NgramModel._ctx_keys``: same 3-token chunks, the
    same multipliers, the same XOR combination.  ``self_test`` asserts this.
    """
    n, k = rows.shape
    keys = np.zeros(n, dtype=np.uint64)
    for c in range(0, k, 3):
        ln = min(3, k - c)
        acc = np.zeros(n, dtype=np.int64)
        for j in range(ln):
            acc *= vocab
            acc += rows[:, c + j].astype(np.int64)
        keys ^= acc.astype(np.uint64) * mults[(c // 3) % mults.shape[0]]
        del acc
    return keys


def ngram_hash_keys(stream: np.ndarray, n: int, vocab: int,
                    mults: np.ndarray) -> np.ndarray:
    """Deterministic 64-bit hash of every length-``n`` window of ``stream``.

    Windows are folded in chunks of three symbols (3 x 18 bits fits exactly in
    int64 for token ids, and 3 bytes for text), each chunk mixed with an odd
    64-bit multiplier and XOR-combined.  Collision probability for ~1e6 reference
    keys against ~1e6 queries is ~1e-7, i.e. far below any effect measured here.

    This is the single canonical implementation; ``build_wikitext_heldout.py``
    imports it rather than keeping a second copy.
    """
    N = int(stream.shape[0]) - n + 1
    if N <= 0:
        return np.zeros(0, dtype=np.uint64)
    keys = np.zeros(N, dtype=np.uint64)
    chunk = 3
    for c in range(0, n, chunk):
        ln = min(chunk, n - c)
        win = np.lib.stride_tricks.sliding_window_view(stream, ln)[c : c + N]
        acc = np.zeros(N, dtype=np.int64)
        for j in range(ln):
            acc *= vocab
            acc += win[:, j].astype(np.int64)
        keys ^= acc.astype(np.uint64) * np.uint64(mults[(c // chunk) % mults.shape[0]])
        del win, acc
    return keys


# --------------------------------------------------------------------------
# distribution analysis (round 159): tail buckets, verbatim memorisation,
# tokenizer-normalised storage.  Additive: nothing below runs unless the new
# flags are given, so every round-158 number keeps its meaning.
# --------------------------------------------------------------------------
PRE_REGISTERED_RULE = """PRE-REGISTERED INTERPRETATION RULE (fixed before any number below was computed)
  * tail share of NLL large AND verbatim hit rate high at k>=4  => strong candidate for external memory; quantify how much of the tail is memorisable.
  * tail share small OR verbatim hit rate ~0                     => external memory has no niche on this distribution regardless of how good the read-out is.
  * if NO corpus in the set beats PURE_WIKI materially on those two axes, the honest conclusion is that the limitation is a property of n-gram memory itself, not of prose, and the round-157/158 verdict generalises."""

# The rule above is not decidable as written: "large", "high" and "materially"
# have no thresholds.  They are operationalised here, BEFORE the numbers were
# seen, so the verdict is not chosen after the fact.  Both the raw numbers and
# this operationalisation are reported so a reader can apply their own.
PREREGISTERED_OPERATIONALISATION = {
    "tail_share_large": "share_nll of buckets with train trigram-count <= 10 is >= 0.50",
    "verbatim_high_k_ge_4": "hit rate at k=4 over all scored positions is >= 0.50",
    "beats_wiki_materially": ("on BOTH axes: tail share (count<=10) at least 0.10 higher "
                              "than PURE_WIKI, or verbatim hit rate at k=4 at least 0.10 "
                              "higher, and not worse on the other axis by more than 0.05"),
    "bucket_edges": "train count of the trigram context ending at the previous token: 0 / 1 / 2-4 / 5-10 / 11-100 / >100",
}

DEFAULT_BUCKETS = "0,1,2-4,5-10,11-100,101+"


def parse_buckets(spec: str):
    """Parse ``0,1,2-4,5-10,11-100,101+`` into [(lo, hi, label), ...]."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part.endswith("+"):
            lo = int(part[:-1])
            out.append((lo, None, part))
        elif "-" in part:
            lo, hi = part.split("-", 1)
            out.append((int(lo), int(hi), part))
        else:
            out.append((int(part), int(part), part))
    if not out:
        raise SystemExit("empty --tail-buckets")
    return out


def bucket_index(counts: np.ndarray, buckets) -> np.ndarray:
    """Map per-position counts to bucket indices (last matching bucket wins)."""
    idx = np.full(counts.shape[0], -1, dtype=np.int64)
    for b, (lo, hi, _label) in enumerate(buckets):
        m = counts >= lo
        if hi is not None:
            m &= counts <= hi
        idx[m] = b
    return idx


def trigram_context_counts(stream: np.ndarray, positions: np.ndarray,
                           raw3: "LevelTable", model: "NgramModel") -> np.ndarray:
    """Raw TRAIN count of the trigram context that ends at ``position - 1``.

    The graft's row id is a function of (token[t-2], token[t-1], token[t]) and is
    used to predict token[t+1], so for a scored target at index j the relevant
    context is the trigram ending at j-1.
    """
    if positions.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    pos_ctx = positions - 1
    if int(pos_ctx.min()) < 2:
        raise SystemExit("positions too early for a trigram context")
    ck = context_keys_at(stream, pos_ctx, 2, model.vocab)
    idx, valid = model.row_index(raw3, ck)
    return model.row_counts(raw3, idx, valid, stream[pos_ctx].astype(np.int64))


def ngram_present_mask(train_stream: np.ndarray, query_stream: np.ndarray,
                       positions: np.ndarray, n: int, vocab: int,
                       mults: np.ndarray):
    """For each scored position: does its length-``n`` window (context+gold) occur in train?

    The window for the target at index j is query[j-n+1 : j+1], i.e. the n-1
    preceding tokens plus the gold continuation.  Positions without a full window
    are reported as not applicable (``None`` in the returned mask's companion).
    """
    usable = positions >= (n - 1)
    out = np.zeros(positions.shape[0], dtype=bool)
    if not usable.any():
        return out, usable
    ref = np.unique(ngram_hash_keys(train_stream, n, vocab, mults))
    keys = ngram_hash_keys(query_stream, n, vocab, mults)
    start = positions[usable] - (n - 1)
    out[usable] = np.isin(keys[start], ref)
    del ref, keys
    return out, usable



def self_test() -> int:
    """Cheap invariants for the two places that are easy to get silently wrong.

    1. the vectorised ragged binary search must agree with a linear scan,
       including rows that end at the very end of the value array;
    2. every level must be a proper distribution: sum_w disc_w + gamma == 1.
    """
    notes: List[str] = []
    rows = np.array([[1, 5], [1, 7], [2, 3], [2, 9], [3, 4]], dtype=np.int32)
    counts = np.array([3, 1, 2, 4, 1], dtype=np.int64)
    l2 = build_level(rows, counts, 16)
    finalize_level(l2, "mkn", 0.1, 16, notes)
    uniq, cnt = unique_rows(np.array([[1], [2], [3]], dtype=np.int32))
    l1 = build_level(uniq, np.array([5, 6, 1], dtype=np.int64), 16)
    finalize_level(l1, "mkn", 0.1, 16, notes)
    model = NgramModel(2, {1: l1, 2: l2}, 16, "mkn", 0.1, notes)
    ctx = np.array([3, 2, 1, 3, 0], dtype=np.int64)
    w = np.array([4, 9, 5, 8, 0], dtype=np.int64)
    idx, valid = model.row_index(l2, ctx)
    got = model.row_counts(l2, idx, valid, w)
    want = np.zeros_like(got)
    for j in range(ctx.shape[0]):
        if not valid[j]:
            continue
        s, e = int(l2.starts[idx[j]]), int(l2.starts[idx[j] + 1])
        hit = np.flatnonzero(l2.next_tokens[s:e] == w[j])
        want[j] = int(l2.counts[s + hit[0]]) if hit.size else 0
    ok_search = bool(np.array_equal(got, want))
    # unigram floor must be positive and the unigram must sum to 1 over V
    total = float(model.p1_dense.sum())
    print("[self-test] ragged search == linear scan: {}".format(ok_search))
    print("[self-test] floor={:.3e}  sum_w p1(w)={:.12f}".format(model.floor, total))
    print("[self-test] normalization notes: {}".format(notes or "none"))
    # 3. tail buckets: edges, ordering, and that every count lands somewhere
    buckets = parse_buckets(DEFAULT_BUCKETS)
    probe = np.array([0, 1, 2, 4, 5, 10, 11, 100, 101, 10 ** 6], dtype=np.int64)
    got = bucket_index(probe, buckets)
    want = np.array([0, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int64)
    ok_buckets = bool(np.array_equal(got, want))
    # 4. verbatim-continuation detection on a stream with a known repeat
    stream = np.array([1, 2, 3, 4, 5, 9, 9, 1, 2, 3, 4, 7], dtype=np.int64)
    train_s = stream[:6]          # contains 1,2,3,4,5
    positions = np.array([4, 11], dtype=np.int64)   # targets 5 (seen 4-gram) and 7 (unseen)
    mv = np.random.default_rng(0).integers(1, 2 ** 63, size=512, dtype=np.uint64) | np.uint64(1)
    hit, usable = ngram_present_mask(train_s, stream, positions, 5, 16, mv)
    ok_verbatim = bool(usable.all() and hit[0] and not hit[1])
    # 5. position decontamination only drops windows that really occur in train
    keep, _u = ngram_present_mask(train_s, stream, positions, 5, 16, mv)
    ok_decon = bool(keep[0] and not keep[1])
    print("[self-test] bucket edges: {}".format(ok_buckets))
    print("[self-test] verbatim hit (seen 5-gram True / unseen False): {}".format(ok_verbatim))
    print("[self-test] position decontamination: {}".format(ok_decon))
    # 6. round-160: hashed rows must equal the model's own context hashing
    rng2 = np.random.default_rng(7)
    probe = rng2.integers(0, 100, size=(50, 5)).astype(np.int32)
    mv2 = np.random.default_rng(0).integers(1, 2 ** 63, size=512, dtype=np.uint64) | np.uint64(1)
    rows_hash = ngram_hash_rows(probe, 128, mv2)
    probe_stream = np.concatenate([probe[0], probe[1]])
    model_hash = NgramModel(2, {1: l1, 2: l2}, 128, "mkn", 0.1, notes,
                            key_mode="hash")._ctx_keys(probe_stream[:6], np.array([5]), 5)
    ok_hash = bool(np.array_equal(rows_hash[0], model_hash[0]))
    # 7. pruning: bytes shrink, nothing below the threshold survives, still a
    #    proper distribution (finalize_level's per-row check must not complain)
    notes_prune: List[str] = []
    big = build_level(np.array([[1, 5], [1, 7], [2, 3], [2, 9], [3, 4], [3, 6]],
                               dtype=np.int32),
                      np.array([3, 1, 2, 4, 1, 1], dtype=np.int64), 16)
    finalize_level(big, "mkn", 0.1, 16, notes_prune)
    small = prune_level(big, 2, "mkn", 0.1, 16, notes_prune)
    ok_prune = (small.nbytes < big.nbytes and small.n_entries < big.n_entries
                and int(small.counts.min()) >= 2 and not notes_prune)
    # 8. longest-suffix lookup must differ from interpolation where a long context
    #    is observed once but a shorter one is reliably different
    # a long context seen once with a misleading continuation, while the shorter
    # context is *reliably* continued by the gold token: interpolation can lean on
    # the shorter context, longest-match cannot
    tr2_list = []
    for pre in (20, 21, 22, 23, 24):      # five distinct left extensions ...
        tr2_list += [pre, 1, 2, 3, 4]     # ... so cont_4(1,2,3,4) = 5 (KN-counted)
    tr2_list += [30, 1, 2, 3, 31]         # one long context (30,1,2,3) -> 31
    tr2 = np.array(tr2_list, dtype=np.int32)
    raw2, cont2 = build_counts(tr2, [1, 2, 3, 4, 5], 32)
    lvls2 = {}
    for k in range(1, 6):
        uq, ct = raw2[k] if k == 5 else cont2[k]
        lvls2[k] = build_level(uq, ct, 32)
        finalize_level(lvls2[k], "mkn", 0.1, 32, notes)
    q2 = np.array([30, 1, 2, 3, 4], dtype=np.int32)
    ppos2 = np.array([4], dtype=np.int64)   # context (30,1,2,3), gold 4
    m_i = NgramModel(5, lvls2, 32, "mkn", 0.1, notes, lookup="interpolate")
    m_l = NgramModel(5, lvls2, 32, "mkn", 0.1, notes, lookup="longest")
    lp_i, tl_i, _, _ = m_i.stream_logprob(q2, ppos2)
    lp_l, tl_l, _, _ = m_l.stream_logprob(q2, ppos2)
    ok_lookup = bool(lp_i[0] > lp_l[0] + 0.5 and int(tl_l[0]) == 5 and int(tl_i[0]) == 5)
    print("[self-test] hashed rows == model context hash: {}".format(ok_hash))
    print("[self-test] pruning (bytes {} -> {}, min count >= 2): {}".format(
        big.nbytes, small.nbytes, ok_prune))
    print("[self-test] longest lookup differs (interp ln p={:.4f} vs longest {:.4f}, "
          "gap {:.4f} nats): {}".format(lp_i[0], lp_l[0], lp_i[0] - lp_l[0], ok_lookup))
    ok = (ok_search and abs(total - 1.0) < 1e-9 and model.floor > 0 and not notes
          and ok_buckets and ok_verbatim and ok_decon and ok_hash and ok_prune
          and ok_lookup)
    print("[self-test] {}".format("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def render_markdown(out: Dict[str, object]) -> str:
    K = out["acc_candidates"]
    lines = ["# Count-based n-gram reference ({})".format(out["schema"]), ""]
    an = out.get("analysis") or {}
    if an.get("enabled"):
        lines.append("## Pre-registered interpretation rule (printed before the numbers)")
        lines.append("")
        lines.append("```")
        lines.append(an["pre_registered_rule"])
        lines.append("```")
        lines.append("")
    lines.append("split: {} | train {:,} tokens | eval {:,} tokens | V_uni={:,}".format(
        out["split"]["kind"], out["split"]["train_tokens"], out["split"]["eval_tokens"],
        out["uniform_vocab"]))
    lines.append("")
    lines.append("smoothing: {} | positions scored: {:,} (identical for all orders) | "
                 "top-K accuracy positions: {:,} | full-vocab accuracy positions: {:,} | "
                 "peak RSS {:.2f} GiB".format(
                     out["smoothing"], out["n_scored_positions_per_model"],
                     out["n_accuracy_positions"], out["n_accuracy_positions_full"],
                     out["peak_rss_bytes"] / GIB))
    lines.append("")
    lines.append("| order | NLL (nats) | ppl | bits/tok | top-1 | top-5 | top-1 (top{}) | "
                 "top-5 (top{}) | distinct n-grams | bytes | bytes/tok |".format(K, K))
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in out["results"]:
        m = r["metrics"]
        a_full = m["accuracy"].get("full_vocab", {})
        a_top = m["accuracy"].get("top{}".format(K), {})
        distinct = sum(l["distinct_ngrams"] for l in r["storage"]["per_order"])
        lines.append("| {} | {:.4f} | {:.2f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | "
                     "{:,} | {:,} | {:.2f} |".format(
                         r["order"], m["nll_nats"], m["perplexity"], m["bits_per_token"],
                         a_full.get("top1", float("nan")), a_full.get("top5", float("nan")),
                         a_top.get("top1", float("nan")), a_top.get("top5", float("nan")),
                         distinct, r["storage"]["packed_bytes_total"],
                         r["storage"]["packed_bytes_per_train_token"]))
    lines.append("")
    lines.append("majority-class (unigram) rate: {:.4f} (token id {})".format(
        out["sanity"]["majority_token_rate"], out["sanity"]["majority_token_id"]))
    if an.get("enabled"):
        lines.append("")
        lines.append("operationalisation of the rule (thresholds fixed before the run): "
                     "`{}`".format(json.dumps(an["pre_registered_operationalisation"])))
        pd = an.get("position_decontamination") or {}
        if pd.get("applied"):
            lines.append("")
            lines.append("position decontamination: dropped {:,} of {:,} scored positions "
                         "({:.4f}) whose {}-token window occurs in the training stream; "
                         "dropped-count sensitivity by window: {}".format(
                             pd["positions_dropped"], pd["positions_with_full_window"],
                             pd["positions_dropped_share_of_full_window"],
                             pd["window_tokens"],
                             json.dumps(pd["sensitivity_dropped_by_window"])))
        tb = an.get("tail_buckets")
        if tb:
            lines.append("")
            lines.append("## Long-tail decomposition (order {}; bucketed by the raw TRAIN "
                         "count of the trigram context)".format(tb["order_used"]))
            lines.append("")
            lines.append("| train count of trigram context | positions | share of positions | "
                         "share of total NLL | mean NLL | top-1 (cand={}) | n acc |".format(
                             tb["top1_by_bucket_candidate_set"]))
            lines.append("|---|---|---|---|---|---|---|")
            t1map = {r["bucket"]: r for r in tb["top1_by_bucket"].get(
                tb["top1_by_bucket_candidate_set"], [])}
            for r in tb["buckets"]:
                m = t1map.get(r["bucket"], {})
                lines.append("| {} | {:,} | {:.4f} | {:.4f} | {} | {} | {:,} |".format(
                    r["bucket"], r["n_positions"], r["share_positions"], r["share_nll"],
                    ("{:.4f}".format(r["nll_mean"]) if r["nll_mean"] is not None else "n/a"),
                    ("{:.4f}".format(m["top1"]) if m.get("top1") is not None else "n/a"),
                    m.get("n_scored", 0)))
            lines.append("")
            lines.append("tail share of NLL for contexts with train count <= 10: "
                         "**{:.4f}** (positions: {:.4f}); order used: {}; NLL positions: "
                         "{:,}; top-1 positions: {:,}".format(
                             tb["head_tail_split"]["tail_share_nll_count_le_10"],
                             tb["head_tail_split"]["tail_share_positions_count_le_10"],
                             tb["order_used"], tb["positions_scored"],
                             tb["n_accuracy_positions"]))
        vb = an.get("verbatim_continuation")
        if vb:
            lines.append("")
            lines.append("## Verbatim-continuation hit rate (gold continuation follows the "
                         "same k-token context in TRAINING)")
            lines.append("")
            lines.append("| k | positions with full context | hit rate |")
            lines.append("|---|---|---|")
            for k, v in sorted(vb["per_k"].items(), key=lambda kv: int(kv[0])):
                lines.append("| {} | {:,} | {} |".format(
                    k, v["n_positions_with_full_context"],
                    ("{:.4f}".format(v["hit_rate"]) if v["hit_rate"] is not None else "n/a")))
            lines.append("")
            lines.append("hit rate broken down by the tail buckets (rows = k, columns = "
                         "train trigram count):")
            lines.append("")
            blabels = [r["bucket"] for r in tb["buckets"]] if tb else []
            lines.append("| k | " + " | ".join(blabels) + " |")
            lines.append("|---|" + "---|" * len(blabels))
            for k, rows in sorted(vb["per_k_by_bucket"].items(), key=lambda kv: int(kv[0])):
                cells = []
                for r in rows:
                    cells.append("{:.4f} (n={:,})".format(r["hit_rate"], r["n_positions"])
                                 if r["hit_rate"] is not None else "n/a")
                lines.append("| {} | {} |".format(k, " | ".join(cells)))
        sn = an.get("storage_normalised")
        if sn:
            lines.append("")
            lines.append("## Tokenizer-normalised storage ({})".format(sn["source_text"]))
            lines.append("")
            lines.append("source bytes {:,}; train tokens {:,}; tokens per source byte "
                         "{:.4f}".format(sn["source_bytes"], sn["train_tokens"],
                                         sn["tokens_per_source_byte"]))
            lines.append("")
            lines.append("| order | packed bytes | bytes per source byte | bytes per train token |")
            lines.append("|---|---|---|---|")
            for k, r in sorted(sn["per_order"].items(), key=lambda kv: int(kv[0])):
                lines.append("| {} | {:,} | {:.6f} | {:.2f} |".format(
                    k, r["packed_bytes"], r["bytes_per_source_byte"], r["bytes_per_train_token"]))
    lines.append("")
    lines.append("## Projection onto {:.0f} GiB ({} bytes)".format(
        out["projection"]["target_gib"], int(out["projection"]["target_bytes"])))
    lines.append("")
    lines.append("| order | bytes @ {:,} tok | linear tok -> target | power-law b (R2) | "
                 "power-law tok -> target | copies of table in target |".format(
                     out["split"]["train_tokens"]))
    lines.append("|---|---|---|---|---|---|")
    for row in out["projection"]["per_order"]:
        fb = row["bytes_power_law_fit"] or {}
        pl = row["train_tokens_to_fill_target_power_law"]
        lines.append("| {} | {:,} | {:,.0f} | {:.3f} ({:.3f}) | {} | {:,.0f} |".format(
            row["order"], row["packed_bytes_at_train_tokens"],
            row["train_tokens_to_fill_target_linear"],
            fb.get("b", float("nan")), fb.get("r2", float("nan")),
            "{:,.0f}".format(pl) if pl else "n/a",
            row["copies_of_this_table_in_target"]))
    lines.append("")
    lines.append("PLE-equivalent rows in the target: {:,.0f} rows @ {} bytes/row".format(
        out["projection"]["ple_rows_equivalent"], out["projection"]["ple_row_bytes"]))
    lines.append("")
    lines.append("## Quality at each byte budget (same eval stream and positions)")
    lines.append("")
    lines.append("| order | train tokens | table bytes | NLL | ppl | top-1 (full vocab) | "
                 "bytes / stored n-gram | n-grams storable in the target |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in out["byte_budget"]["per_order"]:
        for c in row["curve"]:
            bpn = out["byte_budget"]["measured_count_bytes_per_stored_ngram"][str(row["order"])]
            lines.append("| {} | {:,} | {:,} | {:.4f} | {:.2f} | {} | {:.2f} | {:,.0f} |".format(
                row["order"], c["train_tokens"], c["bytes"], c["nll"], c["perplexity"],
                ("{:.4f}".format(c["top1_full_vocab"])
                 if c.get("top1_full_vocab") is not None else "n/a"),
                bpn, out["byte_budget"]["ngrams_storable_in_target_at_count_density"][str(row["order"])]))
    lines.append("")
    lines.append("| order | learning-curve asymptote NLL (a + b/sqrt(N)) | R2 | "
                 "asymptote top-1 (same fit) | R2 | "
                 "NLL predicted at the target byte budget (EXTRAPOLATION) | factor |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in out["byte_budget"]["per_order"]:
        f = row["learning_curve_fit"] or {}
        fa = row.get("learning_curve_fit_top1") or {}
        e = row["quality_at_target_bytes_extrapolated"]
        lines.append("| {} | {} | {} | {} | {} | {} | {}x |".format(
            row["order"],
            "{:.4f}".format(f["asymptote_nll"]) if f else "n/a",
            "{:.4f}".format(f["r2"]) if f else "n/a",
            "{:.4f}".format(fa["asymptote_top1"]) if fa else "n/a",
            "{:.4f}".format(fa["r2"]) if fa else "n/a",
            "{:.4f}".format(e.get("nll_predicted_at_target", float("nan")))
            if e.get("available") else "n/a",
            "{:,.0f}".format(e.get("extrapolation_factor_in_bytes", float("nan")))
            if e.get("available") else "n/a"))
    lines.append("")
    lines.append("Count tables need ~{:.1f} bytes per stored n-gram (order 4, measured), "
                 "versus {} bytes per frozen PLE row: at equal bytes the count format "
                 "holds ~{:.1f}x more n-grams.".format(
                     out["byte_budget"]["measured_count_bytes_per_stored_ngram"].get("4", float("nan")),
                     out["projection"]["ple_row_bytes"],
                     (out["byte_budget"]["ngrams_storable_in_target_at_count_density"].get("4", 0)
                      / max(1.0, out["projection"]["ple_rows_equivalent"]))))
    lines.append("")
    lines.append("sanity: `{}`".format(json.dumps(out["sanity"])))
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
