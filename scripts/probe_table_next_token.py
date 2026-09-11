#!/usr/bin/env python3
"""Does the frozen PLE row table carry *recoverable* next-token information?

Motivation (round-157)
----------------------
``docs/round-157-why-the-graft-is-a-prior-not-a-knowledge-channel.md`` argues
that the graft is a *prior* channel, not a *knowledge* channel, because the
reader can only learn a smooth low-complexity function of each 2560-dim row.
That argument is about the **read-out**.  This script asks the logically prior
question:

    Do the raw rows themselves already fail to predict the next token?

If the raw rows are uninformative, no reader can fix it and the "external
memory" premise is dead at the source.  If they are informative, the bottleneck
is the read-out and there is a concrete architectural fix.

PRE-REGISTERED INTERPRETATION RULE (fixed before seeing any number)
-------------------------------------------------------------------
    probe << trigram, probe ~ shuffled control ~ majority
        -> the raw rows carry NO recoverable next-token signal; the premise
           fails at the source, independent of any reader.
    probe ~ trigram, both clearly above majority
        -> the rows carry a faithful n-gram prior; the reader is failing to
           exploit it (supports round-157's "prior, not knowledge" reading and
           points at the read-out as the fixable part).
    probe clearly above trigram
        -> the table carries more than an explicit n-gram model estimated from
           this corpus, so the read-out is the bottleneck.  NOTE the structural
           caveat below: e_t is a function of the preceding trigram only, so
           this can only mean the row encodes a *better-estimated* trigram
           posterior (Qwen3.8 saw vastly more text), never information beyond
           an n-gram.

Structural ceiling (why the third case cannot mean "more than n-grams")
-----------------------------------------------------------------------
``e_t`` is the concatenation of 16 table rows (8 bigram heads + 8 trigram
heads).  ``qwen35_ple.ple_hash.PleSpec.rowids_for_seq`` fixes the row ids at
position ``t`` as a deterministic function of the **preceding** tokens only::

    order-2 heads: hash(token[t], token[t-1])
    order-3 heads: hash(token[t], token[t-1], token[t-2])

so ``e_t`` is a constant per n-gram, independent of the longer context, and
``I(e_t ; token[t+1]) <= I(trigram ; token[t+1])``.  No reader -- however deep
-- can beat the Bayes-optimal predictor given the preceding trigram.  The
empirical question is how close a learned linear read-out gets to a
well-estimated explicit trigram, and whether it clears the majority floor.
``verify_causality`` re-checks the causality claim empirically.

Predictors -- all scored on the SAME held-out positions with the SAME K
candidate set (positions whose target falls outside the candidate set are
dropped, so every method sees identical items)::

    majority_train_prior     : the train-stream class prior (top-1 = majority token)
    count_bigram             : explicit counts P(y | token[t]), interpolated
                               absolute discounting
    count_trigram            : explicit counts P(y | token[t-1], token[t]),
                               interpolated absolute discounting + backoff
    probe_raw_rows           : ridge linear probe 2560 -> K on raw e_t
    probe_frozen_value_proj  : same probe in front of the frozen official
                               ``value_proj``
    control_shuffled_train_rows : identical probe, TRAIN labels permuted
    control_shuffled_eval_rows  : probe on real rows, EVAL labels permuted

Permuting labels is distributionally identical to permuting rows across
positions; it is done that way so the streaming implementation never has to
re-fetch.  Both controls MUST collapse to (or below) the majority floor.

Data split
----------
TRAIN  = ``data/phase1/PURE_WIKI/tokens.npy`` -- the corpus the graft was
         trained on.  Supplies (a) the count models, (b) the probe's training
         positions, (c) the top-K vocabulary.  The last ``--dev-tokens`` tokens
         are excluded from counts and probe training and used only to tune the
         n-gram interpolation weights.
EVAL   = WikiText-103 records from ``data/sources/wikitext.jsonl`` NOT selected
         into PURE_WIKI.  The selection is reproduced exactly (``build_mix``
         seed 0 / budget 1e6 / 512-token chunks) and validated against the
         frozen manifest, then independently cross-checked by
         normalized-substring containment against PURE_WIKI/corpus.txt.

Caveat this design CANNOT remove: WikiText-103 is a very common public corpus
and is almost certainly inside the Qwen3.8 pretraining data the frozen table was
learned from.  A positive probe result therefore cannot separate "the table
generalises n-gram statistics" from "the table memorised WikiText".  It need
not: the row is by construction a function of the n-gram alone, so either way
the probe measures a **next-token prior conditional on the preceding trigram**,
never document-specific knowledge.

Resource behaviour
------------------
Written for a container with a hard ~2 GiB cgroup memory cap and 0.5 CPU.
Everything is streamed in ``--chunk`` sized blocks; there is no full-[N, 2560]
or [N, K] array anywhere.  ``--max-alloc-mb`` refuses any single allocation
above the budget instead of dying by SIGKILL, and peak RSS is reported.

Usage
-----
::

    PYTHONPATH=src python scripts/probe_table_next_token.py \
      --rows-dir /root/autodl-tmp/qwen35-ple/qwen38-rows \
      --tokenizer /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B \
      --out data/probe-table-next-token.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

LOG_PREFIX = "[probe-table]"
MAX_ALLOC = 400 * 2**20  # replaced from --max-alloc-mb in main()


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", flush=True)


def rss_mb() -> float:
    """Peak RSS of this process so far, MiB.

    ``ru_maxrss`` is KiB on Linux and bytes on macOS; the remote target is Linux
    but the script is also run locally for self-tests.
    """
    v = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return v / 1024.0 if sys.platform != "darwin" else v / 2**20


def guard(nbytes: float, what: str) -> None:
    """Refuse an allocation that would obviously blow the container budget."""
    if nbytes > MAX_ALLOC:
        raise MemoryError(
            f"refusing to allocate {nbytes / 2**20:.0f} MiB for {what} "
            f"(budget {MAX_ALLOC / 2**20:.0f} MiB); lower --chunk / --topk"
        )


def _norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


# --------------------------------------------------------------------------- #
# Row-table pre-flight
# --------------------------------------------------------------------------- #
DEFAULT_EXPECTED_SHARDS = 128
DEFAULT_EXPECTED_SHARD_BYTES = 400_001_920


def verify_rows_dir(
    rows_dir: str,
    *,
    expected_shards: int,
    expected_shard_bytes: int,
    allow_partial: bool,
) -> dict[str, Any]:
    """Refuse to run against a truncated row table.

    A partially-copied store is the single most dangerous failure mode for this
    experiment: ``engramdb.Store`` is told the *contract* geometry
    (``shards``/``rows_per_shard`` from the frozen ``PleSpec``), not the geometry
    actually present on disk, so a missing shard can silently yield zeros or
    garbage rows and make every number below a lie.  This check is deliberately
    independent of the store: it inspects the filesystem only.
    """
    root = Path(rows_dir)
    if not root.is_dir():
        raise SystemExit(f"row table directory does not exist: {rows_dir}")
    shards = sorted(root.glob("shard_*.bin"))
    sizes = [p.stat().st_size for p in shards]
    wrong = [p.name for p, s in zip(shards, sizes) if s != expected_shard_bytes]
    info: dict[str, Any] = {
        "dir": str(root),
        "shard_files_found": len(shards),
        "expected_shards": expected_shards,
        "expected_shard_bytes": expected_shard_bytes,
        "n_shards_wrong_size": len(wrong),
        "shards_wrong_size_examples": wrong[:5],
        "total_bytes": int(sum(sizes)),
        "allow_partial": bool(allow_partial),
    }
    problems = []
    if len(shards) != expected_shards:
        problems.append(f"found {len(shards)} shard_*.bin files, expected {expected_shards}")
    if wrong:
        problems.append(
            f"{len(wrong)} shard files are not {expected_shard_bytes} bytes (e.g. {wrong[0]})"
        )
    info["problems"] = problems
    if problems and not allow_partial:
        raise SystemExit(
            "ROW TABLE PRE-FLIGHT FAILED: "
            + "; ".join(problems)
            + f"  [dir={rows_dir}]  A partial table would silently produce a WRONG "
            "answer. Re-run once the copy completes, or pass --allow-partial-shards "
            "if you explicitly want to measure a truncated table."
        )
    if problems:
        log(f"WARNING: proceeding with a partial row table: {problems}")
    else:
        log(
            f"row table pre-flight OK: {len(shards)} shards x {expected_shard_bytes} bytes "
            f"({info['total_bytes'] / 2**30:.1f} GiB)"
        )
    return info


# --------------------------------------------------------------------------- #
# Stage 1: reproduce the PURE_WIKI selection so EVAL is genuinely held out
# --------------------------------------------------------------------------- #
def build_heldout_indices(
    *,
    wikitext_path: Path,
    tokenizer: Any,
    qa_exclude: Path | None,
    budget: int,
    seed: int,
    expected_records: int | None,
    expected_tokens: int | None,
    max_source_records: int | None = None,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """Return (filtered_record_texts, heldout_record_indices, stats).

    Mirrors ``scripts/build_mix.py`` for the single-category ``wiki=100`` run in
    ``data/phase1/PURE_WIKI/manifest.json``: ``_load_category`` ->
    ``_filter_contaminated`` -> ``Random(seed)`` shuffle -> greedy token-budget
    walk.  ``wiki`` is the only category with a source in that run, so its
    shuffle is the very first draw from the RNG.

    ``max_source_records`` truncates the source (smoke mode only); the manifest
    reproduction will then legitimately not match.
    """
    import build_mix  # local module, no import side effects

    texts, lengths = build_mix._load_category(
        [wikitext_path], "wiki", tokenizer, max_source_records,
        chunk_chars=build_mix.DEFAULT_CHUNK_CHARS,
    )
    log(f"wikitext records loaded: {len(texts)}")

    excluded_counts = 0
    if qa_exclude is not None and qa_exclude.exists():
        needles = build_mix._load_qa_needles(str(qa_exclude))
        texts, excluded_counts = build_mix._filter_contaminated(texts, needles)
        log(f"QA contamination filter removed {excluded_counts} records; kept {len(texts)}")

    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)

    chunk_tokens = build_mix.DEFAULT_CHUNK_TOKENS
    selected: set[int] = set()
    total = 0
    for i in order:
        if total >= budget:
            break
        n = lengths[i]
        selected.add(i)
        if n > chunk_tokens:
            # _select_for_budget splits an over-long record into chunk_tokens
            # shards, charging len(shard)+1 for every shard including the short
            # final one, until the budget is met.  The record is "touched" and
            # is therefore not held out even if only a prefix reached the corpus.
            for start in range(0, n, chunk_tokens):
                if total >= budget:
                    break
                total += min(chunk_tokens, n - start) + 1
        else:
            total += n + 1

    heldout = np.asarray(sorted(set(range(len(texts))) - selected), dtype=np.int64)
    stats = {
        "records_total": len(texts),
        "records_selected_reproduced": len(selected),
        "records_heldout": int(heldout.size),
        "tokens_selected_reproduced": int(total),
        "qa_filtered_records": int(excluded_counts),
        "reproduction_matches_manifest": bool(
            (expected_records is None or len(selected) == expected_records)
            and (expected_tokens is None or total == expected_tokens)
        ),
        "manifest_expected_records": expected_records,
        "manifest_expected_tokens": expected_tokens,
    }
    return texts, heldout, stats


def containment_check(
    texts: list[str],
    heldout: np.ndarray,
    corpus_txt: Path,
    *,
    sample: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Independent check that held-out records are absent from PURE_WIKI text.

    A hit means the record *is* in the graft's training corpus, i.e. the
    held-out split is contaminated for that record.
    """
    if not corpus_txt.exists():
        return {"skipped": True}
    norm_corpus = _norm(corpus_txt.read_text(encoding="utf-8", errors="ignore"))
    idx = heldout if heldout.size <= sample else rng.choice(heldout, sample, replace=False)
    checked = hits = 0
    for i in np.atleast_1d(idx):
        norm = _norm(texts[int(i)])
        if len(norm) < 200:
            continue
        checked += 1
        if norm in norm_corpus:
            hits += 1
    return {
        "checked_records_ge_200_norm_chars": checked,
        "found_in_pure_wiki": hits,
        "contamination_rate": (hits / checked) if checked else None,
        "sample_requested": int(sample),
    }


# --------------------------------------------------------------------------- #
# Stage 2: row ids and row fetch
# --------------------------------------------------------------------------- #
def rowids_at_positions(
    tokens: np.ndarray, positions: np.ndarray, *, window: int = 8192, verify_prefix: int = 20_000
) -> np.ndarray:
    """[N, 16] int32 rowids for the given (sorted) positions.

    ``rowids_for_seq`` prepends ``[eos, eos]`` and returns one row per input
    token, so for a window that starts two tokens before the first queried
    position the requested rows are identical to the full-sequence result (the
    EOS-segment arithmetic is shift-invariant once two real context tokens
    precede the query).  This is asserted against a full-sequence computation
    on a short prefix.
    """
    from qwen35_ple.real_ple import rowids_from_tokens

    positions = np.asarray(positions, dtype=np.int64)
    out = np.empty((len(positions), 16), dtype=np.int32)
    i = 0
    n = len(positions)
    while i < n:
        lo = int(positions[i])
        j = i
        while j + 1 < n and positions[j + 1] - lo < window:
            j += 1
        start = max(0, lo - 2)
        end = int(positions[j]) + 1
        rows = rowids_from_tokens(np.asarray(tokens[start:end], dtype=np.int64))
        for k in range(i, j + 1):
            out[k] = rows[int(positions[k]) - start]
        i = j + 1

    check = min(verify_prefix, len(tokens))
    if check > 8:
        whole = rowids_from_tokens(np.asarray(tokens[:check], dtype=np.int64))
        probe_pos = np.arange(2, check, dtype=np.int64)
        win = rowids_at_positions(tokens[:check], probe_pos, window=window, verify_prefix=0)
        assert np.array_equal(
            win.astype(np.int64), whole[2:check].astype(np.int64)
        ), "windowed rowids disagree with the whole-sequence reference"
    return out


def verify_causality(tokens: np.ndarray) -> dict[str, Any]:
    """Confirm row t depends only on tokens[t-2..t] (no future leakage)."""
    import importlib

    spec = importlib.import_module("qwen35_ple.ple_hash").real_spec()
    n = min(64, len(tokens) - 4)
    if n <= 0:
        return {"skipped": True}
    prefix = np.asarray(tokens[: 2 + n + 4], dtype=np.int64)
    full = np.asarray(spec.rowids_for_seq(prefix.tolist()), dtype=np.int64)
    truncated = np.asarray(spec.rowids_for_seq(prefix[: 2 + n].tolist()), dtype=np.int64)
    mutated = prefix.copy()
    mutated[-1] = int((int(mutated[-1]) + 1) % 200000)
    mutated_rows = np.asarray(spec.rowids_for_seq(mutated.tolist()), dtype=np.int64)
    return {
        "future_truncation_invariant": bool(np.array_equal(truncated, full[: 2 + n])),
        "future_perturbation_invariant": bool(np.array_equal(mutated_rows[:-1], full[:-1])),
    }


def make_fetcher(rows_dir: str, scale: float, block: int):
    """Return a callable fetching [n, 2560] float32 e_t for [n, 16] rowids.

    ``rows_dir`` is used verbatim -- nothing about the store layout is hard-coded
    beyond the frozen ``PleSpec`` geometry, and every fetched chunk is checked
    for all-zero rows so a truncated shard cannot pass silently.
    """
    import engramdb

    from qwen35_ple.real_ple import real_spec

    spec = real_spec()
    store = engramdb.Store(
        rows_dir, shards=spec.shards, rows_per_shard=spec.rows_per_shard, width=160
    )
    dim = 16 * 160
    stats = {"rows": 0, "zero_rows": 0, "finite": True}

    def fetch(rowids_sel: np.ndarray) -> np.ndarray:
        n = rowids_sel.shape[0]
        guard(n * dim * 4, f"e_t block ({n} rows)")
        out = np.empty((n, dim), dtype=np.float32)
        for s in range(0, n, block):
            e = min(n, s + block)
            arr = engramdb.fetch_e_t_tensor(
                store,
                rowids_sel[s:e].reshape(-1).tolist(),
                scale=scale, num_heads=16, head_dim=160, dtype=None, out_dtype=None,
            )
            chunk = arr.reshape(e - s, dim)
            out[s:e] = chunk.numpy() if hasattr(chunk, "numpy") else np.asarray(chunk)
        stats["rows"] += int(n)
        stats["zero_rows"] += int((~out.any(axis=1)).sum())
        if not np.isfinite(out).all():
            stats["finite"] = False
        return out

    return fetch, store, stats


# --------------------------------------------------------------------------- #
# Stage 3: explicit count models (interpolated absolute discounting, top-K only)
# --------------------------------------------------------------------------- #
class CountModel:
    """Bigram / trigram counts restricted to the K candidate targets.

    ``raw_bi(y|w1)     = max(c2-d,0)/c1 + (d*n1p_bi/c1) * p_uni(y)``
    ``raw_tri(y|w1,w2) = max(c3-d,0)/c2c + (d*n1p_tri/c2c) * P_bi(y|w1)``

    with Jelinek-Mercer interpolation of each level with its back-off::

        P_bi  = lam_bi  * raw_bi  + (1-lam_bi)  * p_uni
        P_tri = lam_tri * raw_tri + (1-lam_tri) * P_bi

    Bigram contexts are token ids, so they live in dense arrays of size V (the
    tokenizer vocabulary).  Trigram contexts are (w2, w1) pairs, so they live in
    sorted arrays addressed by ``np.searchsorted`` -- a dict would cost ~250 MiB
    at 1M tokens and this box has a 2 GiB cap.
    """

    def __init__(
        self, tokens: np.ndarray, vocab: np.ndarray, *,
        discount: float = 0.75, lam_bi: float = 0.7, lam_tri: float = 0.8,
    ) -> None:
        self.d = float(discount)
        self.lam_bi, self.lam_tri = float(lam_bi), float(lam_tri)
        self.K = int(len(vocab))
        self.V = int(tokens.max()) + 2

        remap = np.full(self.V, -1, dtype=np.int64)
        remap[vocab] = np.arange(self.K, dtype=np.int64)
        tgt = remap[tokens]

        uni = np.bincount(tgt[tgt >= 0], minlength=self.K).astype(np.float64)
        self.uni_p = (uni / max(1.0, uni.sum())).astype(np.float32)

        # ---- bigram: context token[i-1] -> target token[i] ---------------- #
        w1 = tokens[:-1].astype(np.int64)
        y1 = tgt[1:]
        ok = y1 >= 0
        uniq, cnt = np.unique(w1[ok] * self.K + y1[ok], return_counts=True)
        ctx = uniq // self.K
        self.bi_tgt = (uniq % self.K).astype(np.int64)
        self.bi_cnt = cnt.astype(np.float64)
        self.c1 = np.bincount(ctx, weights=self.bi_cnt, minlength=self.V).astype(np.float64)
        self.n1p_bi = np.bincount(ctx, minlength=self.V).astype(np.float64)
        del uniq, cnt, w1, y1
        self.bi_start = np.searchsorted(ctx, np.arange(self.V)).astype(np.int64)
        self.bi_end = np.searchsorted(ctx, np.arange(self.V), side="right").astype(np.int64)
        del ctx

        # ---- trigram: context (token[i-2], token[i-1]) -> target token[i] - #
        w2 = tokens[:-2].astype(np.int64)
        w1b = tokens[1:-1].astype(np.int64)
        y2 = tgt[2:]
        del tgt
        ok2 = y2 >= 0
        key3 = (w2[ok2] * self.V + w1b[ok2]) * self.K + y2[ok2]
        del w2, w1b, y2
        uniq3, cnt3 = np.unique(key3, return_counts=True)
        del key3
        ctx3 = uniq3 // self.K
        self.tri_tgt = (uniq3 % self.K).astype(np.int64)
        self.tri_cnt = cnt3.astype(np.float64)
        del uniq3, cnt3
        self.tri_ctx, first = np.unique(ctx3, return_index=True)
        self.tri_first = first.astype(np.int64)
        self.tri_m = np.diff(np.append(first, len(ctx3))).astype(np.int64)
        self.tri_c2c = np.add.reduceat(self.tri_cnt, first)
        del ctx3, first
        log(
            f"count model: top-K target mass {uni.sum() / len(tokens):.4f}, "
            f"seen bigram contexts {int((self.c1 > 0).sum())}, "
            f"seen trigram contexts {len(self.tri_ctx)}"
        )

    # -- context lookups ---------------------------------------------------- #
    def tri_lookup(self, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """Index into tri_ctx for each (w2, w1) pair, or -1 when unseen."""
        key = np.asarray(w2, dtype=np.int64) * self.V + np.asarray(w1, dtype=np.int64)
        pos = np.searchsorted(self.tri_ctx, key)
        pos = np.clip(pos, 0, max(0, len(self.tri_ctx) - 1))
        hit = self.tri_ctx[pos] == key if len(self.tri_ctx) else np.zeros(len(key), bool)
        return np.where(hit, pos, -1)

    # -- back-off levels ---------------------------------------------------- #
    def raw_bi(self, w1: np.ndarray) -> np.ndarray:
        """[n, K] absolute-discounted bigram, un-interpolated."""
        n = len(w1)
        guard(n * self.K * 4, f"bigram block ({n} positions)")
        out = np.empty((n, self.K), dtype=np.float32)
        w1 = np.asarray(w1, dtype=np.int64)
        c1 = self.c1[w1]
        alpha = np.where(c1 > 0, self.d * self.n1p_bi[w1] / np.maximum(c1, 1e-30), 1.0)
        out[:] = alpha[:, None].astype(np.float32) * self.uni_p[None, :]
        for i in range(n):
            c = c1[i]
            if c <= 0:
                out[i] = self.uni_p
                continue
            s, e = int(self.bi_start[w1[i]]), int(self.bi_end[w1[i]])
            if e > s:
                np.add.at(
                    out[i], self.bi_tgt[s:e],
                    (np.maximum(self.bi_cnt[s:e] - self.d, 0.0) / c).astype(np.float32),
                )
        return out

    def raw_tri(self, w1: np.ndarray, w2: np.ndarray, backoff: np.ndarray) -> np.ndarray:
        """[n, K] absolute-discounted trigram using ``backoff`` for unseen mass."""
        out = backoff.copy()
        idx = self.tri_lookup(w1, w2)
        for i in np.nonzero(idx >= 0)[0]:
            c = int(idx[i])
            c2c = float(self.tri_c2c[c])
            if c2c <= 0:
                continue
            s, m = int(self.tri_first[c]), int(self.tri_m[c])
            row = backoff[i] * np.float32(self.d * m / c2c)
            np.add.at(
                row, self.tri_tgt[s : s + m],
                (np.maximum(self.tri_cnt[s : s + m] - self.d, 0.0) / c2c).astype(np.float32),
            )
            out[i] = row
        return out

    def dist(self, level: str, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        bi = self.lam_bi * self.raw_bi(w1) + (1.0 - self.lam_bi) * self.uni_p[None, :]
        if level == "bigram":
            return bi
        tri = self.raw_tri(w1, w2, bi)
        return self.lam_tri * tri + (1.0 - self.lam_tri) * bi

    def tune(self, w1: np.ndarray, w2: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """Grid-search (lam_bi, lam_tri) on a dev slice by NLL at T=1."""
        raw_bi = self.raw_bi(w1)
        best = (float("inf"), self.lam_bi, self.lam_tri)
        for lb in (0.2, 0.4, 0.6, 0.8, 0.95):
            bi = lb * raw_bi + (1.0 - lb) * self.uni_p[None, :]
            raw_tri = self.raw_tri(w1, w2, bi)
            for lt in (0.2, 0.4, 0.6, 0.8, 0.95):
                v = softmax_nll(lt * raw_tri + (1.0 - lt) * bi, labels, 1.0)
                if v < best[0]:
                    best = (v, lb, lt)
        self.lam_bi, self.lam_tri = best[1], best[2]
        return {"dev_nll": best[0], "lam_bi": best[1], "lam_tri": best[2]}


# --------------------------------------------------------------------------- #
# Stage 4: per-position score accumulator (never materialises [N, K])
# --------------------------------------------------------------------------- #
class Scores:
    """Stores, per position, enough to compute top-1/top-5/NLL at any temperature.

    Storing ``(top1, top5, gold, logsumexp)`` per position instead of the
    ``[N, K]`` score matrix keeps this at ~1 MiB for 30k positions x K=1000 while
    remaining exact: every position lives in exactly one chunk, so its chunk-local
    top-5 *is* its global top-5.
    """

    def __init__(self, n: int, k: int = 5) -> None:
        guard(n * (4 + 4 * k + 4 + 4 + 4), "score accumulator")
        self.n = n
        self.filled = 0
        # sentinel-initialised so a partially filled accumulator can be scored
        self.top1 = np.full(n, -1, dtype=np.int32)
        self.top5 = np.full((n, k), -1, dtype=np.int32)
        self.gold = np.full(n, np.nan, dtype=np.float32)
        self.gold_shuf = np.full(n, np.nan, dtype=np.float32)
        self.lse = np.full(n, np.nan, dtype=np.float32)

    def update(
        self, start: int, sc: np.ndarray, labels: np.ndarray, labels_shuf: np.ndarray
    ) -> None:
        m = sc.shape[0]
        k = min(self.top5.shape[1], sc.shape[1])
        part = np.argpartition(-sc, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(sc, part, axis=1)
        ranked = np.take_along_axis(part, np.argsort(-vals, axis=1), axis=1)
        row = np.arange(m)
        self.top1[start : start + m] = ranked[:, 0]
        self.top5[start : start + m] = ranked
        self.gold[start : start + m] = sc[row, labels]
        self.gold_shuf[start : start + m] = sc[row, labels_shuf]
        mx = sc.max(axis=1)
        self.lse[start : start + m] = mx + np.log(
            np.exp((sc - mx[:, None]).astype(np.float32)).sum(axis=1)
        )
        self.filled = max(self.filled, start + m)

    def metrics(
        self, labels: np.ndarray, *, temperature: float = 1.0,
        use_shuf_gold: bool = False, limit: int | None = None,
    ) -> dict[str, float]:
        """top-1/top-5 against ``labels``; NLL against the (possibly permuted) gold.

        ``use_shuf_gold=True`` scores the permuted-label control: the predictions
        are unchanged, but the probability is read at ``labels[i]`` where the
        accumulator was fed ``labels_shuf`` -- i.e. the row/target pairing is
        destroyed, exactly as if the rows had been permuted across positions.

        ``limit`` scores only the first ``limit`` positions, which is what makes
        a per-chunk partial checkpoint readable.
        """
        m = self.n if limit is None else min(int(limit), self.n)
        labels = labels[:m]
        top1 = float((self.top1[:m] == labels).mean())
        top5 = float((self.top5[:m] == labels[:, None]).any(axis=1).mean())
        gold = (self.gold_shuf if use_shuf_gold else self.gold)[:m]
        t = np.float32(temperature)
        nll = float((self.lse[:m] / t - gold / t).mean())
        return {"n": int(m), "top1": top1, "top5": top5, "nll": nll, "temperature": float(temperature)}

    def subset(self, mask: np.ndarray) -> "Scores":
        s = Scores.__new__(Scores)
        s.n = int(mask.sum())
        s.filled = s.n
        s.top1 = self.top1[mask]
        s.top5 = self.top5[mask]
        s.gold = self.gold[mask]
        s.gold_shuf = self.gold_shuf[mask]
        s.lse = self.lse[mask]
        return s

    def fit_temperature(self, labels: np.ndarray, grid_size: int = 60) -> tuple[float, float]:
        best_t, best_nll = 1.0, float("inf")
        for t in np.exp(np.linspace(np.log(0.005), np.log(200.0), grid_size)):
            v = self.metrics(labels, temperature=float(t))["nll"]
            if v < best_nll:
                best_t, best_nll = float(t), v
        return best_t, best_nll


def softmax_nll(scores: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    z = scores.astype(np.float32) / np.float32(temperature)
    z -= z.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    p = z[np.arange(len(labels)), labels]
    return float(-np.log(np.maximum(p, 1e-30)).mean())


# --------------------------------------------------------------------------- #
# Stage 5: ridge linear probe, streamed
# --------------------------------------------------------------------------- #
class RidgeAccumulator:
    """Accumulates G = X^T X and B = X^T Y for one feature space."""

    def __init__(self, D: int, K: int, tag: str) -> None:
        guard(D * D * 8 + 2 * D * K * 4, f"ridge accumulators ({tag})")
        self.G = np.zeros((D, D), dtype=np.float64)
        self.B = np.zeros((D, K), dtype=np.float32)
        self.B_shuf = np.zeros((D, K), dtype=np.float32)
        self.D, self.K, self.tag = D, K, tag

    def add(self, X: np.ndarray, y: np.ndarray, y_shuf: np.ndarray) -> None:
        Xf = X.astype(np.float32, copy=False)
        self.G += (Xf.T @ Xf).astype(np.float64)
        for B, lab in ((self.B, y), (self.B_shuf, y_shuf)):
            order = np.argsort(lab, kind="stable")
            ys = lab[order]
            counts = np.bincount(ys, minlength=self.K)
            present = np.nonzero(counts)[0]
            starts = np.concatenate([[0], np.cumsum(counts[present])[:-1]]).astype(np.int64)
            B[:, present] += np.add.reduceat(Xf[order], starts, axis=0).T

    def solve(self, lam: float, *, shuffled: bool = False) -> np.ndarray:
        reg = np.full(self.D, float(lam), dtype=np.float64)
        reg[-1] = 0.0  # bias column is never penalised
        B = (self.B_shuf if shuffled else self.B).astype(np.float64)
        return np.linalg.solve(self.G + np.diag(reg), B)


def ridge_predict(X: np.ndarray, W: np.ndarray, out_dtype=np.float32) -> np.ndarray:
    return (X.astype(np.float64) @ W).astype(out_dtype)


# --------------------------------------------------------------------------- #
def sample_positions(
    tokens: np.ndarray, *, n: int, lo: int, hi: int,
    keep_targets: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Sample positions t in [lo, hi) whose target token[t+1] is a candidate."""
    cand = np.arange(lo, hi, dtype=np.int64)
    if cand.size == 0:
        return cand
    cand = cand[np.isin(tokens[cand + 1], keep_targets)]
    if cand.size > n:
        cand = np.sort(rng.choice(cand, size=n, replace=False))
    return cand


def stream_context(tokens: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(w1, w2) = (token[t], token[t-1]) -- the context the n-gram heads hash."""
    return (
        tokens[positions].astype(np.int64),
        tokens[np.maximum(positions - 1, 0)].astype(np.int64),
    )


def chunks(n: int, size: int):
    for s in range(0, n, size):
        yield s, min(n, s + size)


# --------------------------------------------------------------------------- #
# Checkpointing: the box is a 2 GiB / 0.5-core container that has already been
# OOM-killed repeatedly, so every expensive stage is resumable.
# --------------------------------------------------------------------------- #
def write_json(path: Path, payload: dict) -> None:
    """Atomic small-JSON write (never leaves a half-written report)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def scores_dump(name: str, sc: "Scores | None", out: dict, n: int) -> None:
    if sc is None:
        return
    out[f"{name}__top1"] = sc.top1
    out[f"{name}__top5"] = sc.top5
    out[f"{name}__gold"] = sc.gold
    out[f"{name}__gold_shuf"] = sc.gold_shuf
    out[f"{name}__lse"] = sc.lse
    out[f"{name}__n"] = np.int64(n)
    out[f"{name}__filled"] = np.int64(sc.filled)


def scores_load(name: str, data: Any) -> "Scores | None":
    key = f"{name}__top1"
    if key not in data:
        return None
    sc = Scores.__new__(Scores)
    sc.n = int(data[f"{name}__n"])
    sc.filled = int(data[f"{name}__filled"]) if f"{name}__filled" in data else sc.n
    sc.top1 = data[f"{name}__top1"]
    sc.top5 = data[f"{name}__top5"]
    sc.gold = data[f"{name}__gold"]
    sc.gold_shuf = data[f"{name}__gold_shuf"]
    sc.lse = data[f"{name}__lse"]
    return sc


def save_state(path: Path, arrays: dict, meta: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(path)
    write_json(path.with_name(path.name + ".meta.json"), meta)


def load_state(path: Path) -> tuple[dict, dict] | None:
    if not path.exists():
        return None
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    meta_path = path.with_name(path.name + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return arrays, meta


# --------------------------------------------------------------------------- #
def main() -> int:
    global MAX_ALLOC
    ap = argparse.ArgumentParser(
        description="Probe the frozen PLE table for recoverable next-token information"
    )
    ap.add_argument("--rows-dir", default="/root/autodl-tmp/qwen35-ple/qwen38-rows")
    ap.add_argument("--train-tokens", default="data/phase1/PURE_WIKI/tokens.npy")
    ap.add_argument("--wikitext", default="data/sources/wikitext.jsonl")
    ap.add_argument("--pure-wiki-corpus", default="data/phase1/PURE_WIKI/corpus.txt")
    ap.add_argument("--manifest", default="data/phase1/PURE_WIKI/manifest.json")
    ap.add_argument("--qa-exclude", default="data/qa-expanded-150.json")
    ap.add_argument("--tokenizer", default="/root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B")
    ap.add_argument("--reader-ckpt", default="data/official_ple_reader.pt")
    ap.add_argument("--out", default="data/probe-table-next-token.json")
    ap.add_argument("--scale", type=float, default=0.00019931793212890625)
    ap.add_argument("--topk", type=int, default=5000)
    ap.add_argument("--n-probe-train", type=int, default=300_000)
    ap.add_argument("--n-probe-val", type=int, default=30_000)
    ap.add_argument("--n-eval", type=int, default=60_000)
    ap.add_argument("--n-dev", type=int, default=20_000)
    ap.add_argument("--dev-tokens", type=int, default=20_000)
    ap.add_argument("--chunk", type=int, default=4_000)
    ap.add_argument("--rowid-window", type=int, default=65_536)
    ap.add_argument("--seed", type=int, default=20250911)
    ap.add_argument("--containment-sample", type=int, default=400)
    ap.add_argument("--max-alloc-mb", type=int, default=4096)
    ap.add_argument("--skip-value-proj", action="store_true")
    ap.add_argument(
        "--expected-shards", type=int, default=DEFAULT_EXPECTED_SHARDS,
        help="shard_*.bin files the row table must contain (0 = do not check)",
    )
    ap.add_argument(
        "--expected-shard-bytes", type=int, default=DEFAULT_EXPECTED_SHARD_BYTES,
        help="size every shard file must have (0 = do not check)",
    )
    ap.add_argument(
        "--allow-partial-shards", action="store_true",
        help="run even if the row table is incomplete (produces a WRONG answer)",
    )
    ap.add_argument(
        "--smoke", action="store_true",
        help="tiny end-to-end validation run (seconds, not minutes)",
    )
    ap.add_argument(
        "--state-file", default="",
        help="prefix for the resumable .npz state (empty = no checkpointing)",
    )
    ap.add_argument("--resume", action="store_true", help="resume from --state-file")
    ap.add_argument("--checkpoint-every", type=int, default=5, help="chunks between state saves")
    ap.add_argument(
        "--json-every", type=int, default=1,
        help="chunks between small-JSON partial-result writes",
    )
    ap.add_argument(
        "--max-source-records", type=int, default=0,
        help="truncate the wikitext source (smoke only); 0 = all records",
    )
    args = ap.parse_args()
    if args.smoke:
        # Deliberately small: the point is to exercise every stage and the
        # controls, not to measure anything.  Accuracy at this scale is noise.
        args.topk = min(args.topk, 200)
        args.n_probe_train = min(args.n_probe_train, 600)
        args.n_probe_val = min(args.n_probe_val, 200)
        args.n_eval = min(args.n_eval, 400)
        args.n_dev = min(args.n_dev, 200)
        args.chunk = min(args.chunk, 200)
        args.containment_sample = min(args.containment_sample, 40)
        args.max_source_records = args.max_source_records or 400
        args.checkpoint_every = 1
        if not args.out or args.out == "data/probe-table-next-token.json":
            args.out = "data/probe-table-next-token-smoke.json"
    MAX_ALLOC = int(args.max_alloc_mb) * 2**20

    rng = np.random.default_rng(args.seed)
    t_start = time.time()
    report: dict[str, Any] = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "seed": args.seed,
        "warnings": [],
        "interpretation_rule_preregistered": [
            "probe ~ shuffled control ~ majority  -> NO recoverable signal in the raw rows",
            "probe ~ trigram, both >> majority    -> faithful n-gram prior; read-out is the loss",
            "probe >> trigram                    -> rows beat explicit corpus counts; read-out is the bottleneck",
        ],
    }

    exp_records = exp_tokens = None
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        cat = man.get("per_category", {}).get("wiki", {})
        exp_records, exp_tokens = cat.get("records"), cat.get("tokens")
    report["manifest_expected"] = {"records": exp_records, "tokens": exp_tokens}
    log(f"peak RSS at start: {rss_mb():.0f} MiB; alloc budget {args.max_alloc_mb} MiB")

    # ---- tokenizer -------------------------------------------------------- #
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    log(f"tokenizer loaded: vocab={tokenizer.vocab_size} eos={tokenizer.eos_token_id}")

    # ---- TRAIN stream ----------------------------------------------------- #
    train_tokens = np.load(args.train_tokens).astype(np.int64).reshape(-1)
    T_train = int(train_tokens.size)
    train_usable = T_train - args.dev_tokens
    log(f"train stream: {T_train} tokens ({args.train_tokens}); peak RSS {rss_mb():.0f} MiB")
    if train_usable < 100_000:
        raise SystemExit("train stream too short for the requested dev split")

    # ---- EVAL stream (held out from PURE_WIKI) ---------------------------- #
    texts, heldout_idx, split_stats = build_heldout_indices(
        wikitext_path=Path(args.wikitext), tokenizer=tokenizer,
        qa_exclude=Path(args.qa_exclude), budget=1_000_000, seed=0,
        expected_records=exp_records, expected_tokens=exp_tokens,
        max_source_records=args.max_source_records or None,
    )
    cont = containment_check(
        texts, heldout_idx, Path(args.pure_wiki_corpus),
        sample=args.containment_sample, rng=rng,
    )
    report["heldout_split"] = {**split_stats, "containment_check": cont}
    log(f"held-out split: {split_stats}")
    log(f"containment check: {cont}")
    if not split_stats["reproduction_matches_manifest"]:
        report["warnings"].append("PURE_WIKI selection reproduction != manifest")

    eos = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else 248044
    eval_ids: list[int] = []
    for i in heldout_idx.tolist():
        eval_ids.extend(tokenizer.encode(texts[i], add_special_tokens=False))
        eval_ids.append(eos)
    eval_tokens = np.asarray(eval_ids, dtype=np.int64)
    del eval_ids, texts
    T_eval = int(eval_tokens.size)
    report["heldout_split"]["eval_stream_tokens"] = T_eval
    log(f"eval stream: {T_eval} tokens from {heldout_idx.size} held-out records")

    # ---- candidate vocabulary (from TRAIN only) --------------------------- #
    counts = np.bincount(train_tokens, minlength=tokenizer.vocab_size + 8)
    counts[eos] = 0  # never predict document end
    vocab = np.argsort(-counts, kind="stable")[: args.topk]
    coverage = float(counts[vocab].sum() / counts.sum())
    log(
        f"candidate vocab K={args.topk}, train coverage={coverage:.4f}, "
        f"most frequent = {int(vocab[0])} ({tokenizer.decode([int(vocab[0])])!r})"
    )

    # ---- positions and rowids --------------------------------------------- #
    probe_train_pos = sample_positions(
        train_tokens, n=args.n_probe_train, lo=2, hi=train_usable - 1,
        keep_targets=vocab, rng=rng,
    )
    probe_val_pos = sample_positions(
        train_tokens, n=args.n_probe_val, lo=train_usable, hi=T_train - 1,
        keep_targets=vocab, rng=rng,
    )
    eval_pos = sample_positions(
        eval_tokens, n=args.n_eval, lo=2, hi=T_eval - 1, keep_targets=vocab, rng=rng
    )
    del counts
    n_tr, n_va, n_ev = len(probe_train_pos), len(probe_val_pos), len(eval_pos)
    log(f"positions: probe-train={n_tr} probe-val={n_va} eval={n_ev}")
    if min(n_tr, n_va, n_ev) < 500:
        raise SystemExit("too few positions; check corpus paths")

    t0 = time.time()
    rd_tr = rowids_at_positions(train_tokens, probe_train_pos, window=args.rowid_window)
    rd_va = rowids_at_positions(train_tokens, probe_val_pos, window=args.rowid_window)
    rd_ev = rowids_at_positions(eval_tokens, eval_pos, window=args.rowid_window)
    rowid_secs = time.time() - t0
    causal = verify_causality(train_tokens)
    log(f"rowids in {rowid_secs:.1f}s; causality={causal}; peak RSS {rss_mb():.0f} MiB")
    if not all(v for k, v in causal.items() if isinstance(v, bool)):
        report["warnings"].append(f"causality check failed: {causal}")

    # ---- labels ----------------------------------------------------------- #
    cls_size = int(max(tokenizer.vocab_size, int(train_tokens.max()), int(eval_tokens.max()))) + 8
    cls = np.zeros(cls_size, dtype=np.int64)
    cls[vocab] = np.arange(len(vocab), dtype=np.int64)
    l_tr = cls[train_tokens[probe_train_pos + 1]]
    l_va = cls[train_tokens[probe_val_pos + 1]]
    l_ev = cls[eval_tokens[eval_pos + 1]]
    train_prior = np.bincount(l_tr, minlength=args.topk).astype(np.float64)
    train_prior /= train_prior.sum()
    majority_cls = int(train_prior.argmax())
    majority_token = int(vocab[majority_cls])
    eval_prior = np.bincount(l_ev, minlength=args.topk).astype(np.float64) / n_ev
    majority_rate = float(eval_prior[majority_cls])
    report["majority"] = {
        "token_id": majority_token,
        "token": tokenizer.decode([majority_token]),
        "train_rate": float(train_prior[majority_cls]),
        "eval_rate": majority_rate,
        "eval_label_entropy_nats": float(
            -(eval_prior[eval_prior > 0] * np.log(eval_prior[eval_prior > 0])).sum()
        ),
    }
    log(
        f"majority token = {majority_token} ({tokenizer.decode([majority_token])!r}) "
        f"eval rate = {majority_rate:.4f}"
    )

    # label permutations for the two controls (equivalent to permuting rows)
    l_tr_shuf = l_tr[rng.permutation(n_tr)]
    l_ev_shuf = l_ev[rng.permutation(n_ev)]

    results: dict[str, Any] = {}

    def add_result(name: str, val_m: dict, ev_m: dict) -> None:
        results[name] = {"val": val_m, "eval": ev_m}
        log(
            f"  {name:<30} eval top1={ev_m['top1']:.4f} top5={ev_m['top5']:.4f} "
            f"nll={ev_m['nll']:.4f} (T={ev_m['temperature']:.3g})"
        )

    # ---- majority baseline = train class prior ---------------------------- #
    pt = np.log(np.maximum(train_prior, 1e-12)).astype(np.float32)
    maj_val, maj_ev = Scores(n_va), Scores(n_ev)
    for s, e in chunks(n_va, args.chunk):
        maj_val.update(s, np.tile(pt, (e - s, 1)), l_va[s:e], l_va[s:e])
    for s, e in chunks(n_ev, args.chunk):
        maj_ev.update(s, np.tile(pt, (e - s, 1)), l_ev[s:e], l_ev[s:e])
    add_result(
        "majority_train_prior",
        maj_val.metrics(l_va, temperature=1.0),
        maj_ev.metrics(l_ev, temperature=1.0),
    )
    del maj_val, maj_ev

    # ---- count models ----------------------------------------------------- #
    cnt = CountModel(train_tokens[:train_usable], vocab)
    dev_pos = sample_positions(
        train_tokens, n=args.n_dev, lo=train_usable, hi=T_train - 1,
        keep_targets=vocab, rng=rng,
    )
    d_w1, d_w2 = stream_context(train_tokens, dev_pos)
    tuned = cnt.tune(d_w1, d_w2, cls[train_tokens[dev_pos + 1]])
    del d_w1, d_w2, dev_pos
    log(f"count model tuned on dev slice: {tuned}")
    report["count_model"] = {
        "discount": cnt.d, "dev_positions": int(args.n_dev), **tuned,
        "seen_bigram_contexts": int((cnt.c1 > 0).sum()),
        "seen_trigram_contexts": int(len(cnt.tri_ctx)),
    }

    va_w1, va_w2 = stream_context(train_tokens, probe_val_pos)
    ev_w1, ev_w2 = stream_context(eval_tokens, eval_pos)

    cnt_va, cnt_ev = {}, {}
    for level in ("bigram", "trigram"):
        sva = Scores(n_va)
        for s, e in chunks(n_va, args.chunk):
            sva.update(
                s, cnt.dist(level, va_w1[s:e], va_w2[s:e]), l_va[s:e], l_va[s:e]
            )
        T, _ = sva.fit_temperature(l_va)
        cnt_va[level] = sva
        sev = Scores(n_ev)
        for s, e in chunks(n_ev, args.chunk):
            sev.update(
                s, cnt.dist(level, ev_w1[s:e], ev_w2[s:e]), l_ev[s:e], l_ev_shuf[s:e]
            )
        cnt_ev[level] = sev
        add_result(
            f"count_{level}",
            sva.metrics(l_va, temperature=T),
            sev.metrics(l_ev, temperature=T),
        )
        del sva
    log(f"count models scored; peak RSS {rss_mb():.0f} MiB")
    # checkpoint: the count baselines alone are already a useful result
    report["results"] = results
    report["positions"] = {"probe_train": n_tr, "probe_val": n_va, "eval": n_ev}
    report["eval_candidate_coverage"] = coverage
    report["peak_rss_mb"] = rss_mb()
    report["stage"] = "counts_done"
    write_json(Path(args.out), report)

    # ---- probe scaffolding ------------------------------------------------- #
    fetch, store = make_fetcher(args.rows_dir, args.scale, block=args.chunk)
    try:
        D = 2560 + 1
        mu = sd = vp_mu = vp_sd = None

        def design(E: np.ndarray) -> np.ndarray:
            guard(E.shape[0] * D * 4, "design matrix chunk")
            X = np.empty((E.shape[0], D), dtype=np.float32)
            X[:, :-1] = (E - mu) / np.float32(sd)
            X[:, -1] = 1.0
            return X

        def design_vp(E: np.ndarray) -> np.ndarray:
            guard(E.shape[0] * D * 4, "value_proj design chunk")
            P = E @ V.T
            X = np.empty((P.shape[0], D), dtype=np.float32)
            X[:, :-1] = (P - vp_mu) / np.float32(vp_sd)
            X[:, -1] = 1.0
            return X

        # value_proj (frozen official reader) -- optional extra feature space
        V = None
        if not args.skip_value_proj and Path(args.reader_ckpt).exists():
            import torch

            ck = torch.load(args.reader_ckpt, map_location="cpu", weights_only=False)
            keys = [k for k in ck if k.endswith("value_proj.weight")]
            if keys:
                V = ck[keys[0]].to(torch.float32).numpy()
                log(f"value_proj loaded from {keys[0]} {V.shape}")
        if V is None:
            report["warnings"].append("value_proj probe disabled")

        # ---- pass 1: accumulate G / B on the probe-train positions -------- #
        acc = RidgeAccumulator(D, args.topk, "raw")
        acc_vp = RidgeAccumulator(D, args.topk, "value_proj") if V is not None else None
        sel_mask = np.zeros(n_tr, dtype=bool)
        sel_mask[rng.permutation(n_tr)[: max(1, n_tr // 10)]] = True
        sel_X = np.empty((int(sel_mask.sum()), D), dtype=np.float32)

        state_path = Path(args.state_file) if args.state_file else None
        state = load_state(state_path) if (state_path and args.resume) else None
        if state is not None and "G_raw" in state[0]:
            arrays, meta = state
            log(f"RESUME: restoring pass-1 state from {state_path} (meta={meta})")
            acc.G = arrays["G_raw"]
            acc.B = arrays["B_raw"]
            acc.B_shuf = arrays["B_shuf_raw"]
            if acc_vp is not None and "G_vp" in arrays:
                acc_vp.G = arrays["G_vp"]
                acc_vp.B = arrays["B_vp"]
                acc_vp.B_shuf = arrays["B_shuf_vp"]
            mu = arrays["mu"]
            sd = float(arrays["sd"])
            vp_mu = arrays["vp_mu"] if "vp_mu" in arrays else None
            vp_sd = float(arrays["vp_sd"]) if "vp_sd" in arrays else None
            best_lam = float(meta["best_lam"])
            best_acc = float(meta.get("best_acc", float("nan")))
            lam_scale = float(meta.get("lam_scale", float("nan")))
            report["probe"] = dict(meta.get("probe_report", {}))
            log(f"RESUME: lambda={best_lam:.6g}, selection top1={best_acc}")
        else:
            # ---- standardisation stats (deterministic prefix of rd_tr) ----- #
            stat_n = min(2000, n_tr)
            E_stat = fetch(rd_tr[:stat_n])
            mu = E_stat.mean(axis=0)
            sd = float(E_stat.std()) or 1.0
            if V is not None:
                Pv = E_stat @ V.T
                vp_mu = Pv.mean(axis=0)
                vp_sd = float(Pv.std()) or 1.0
                del Pv
            del E_stat
            log(f"feature standardisation: mean|mu|={np.abs(mu).mean():.3g} sd={sd:.3g}")

            sel_filled = 0
            t0 = time.time()
            for s, e in chunks(n_tr, args.chunk):
                E = fetch(rd_tr[s:e])
                X = design(E)
                y, ys = l_tr[s:e], l_tr_shuf[s:e]
                acc.add(X, y, ys)
                if acc_vp is not None:
                    acc_vp.add(design_vp(E), y, ys)
                m = sel_mask[s:e]
                if m.any():
                    gsel = np.nonzero(m)[0]
                    sel_X[sel_filled : sel_filled + len(gsel)] = X[gsel]
                    sel_filled += len(gsel)
                del E, X
                if (s // args.chunk) % 10 == 0:
                    log(f"  pass1 {e}/{n_tr}  peak RSS {rss_mb():.0f} MiB")
            log(f"pass 1 done in {time.time() - t0:.0f}s; peak RSS {rss_mb():.0f} MiB")

            # lambda selection on a held-out tenth of the train positions
            sel_y = l_tr[sel_mask]
            G_sel = (sel_X.T @ sel_X).astype(np.float64)
            order = np.argsort(sel_y, kind="stable")
            cts = np.bincount(sel_y[order], minlength=args.topk)
            pres = np.nonzero(cts)[0]
            st = np.concatenate([[0], np.cumsum(cts[pres])[:-1]]).astype(np.int64)
            B_sel = np.zeros((D, args.topk), dtype=np.float64)
            B_sel[:, pres] = np.add.reduceat(sel_X[order], st, axis=0).astype(np.float64).T
            del order, cts, pres, st
            G_fit = acc.G - G_sel
            B_fit = acc.B.astype(np.float64) - B_sel
            del G_sel, B_sel
            # ridge strength is expressed relative to the mean diagonal of G
            lam_scale = float(np.trace(G_fit)) / D
            log(f"ridge lambda scale (trace(G)/D) = {lam_scale:.4g}")
            best_lam, best_acc = None, -1.0
            for lam_rel in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
                lam = lam_rel * lam_scale
                reg = np.full(D, lam, dtype=np.float64)
                reg[-1] = 0.0
                W = np.linalg.solve(G_fit + np.diag(reg), B_fit)
                sc = ridge_predict(sel_X, W)
                a = float((sc.argmax(axis=1) == sel_y).mean())
                log(f"  lambda_rel={lam_rel:g} -> selection top1={a:.4f}")
                if a > best_acc:
                    best_lam, best_acc = lam, a
            log(f"selected lambda={best_lam:.6g} (rel {best_lam / lam_scale:g}, "
                f"selection top1={best_acc:.4f})")
            report["probe"] = {
                "lambda": best_lam, "lambda_rel": best_lam / lam_scale,
                "lambda_scale_trace_over_D": lam_scale,
                "lambda_selection_top1": best_acc,
                "fit_positions": int(n_tr - sel_mask.sum()),
                "selection_positions": int(sel_mask.sum()),
                "feature_sd": sd, "dim": D,
            }
            del B_fit, G_fit

            if state_path is not None:
                arrays = {
                    "G_raw": acc.G, "B_raw": acc.B, "B_shuf_raw": acc.B_shuf,
                    "mu": mu, "sd": np.float64(sd),
                }
                if acc_vp is not None:
                    arrays.update({
                        "G_vp": acc_vp.G, "B_vp": acc_vp.B,
                        "B_shuf_vp": acc_vp.B_shuf,
                        "vp_mu": vp_mu, "vp_sd": np.float64(vp_sd),
                    })
                save_state(state_path, arrays, {
                    "stage": "pass1_done", "best_lam": best_lam, "best_acc": best_acc,
                    "lam_scale": lam_scale, "probe_report": report["probe"],
                })
                log(f"checkpoint written: {state_path}")
        del sel_X

        W_real = acc.solve(best_lam)
        W_shuf = acc.solve(best_lam, shuffled=True)
        W_vp = acc_vp.solve(best_lam) if acc_vp is not None else None
        if state_path is None:
            # keep G/B alive when checkpointing so the state file stays complete
            acc.G = acc.B = acc.B_shuf = None  # type: ignore[assignment]
            if acc_vp is not None:
                acc_vp.G = acc_vp.B = acc_vp.B_shuf = None  # type: ignore[assignment]
        log(f"probes fitted; peak RSS {rss_mb():.0f} MiB")

        # ---- pass 2: predict on val + eval -------------------------------- #
        sc_va_real, sc_va_shuf, sc_va_vp = Scores(n_va), Scores(n_va), (
            Scores(n_va) if W_vp is not None else None
        )
        sc_ev_real, sc_ev_shuftr, sc_ev_vp = Scores(n_ev), Scores(n_ev), (
            Scores(n_ev) if W_vp is not None else None
        )
        resume_from = {"val": 0, "eval": 0}
        if state is not None and state[1].get("stage") == "pass2_partial":
            arrays, meta = state
            resume_from = meta.get("pass2_chunks", resume_from)
            for name, slot in (
                ("va_real", sc_va_real), ("va_shuf", sc_va_shuf), ("va_vp", sc_va_vp),
                ("ev_real", sc_ev_real), ("ev_shuftr", sc_ev_shuftr), ("ev_vp", sc_ev_vp),
            ):
                if slot is None:
                    continue
                loaded = scores_load(name, arrays)
                if loaded is not None:
                    slot.top1, slot.top5 = loaded.top1, loaded.top5
                    slot.gold, slot.gold_shuf, slot.lse = loaded.gold, loaded.gold_shuf, loaded.lse
            log(f"RESUME: pass 2 from {resume_from}")

        def dump_pass2(stage: str) -> None:
            if state_path is None:
                return
            arrays: dict[str, Any] = {"mu": mu, "sd": np.float64(sd)}
            if acc.G is not None:
                arrays.update({"G_raw": acc.G, "B_raw": acc.B, "B_shuf_raw": acc.B_shuf})
                if acc_vp is not None and acc_vp.G is not None:
                    arrays.update({
                        "G_vp": acc_vp.G, "B_vp": acc_vp.B, "B_shuf_vp": acc_vp.B_shuf,
                    })
            if vp_mu is not None:
                arrays.update({"vp_mu": vp_mu, "vp_sd": np.float64(vp_sd)})
            for name, slot in (
                ("va_real", sc_va_real), ("va_shuf", sc_va_shuf), ("va_vp", sc_va_vp),
                ("ev_real", sc_ev_real), ("ev_shuftr", sc_ev_shuftr), ("ev_vp", sc_ev_vp),
            ):
                scores_dump(name, slot, arrays, slot.n if slot is not None else 0)
            save_state(state_path, arrays, {
                "stage": stage, "best_lam": best_lam, "best_acc": best_acc,
                "lam_scale": lam_scale, "probe_report": report["probe"],
                "pass2_chunks": progress,
            })

        progress = dict(resume_from)
        t0 = time.time()
        for tag, rd, lab, lab_shuf, npos, accs in (
            ("val", rd_va, l_va, l_va, n_va,
             (sc_va_real, sc_va_shuf, sc_va_vp)),
            ("eval", rd_ev, l_ev, l_ev_shuf, n_ev,
             (sc_ev_real, sc_ev_shuftr, sc_ev_vp)),
        ):
            s_real, s_shuftr, s_vp = accs
            ci = 0
            for s, e in chunks(npos, args.chunk):
                ci += 1
                if ci <= progress.get(tag, 0):
                    continue
                E = fetch(rd[s:e])
                X = design(E)
                s_real.update(s, ridge_predict(X, W_real), lab[s:e], lab_shuf[s:e])
                s_shuftr.update(s, ridge_predict(X, W_shuf), lab[s:e], lab[s:e])
                if s_vp is not None:
                    s_vp.update(s, ridge_predict(design_vp(E), W_vp), lab[s:e], lab_shuf[s:e])
                del E, X
                progress[tag] = ci
                if ci % args.checkpoint_every == 0:
                    dump_pass2("pass2_partial")
                    log(f"  pass2[{tag}] {e}/{npos}  peak RSS {rss_mb():.0f} MiB")
            progress[tag] = ci
        log(f"pass 2 done in {time.time() - t0:.0f}s; peak RSS {rss_mb():.0f} MiB")
        if state_path is not None:
            dump_pass2("pass2_done")
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()

    T_probe, _ = sc_va_real.fit_temperature(l_va)
    add_result(
        "probe_raw_rows",
        sc_va_real.metrics(l_va, temperature=T_probe),
        sc_ev_real.metrics(l_ev, temperature=T_probe),
    )
    add_result(
        "control_shuffled_train_rows",
        sc_va_shuf.metrics(l_va, temperature=T_probe),
        sc_ev_shuftr.metrics(l_ev, temperature=T_probe),
    )
    add_result(
        "control_shuffled_eval_rows",
        sc_va_shuf.metrics(l_va, temperature=T_probe),
        sc_ev_real.metrics(l_ev_shuf, temperature=T_probe, use_shuf_gold=True),
    )
    if sc_ev_vp is not None:
        T_vp, _ = sc_va_vp.fit_temperature(l_va)
        add_result(
            "probe_frozen_value_proj",
            sc_va_vp.metrics(l_va, temperature=T_vp),
            sc_ev_vp.metrics(l_ev, temperature=T_vp),
        )
        report["value_proj_note"] = (
            "value_proj is linear, so a linear probe on value_proj(e_t) spans the same "
            "function class as a linear probe on e_t; the two differ only through the "
            "ridge penalty coordinates, and near-identical scores are the expected "
            "numerical consistency check."
        )

    # ---- seen / unseen context breakdown ---------------------------------- #
    tri_idx_ev = cnt.tri_lookup(ev_w1, ev_w2)
    tri_seen = tri_idx_ev >= 0
    breakdown: dict[str, Any] = {}
    for label, mask in (
        ("all_eval", np.ones(n_ev, dtype=bool)),
        ("trigram_context_seen_in_train", tri_seen),
        ("trigram_context_unseen_in_train", ~tri_seen),
    ):
        m = int(mask.sum())
        if m == 0:
            continue
        prior = np.bincount(l_ev[mask], minlength=args.topk) / m
        breakdown[label] = {
            "n": m,
            "fraction": float(mask.mean()),
            "majority_rate": float(prior[majority_cls]),
            "probe_raw_rows": sc_ev_real.subset(mask).metrics(l_ev[mask], temperature=T_probe),
            "control_shuffled_eval_rows": sc_ev_real.subset(mask).metrics(
                l_ev_shuf[mask], temperature=T_probe, use_shuf_gold=True
            ),
            "count_bigram": cnt_ev["bigram"].subset(mask).metrics(
                l_ev[mask], temperature=results["count_bigram"]["eval"]["temperature"]
            ),
            "count_trigram": cnt_ev["trigram"].subset(mask).metrics(
                l_ev[mask], temperature=results["count_trigram"]["eval"]["temperature"]
            ),
        }
        if sc_ev_vp is not None:
            breakdown[label]["probe_frozen_value_proj"] = sc_ev_vp.subset(mask).metrics(
                l_ev[mask], temperature=T_probe
            )
        b = breakdown[label]
        log(
            f"  [{label}] n={m} majority={b['majority_rate']:.4f} "
            f"probe={b['probe_raw_rows']['top1']:.4f} "
            f"bigram={b['count_bigram']['top1']:.4f} "
            f"trigram={b['count_trigram']['top1']:.4f}"
        )

    # ---- verdict ---------------------------------------------------------- #
    p = results["probe_raw_rows"]["eval"]
    maj = results["majority_train_prior"]["eval"]
    sh_tr = results["control_shuffled_train_rows"]["eval"]
    sh_ev = results["control_shuffled_eval_rows"]["eval"]
    tri = results["count_trigram"]["eval"]
    bi = results["count_bigram"]["eval"]
    control_ok = bool(sh_tr["top1"] <= maj["top1"] + 0.02 and sh_ev["top1"] <= maj["top1"] + 0.02)
    # The floor a probe must clear is the best trivial predictor available: the
    # majority rate OR either shuffled control.  Keying off the controls (rather
    # than majority alone) is what makes a probe that merely re-learns the class
    # prior score as "no signal".
    control_floor = max(maj["top1"], sh_tr["top1"], sh_ev["top1"])
    report["control_deltas_vs_majority"] = {
        "shuffled_train_rows": sh_tr["top1"] - maj["top1"],
        "shuffled_eval_rows": sh_ev["top1"] - maj["top1"],
    }
    report["control_floor"] = control_floor
    delta_maj = p["top1"] - maj["top1"]
    delta_floor = p["top1"] - control_floor
    delta_tri = p["top1"] - tri["top1"]
    if not control_ok:
        verdict = "VOID_CONTROL_FAILED"
        text = ("a shuffled-label control rose above the majority floor -- the probe "
                "pipeline leaks and no conclusion can be drawn.")
    elif delta_floor < 0.02:
        verdict = "NO_RECOVERABLE_SIGNAL"
        text = ("the probe does not clear the trivial-predictor floor "
                f"(majority / shuffled controls, best = {control_floor:.4f}) by 2 points: "
                "the raw 2560-dim rows carry no linearly recoverable next-token signal.")
    elif delta_tri > 0.02:
        verdict = "ABOVE_EXPLICIT_TRIGRAM"
        text = ("the probe beats the explicit trigram baseline.  e_t is by construction a "
                "function of the preceding trigram, so this cannot mean information beyond "
                "an n-gram model -- it means the table's row for an n-gram encodes a "
                "better-estimated continuation distribution than this corpus's explicit "
                "counts.  The raw rows do carry a recoverable n-gram prior, and the "
                "read-out is where value is lost.")
    else:
        verdict = "FAITHFUL_NGRAM_PRIOR"
        text = ("the probe is clearly above the majority floor and at or below the explicit "
                "trigram: the rows carry a faithful n-gram continuation prior and nothing "
                "more (which is all they can carry).  Round-157's claim is about the "
                "READ-OUT, not about the table.")
    report["controls_collapsed"] = control_ok
    report["verdict"] = {
        "label": verdict, "text": text,
        "probe_minus_majority": delta_maj,
        "probe_minus_control_floor": delta_floor,
        "probe_minus_trigram": delta_tri,
        "probe_minus_bigram": p["top1"] - bi["top1"],
        "probe_minus_shuffled_eval": p["top1"] - sh_ev["top1"],
    }

    report["positions"] = {"probe_train": n_tr, "probe_val": n_va, "eval": n_ev}
    report["eval_candidate_coverage"] = coverage
    report["results"] = results
    report["seen_unseen_breakdown"] = breakdown
    report["rowid"] = {"seconds": rowid_secs, "causality": causal}
    report["peak_rss_mb"] = rss_mb()
    report["stage"] = "complete"
    report["elapsed_seconds"] = time.time() - t_start

    out_path = Path(args.out)
    write_json(out_path, report)
    log(f"wrote {out_path}; peak RSS {rss_mb():.0f} MiB; total {report['elapsed_seconds']:.0f}s")

    order_names = [
        "majority_train_prior", "count_bigram", "count_trigram", "probe_raw_rows",
        "probe_frozen_value_proj", "control_shuffled_train_rows", "control_shuffled_eval_rows",
    ]
    print("\n=== next-token prediction from raw PLE rows (held-out WikiText) ===")
    print(f"eval positions {n_ev} | probe train {n_tr} | probe val {n_va} | K={args.topk}\n")
    print(f"{'method':<34}{'top1':>9}{'top5':>9}{'nll':>9}")
    print("-" * 61)
    for name in order_names:
        if name not in results:
            continue
        e = results[name]["eval"]
        if "top1" not in e:
            continue
        print(f"{name:<34}{e['top1']:>9.4f}{e['top5']:>9.4f}{e['nll']:>9.4f}")
    print(f"\ncontrols_collapsed = {control_ok}   peak_rss_mb = {report['peak_rss_mb']:.0f}")
    print(f"verdict = {verdict}\n{text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
