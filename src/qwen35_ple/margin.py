"""Stage 1.5c: the marginal-advantage (``margin``) distribution.

Why this module exists
----------------------
Round 167 established a *proven, decoder-independent* bound: every PLE row id is
a function of at most three tokens (``k * n = 12`` characters, ``r = 9``), so

    I(Y ; e_{t-r:t} | h_t)  <=  I(Y ; w_t | h_t) ,      |w_t| <= 3 tokens.

Both sides live in "how much can the memory add", but neither was ever measured
directly.  ``docs/round-167-design-corpus-selection-by-marginal-advantage.md``
proposed the cheapest instrument that speaks to it *without training a reader*:

    margin(t) = L_bb(t) - L_cnt(t)

the per-position NLL difference between the backbone and a count-based trigram
model trained on the corpus whose n-grams the table indexes.  Where ``margin``
is positive the trigram alone predicts better than the backbone alone, so the
trigram is *not redundant* with the backbone state at that position; where the
whole distribution is negative there is no position a "read the trigram answer
out of the table" memory could improve, and the line can be closed for free.

What this module is, and is not
-------------------------------
It is a **ranking / diagnostic** instrument:

* ``margin`` uses ``L_cnt`` from *a* count model, which is a proxy for the best
  achievable trigram predictor.  A weak proxy makes ``L_cnt`` too high and
  ``margin`` too low, so the instrument is **pessimistic about the memory**:
  negative results are strong, positive ones are weak.
* ``oracle_gain`` answers "what if at the selected positions the prediction error
  dropped from the backbone's to exactly the count model's?".  That is the upper
  bound for a *prediction-replacing* memory (retrieve the trigram's answer and
  use it).  It is **not** an upper bound on a *complementary* memory, because
  combining two predictors can beat both; the mutual-information ceiling for
  that case needs the probe in :mod:`qwen35_ple.cmi_probe`.
* Deterministic and torch-free, so the Stage 1.5c analysis is unit-testable
  without a GPU (see ``tests/test_margin.py``).

Pre-registered decision rule
----------------------------
Frozen in ``docs/round-168-stage1.5c-preregistration.md`` before any number was
read: on the *realizable* curve (trigram context attested at least
``MIN_CONTEXT_COUNT`` times in the count model's training stream) and for
``q <= 20%``,

    best corpus-level NLL reduction  <  0.05 nats   ->  MEMORY_CEILING_EMPTY
    best corpus-level NLL reduction  <  0.20 nats   ->  MEMORY_CEILING_THIN
    otherwise                                       ->  MEMORY_CEILING_ROOM

with ``NO_POSITIVE_POSITION`` taking precedence when no position at all has a
positive margin.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "COUNT_EDGES",
    "MARGIN_QUANTILES",
    "MIN_CONTEXT_COUNT",
    "VERDICT_FLOOR_NATS",
    "VERDICT_Q_CEILING",
    "VERDICT_ROOM_NATS",
    "GainEntry",
    "MarginVerdict",
    "bucket_labels",
    "bucket_summary",
    "cross_tab",
    "gain_curve",
    "margin_from_nll",
    "oracle_gain",
    "positive_share",
    "top_fraction_mask",
    "verdict_from_curve",
]

#: Fractions (percent) of positions selected by margin, from most to least
#: selective.  Fixed before the run so the verdict cannot be moved by adding a
#: friendlier ``q`` afterwards.
MARGIN_QUANTILES: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)

#: A trigram context must be attested at least this many times in the count
#: model's training stream for its row to count as "well estimated"
#: (``docs/round-167-design-corpus-selection-by-marginal-advantage.md``
#: condition 3, the Nishida et al. 2025 warning).
MIN_CONTEXT_COUNT: int = 5

#: Best realizable reduction below this is treated as "no room".
VERDICT_FLOOR_NATS: float = 0.05
#: Best realizable reduction below this is "thin"; at or above it, "room".
VERDICT_ROOM_NATS: float = 0.20
#: Only selections at or below this percentile are allowed to drive the verdict
#: (a 50%-of-the-corpus memorisation programme is not a corpus-level argument).
VERDICT_Q_CEILING: float = 20.0

#: Bucket edges (in training occurrences) for the frequency axes.  The lower
#: edge is 0 so that unseen targets / contexts land in the first bucket.
COUNT_EDGES: tuple[int, ...] = (0, 1, 2, 3, 5, 10, 50, 200, 1000)


# --------------------------------------------------------------------------
# per-position quantities
# --------------------------------------------------------------------------
def margin_from_nll(bb_nll: np.ndarray, cnt_nll: np.ndarray) -> np.ndarray:
    """``margin(t) = L_bb(t) - L_cnt(t)`` in nats.

    Both inputs are per-position negative log-likelihoods over the *same*
    position set; the caller is responsible for that alignment, which is the
    only way this instrument can be silently wrong.
    """
    bb = np.asarray(bb_nll, dtype=np.float64)
    cnt = np.asarray(cnt_nll, dtype=np.float64)
    if bb.shape != cnt.shape:
        raise ValueError(f"shape mismatch: bb {bb.shape} vs cnt {cnt.shape}")
    if bb.ndim != 1:
        raise ValueError(f"expected 1-D per-position NLL, got ndim={bb.ndim}")
    if not (np.isfinite(bb).all() and np.isfinite(cnt).all()):
        raise ValueError("non-finite NLL in margin_from_nll")
    return bb - cnt


def positive_share(margin: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Fraction of (selected) positions with a strictly positive margin."""
    m = np.asarray(margin, dtype=np.float64)
    if mask is not None:
        m = m[np.asarray(mask, dtype=bool)]
    if m.size == 0:
        return 0.0
    return float(np.count_nonzero(m > 0.0) / m.size)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def top_fraction_mask(
    margin: np.ndarray, q: float, *, context_counts: np.ndarray | None = None,
    min_context_count: int | None = None,
) -> np.ndarray:
    """Boolean mask of the top ``q`` percent of positions by margin.

    Ties are broken by position order (``np.argsort(kind="stable")``), so the
    selection is deterministic.  When ``context_counts`` and
    ``min_context_count`` are given, positions whose trigram context is attested
    fewer than ``min_context_count`` times are excluded *before* the quantile is
    taken, i.e. ``q`` is a fraction of the well-attested positions -- the
    "realizable" curve.
    """
    m = np.asarray(margin, dtype=np.float64)
    if m.ndim != 1:
        raise ValueError("margin must be 1-D")
    q = float(q)
    if not 0.0 < q <= 100.0:
        raise ValueError(f"q must be in (0, 100], got {q}")
    eligible = np.ones(m.shape[0], dtype=bool)
    if min_context_count is not None:
        if context_counts is None:
            raise ValueError("min_context_count requires context_counts")
        cc = np.asarray(context_counts, dtype=np.int64)
        if cc.shape != m.shape:
            raise ValueError(f"context_counts shape {cc.shape} != margin {m.shape}")
        eligible = cc >= int(min_context_count)
    n_eligible = int(np.count_nonzero(eligible))
    if n_eligible == 0:
        return np.zeros(m.shape[0], dtype=bool)
    n_take = max(1, math.ceil(n_eligible * q / 100.0))
    order = np.argsort(-m, kind="stable")
    keep = np.zeros(m.shape[0], dtype=bool)
    taken = 0
    for idx in order:
        if not eligible[idx]:
            continue
        keep[idx] = True
        taken += 1
        if taken >= n_take:
            break
    return keep


@dataclass(frozen=True)
class GainEntry:
    """One point of the oracle-gain curve."""

    q_percent: float
    n_selected: int
    n_eligible: int
    share_of_positions: float
    mean_margin_selected: float
    share_positive_selected: float
    corpus_nll_reduction: float
    realizable: bool
    label: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "q_percent": self.q_percent,
            "label": self.label,
            "realizable": self.realizable,
            "n_selected": self.n_selected,
            "n_eligible": self.n_eligible,
            "share_of_positions": self.share_of_positions,
            "mean_margin_selected": self.mean_margin_selected,
            "share_positive_selected": self.share_positive_selected,
            "corpus_nll_reduction": self.corpus_nll_reduction,
        }


def oracle_gain(
    margin: np.ndarray, mask: np.ndarray, *, realizable: bool = False,
    label: str = "", n_eligible: int | None = None, q_percent: float | None = None,
) -> GainEntry:
    """Corpus-level NLL reduction if the selected positions matched ``L_cnt``.

    ``corpus_nll_reduction = (1 / N_total) * sum_{t in selected} max(0, margin(t))``

    Normalising by the **whole** position set (not just the selection) is what
    makes the entries of a curve comparable across ``q``: it is the nats per
    position the whole corpus would save.
    """
    m = np.asarray(margin, dtype=np.float64)
    sel = np.asarray(mask, dtype=bool)
    if sel.shape != m.shape:
        raise ValueError(f"mask shape {sel.shape} != margin {m.shape}")
    n_total = int(m.shape[0])
    n_sel = int(np.count_nonzero(sel))
    n_elig = n_total if n_eligible is None else int(n_eligible)
    # ``q_percent`` is the *requested* selection fraction (what the pre-registered
    # rule is stated over); the realised fraction can differ by one position
    # because the take count is rounded up.
    realised = 100.0 * n_sel / max(n_elig, 1)
    q = realised if q_percent is None else float(q_percent)
    if n_total == 0 or n_sel == 0:
        return GainEntry(q, 0, n_elig, 0.0, 0.0, 0.0, 0.0, realizable, label)
    picked = np.maximum(m[sel], 0.0)
    return GainEntry(
        q_percent=q,
        n_selected=n_sel,
        n_eligible=n_elig,
        share_of_positions=n_sel / n_total,
        mean_margin_selected=float(m[sel].mean()),
        share_positive_selected=float(np.count_nonzero(m[sel] > 0.0) / n_sel),
        corpus_nll_reduction=float(picked.sum() / n_total),
        realizable=realizable,
        label=label,
    )


def gain_curve(
    margin: np.ndarray,
    *,
    quantiles: Sequence[float] = MARGIN_QUANTILES,
    context_counts: np.ndarray | None = None,
    min_context_count: int = MIN_CONTEXT_COUNT,
) -> list[GainEntry]:
    """Both curves: all positions, and the well-attested ("realizable") subset.

    ``q`` is *redefined* on the realizable subset (fractions of well-attested
    positions) so the two curves answer the same question on different pools.
    """
    out: list[GainEntry] = []
    for q in quantiles:
        out.append(oracle_gain(
            margin, top_fraction_mask(margin, q), label="all", realizable=False,
            q_percent=q,
        ))
    if context_counts is not None:
        cc = np.asarray(context_counts, dtype=np.int64)
        if cc.shape != np.asarray(margin).shape:
            raise ValueError("context_counts must match margin")
        n_eligible = int(np.count_nonzero(cc >= min_context_count))
        for q in quantiles:
            out.append(oracle_gain(
                margin,
                top_fraction_mask(
                    margin, q, context_counts=context_counts,
                    min_context_count=min_context_count,
                ),
                realizable=True,
                label=f"ctx_count>={min_context_count}",
                n_eligible=n_eligible,
                q_percent=q,
            ))
    return out


# --------------------------------------------------------------------------
# frequency axes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class _Bucket:
    lo: int
    hi: int | None

    @property
    def label(self) -> str:
        if self.hi is None:
            return f"{self.lo}"
        return f"{self.lo}" if self.hi - 1 == self.lo else f"{self.lo}-{self.hi - 1}"


def bucket_labels(
    counts: np.ndarray, *, edges: Sequence[int] = COUNT_EDGES,
) -> tuple[np.ndarray, list[str]]:
    """Bucket index per position, plus the labels in :func:`bucket_summary` order.

    Shared by the Stage 1.5c summaries and the Stage 1.5e grouped fusion probe so
    that "which frequency band" means the same thing in both.
    """
    c = np.asarray(counts, dtype=np.int64)
    e = sorted({int(x) for x in edges})
    if not e or e[0] != 0:
        raise ValueError("bucket edges must start at 0")
    # np.digitize(c, e) returns i with e[i-1] <= c < e[i]; shifting by one turns
    # that into the [e[i], e[i+1]) bucket index used everywhere else.
    index = np.clip(np.digitize(c, np.asarray(e, dtype=np.int64)) - 1, 0, len(e) - 1)
    return index.astype(np.int64), [b.label for b in _buckets(e)]


def _buckets(edges: Sequence[int]) -> list[_Bucket]:
    e = sorted({int(x) for x in edges})
    if not e or e[0] != 0:
        raise ValueError("bucket edges must start at 0")
    out: list[_Bucket] = []
    for i, lo in enumerate(e):
        hi = e[i + 1] if i + 1 < len(e) else None
        out.append(_Bucket(lo, hi))
    return out


def bucket_summary(
    margin: np.ndarray, counts: np.ndarray, *, edges: Sequence[int] = COUNT_EDGES,
    name: str = "count", bb_nll: np.ndarray | None = None,
    cnt_nll: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Margin statistics per bucket of ``counts`` (target or context frequency).

    ``mean_bb_nll`` / ``mean_cnt_nll`` are filled in when the two NLL arrays are
    supplied: they say whether a bucket's margin comes from the backbone doing
    badly or from the count model doing well, which the margin alone hides.
    """
    m = np.asarray(margin, dtype=np.float64)
    c = np.asarray(counts, dtype=np.int64)
    if c.shape != m.shape:
        raise ValueError(f"counts shape {c.shape} != margin {m.shape}")
    bb = None if bb_nll is None else np.asarray(bb_nll, dtype=np.float64)
    cn = None if cnt_nll is None else np.asarray(cnt_nll, dtype=np.float64)
    for arr, label in ((bb, "bb_nll"), (cn, "cnt_nll")):
        if arr is not None and arr.shape != m.shape:
            raise ValueError(f"{label} shape {arr.shape} != margin {m.shape}")
    rows: list[dict[str, object]] = []
    for b in _buckets(edges):
        sel = c >= b.lo if b.hi is None else (c >= b.lo) & (c < b.hi)
        n = int(np.count_nonzero(sel))
        if n == 0:
            rows.append({
                "axis": name, "bucket": b.label, "lo": b.lo, "hi": b.hi,
                "n": 0, "share": 0.0, "mean_margin": None,
                "mean_bb_nll": None, "mean_cnt_nll": None, "share_positive": None,
            })
            continue
        mv = m[sel]
        rows.append({
            "axis": name,
            "bucket": b.label,
            "lo": b.lo,
            "hi": b.hi,
            "n": n,
            "share": n / m.shape[0] if m.shape[0] else 0.0,
            "mean_margin": float(mv.mean()),
            "mean_bb_nll": None if bb is None else float(bb[sel].mean()),
            "mean_cnt_nll": None if cn is None else float(cn[sel].mean()),
            "share_positive": float(np.count_nonzero(mv > 0.0) / n),
        })
    return rows


def cross_tab(
    margin: np.ndarray,
    target_counts: np.ndarray,
    context_counts: np.ndarray,
    *,
    target_edges: Sequence[int] = COUNT_EDGES,
    context_edges: Sequence[int] = COUNT_EDGES,
) -> dict[str, object]:
    """Margin over the 2-D (target frequency) x (context frequency) grid.

    This separates the two failure modes the literature predicts:
    * rare **target** -> the value itself is not in the table / is noisy
      (Nishida et al. 2025, next-token prediction);
    * rare **context** -> the *row* is estimated from few occurrences
      (the PLE-specific version of the same warning).
    """
    m = np.asarray(margin, dtype=np.float64)
    tc = np.asarray(target_counts, dtype=np.int64)
    cc = np.asarray(context_counts, dtype=np.int64)
    if not (tc.shape == cc.shape == m.shape):
        raise ValueError("margin / target_counts / context_counts must share a shape")
    tb = _buckets(target_edges)
    cb = _buckets(context_edges)
    cells: list[dict[str, object]] = []
    for tbucket in tb:
        tsel = tc >= tbucket.lo if tbucket.hi is None else (tc >= tbucket.lo) & (tc < tbucket.hi)
        for cbucket in cb:
            csel = cc >= cbucket.lo if cbucket.hi is None else (cc >= cbucket.lo) & (cc < cbucket.hi)
            sel = tsel & csel
            n = int(np.count_nonzero(sel))
            if n == 0:
                cells.append({
                    "target_bucket": tbucket.label, "context_bucket": cbucket.label,
                    "n": 0, "share": 0.0, "mean_margin": None, "share_positive": None,
                })
                continue
            mv = m[sel]
            cells.append({
                "target_bucket": tbucket.label,
                "context_bucket": cbucket.label,
                "n": n,
                "share": n / m.shape[0] if m.shape[0] else 0.0,
                "mean_margin": float(mv.mean()),
                "share_positive": float(np.count_nonzero(mv > 0.0) / n),
            })
    return {
        "target_edges": list(target_edges),
        "context_edges": list(context_edges),
        "cells": cells,
    }


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MarginVerdict:
    label: str
    reason: str
    best_reduction: float
    best_q: float
    best_label: str
    positive_share: float
    n_positions: int
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "reason": self.reason,
            "best_reduction": self.best_reduction,
            "best_q": self.best_q,
            "best_label": self.best_label,
            "positive_share": self.positive_share,
            "n_positions": self.n_positions,
            "details": self.details,
        }


def verdict_from_curve(
    curve: Iterable[GainEntry],
    *,
    positive_share_all: float,
    n_positions: int,
    q_ceiling: float = VERDICT_Q_CEILING,
    floor_nats: float = VERDICT_FLOOR_NATS,
    room_nats: float = VERDICT_ROOM_NATS,
) -> MarginVerdict:
    """Apply the pre-registered rule to a :func:`gain_curve` output."""
    entries = list(curve)
    realizable = [e for e in entries if e.realizable and e.q_percent <= q_ceiling]
    pool = realizable or [e for e in entries if not e.realizable and e.q_percent <= q_ceiling]
    if not pool:
        raise ValueError("gain curve has no entry within the q ceiling")
    best = max(pool, key=lambda e: e.corpus_nll_reduction)
    common = {
        "best_reduction": best.corpus_nll_reduction,
        "best_q": best.q_percent,
        "best_label": best.label,
        "positive_share": positive_share_all,
        "n_positions": n_positions,
        "details": {
            "q_ceiling": q_ceiling,
            "floor_nats": floor_nats,
            "room_nats": room_nats,
            "used_realizable_pool": bool(realizable),
            "best_share_positive_selected": best.share_positive_selected,
            "best_mean_margin_selected": best.mean_margin_selected,
        },
    }
    if positive_share_all <= 0.0:
        return MarginVerdict(
            "NO_POSITIVE_POSITION",
            "no position at all has L_bb > L_cnt: the count model never beats the "
            "backbone, so a prediction-replacing memory has nothing to add anywhere",
            **common,
        )
    if best.corpus_nll_reduction < floor_nats:
        return MarginVerdict(
            "MEMORY_CEILING_EMPTY",
            f"best realizable corpus-level reduction {best.corpus_nll_reduction:.4f} "
            f"nats < {floor_nats} nats at q<={q_ceiling}%",
            **common,
        )
    if best.corpus_nll_reduction < room_nats:
        return MarginVerdict(
            "MEMORY_CEILING_THIN",
            f"best realizable corpus-level reduction {best.corpus_nll_reduction:.4f} "
            f"nats is in [{floor_nats}, {room_nats}) at q<={q_ceiling}%",
            **common,
        )
    return MarginVerdict(
        "MEMORY_CEILING_ROOM",
        f"best realizable corpus-level reduction {best.corpus_nll_reduction:.4f} "
        f"nats >= {room_nats} nats at q={best.q_percent}%",
        **common,
    )
