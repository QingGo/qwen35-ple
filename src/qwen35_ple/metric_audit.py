"""Nuisance-variable audit for evaluation metrics (Round 167, TD-1).

Motivation
----------
Round 166 found that ``classify_format.domain_fingerprint`` counts how many
*distinct* markers occur in a generation::

    prose = sum(1 for m in _PROSE_MARKERS if m in low)      # 0..6

A longer generation is more likely to contain each marker, so an arm that
generates more tokens accumulates higher marker counts *regardless of the corpus
behind its table*.  Round 162's directional predictions compare marker counts
across arms whose generated lengths differ, so part of the reported "effect" was
generated length: the 0.8B prose contrast was inflated ~40%
(raw +0.6933 -> +0.4159 at equal generated length).

The general lesson, which round-153's debt list did not contain: round-153's debt
was *experiments that silently did not run*; this is **measurements that silently
measured the wrong thing**.  A metric that moves when only a nuisance variable
moves is not evidence about the variable of interest.

What this module does
---------------------
Given a paired contrast between two arms, it separates the effect of interest
from the effect of a nuisance covariate by regressing the *difference* of the
metric on the *difference* of the nuisance::

    d_i = c + b * l_i + e_i,   d_i = m_a(i) - m_b(i),   l_i = n_a(i) - n_b(i)

``c`` is the arm difference at equal nuisance, ``b`` is the metric-per-nuisance
slope.  ``classify_contrast`` then says whether the raw difference survives the
adjustment.

The pairing is what makes this work: both arms answer the same items in the same
order, so index ``i`` is the same item in both and a per-item nuisance
covariate (generated length) can be differenced away without item fixed effects.

The set of verdicts
-------------------
``SURVIVES_ADJUSTMENT``
    There was a raw effect and it is still there at equal nuisance.
``NUISANCE_SENSITIVE``
    There was a raw effect and it is gone at equal nuisance: the contrast was
    measuring the nuisance, not the arms.
``NO_EFFECT``
    No raw effect to explain.
``UNDETERMINED``
    Degenerate input (a single pair, or a zero-variance nuisance with no raw
    effect); refuse to classify rather than guess.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import erfc, sqrt
from typing import Any

import numpy as np

__all__ = [
    "INVARIANT",
    "METRIC_DECLARATIONS",
    "NO_EFFECT",
    "NUISANCE_SENSITIVE",
    "SURVIVES_ADJUSTMENT",
    "UNDETERMINED",
    "MetricDeclaration",
    "NuisanceContrast",
    "classify_contrast",
    "declared_nuisance_sensitive",
    "paired_nuisance_contrast",
    "recompute_verdict",
    "synth_content_only",
    "synth_length_only",
    "synth_mixed",
]

SURVIVES_ADJUSTMENT = "SURVIVES_ADJUSTMENT"
NUISANCE_SENSITIVE = "NUISANCE_SENSITIVE"
NO_EFFECT = "NO_EFFECT"
UNDETERMINED = "UNDETERMINED"

#: Alias kept because the plan document (round-167 Stage 0.1) talks about a
#: metric being "invariant" to the nuisance.  Semantically the same verdict as
#: ``SURVIVES_ADJUSTMENT``: the contrast is still there once the nuisance is held
#: fixed, so the metric is not merely reporting the nuisance.
INVARIANT = SURVIVES_ADJUSTMENT


@dataclass(frozen=True)
class MetricDeclaration:
    """What a metric claims about a nuisance variable, *before* it is audited.

    The point of declaring this up front is that the audit can then disagree:
    ``test_metric_audit.py`` asserts that the declaration and the measured
    verdict agree, so a newly added metric that quietly tracks generated length
    fails CI instead of entering the lockedsuite.
    """

    nuisance: str
    requires_adjustment: bool
    note: str = ""


#: Registry of the metrics that feed the round-162/166 directional verdicts.
#: ``requires_adjustment`` is a *claim*; the audit is what decides.
METRIC_DECLARATIONS: dict[str, MetricDeclaration] = {
    "prose_markers": MetricDeclaration(
        nuisance="generated_length",
        requires_adjustment=True,
        note="distinct-marker count (0..6) rises with length; this is the round-166 confound",
    ),
    "code_markers": MetricDeclaration(
        nuisance="generated_length",
        requires_adjustment=True,
        note="same distinct-marker construction as prose_markers",
    ),
    "math_markers": MetricDeclaration(
        nuisance="generated_length",
        requires_adjustment=True,
        note="same distinct-marker construction as prose_markers",
    ),
    "empty_rate": MetricDeclaration(
        nuisance="generated_length",
        requires_adjustment=True,
        note="a generation is empty iff it produced no tokens, so this is exactly a length readout",
    ),
    "chat_scaffold_rate": MetricDeclaration(
        nuisance="generated_length",
        requires_adjustment=True,
        note=(
            "REVISED after the first audit: a generation must run long enough to "
            "emit the marker, so the rate is length-exposed like any other "
            "marker-presence count.  The audit found it confounded on "
            "0.8B-sft/ple-off|wiki (share +1.16: the 19.5pp format gain is "
            "entirely generated length) while the 4B-sft contrast survives "
            "untouched (share -0.02)."
        ),
    ),
    "scaffold_in_first_32_chars": MetricDeclaration(
        nuisance="generated_length",
        requires_adjustment=True,
        note=(
            "Meant as a position-0 control but REVISED after the audit: it is "
            "exposed too, because in an autoregressive model even the opening "
            "token correlates with the eventual length (a model that opens with "
            "<think> keeps going).  It nonetheless reproduces the same per-regime "
            "conclusion as the full-text metric (0.8B share +1.16, 4B share "
            "-0.02), which is what makes that conclusion robust.  There is no "
            "structurally immune metric here; the audit quantifies each contrast "
            "instead of certifying any metric."
        ),
    ),
}


def declared_nuisance_sensitive(name: str) -> bool:
    """Whether ``name`` was declared to need equal-nuisance adjustment."""
    try:
        return METRIC_DECLARATIONS[name].requires_adjustment
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"metric {name!r} is not declared in METRIC_DECLARATIONS; "
            "add a declaration (and a nuisance test) before using it"
        ) from exc


@dataclass(frozen=True)
class NuisanceContrast:
    """A paired arm contrast decomposed into nuisance-free and nuisance parts."""

    n_pairs: int
    raw_diff: float
    raw_se: float
    adjusted_diff: float
    adjusted_se: float
    nuisance_slope: float
    nuisance_slope_se: float
    nuisance_share: float | None
    corr_metric_nuisance_a: float | None
    corr_metric_nuisance_b: float | None
    nuisance_has_variance: bool
    verdict: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nan_safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _t_one_sided(est: float, se: float) -> float:
    """One-sided p for ``est > 0`` under a normal approximation."""
    if not np.isfinite(est) or not np.isfinite(se) or se <= 0:
        return 1.0
    return 0.5 * erfc(est / (se * sqrt(2.0)))


def paired_nuisance_contrast(
    metric_a: Sequence[float],
    metric_b: Sequence[float],
    nuisance_a: Sequence[float],
    nuisance_b: Sequence[float],
    *,
    tol_sigma: float = 2.0,
    share_threshold: float = 0.5,
) -> NuisanceContrast:
    """Decompose a paired arm difference into effect and nuisance.

    ``metric_*`` and ``nuisance_*`` are per-item values for two arms over the
    *same items in the same order* (that is what makes the differencing valid).
    ``tol_sigma`` is how many standard errors an estimate must clear to count as
    distinguishable from zero; ``share_threshold`` is how much of the raw effect
    the nuisance may absorb before the contrast is called confounded.
    """
    m_a = np.asarray(metric_a, dtype=float)
    m_b = np.asarray(metric_b, dtype=float)
    n_a = np.asarray(nuisance_a, dtype=float)
    n_b = np.asarray(nuisance_b, dtype=float)

    if not (m_a.shape == m_b.shape == n_a.shape == n_b.shape):
        raise ValueError(
            "metric/nuisance arrays must share a shape, got "
            f"{m_a.shape}, {m_b.shape}, {n_a.shape}, {n_b.shape}"
        )
    n_pairs = int(m_a.size)
    if n_pairs < 2:
        return NuisanceContrast(
            n_pairs=n_pairs,
            raw_diff=float("nan"),
            raw_se=float("nan"),
            adjusted_diff=float("nan"),
            adjusted_se=float("nan"),
            nuisance_slope=float("nan"),
            nuisance_slope_se=float("nan"),
            nuisance_share=None,
            corr_metric_nuisance_a=None,
            corr_metric_nuisance_b=None,
            nuisance_has_variance=False,
            verdict=UNDETERMINED,
        )

    d = m_a - m_b
    l = n_a - n_b
    raw_diff = float(d.mean())
    raw_se = float(d.std(ddof=1) / sqrt(n_pairs))
    nuisance_has_variance = bool(l.std() > 0)

    if nuisance_has_variance:
        X = np.column_stack([np.ones_like(l), l])
        coef, *_ = np.linalg.lstsq(X, d, rcond=None)
        c_hat, b_hat = float(coef[0]), float(coef[1])
        resid = d - X @ coef
        dof = max(1, n_pairs - 2)
        s2 = float(resid @ resid) / dof
        xtx_inv = np.linalg.pinv(X.T @ X)
        se_c = float(np.sqrt(max(s2 * xtx_inv[0, 0], 0.0)))
        se_b = float(np.sqrt(max(s2 * xtx_inv[1, 1], 0.0)))
    else:
        # The nuisance is identical in both arms, so it cannot explain any part
        # of the difference: the adjusted estimate is the raw one.
        c_hat, b_hat = raw_diff, 0.0
        se_c, se_b = raw_se, 0.0

    nuisance_share: float | None
    if raw_diff != 0.0:
        nuisance_share = float((raw_diff - c_hat) / raw_diff)
    else:
        nuisance_share = None

    contrast = NuisanceContrast(
        n_pairs=n_pairs,
        raw_diff=raw_diff,
        raw_se=raw_se,
        adjusted_diff=c_hat,
        adjusted_se=se_c,
        nuisance_slope=b_hat,
        nuisance_slope_se=se_b,
        nuisance_share=nuisance_share,
        corr_metric_nuisance_a=_nan_safe_corr(n_a, m_a),
        corr_metric_nuisance_b=_nan_safe_corr(n_b, m_b),
        nuisance_has_variance=nuisance_has_variance,
        verdict=UNDETERMINED,
    )
    return _with_verdict(contrast, tol_sigma=tol_sigma, share_threshold=share_threshold)


def recompute_verdict(
    ct: NuisanceContrast,
    *,
    tol_sigma: float = 2.0,
    share_threshold: float = 0.5,
) -> str:
    """The verdict implied by a contrast's numbers (no side effects).

    Two things count as "the headline number is mostly the nuisance":

    1. the adjusted estimate is indistinguishable from zero at ``tol_sigma``
       standard errors, or
    2. the nuisance accounts for at least ``share_threshold`` of the raw effect.

    Criterion 2 exists because criterion 1 alone has the ~5% miss rate of any
    2-sigma screen.  For a *discipline gate* that is too leaky: a confounded
    contrast should be caught reliably, not 19 times out of 20.  It is also the
    more interpretable statement -- "more than half of what you reported is the
    nuisance" -- and it is deterministic given the data.
    """
    raw_sig = np.isfinite(ct.raw_se) and ct.raw_se > 0 and abs(ct.raw_diff) > tol_sigma * ct.raw_se
    if not raw_sig:
        return NO_EFFECT

    adj_sig = (
        np.isfinite(ct.adjusted_se)
        and ct.adjusted_se > 0
        and abs(ct.adjusted_diff) > tol_sigma * ct.adjusted_se
    )
    share = ct.nuisance_share
    share_dominated = share is not None and share >= share_threshold

    if not adj_sig or share_dominated:
        return NUISANCE_SENSITIVE
    return SURVIVES_ADJUSTMENT


def classify_contrast(
    ct: NuisanceContrast,
    *,
    tol_sigma: float = 2.0,
    share_threshold: float = 0.5,
) -> str:
    """Public entry point: classify an already-computed contrast."""
    return recompute_verdict(ct, tol_sigma=tol_sigma, share_threshold=share_threshold)


def _with_verdict(
    ct: NuisanceContrast,
    *,
    tol_sigma: float,
    share_threshold: float = 0.5,
) -> NuisanceContrast:
    return NuisanceContrast(
        **{
            **ct.as_dict(),
            "verdict": recompute_verdict(
                ct, tol_sigma=tol_sigma, share_threshold=share_threshold
            ),
        }
    )


# --------------------------------------------------------------------------
# Synthetic controls.
#
# These construct paired arm data with a *known* answer, so the auditor itself
# can be falsified.  They are the reason Stage 0.1 is verifiable rather than
# aspirational: `synth_length_only` reproduces the round-166 shape (a real raw
# effect that is entirely generated length) and the audit must call it out.
# --------------------------------------------------------------------------


def _synth_lengths(
    n: int, *, rng: np.random.Generator, mean_a: float, mean_b: float, sd: float
) -> tuple[np.ndarray, np.ndarray]:
    n_a = np.clip(rng.normal(mean_a, sd, n), 1, None)
    n_b = np.clip(rng.normal(mean_b, sd, n), 1, None)
    return n_a, n_b


def synth_length_only(
    n: int = 600,
    *,
    slope: float = 0.08,
    noise: float = 0.35,
    mean_a: float = 31.0,
    mean_b: float = 23.0,
    sd: float = 4.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two arms with the *same* content but different generated lengths.

    Truth: the whole difference is nuisance.  A metric built as a distinct-marker
    count will show a raw effect here, and the audit must report
    ``NUISANCE_SENSITIVE``.
    """
    rng = np.random.default_rng(seed)
    n_a, n_b = _synth_lengths(n, rng=rng, mean_a=mean_a, mean_b=mean_b, sd=sd)
    m_a = slope * n_a + rng.normal(0.0, noise, n)
    m_b = slope * n_b + rng.normal(0.0, noise, n)
    return m_a, m_b, n_a, n_b


def synth_content_only(
    n: int = 600,
    *,
    effect: float = 0.6,
    slope: float = 0.08,
    noise: float = 0.35,
    mean_len: float = 27.0,
    sd: float = 4.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two arms of *equal* length whose content genuinely differs by ``effect``.

    Truth: the whole difference is real.  The audit must not cry wolf, i.e. it
    must report ``SURVIVES_ADJUSTMENT``.
    """
    rng = np.random.default_rng(seed)
    n_shared = np.clip(rng.normal(mean_len, sd, n), 1, None)
    m_a = effect + slope * n_shared + rng.normal(0.0, noise, n)
    m_b = slope * n_shared + rng.normal(0.0, noise, n)
    return m_a, m_b, n_shared.copy(), n_shared.copy()


def synth_mixed(
    n: int = 600,
    *,
    effect: float = 0.4,
    slope: float = 0.08,
    noise: float = 0.35,
    mean_a: float = 31.0,
    mean_b: float = 23.0,
    sd: float = 4.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Both a real content effect and a length difference.

    Truth: the adjusted estimate should recover ``effect`` and the slope should
    recover ``slope``.  This is the round-166 case in full.
    """
    rng = np.random.default_rng(seed)
    n_a, n_b = _synth_lengths(n, rng=rng, mean_a=mean_a, mean_b=mean_b, sd=sd)
    m_a = effect + slope * n_a + rng.normal(0.0, noise, n)
    m_b = slope * n_b + rng.normal(0.0, noise, n)
    return m_a, m_b, n_a, n_b
