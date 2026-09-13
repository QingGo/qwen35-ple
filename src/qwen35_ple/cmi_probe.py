"""Conditional-mutual-information probe (Round 167).

The question this answers
-------------------------
Every read-out design we have tried -- ours, the official one, and any future
one -- is bounded by the same quantity::

    CMI(t) = I( x_{t+1} ; e_t | h_t )

Round-157 proved an *upper bound* on it (a trigram addressing can carry at most
the information of the trigram).  What we have never done is *measure* it.  Every
number we own is a proxy: probe accuracy, participation ratio, a necessary-condition
term, a cosine.  This module measures the quantity itself, and the measurement is
architecture-agnostic -- it says whether content *can* reach the output through
any read-out, not whether ours did.

How it is estimated
-------------------
A probe predicts the next token from ``h_t`` alone, and a second probe of the
same capacity predicts it from ``(h_t, e_t)``.  The held-out cross-entropy gap

    delta = NLL(h) - NLL(h, e)

is an **information gain attributable to e_t given h**: it is non-negative when
e_t helps and ~0 when e_t is redundant with h_t.  With finite probe capacity both
losses are too high, so ``delta`` **understates** the true CMI -- it is a lower
bound on what any read-out could extract.

That direction matters: a *small* delta is only evidence that the channel is empty
if the probe demonstrably had the power to detect a large one.  Hence the
synthetic controls below, and hence they must be run before any real-data number
is believed.

Controls
--------
``synth_benchmark(..., dose)`` plants a known amount of information in ``e``.
Sweeping ``dose`` traces the probe's power curve.  ``dose=0`` is the null: ``e``
is independent of the label and the measured gap must collapse to zero.  If the
curve does not rise with the dose, the instrument is broken and no negative
real-data result from it counts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "CmiResult",
    "CmiVerdict",
    "cmi_gap",
    "cmi_with_null",
    "fit_linear_probe",
    "fit_mlp_probe",
    "held_out_nll",
    "mlp_scores",
    "synth_benchmark",
    "synth_features_for_labels",
]


@dataclass(frozen=True)
class CmiResult:
    n: int
    n_classes: int
    nll_h: float
    nll_he: float
    delta: float
    delta_ci95_lo: float
    delta_ci95_hi: float
    acc_h: float
    acc_he: float
    delta_per_position: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("delta_per_position")
        return d

    @property
    def positive(self) -> bool:
        """Whether the gain is distinguishable from zero at 95%."""
        return self.delta_ci95_lo > 0.0


def fit_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    ridge: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ridge least-squares classifier, closed form.

    Returns ``(W, mu, sigma)`` where the caller must apply ``(X - mu) / sigma``
    before ``X @ W``.  Closed form keeps the probe deterministic and torch-free,
    which matters because this instrument has to be trusted before it is used.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma

    # One-hot targets for a squared-error fit; the softmax below reads it as a
    # classifier, which is the usual linear-probe convention.
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    A = Xs.T @ Xs + ridge * np.eye(d)
    B = Xs.T @ Y
    W = np.linalg.solve(A, B)
    return W, mu, sigma


def fit_mlp_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    hidden: int = 128,
    epochs: int = 150,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """A small 2-layer MLP, trained with Adam, in numpy.

    Why not the linear probe: a single linear map cannot express *complementary*
    information.  If ``h`` reveals one factor and ``e`` another, and the label
    depends on their combination, then no linear function of the concatenation
    can compute it -- only an interaction term can.  A linear probe would
    therefore report a near-zero CMI precisely in the case this instrument exists
    to detect, which would turn a capacity artefact into a false scientific
    conclusion.  Kept in numpy so the instrument stays torch-free and testable
    without a GPU.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n, d = X.shape
    rng = np.random.default_rng(seed)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma

    W1 = rng.normal(0.0, np.sqrt(2.0 / d), size=(d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden, n_classes))
    b2 = np.zeros(n_classes)
    params = [W1, b1, W2, b2]

    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1t, b2t, eps = 0.9, 0.999, 1e-8

    for step in range(1, epochs + 1):
        H = np.maximum(Xs @ W1 + b1, 0.0)
        S = H @ W2 + b2
        P = np.exp(_log_softmax(S))
        dS = P.copy()
        dS[np.arange(n), y] -= 1.0
        dS /= n
        gW2 = H.T @ dS + weight_decay * W2
        gb2 = dS.sum(axis=0)
        dH = dS @ W2.T
        dH[H <= 0] = 0.0
        gW1 = Xs.T @ dH + weight_decay * W1
        gb1 = dH.sum(axis=0)
        grads = [gW1, gb1, gW2, gb2]

        for i, (prm, g) in enumerate(zip(params, grads)):
            m[i] = b1t * m[i] + (1 - b1t) * g
            v[i] = b2t * v[i] + (1 - b2t) * g * g
            mh = m[i] / (1 - b1t**step)
            vh = v[i] / (1 - b2t**step)
            prm -= lr * mh / (np.sqrt(vh) + eps)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "mu": mu, "sigma": sigma}


def mlp_scores(params: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    Xs = (np.asarray(X, dtype=np.float64) - params["mu"]) / params["sigma"]
    H = np.maximum(Xs @ params["W1"] + params["b1"], 0.0)
    return H @ params["W2"] + params["b2"]


def _log_softmax(scores: np.ndarray) -> np.ndarray:
    m = scores.max(axis=1, keepdims=True)
    z = scores - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def _fit_temperature(scores: np.ndarray, y: np.ndarray) -> float:
    """Pick the scalar that makes a least-squares score vector a proper logit.

    A squared-error fit to one-hot targets produces scores that rank correctly but
    are not scaled like logits, so their raw softmax is badly calibrated and the
    absolute NLL is not interpretable -- which matters, because the whole reading
    of a measured gap depends on comparing it against the remaining headroom
    ``nll_h``.  One scalar fitted on the training fold fixes the scale without
    adding capacity, so the comparison between the two probes stays fair.
    """
    best_tau, best_nll = 1.0, float("inf")
    for tau in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0):
        logp = _log_softmax(scores / tau)
        nll = float(-logp[np.arange(len(y)), y].mean())
        if nll < best_nll:
            best_tau, best_nll = tau, nll
    return best_tau


def held_out_nll(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    ridge: float = 1.0,
    folds: int = 5,
    seed: int = 0,
    probe: str = "mlp",
    mlp_hidden: int = 128,
    mlp_epochs: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold per-position NLL and correctness.

    Standardisation statistics and the score temperature come from the training
    folds only; using the whole matrix would leak and would flatter whichever
    probe has more inputs.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n = X.shape[0]
    if n < folds * 2:
        raise ValueError(f"need at least {folds * 2} positions, got {n}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold_of = np.empty(n, dtype=np.int64)
    for k, idx in enumerate(np.array_split(order, folds)):
        fold_of[idx] = k

    nll = np.full(n, np.nan)
    correct = np.zeros(n, dtype=bool)
    for k in range(folds):
        te = fold_of == k
        tr = ~te
        if probe == "linear":
            W, mu, sigma = fit_linear_probe(X[tr], y[tr], n_classes, ridge=ridge)
            train_scores = ((X[tr] - mu) / sigma) @ W
            tau = _fit_temperature(train_scores, y[tr])
            scores = ((X[te] - mu) / sigma) @ W
        elif probe == "mlp":
            params = fit_mlp_probe(
                X[tr],
                y[tr],
                n_classes,
                hidden=mlp_hidden,
                epochs=mlp_epochs,
                seed=seed,
            )
            # An MLP trained to convergence on cross-entropy is already
            # calibrated, so no temperature is fitted.
            tau = 1.0
            scores = mlp_scores(params, X[te])
        else:
            raise ValueError(f"probe must be 'linear' or 'mlp', got {probe!r}")
        logp = _log_softmax(scores / tau)
        nll[te] = -logp[np.arange(te.sum()), y[te]]
        correct[te] = scores.argmax(axis=1) == y[te]
    return nll, correct


def cmi_gap(
    X_h: np.ndarray,
    X_e: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int | None = None,
    ridge: float = 1.0,
    folds: int = 5,
    seed: int = 0,
    n_bootstrap: int = 2000,
    probe: str = "mlp",
    mlp_hidden: int = 128,
    mlp_epochs: int = 150,
) -> CmiResult:
    """Estimate ``I(x ; e | h)`` as the held-out NLL gap between two probes."""
    X_h = np.asarray(X_h, dtype=np.float64)
    X_e = np.asarray(X_e, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if X_h.shape[0] != X_e.shape[0] or X_h.shape[0] != y.shape[0]:
        raise ValueError("X_h, X_e and y must share their first dimension")
    n_classes = int(n_classes if n_classes is not None else y.max() + 1)

    # Same folds and same seed for both designs, so the comparison is paired.
    kw = {
        "ridge": ridge,
        "folds": folds,
        "seed": seed,
        "probe": probe,
        "mlp_hidden": mlp_hidden,
        "mlp_epochs": mlp_epochs,
    }
    nll_h, ok_h = held_out_nll(X_h, y, n_classes, **kw)
    nll_he, ok_he = held_out_nll(np.hstack([X_h, X_e]), y, n_classes, **kw)

    d = nll_h - nll_he  # positive => e_t helped
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_bootstrap, d.size))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return CmiResult(
        n=int(d.size),
        n_classes=n_classes,
        nll_h=float(nll_h.mean()),
        nll_he=float(nll_he.mean()),
        delta=float(d.mean()),
        delta_ci95_lo=float(lo),
        delta_ci95_hi=float(hi),
        acc_h=float(ok_h.mean()),
        acc_he=float(ok_he.mean()),
        delta_per_position=d,
    )


@dataclass(frozen=True)
class CmiVerdict:
    """The measured gain, and the same measurement against a shuffled-row null.

    The width bias, and why the raw gap cannot be read on its own
    -----------------------------------------------------------
    The ``(h, e)`` probe has more inputs than the ``h`` probe, so when ``e``
    carries nothing it overfits and its held-out loss is *worse*.  Measured on
    the synthetic null that bias is about ``-1.44`` nats, i.e. a zero-information
    row produces a clearly **negative** gap.  Reporting the raw gap would then
    make an empty channel look like a negative one and a weak channel look like
    an empty one -- a capacity artefact dressed as a scientific finding.

    So the reported quantity is the **excess** over the null measured on the very
    same features, labels, folds and probe, with only the pairing between ``e``
    and its label destroyed.  That cancels the width bias and leaves the part
    attributable to information in the row.
    """

    observed: CmiResult
    null: CmiResult
    delta_excess: float
    delta_excess_ci95_lo: float
    delta_excess_ci95_hi: float
    width_bias: float

    @property
    def positive(self) -> bool:
        """Excess gain distinguishable from zero at 95%."""
        return self.delta_excess_ci95_lo > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed": self.observed.as_dict(),
            "null": self.null.as_dict(),
            "delta_excess": self.delta_excess,
            "delta_excess_ci95": [self.delta_excess_ci95_lo, self.delta_excess_ci95_hi],
            "width_bias": self.width_bias,
            "positive": self.positive,
        }


def cmi_with_null(
    X_h: np.ndarray,
    X_e: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int | None = None,
    seed: int = 0,
    n_bootstrap: int = 2000,
    **kwargs: Any,
) -> CmiVerdict:
    """Measure ``I(x ; e | h)`` as the excess over a shuffled-row null."""
    X_h = np.asarray(X_h, dtype=np.float64)
    X_e = np.asarray(X_e, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    observed = cmi_gap(
        X_h, X_e, y, n_classes=n_classes, seed=seed, n_bootstrap=n_bootstrap, **kwargs
    )
    perm = np.random.default_rng(seed + 9973).permutation(len(y))
    null = cmi_gap(
        X_h,
        X_e[perm],
        y,
        n_classes=n_classes,
        seed=seed,
        n_bootstrap=n_bootstrap,
        **kwargs,
    )

    # The two designs share folds and labels, so their per-position differences
    # are paired and the excess can be bootstrapped directly.
    per_pos = observed.delta_per_position - null.delta_per_position
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, per_pos.size, size=(n_bootstrap, per_pos.size))
    boot = per_pos[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return CmiVerdict(
        observed=observed,
        null=null,
        delta_excess=float(per_pos.mean()),
        delta_excess_ci95_lo=float(lo),
        delta_excess_ci95_hi=float(hi),
        width_bias=float(null.delta),
    )


# --------------------------------------------------------------------------
# Synthetic controls.
#
# These are the instrument's own falsification tests.  `dose` sets how much of
# the label is planted in `e`; dose=0 is the null, and the measured gap must
# track the dose or the probe cannot be trusted on real data.
# --------------------------------------------------------------------------


def synth_features_for_labels(
    y: np.ndarray,
    dim: int,
    *,
    signal: float,
    seed: int = 0,
) -> np.ndarray:
    """A random projection of the label plus noise.

    ``signal=0`` gives features that are independent of ``y`` (the null);
    ``signal`` large gives features that determine it.  The projection is fixed
    across the dataset, exactly as a real ``e_t`` is a fixed function of the
    addressed n-gram.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int64)
    n_classes = int(y.max()) + 1
    P = rng.normal(0.0, 1.0, size=(n_classes, dim))
    noise = rng.normal(0.0, 1.0, size=(len(y), dim))
    return signal * P[y] + noise


def synth_benchmark(
    n: int = 4000,
    *,
    n_classes: int = 16,
    dim_h: int = 32,
    dim_e: int = 32,
    dose: float = 1.0,
    h_signal: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labelled features with a *known* complementary-information structure.

    The first draft of this benchmark had ``e`` carry the *same* latent factor as
    ``h``, which validates the wrong thing: it tests whether a probe notices a
    redundant input, whereas the question on real data is whether the row carries
    information the hidden state does **not** already have.  With a redundant
    design the gap vanishes as soon as ``h`` alone solves the task, which is an
    artefact of the construction rather than a property of the estimator.

    So the label is instead a composition of two latents::

        y = (a + b) mod K

    where ``a`` is readable only from ``h`` and ``b`` only from ``e``.  Neither
    view can predict ``y`` alone, so the pair has genuinely complementary
    information and the true CMI is large -- of order ``log K``.  ``dose`` scales
    how recoverable ``b`` is: at ``dose=0`` the ``e`` features are independent
    noise, ``b`` is unrecoverable, and the measured gap must collapse to zero even
    though the absolute loss stays high.  That is the null this instrument
    actually needs to pass.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n_classes, size=n)
    b = rng.integers(0, n_classes, size=n)
    y = (a + b) % n_classes
    X_h = synth_features_for_labels(a, dim_h, signal=h_signal, seed=seed + 1)
    X_e = synth_features_for_labels(b, dim_e, signal=dose, seed=seed + 2)
    return X_h, X_e, y
