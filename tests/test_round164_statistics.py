"""Unit tests for the round-164 Part B statistics.

Why this file exists
--------------------
Part B needs a GPU and the 47 GiB row store, so it runs on the compute box.
The decision rule it implements is pre-registered
(``docs/round-164-window-composition-preregistration.md`` §5) and must not be
adjusted once numbers exist.  That makes it exactly the wrong place to discover
a sign error or a threshold typo *after* the box has been booted -- so the
statistics are pure numpy and are exercised here, on synthetic data where the
answer is known by construction.

The four cases that matter:

* boolq genuinely more similar, hidden states not  -> WINDOW_DOMINATED
* boolq not more similar                           -> ITEM_DOMINATED
* both more similar                                -> CONFOUNDED
* labels carry no information                      -> not significant

Note the last one: a permutation test that never fails to reject would make
every verdict above meaningless, so it is asserted explicitly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "round164_window_dominance.py"


def _load():
    spec = importlib.util.spec_from_file_location("_r164", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = _load()


def _diverse(rng, n, d):
    return rng.normal(size=(n, d))


def _tight(rng, n, d, jitter=0.01):
    base = rng.normal(size=(1, d))
    return base + jitter * rng.normal(size=(n, d))


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def test_mean_pairwise_cosine_identical_is_one():
    v = np.tile(np.array([[1.0, 2.0, 3.0]]), (5, 1))
    assert m._mean_pairwise_cosine(v, np.ones(5, dtype=bool)) == pytest.approx(1.0)


def test_mean_pairwise_cosine_orthogonal_is_zero():
    v = np.eye(2)
    assert m._mean_pairwise_cosine(v, np.ones(2, dtype=bool)) == pytest.approx(0.0)


def test_mean_pairwise_cosine_raises_on_zero_norm():
    """A zero contribution means the hook is not recording; refuse to score it."""
    v = np.array([[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError):
        m._mean_pairwise_cosine(v, np.ones(2, dtype=bool))


def test_verdict_branches():
    assert m._verdict(0.10, 0.001, -0.05) == "WINDOW_DOMINATED"
    assert m._verdict(-0.10, 0.001, -0.05) == "ITEM_DOMINATED"
    assert m._verdict(0.10, 0.001, 0.05) == "CONFOUNDED"
    # significant direction but the hidden-state control is unfavourable
    assert m._verdict(0.10, 0.50, -0.05) == "INCONCLUSIVE"
    # dS_c <= 0 short-circuits regardless of anything else
    assert m._verdict(0.0, 0.001, -0.05) == "ITEM_DOMINATED"


# --------------------------------------------------------------------------
# End to end on synthetic vectors
# --------------------------------------------------------------------------


def test_window_dominated_case_is_detected():
    """boolq vectors collapse onto one direction; short ones do not."""
    rng = np.random.default_rng(0)
    is_boolq = np.array([True] * 30 + [False] * 30)
    C = np.vstack([_tight(rng, 30, 8), _diverse(rng, 30, 8)])
    H = _diverse(rng, 60, 8)

    ds_c = m._delta(C, is_boolq)
    ds_h = m._delta(H, is_boolq)
    p_c = m._permutation_p(C, is_boolq, ds_c, seed=0)

    assert ds_c > 0.5, "tight group should be far more similar"
    assert p_c < 0.01
    assert m._verdict(ds_c, p_c, ds_h) == "WINDOW_DOMINATED"


def test_confounded_case_is_not_reported_as_window_dominated():
    """If the hidden states are also more similar, the window claim is unsupported."""
    rng = np.random.default_rng(1)
    is_boolq = np.array([True] * 30 + [False] * 30)
    C = np.vstack([_tight(rng, 30, 8), _diverse(rng, 30, 8)])
    H = np.vstack([_tight(rng, 30, 8), _diverse(rng, 30, 8)])

    ds_c = m._delta(C, is_boolq)
    ds_h = m._delta(H, is_boolq)
    p_c = m._permutation_p(C, is_boolq, ds_c, seed=0)

    assert ds_h > 0
    assert m._verdict(ds_c, p_c, ds_h) == "CONFOUNDED"


def test_item_dominated_case_is_detected():
    """The refutation path: boolq no more similar than short."""
    rng = np.random.default_rng(2)
    is_boolq = np.array([True] * 30 + [False] * 30)
    C = np.vstack([_diverse(rng, 30, 8), _tight(rng, 30, 8)])
    ds_c = m._delta(C, is_boolq)
    assert ds_c <= 0
    assert m._verdict(ds_c, 0.99, 0.0) == "ITEM_DOMINATED"


def test_permutation_test_does_not_always_reject():
    """Anti-vacuity: under exchangeable labels the test must stay quiet.

    Without this, `p_c < 0.01` would be free and every other assertion here
    would be theatre.
    """
    rng = np.random.default_rng(3)
    C = _diverse(rng, 60, 8)
    is_boolq = np.zeros(60, dtype=bool)
    is_boolq[:30] = True
    ds_c = m._delta(C, is_boolq)
    p_c = m._permutation_p(C, is_boolq, ds_c, seed=0)
    assert p_c > 0.01, f"permutation test rejected pure noise (p={p_c})"


def test_permutation_test_is_seed_reproducible():
    rng = np.random.default_rng(4)
    C = _diverse(rng, 40, 6)
    is_boolq = np.zeros(40, dtype=bool)
    is_boolq[:20] = True
    ds = m._delta(C, is_boolq)
    assert m._permutation_p(C, is_boolq, ds, seed=7) == m._permutation_p(
        C, is_boolq, ds, seed=7
    )


# --------------------------------------------------------------------------
# scripts/analyze_round164_vectors.py -- the exploratory half
# --------------------------------------------------------------------------

ANALYZE = ROOT / "scripts" / "analyze_round164_vectors.py"
VECTORS = ROOT / "outputs" / "round164" / "vectors"


def _load_analyze():
    spec = importlib.util.spec_from_file_location("_r164a", ANALYZE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_participation_ratio_rank_one_is_one():
    """A cloud on a single line has PR 1 -- the round's headline is this number."""
    rng = np.random.default_rng(0)
    direction = rng.normal(size=(1, 16))
    cloud = rng.normal(size=(50, 1)) * direction
    pr, top = _load_analyze().participation_ratio(cloud)
    assert pr == pytest.approx(1.0, abs=1e-6)
    assert top == pytest.approx(1.0, abs=1e-6)


def test_participation_ratio_isotropic_is_the_dimension():
    rng = np.random.default_rng(1)
    pr, top = _load_analyze().participation_ratio(rng.normal(size=(4000, 8)))
    assert pr == pytest.approx(8.0, rel=0.2)
    assert top < 0.3


def test_participation_ratio_discriminates_one_from_two_dimensions():
    """The 1.001-vs-1.480 claim is only meaningful if PR separates these."""
    rng = np.random.default_rng(2)
    rank1, _ = _load_analyze().participation_ratio(rng.normal(size=(400, 1)) * rng.normal(size=(1, 32)))
    rank2, _ = _load_analyze().participation_ratio(rng.normal(size=(400, 2)) @ rng.normal(size=(2, 32)))
    assert rank1 == pytest.approx(1.0, abs=1e-6)
    assert rank2 == pytest.approx(2.0, rel=0.1)


def test_offline_recomputation_matches_the_frozen_verdicts():
    """Recompute Part B from the saved vectors and check the frozen labels.

    ``outputs/`` is gitignored, so this skips in CI (``_skip_external``
    convention) -- but where the vectors exist it is the check that the
    write-up's verdicts follow from the artifacts rather than from a hand copy.
    """
    if not VECTORS.exists():
        pytest.skip(f"round-164 vectors not found: {VECTORS}")
    import json

    payload = json.loads(
        (ROOT / "outputs" / "round164" / "window-dominance.json").read_text()
    )
    tasks = [
        json.loads(line)["task"]
        for line in (ROOT / "outputs" / "round164" / "items-eval-600b.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    is_boolq = np.array(tasks) == "boolq"
    mod = _load_analyze()
    for name, frozen in payload["per_reader"].items():
        C = np.load(VECTORS / f"{name}-c.npy")
        H = np.load(VECTORS / f"{name}-h.npy")
        ds_c = mod.mean_pairwise_cosine(C[is_boolq]) - mod.mean_pairwise_cosine(C[~is_boolq])
        ds_h = mod.mean_pairwise_cosine(H[is_boolq]) - mod.mean_pairwise_cosine(H[~is_boolq])
        assert ds_c == pytest.approx(frozen["dS_c"], abs=1e-6)
        assert ds_h == pytest.approx(frozen["dS_h"], abs=1e-6)
        # The frozen label must be the one the frozen rule produces.
        assert mod.__dict__ and m._verdict(ds_c, frozen["p_c"], ds_h) == frozen["verdict"]


def test_injected_vector_is_near_rank_one():
    """The claim the write-up rests on, asserted against the artifacts."""
    if not VECTORS.exists():
        pytest.skip(f"round-164 vectors not found: {VECTORS}")
    mod = _load_analyze()
    pr_c, top_c = mod.participation_ratio(np.load(VECTORS / "wiki-c.npy"))
    pr_h, top_h = mod.participation_ratio(np.load(VECTORS / "wiki-h.npy"))
    assert pr_c < 1.05, f"c is not near rank-1 (PR={pr_c})"
    assert pr_h > 1.3, f"h is not measurably less collapsed (PR={pr_h})"
    assert top_c > 0.999 and top_h < 0.85
