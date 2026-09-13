"""Round 167 Stage 0.1: the nuisance audit must catch the round-166 confound.

The important tests here are the *falsification* tests: synthetic paired arms are
constructed with a known ground truth, and the auditor is required to recover it.
If ``synth_length_only`` ever classifies as anything other than
``NUISANCE_SENSITIVE``, the auditor cannot be trusted to have caught the real
round-166 confound, and the whole TD-1 fix is decorative.
"""

from __future__ import annotations

import numpy as np
import pytest

from qwen35_ple.metric_audit import (
    METRIC_DECLARATIONS,
    NO_EFFECT,
    NUISANCE_SENSITIVE,
    SURVIVES_ADJUSTMENT,
    UNDETERMINED,
    classify_contrast,
    declared_nuisance_sensitive,
    paired_nuisance_contrast,
    synth_content_only,
    synth_length_only,
    synth_mixed,
)

# ---------------------------------------------------------------------------
# The meta-tests: does the auditor find the confound, and only the confound?
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_length_only_is_flagged(seed: int) -> None:
    """A raw effect that is entirely generated length must be flagged."""
    ct = paired_nuisance_contrast(*synth_length_only(seed=seed))
    assert ct.verdict == NUISANCE_SENSITIVE, (
        f"seed {seed}: auditor missed a pure-length effect "
        f"(raw={ct.raw_diff:+.4f}, adjusted={ct.adjusted_diff:+.4f})"
    )
    # The nuisance explains the large majority of it.  Over 1000 seeds the
    # estimate's minimum is 0.829 (mean 1.000), so 0.8 is a bound the statistic
    # actually holds rather than one picked to make the test pass.
    assert ct.nuisance_share is not None and ct.nuisance_share > 0.8


@pytest.mark.parametrize("seed", range(12))
def test_content_only_is_not_flagged(seed: int) -> None:
    """A genuine equal-length effect must survive, not be dismissed."""
    ct = paired_nuisance_contrast(*synth_content_only(seed=seed))
    assert ct.verdict == SURVIVES_ADJUSTMENT, (
        f"seed {seed}: auditor dismissed a real effect "
        f"(raw={ct.raw_diff:+.4f}, adjusted={ct.adjusted_diff:+.4f})"
    )


@pytest.mark.parametrize("seed", range(12))
def test_mixed_recovers_effect_and_slope(seed: int) -> None:
    """With both present, the intercept recovers the content effect.

    Here the length difference dominates (share ~0.6), so the contrast is
    reported as confounded *and* the recovered numbers are still correct: the
    adjusted estimate is the honest one to quote.
    """
    ct = paired_nuisance_contrast(*synth_mixed(effect=0.4, slope=0.08, seed=seed))
    assert ct.verdict == NUISANCE_SENSITIVE  # share ~0.6 >= 0.5
    assert ct.adjusted_diff == pytest.approx(0.4, abs=0.12)
    assert ct.nuisance_slope == pytest.approx(0.08, abs=0.03)
    # And the raw number is materially larger than the true effect.
    assert ct.raw_diff > ct.adjusted_diff
    assert ct.nuisance_share is not None and 0.3 < ct.nuisance_share < 0.8


@pytest.mark.parametrize("seed", range(12))
def test_mixed_survives_when_the_effect_dominates(seed: int) -> None:
    """A small length difference must not get a real effect dismissed."""
    ct = paired_nuisance_contrast(
        *synth_mixed(effect=0.4, slope=0.08, mean_a=30.0, mean_b=28.0, seed=seed)
    )
    assert ct.verdict == SURVIVES_ADJUSTMENT
    assert ct.nuisance_share is not None and ct.nuisance_share < 0.5
    assert ct.adjusted_diff == pytest.approx(0.4, abs=0.12)


def test_auditor_has_power_on_the_real_round166_shape() -> None:
    """Round 166's real numbers: raw +0.6933, slope ~0.079, ~8 tokens longer.

    Detection must not be a coin flip.  With the share criterion this is exact,
    so require it on every one of 40 seeds -- not 19 in 20.
    """
    detected = 0
    seeds = 40
    for seed in range(seeds):
        ct = paired_nuisance_contrast(
            *synth_length_only(slope=0.079, mean_a=31.7, mean_b=23.8, noise=0.35, seed=seed)
        )
        detected += ct.verdict == NUISANCE_SENSITIVE
    assert detected == seeds, f"detected {detected}/{seeds}"


def test_pure_length_detection_is_exact_over_many_seeds() -> None:
    """Pin the measured operating characteristic of the gate.

    Before the share criterion this was 95.5% at tol_sigma=2 (the ordinary
    2-sigma miss rate).  That is too leaky for a discipline gate, so the gate
    must now catch a pure-length contrast every time.
    """
    seeds = 300
    missed = [
        s
        for s in range(seeds)
        if paired_nuisance_contrast(*synth_length_only(seed=s)).verdict
        != NUISANCE_SENSITIVE
    ]
    assert missed == [], f"missed pure-length contrasts at seeds {missed[:10]}"


def test_content_only_survival_is_exact_over_many_seeds() -> None:
    seeds = 300
    false_alarms = [
        s
        for s in range(seeds)
        if paired_nuisance_contrast(*synth_content_only(seed=s)).verdict
        != SURVIVES_ADJUSTMENT
    ]
    assert false_alarms == [], f"dismissed real effects at seeds {false_alarms[:10]}"


def test_no_false_positive_when_lengths_are_equal() -> None:
    """Zero-variance nuisance must never be reported as the explanation."""
    for seed in range(20):
        ct = paired_nuisance_contrast(*synth_content_only(seed=seed))
        assert ct.nuisance_has_variance is False
        assert ct.verdict != NUISANCE_SENSITIVE


def test_no_effect_is_not_misreported_as_confound() -> None:
    """A contrast with no raw effect is NO_EFFECT, not NUISANCE_SENSITIVE."""
    rng = np.random.default_rng(0)
    n = 400
    m_a = rng.normal(0.0, 1.0, n)
    m_b = rng.normal(0.0, 1.0, n)
    n_a = rng.normal(30.0, 4.0, n)
    n_b = rng.normal(30.0, 4.0, n)
    ct = paired_nuisance_contrast(m_a, m_b, n_a, n_b)
    assert ct.verdict == NO_EFFECT


# ---------------------------------------------------------------------------
# Degenerate inputs and mechanics
# ---------------------------------------------------------------------------


def test_single_pair_is_undetermined() -> None:
    ct = paired_nuisance_contrast([1.0], [0.0], [30.0], [20.0])
    assert ct.verdict == UNDETERMINED
    assert ct.n_pairs == 1


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        paired_nuisance_contrast([1.0, 2.0], [1.0], [3.0, 4.0], [3.0, 4.0])


def test_zero_variance_nuisance_falls_back_to_raw() -> None:
    ct = paired_nuisance_contrast(*synth_content_only(effect=0.5, seed=3))
    assert ct.adjusted_diff == pytest.approx(ct.raw_diff)
    assert ct.nuisance_slope == 0.0


def test_classify_contrast_is_exposed_and_matches_constructor() -> None:
    ct = paired_nuisance_contrast(*synth_length_only(seed=1))
    assert classify_contrast(ct) == ct.verdict


def test_sign_convention_matches_round166() -> None:
    """Arm order must not silently invert the sign of the effect."""
    m_a, m_b, n_a, n_b = synth_mixed(effect=0.4, seed=5)
    forward = paired_nuisance_contrast(m_a, m_b, n_a, n_b)
    reverse = paired_nuisance_contrast(m_b, m_a, n_b, n_a)
    assert forward.adjusted_diff == pytest.approx(-reverse.adjusted_diff)


# ---------------------------------------------------------------------------
# The declaration registry is the discipline gate
# ---------------------------------------------------------------------------


def test_round166_metrics_are_declared_as_needing_adjustment() -> None:
    """The three distinct-marker metrics must carry the declaration."""
    for name in ("prose_markers", "code_markers", "math_markers"):
        assert declared_nuisance_sensitive(name) is True


def test_every_headline_metric_is_declared_length_exposed() -> None:
    """Pin what the round-167 audit learned.

    The first registry draft declared the scaffold metrics as ``as-is`` on the
    reasoning that a per-item format label is not an accumulated count.  The
    audit disagreed on real arms, and the mechanism is clear: a generation has to
    run long enough to emit the marker.  These declarations are now evidence-
    based, so this test is a regression guard on the *learned* fact, not on my
    original guess.
    """
    for name in (
        "prose_markers",
        "code_markers",
        "math_markers",
        "empty_rate",
        "chat_scaffold_rate",
        "scaffold_in_first_32_chars",
    ):
        assert declared_nuisance_sensitive(name) is True, (
            f"{name} was found length-exposed on real arms but is declared as-is"
        )


def test_revised_declarations_record_the_evidence() -> None:
    """A declaration that the audit overturned must say why."""
    for name in ("chat_scaffold_rate", "scaffold_in_first_32_chars"):
        note = METRIC_DECLARATIONS[name].note
        assert "REVISED" in note and "audit" in note, (
            f"{name} changed after being contradicted; the reason must be recorded"
        )


def test_every_declaration_names_a_nuisance_and_a_note() -> None:
    for name, decl in METRIC_DECLARATIONS.items():
        assert decl.nuisance, f"{name} has no nuisance named"
        assert decl.note, f"{name} has no rationale recorded"


def test_undeclared_metric_raises_instead_of_defaulting() -> None:
    """A new metric cannot enter the suite without a declaration."""
    with pytest.raises(KeyError, match="not declared"):
        declared_nuisance_sensitive("some_brand_new_metric")


def test_declaration_agrees_with_measured_verdict_on_synthetic_replica() -> None:
    """End-to-end: declaration for a marker-style metric matches its audit.

    ``synth_length_only`` is the synthetic replica of a distinct-marker metric
    under a length difference, so a metric declared as needing adjustment must be
    measured as ``NUISANCE_SENSITIVE`` on it.
    """
    ct = paired_nuisance_contrast(*synth_length_only(seed=7))
    assert declared_nuisance_sensitive("prose_markers") is (ct.verdict == NUISANCE_SENSITIVE)
