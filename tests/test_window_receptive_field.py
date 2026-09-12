"""Executable evidence for the addressing window (round 165).

Why this file exists
--------------------
The paper's most striking number is the addressing window: twelve tokens for
Engram / Qwen3.8-Flash-Next rather than the two an order-3 key suggests, and
four for V4.1-Flash.  Until now that number rested on reading a config and
multiplying --- which is exactly the kind of claim this project has been wrong
about before (round 163 §5.1: five of one session's corrections were about our
own artifacts and none about the model).

A reviewer is entitled to ask how we know the convolution is in the *executed*
path, and what happens when it is removed.  This file answers by execution:
build the real ``Qwen4ExpTextPLELayer``, perturb one input token, and measure
which output positions change.  The measured receptive field is then compared
against the closed form.

The closed form is simpler than the derivation in the paper makes it look.  The
layer pads by ``(kernel - 1) * dilation`` and the official code sets
``dilation = ngram_size``, so the conv at ``t`` reads ``gated_value`` at
``t - (kernel - 1) * ngram .. t``; each ``gated_value`` at ``s`` reads tokens
``s - (ngram - 1) .. s``; the union is

    window = (kernel - 1) * ngram + ngram = kernel * ngram.

So 4 x 3 = 12 for Engram and Qwen3.8, and a kernel of 1 (no convolution) gives
the key order itself --- 4 tokens for V4.1-Flash's 4-grams.

The check is a delta test, not a numerical-accuracy test: it asserts *support*,
so it is insensitive to dtype, seeds and magnitudes.  Any config that runs the
same code path can be checked this way, which is why the ablation is a loop over
geometries rather than a separate argument.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from qwen35_ple.official_ple_snapshot import Qwen4ExpTextPLELayer

# Long enough that the window is interior to the sequence for every geometry
# tested, so no test depends on left-padding behaviour.
SEQ_LEN = 48
MAX_LAG = 18


def _config(*, hidden_size: int, hc_count: int, ple_embed_dim: int,
            heads_per_ngram: int, ngram_size: int, ple_conv_kernel_size: int):
    """Minimal config carrying the official field names the layer reads.

    ``ngram_vocab_size_base`` is small on purpose: it sizes only the embedding
    table.  Every tensor whose reach we measure is sized by ``hidden_size``,
    ``hc_count`` and ``ple_conv_kernel_size`` alone, so the geometry under test
    is the production geometry even though the table is toy-sized.
    """
    return SimpleNamespace(
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        ple_embed_dim=ple_embed_dim,
        vocab_size=997,
        ngram_vocab_size_base=101,
        hidden_size=hidden_size,
        hc_count=hc_count,
        ple_conv_kernel_size=ple_conv_kernel_size,
        rms_norm_eps=1e-6,
        seed=0,
        eos_token_id=2,
        make_ngram_vocab_size_divisible_by=8,
    )


def _build(*, ngram_size: int, ple_conv_kernel_size: int) -> Qwen4ExpTextPLELayer:
    hidden_size, hc_count, heads_per_ngram, head_dim = 16, 2, 2, 8
    # The n-gram embedding flattens (ngram_size - 1) * heads_per_ngram heads of
    # width head_dim, and value_proj consumes that flattened width, so
    # ple_embed_dim has to be exactly the product -- otherwise the layer cannot
    # be constructed at all for a given ngram_size.
    ple_embed_dim = head_dim * heads_per_ngram * (ngram_size - 1)
    cfg = _config(
        hidden_size=hidden_size, hc_count=hc_count, ple_embed_dim=ple_embed_dim,
        heads_per_ngram=heads_per_ngram, ngram_size=ngram_size,
        ple_conv_kernel_size=ple_conv_kernel_size,
    )
    torch.manual_seed(0)
    layer = Qwen4ExpTextPLELayer(cfg, layer_idx=1, ple_layer_index=0)
    layer.eval()
    return layer


def _influence_profile(layer: Qwen4ExpTextPLELayer, *, seed: int = 0) -> list[float]:
    """Max ``|Δ output[T-1]|`` when the token at lag L is changed.

    Entry ``L`` is zero exactly when position ``T-1`` cannot see token ``T-1-L``.
    The query hidden state is held fixed, so only the memory path varies.
    """
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(1, 997, (SEQ_LEN,), generator=g).unsqueeze(0)
    # hc_count * hidden_size: the official layer consumes the hyper-connection
    # expanded stream, not a single-stream hidden state.
    hidden = torch.randn(1, SEQ_LEN, layer.hc_count * layer.hidden_size, generator=g)

    with torch.no_grad():
        base = layer(hidden, input_ids, None)[0, SEQ_LEN - 1]
        profile = []
        for lag in range(MAX_LAG):
            pos = SEQ_LEN - 1 - lag
            perturbed = input_ids.clone()
            perturbed[0, pos] = (perturbed[0, pos] + 7) % 997
            delta = (layer(hidden, perturbed, None)[0, SEQ_LEN - 1] - base).abs().max()
            profile.append(float(delta))
    return profile


def _measured_window(profile: list[float]) -> int:
    """Number of lags that actually reach the output, i.e. the window length."""
    return sum(1 for d in profile if d > 0.0)


# --------------------------------------------------------------------------
# The closed form, verified by execution across a grid
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kernel", "ngram"),
    [(4, 3), (1, 4), (2, 3), (4, 2), (4, 4), (3, 3), (1, 3), (2, 2)],
)
def test_measured_window_equals_kernel_times_ngram(kernel: int, ngram: int) -> None:
    """window = kernel * ngram, confirmed by perturbing one token at a time.

    This is the claim the paper rests on.  It is checked for a spread of
    geometries precisely so that a passing test cannot be an accident of the
    production configuration.
    """
    layer = _build(ngram_size=ngram, ple_conv_kernel_size=kernel)
    assert layer.short_conv_state_len == (kernel - 1) * ngram

    profile = _influence_profile(layer)
    expected = kernel * ngram
    assert expected <= MAX_LAG, "raise MAX_LAG for this geometry"

    # Support, not magnitude: every lag inside the window is live ...
    for lag in range(expected):
        assert profile[lag] > 0.0, (
            f"kernel={kernel} ngram={ngram}: lag {lag} should be inside the "
            f"window of {expected} but produced no change")
    # ... and every lag outside it is exactly dead.
    for lag in range(expected, MAX_LAG):
        assert profile[lag] == 0.0, (
            f"kernel={kernel} ngram={ngram}: lag {lag} is outside the window of "
            f"{expected} but moved the output by {profile[lag]:.3e}")
    assert _measured_window(profile) == expected


# --------------------------------------------------------------------------
# The three production designs
# --------------------------------------------------------------------------
def test_engram_and_qwen38_geometry_is_twelve_tokens() -> None:
    """kernel 4, dilation = ngram_size 3 -> 12, not the 2 an order-3 key implies."""
    layer = _build(ngram_size=3, ple_conv_kernel_size=4)
    profile = _influence_profile(layer)
    assert profile[11] > 0.0, "token eleven back must still reach the output"
    assert profile[12] == 0.0, "token twelve back must not reach the output"
    assert _measured_window(profile) == 12


def test_removing_the_convolution_leaves_the_key_order() -> None:
    """V4.1-Flash removed the conv; a kernel of 1 is that ablation.

    With no convolution the window is exactly the key order -- four tokens for
    the 4-grams V4.1-Flash uses -- and the twelve-token figure is a property of
    Engram/Qwen3.8 alone, not of the family.
    """
    layer = _build(ngram_size=4, ple_conv_kernel_size=1)
    assert layer.short_conv_state_len == 0
    profile = _influence_profile(layer)
    assert profile[3] > 0.0
    assert profile[4] == 0.0
    assert _measured_window(profile) == 4


def test_window_is_free_of_the_embedding_table_size() -> None:
    """The window is architectural, so it must not depend on the table.

    Guards the reasoning used to justify testing at toy table size: if the
    measured window moved with ``ngram_vocab_size_base``, the small-config tests
    above would not be evidence about the production model.
    """
    small = _influence_profile(_build(ngram_size=3, ple_conv_kernel_size=4))
    big_cfg = _config(hidden_size=16, hc_count=2, ple_embed_dim=8 * 2 * (3 - 1),
                      heads_per_ngram=2, ngram_size=3, ple_conv_kernel_size=4)
    big_cfg.ngram_vocab_size_base = 5003
    big_cfg.vocab_size = 20011
    torch.manual_seed(0)
    big = Qwen4ExpTextPLELayer(big_cfg, layer_idx=1, ple_layer_index=0)
    big.eval()
    assert _measured_window(_influence_profile(big)) == _measured_window(small) == 12
