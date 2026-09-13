#!/usr/bin/env python3
"""Round 167 Stage 2.2: the read-out gate must be genuinely selective.

Round-146 measured the read-out gate saturating open (0.77-0.98 of entries above
0.5) after SFT, which makes injection always-on and *hurts* open QA.  The
structural reason is that the official gate sums the query/key product over all
``d_source`` dimensions, so a branch gets a single scalar with which to control
2560 dimensions.

These tests pin three things: the ``scalar`` mode still computes exactly the
official formula (so every existing checkpoint keeps its behaviour), the
``per_dim`` mode gates each dimension independently, and the two are genuinely
different -- i.e. per-dim really does buy selectivity rather than being a
no-op rename.
"""

from __future__ import annotations

import math

import pytest

# The local macOS venv has no torch; the repo convention is importorskip so the
# suite still collects here and runs fully on the GPU box.
torch = pytest.importorskip("torch")

from qwen35_ple.reader import OfficialSourceQwenReader

D_TARGET = 8
D_SOURCE = 4
D_MEM = 4
HC = 2


def _reader(gate_mode: str) -> OfficialSourceQwenReader:
    torch.manual_seed(0)
    return OfficialSourceQwenReader(
        d_target=D_TARGET,
        d_source=D_SOURCE,
        d_mem=D_MEM,
        hc=HC,
        kernel_size=2,
        dilation=1,
        freeze_source=False,
        # A zero-initialised out_proj makes the output identically zero, which
        # would let "per_dim changed nothing" pass trivially.  Use a real init
        # so the gate change is actually observable at the output.
        zero_init_out=False,
        gate_mode=gate_mode,
    )


def _inputs(seed: int = 0, b: int = 2, t: int = 5):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(b, t, D_TARGET, generator=g)
    e_t = torch.randn(b, t, D_MEM, generator=g)
    return h, e_t


def test_invalid_gate_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="gate_mode"):
        _reader("per_token")


def test_default_is_scalar() -> None:
    """The official behaviour must be the default (additive contract change)."""
    torch.manual_seed(0)
    r = OfficialSourceQwenReader(
        d_target=D_TARGET, d_source=D_SOURCE, d_mem=D_MEM, hc=HC, freeze_source=False
    )
    assert r.gate_mode == "scalar"


@pytest.mark.parametrize("mode", ["scalar", "per_dim"])
def test_output_shape_is_unchanged(mode: str) -> None:
    r = _reader(mode)
    h, e_t = _inputs()
    out = r(h, e_t)
    assert out.shape == (2, 5, D_TARGET)


def test_scalar_gate_reproduces_the_official_formula() -> None:
    """Recompute the published arithmetic from the reader's own submodules."""
    r = _reader("scalar")
    h, e_t = _inputs()
    with torch.no_grad():
        key = r.key_proj(e_t).view(2, 5, HC, D_SOURCE)
        key = r.norm_key(key.view(2, 5, HC * D_SOURCE)).view(2, 5, HC, D_SOURCE)
        query = r.query_bridge(h).view(2, 5, HC, D_SOURCE)
        query = r.norm_query(query.view(2, 5, HC * D_SOURCE)).view(2, 5, HC, D_SOURCE)
        score = (key * query).sum(-1, keepdim=True) / math.sqrt(D_SOURCE)
        score = score.abs().clamp_min(1e-6).sqrt() * score.sign()
        expected = torch.sigmoid(score)
        r(h, e_t)
    torch.testing.assert_close(r.last_gate_raw, expected, atol=0, rtol=0)


def test_scalar_gate_has_one_value_per_branch() -> None:
    r = _reader("scalar")
    h, e_t = _inputs()
    with torch.no_grad():
        r(h, e_t)
    assert r.last_gate_raw.shape == (2, 5, HC, 1)


def test_per_dim_gate_has_one_value_per_dimension() -> None:
    r = _reader("per_dim")
    h, e_t = _inputs()
    with torch.no_grad():
        r(h, e_t)
    assert r.last_gate_raw.shape == (2, 5, HC, D_SOURCE)


def test_per_dim_gate_is_not_constant_across_dimensions() -> None:
    """The whole point: the gate must be able to select within a branch."""
    r = _reader("per_dim")
    h, e_t = _inputs(seed=3)
    with torch.no_grad():
        r(h, e_t)
    per_dim_std = r.last_gate_raw.std(dim=-1)
    assert float(per_dim_std.min()) > 0.0


def test_scalar_gate_is_constant_across_dimensions_by_construction() -> None:
    r = _reader("scalar")
    h, e_t = _inputs(seed=3)
    with torch.no_grad():
        r(h, e_t)
    # shape [B,T,hc,1]: a single value, so there is nothing to vary.
    assert r.last_gate_raw.shape[-1] == 1


def test_the_two_modes_produce_different_gates() -> None:
    """Guards against per_dim being a silent rename of scalar."""
    h, e_t = _inputs(seed=5)
    r_s = _reader("scalar")
    r_p = _reader("per_dim")
    # Same parameters in both readers.
    r_p.load_state_dict(r_s.state_dict(), strict=False)
    with torch.no_grad():
        out_s = r_s(h, e_t)
        out_p = r_p(h, e_t)
    assert not torch.allclose(out_s, out_p), "per_dim changed nothing"


def test_per_dim_mean_is_reported_as_the_headline_diagnostic() -> None:
    """``last_gate`` stays comparable across modes: a per-branch mean."""
    r = _reader("per_dim")
    h, e_t = _inputs()
    with torch.no_grad():
        r(h, e_t)
    assert r.last_gate.shape == (2, 5, HC, 1)
    torch.testing.assert_close(
        r.last_gate, r.last_gate_raw.detach().mean(-1, keepdim=True)
    )


def test_open_fraction_is_reported_for_both_modes() -> None:
    for mode in ("scalar", "per_dim"):
        r = _reader(mode)
        h, e_t = _inputs()
        with torch.no_grad():
            r(h, e_t)
        assert r.last_gate_open_fraction is not None
        assert 0.0 <= r.last_gate_open_fraction <= 1.0


def test_open_fraction_is_zero_for_a_closed_override() -> None:
    r = _reader("per_dim")
    r.gate_override = 0.0
    h, e_t = _inputs()
    with torch.no_grad():
        r(h, e_t)
    assert r.last_gate_open_fraction == 0.0


def test_open_fraction_is_one_for_an_open_override() -> None:
    r = _reader("per_dim")
    r.gate_override = 1.0
    h, e_t = _inputs()
    with torch.no_grad():
        r(h, e_t)
    assert r.last_gate_open_fraction == 1.0


def test_gate_override_still_zeros_the_contribution() -> None:
    """The ple-off switch must still work in the new mode."""
    r = _reader("per_dim")
    h, e_t = _inputs()
    r.gate_override = 0.0
    with torch.no_grad():
        out = r(h, e_t)
    assert torch.allclose(out, torch.zeros_like(out))


def test_gate_mode_is_exposed_in_the_registry_config() -> None:
    from types import SimpleNamespace

    from qwen35_ple.reader_registry import (
        OFFICIAL_SOURCE_QWEN_V1,
        reader_config_from_args,
    )

    cfg = reader_config_from_args(
        SimpleNamespace(gate_mode="per_dim"), 1024, OFFICIAL_SOURCE_QWEN_V1
    )
    assert cfg["gate_mode"] == "per_dim"


def test_registry_config_defaults_to_scalar() -> None:
    from types import SimpleNamespace

    from qwen35_ple.reader_registry import (
        OFFICIAL_SOURCE_QWEN_V1,
        reader_config_from_args,
    )

    cfg = reader_config_from_args(SimpleNamespace(), 1024, OFFICIAL_SOURCE_QWEN_V1)
    assert cfg["gate_mode"] == "scalar"
