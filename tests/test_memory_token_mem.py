"""Tests for P1 TokenMem-style memory modules (skipped without torch)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qwen35_ple.memory.token_mem import (
    MemoryLogitFusion,
    MemoryLogitHead,
    MemoryRouter,
    TokenMemCrossAttention,
)


def test_cross_attention_shape() -> None:
    layer = TokenMemCrossAttention(d_model=32, d_mem=8, n_heads=4)
    h = torch.randn(2, 5, 32)
    mem = torch.randn(2, 5, 3, 8)
    out = layer(h, mem)
    assert out.shape == (2, 5, 32)


def test_memory_logit_head_shape() -> None:
    head = MemoryLogitHead(d_mem=8, vocab_size=11, d_model=16, hidden=8)
    mem = torch.randn(2, 5, 8)
    logits = head(mem)
    assert logits.shape == (2, 5, 11)


def test_router_and_fusion() -> None:
    router = MemoryRouter(d_model=16, d_mem=8)
    h = torch.randn(2, 4, 16)
    mem = torch.randn(2, 4, 8)
    alpha = router(h, mem)
    assert alpha.shape == (2, 4, 1)
    assert float(alpha.min()) >= 0.0 and float(alpha.max()) <= 1.0

    fusion = MemoryLogitFusion(d_model=16, d_mem=8)
    base = torch.randn(2, 4, 11)
    mem_logits = torch.randn(2, 4, 11)
    fused, alpha2 = fusion(h, mem, base, mem_logits)
    assert fused.shape == base.shape
    assert alpha2.shape == (2, 4, 1)


def test_cross_attention_zero_init_keeps_projection_learnable() -> None:
    layer = TokenMemCrossAttention(d_model=16, d_mem=8, n_heads=4, zero_init_out=True)
    assert float(layer.out_proj.weight.abs().sum()) == 0.0
    h = torch.randn(1, 3, 16)
    mem = torch.randn(1, 3, 2, 8)
    layer(h, mem)  # should run without NaN
    assert np.isfinite(float(layer.out_proj.weight.sum()))


def test_p1_memory_module_shape() -> None:
    from qwen35_ple.memory.token_mem import P1MemoryModule

    module = P1MemoryModule(d_model=16, d_mem=8, vocab_size=11, n_heads=4)
    h = torch.randn(1, 5, 16)
    mem = torch.randn(1, 5, 3, 8)
    base = torch.randn(1, 5, 11)
    fused, mem_logits, alpha = module(h, mem, base)
    assert fused.shape == base.shape
    assert mem_logits.shape == base.shape
    assert alpha.shape == (1, 5, 1)


def test_pure_logit_memory_shape_and_zero_scale() -> None:
    from qwen35_ple.memory.token_mem import PureLogitMemoryModule

    module = PureLogitMemoryModule(d_mem=8, vocab_size=11, hidden=8)
    mem = torch.randn(2, 5, 8)
    base = torch.randn(2, 5, 11)
    out = module(mem, base)
    assert out.shape == base.shape
    # With zero-initialized scale, initial output equals base logits.
    assert float((out - base).abs().max()) < 1e-6
