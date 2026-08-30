"""M1 PLE-lite forward golden: engram-peft layer vs Qwen reference math.

This test is optional in dependency-light CI because it needs torch, engram-peft
and a few optional packages.  When the runtime is available it verifies that the
engram-peft ``EngramLayer`` with ``engine='qwen_ple'`` and ``hc_mult=1`` produces
the same value as the Qwen ``Qwen4ExpTextPLELayer`` math.
"""

from __future__ import annotations

import pytest


def _install_rmsnorm_compat() -> None:
    import torch

    if hasattr(torch.nn, "RMSNorm"):
        return

    class _RMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (
                x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
            )

    torch.nn.RMSNorm = _RMSNorm


def _load_engram_peft():
    try:
        import typing

        import typing_extensions
        if not hasattr(typing, "override"):
            typing.override = typing_extensions.override
        import torch
        from engram_peft import EngramConfig
        from engram_peft.hashing import create_hash_mapping
        from engram_peft.layer import EngramLayer
    except Exception as exc:  # noqa: BLE001 - optional deps
        # pragma: no cover
        pytest.skip(f"engram-peft/torch runtime not available: {exc}")
    return torch, EngramConfig, EngramLayer, create_hash_mapping


def _make_layer(torch, EngramConfig, EngramLayer, create_hash_mapping, seq_len):
    prime_sizes = [
        17, 19, 23, 29, 31, 37, 41, 43,
        47, 53, 59, 61, 67, 71, 73, 79,
    ]
    hidden_size = 8
    embedding_dim = 16 * 10  # 16 heads * 10 dims; small but structurally real
    config = EngramConfig(
        hidden_size=hidden_size,
        embedding_dim=embedding_dim,
        ngram_sizes=[2, 3],
        n_head_per_ngram=8,
        target_layers=[1],
        engram_vocab_size_per_ngram=[64, 64],
        compressed_vocab_size=1000,
        pad_id=2,
        engine="qwen_ple",
        table_spec="PLE_QWEN_V1",
        enable_tokenizer_compression=False,
        hc_mult=1,
        conv_kernel_size=2,
        conv_dilation=2,
        gating_zero_init=False,
        conv_zero_init=False,
        prime_sizes=prime_sizes,
    )
    mapping = create_hash_mapping(
        compressed_vocab_size=1000,
        engram_vocab_size_per_ngram=[64, 64],
        ngram_sizes=[2, 3],
        n_head_per_ngram=8,
        layer_ids=[1],
        pad_id=2,
        seed=0,
        engine="qwen_ple",
        table_spec="PLE_QWEN_V1",
        prime_sizes=prime_sizes,
    )
    flat_primes = [p for group in mapping.prime_tables[1] for p in group]
    layer = EngramLayer(config, layer_id=1, primes=flat_primes)
    layer.eval()
    return config, mapping, layer


def _run_comparison(seq_len: int) -> None:
    torch, EngramConfig, EngramLayer, create_hash_mapping = _load_engram_peft()
    _install_rmsnorm_compat()
    config, _, layer = _make_layer(
        torch, EngramConfig, EngramLayer, create_hash_mapping, seq_len
    )

    from qwen35_ple.ple_reference import qwen_ple_forward_reference

    torch.manual_seed(0)
    # A token sequence that crosses EOS to exercise segment reset semantics.
    ids = torch.tensor([[248044, 5, 100, 999, 42, 7] * (seq_len // 6 + 1)])[:, :seq_len]
    hidden = torch.randn(1, seq_len, config.hidden_size)

    with torch.no_grad():
        actual = layer(input_ids=ids, hidden_states=hidden)
        expected = qwen_ple_forward_reference(
            input_ids=ids,
            hidden_states=hidden,
            hash_mapping=layer.hash_mapping,
            multi_head_embedding=layer.multi_head_embedding,
            value_proj=layer.gating.w_v,
            key_proj=layer.gating.w_k[0],
            norm_key=layer.gating.norm_k[0],
            norm_query=layer.gating.norm_h[0],
            norm_conv=layer.short_conv.norms[0],
            conv1d=layer.short_conv.conv,
            hidden_size=config.hidden_size,
            conv_kernel_size=config.conv_kernel_size,
            conv_dilation=config.conv_dilation,
        )

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_ple_forward_matches_reference_short() -> None:
    _run_comparison(seq_len=32)


def test_ple_forward_matches_reference_4096_tokens() -> None:
    _run_comparison(seq_len=4096)
