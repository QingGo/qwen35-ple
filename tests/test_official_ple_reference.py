"""Official Qwen4-Exp PLE reference snapshot and forward golden tests.

The heavy forward test is optional in dependency-light CI: it only runs when
torch and engram-peft are importable.  The snapshot/checksum/golden-shape tests
are lightweight and run in the normal CI job.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "refs" / "qwen4_exp_modeling.py"
MANIFEST = ROOT / "refs" / "qwen4_exp_modeling.manifest.json"
GENERATOR = ROOT / "scripts" / "generate_official_ple_snapshot.py"
GOLDEN_NPZ = ROOT / "tests" / "golden" / "official_ple_forward_4096.npz"
GOLDEN_META = ROOT / "tests" / "golden" / "official_ple_forward_4096.meta.json"


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
        pytest.skip(f"engram-peft/torch runtime not available: {exc}")
    return torch, EngramConfig, EngramLayer, create_hash_mapping


def test_ref_checksum_matches_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual = hashlib.sha256(REFS.read_bytes()).hexdigest()
    assert actual == manifest["sha256"]
    assert manifest["source_commit"]
    assert manifest["file"] == "qwen4_exp_modeling.py"


def test_official_snapshot_is_current() -> None:
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "up to date" in result.stdout


def test_golden_fixture_metadata_and_shapes() -> None:
    meta = json.loads(GOLDEN_META.read_text(encoding="utf-8"))
    data = np.load(GOLDEN_NPZ)

    assert meta["seq_len"] == 4096
    assert meta["hidden_size"] == 8
    assert meta["embedding_dim"] == 160
    assert meta["per_head"] == 10
    assert len(meta["prime_sizes"]) == 16
    assert meta["total_rows"] == sum(meta["prime_sizes"])

    assert data["input_ids"].shape == (1, meta["seq_len"])
    assert data["hidden_states"].shape == (1, meta["seq_len"], meta["hidden_size"])
    assert data["ple_output"].shape == data["hidden_states"].shape
    assert data["expected"].shape == data["hidden_states"].shape
    assert data["embedding_weight"].shape == (
        meta["total_rows"],
        meta["per_head"],
    )
    assert data["value_weight"].shape == (meta["hidden_size"], meta["embedding_dim"])
    assert data["key_weight"].shape == (meta["hidden_size"], meta["embedding_dim"])

    assert np.isfinite(data["hidden_states"]).all()
    assert np.isfinite(data["ple_output"]).all()
    assert np.isfinite(data["expected"]).all()


def test_production_matches_official_golden_4096() -> None:
    """engram-peft EngramLayer must reproduce the frozen official fixture."""
    torch, EngramConfig, EngramLayer, create_hash_mapping = _load_engram_peft()
    _install_rmsnorm_compat()
    meta = json.loads(GOLDEN_META.read_text(encoding="utf-8"))
    data = np.load(GOLDEN_NPZ)

    prime_sizes = list(meta["prime_sizes"])
    hidden_size = meta["hidden_size"]
    embedding_dim = meta["embedding_dim"]

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
        conv_kernel_size=meta["conv_kernel_size"],
        conv_dilation=meta["conv_dilation"],
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

    with torch.no_grad():
        layer.multi_head_embedding.embedding.weight.copy_(
            torch.from_numpy(data["embedding_weight"])
        )
        layer.gating.w_v.weight.copy_(torch.from_numpy(data["value_weight"]))
        layer.gating.w_k[0].weight.copy_(torch.from_numpy(data["key_weight"]))
        layer.short_conv.conv.weight.copy_(torch.from_numpy(data["conv_weight"]))

        # Official Qwen RMSNorm uses weight=0 and output = norm * (1 + weight).
        # The engram-peft nn.RMSNorm uses weight=1; align both and eps.
        layer.gating.norm_k[0].weight.fill_(1.0)
        layer.gating.norm_h[0].weight.fill_(1.0)
        layer.short_conv.norms[0].weight.fill_(1.0)
        layer.gating.norm_k[0].eps = 1e-5
        layer.gating.norm_h[0].eps = 1e-5
        layer.short_conv.norms[0].eps = 1e-5

        input_ids = torch.from_numpy(data["input_ids"])
        hidden = torch.from_numpy(data["hidden_states"])
        actual = layer(input_ids=input_ids, hidden_states=hidden)
        expected = torch.from_numpy(data["expected"])

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
