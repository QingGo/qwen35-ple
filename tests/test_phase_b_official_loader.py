"""Phase B tests for official Qwen4-Exp PLE loading with EngramDB disk adapter.

These tests are optional in dependency-light CI: they only run when torch,
engramdb, and the local qwen35-ple package (containing the frozen official
snapshot) are available.
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
engramdb = pytest.importorskip("engramdb")

from engramdb.official_loader import (  # noqa: E402
    install_disk_ple_in_official_model,
    patch_official_ngram_embedding_for_disk_load,
)
from engramdb.ple_adapter import (  # noqa: E402
    DiskPleNGramEmbedding,
    head_offsets,
    head_vocab_sizes,
    padded_vocab_size,
)

from qwen35_ple.official_ple_snapshot import (  # noqa: E402
    Qwen4ExpTextNGramEmbedding,
    Qwen4ExpTextPLELayer,
)


def _small_config() -> SimpleNamespace:
    return SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=8,
        vocab_size=1000,
        ngram_vocab_size_base=100,
        seed=0,
        eos_token_id=2,
        make_ngram_vocab_size_divisible_by=128,
        hidden_size=8,
        hc_count=1,
        ple_embed_dim=16,
        ple_conv_kernel_size=4,
        rms_norm_eps=1e-5,
    )


def test_placeholder_patch_with_frozen_official_snapshot() -> None:
    config = _small_config()
    with patch_official_ngram_embedding_for_disk_load(
        embedding_class=Qwen4ExpTextNGramEmbedding
    ):
        ple = Qwen4ExpTextNGramEmbedding(config, 16, 0, 0)
    assert ple.ngram_embedding.weight.shape[0] == 1


def test_install_disk_ple_in_frozen_official_ple_layer() -> None:
    config = _small_config()

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.layers = torch.nn.ModuleList()
            with patch_official_ngram_embedding_for_disk_load(
                embedding_class=Qwen4ExpTextNGramEmbedding
            ):
                self.language_model.layers.append(
                    Qwen4ExpTextPLELayer(config, 0, 0)
                )

    model = FakeModel()
    info = {
        "ple_embed_dim": 16,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "weight_scale": 1.0,
        "layer_multipliers": [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071],
    }
    with tempfile.TemporaryDirectory(prefix="engramdb-b2-struct-") as td:
        root = Path(td)
        (root / "shard_000.bin").write_bytes(b"\x00" * 16)
        store = engramdb.Store(str(root), shards=1, rows_per_shard=1, width=16)
        try:
            replaced = install_disk_ple_in_official_model(
                model,
                store,
                info=info,
                scale=1.0,
            )
            assert len(replaced) == 1
            assert type(model.language_model.layers[0].ple_embedding).__name__ == (
                "DiskPleNGramEmbedding"
            )
        finally:
            store.close()


def test_disk_ple_bit_exact_small_batch_eos() -> None:
    base = 100
    ngram_size = 3
    heads_per_ngram = 8
    num_heads = 16
    embed_dim = 16
    eos = 2
    sizes = head_vocab_sizes(base=base, heads=num_heads)
    offsets = head_offsets(sizes)
    total_padded = padded_vocab_size(sizes, divisor=128)

    cfg = SimpleNamespace(
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        vocab_size=1000,
        ngram_vocab_size_base=base,
        seed=0,
        eos_token_id=eos,
        make_ngram_vocab_size_divisible_by=128,
    )
    official = Qwen4ExpTextNGramEmbedding(cfg, embed_dim, 0, 0)
    multipliers = official.layer_multipliers.tolist()
    with torch.no_grad():
        official.ngram_embedding.weight.copy_(
            torch.arange(total_padded, dtype=torch.float32).unsqueeze(1)
        )

    with tempfile.TemporaryDirectory(prefix="engramdb-b2-bit-") as td:
        root = Path(td)
        with open(root / "shard_000.bin", "wb") as f:
            f.writelines(struct.pack("<f", float(i)) for i in range(total_padded))
        store = engramdb.Store(str(root), shards=1, rows_per_shard=total_padded, width=4)
        try:
            disk = DiskPleNGramEmbedding(
                store=store,
                num_embeddings=total_padded,
                embedding_dim=embed_dim,
                num_heads=num_heads,
                layer_multipliers=multipliers,
                scale=1.0,
                dtype=torch.float32,
                cache_size=4096,
                eos=eos,
                prime_sizes=sizes,
                offsets=offsets,
                ngram_size=ngram_size,
                heads_per_ngram=heads_per_ngram,
                divisor=128,
            )
            tokens = torch.tensor([[1, 2, 3], [4, eos, 6]], dtype=torch.long)
            with torch.no_grad():
                expected = official(tokens, None)
                actual = disk(tokens, None)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

            # Chunked/streaming path with per-batch internal context.
            full = torch.tensor([[1, 2, 3, 4, eos, 6]], dtype=torch.long)
            with torch.no_grad():
                expected_stream = official(full, None)[0]
                disk.reset_history()
                part1 = disk(full[:, :2], None)
                part2 = disk(full[:, 2:], None)
                actual_stream = torch.cat([part1[0], part2[0]], dim=0)
            torch.testing.assert_close(actual_stream, expected_stream, atol=0, rtol=0)
        finally:
            store.close()
