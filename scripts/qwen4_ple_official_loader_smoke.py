#!/usr/bin/env python3
"""Phase B1 structural smoke on the frozen official Qwen4-Exp PLE classes.

This does not need a Transformers build that ships Qwen4-Exp.  It loads the
frozen official ``Qwen4ExpTextNGramEmbedding`` / ``Qwen4ExpTextPLELayer`` from
``qwen35_ple.official_ple_snapshot``, demonstrates the placeholder constructor
patch, then installs the EngramDB disk adapter in the same way the full official
model loader would.

Usage:

    PYTHONPATH=src:../EngramDB/python \\
    python scripts/qwen4_ple_official_loader_smoke.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

import engramdb
from engramdb.official_loader import (
    install_disk_ple_in_official_model,
    patch_official_ngram_embedding_for_disk_load,
)
from qwen35_ple.official_ple_snapshot import (
    Qwen4ExpTextNGramEmbedding,
    Qwen4ExpTextPLELayer,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=8,
        vocab_size=1000,
        ngram_vocab_size_base=1000,
        seed=0,
        eos_token_id=2,
        make_ngram_vocab_size_divisible_by=128,
        hidden_size=8,
        hc_count=1,
        ple_embed_dim=160,
        ple_conv_kernel_size=4,
        rms_norm_eps=1e-5,
    )


class FakeOfficialQwen4Model(torch.nn.Module):
    """Minimal carrier matching the official nested PLE module paths."""

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.layers = torch.nn.ModuleList()
        with patch_official_ngram_embedding_for_disk_load(
            embedding_class=Qwen4ExpTextNGramEmbedding
        ):
            self.language_model.layers.append(
                Qwen4ExpTextPLELayer(config, layer_idx=0, ple_layer_index=0)
            )


def main() -> int:
    config = _config()
    model = FakeOfficialQwen4Model(config)

    ple = model.language_model.layers[0].ple_embedding
    assert isinstance(ple, Qwen4ExpTextNGramEmbedding)
    assert ple.ngram_embedding.weight.shape[0] == 1, (
        "placeholder patch did not replace the giant PLE embedding"
    )
    print(f"[smoke] placeholder PLE rows: {ple.ngram_embedding.weight.shape[0]}")

    info = {
        "ple_embed_dim": 160,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "weight_scale": 1.0,
        "layer_multipliers": [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071],
    }

    with tempfile.TemporaryDirectory(prefix="engramdb-official-smoke-") as td:
        root = Path(td)
        (root / "shard_000.bin").write_bytes(b"\x00" * 160)
        store = engramdb.Store(str(root), shards=1, rows_per_shard=1, width=160)
        try:
            replaced = install_disk_ple_in_official_model(
                model,
                store,
                info=info,
                scale=info["weight_scale"],
                cache_size=8,
            )
            print(f"[smoke] replaced: {replaced}")
            assert len(replaced) == 1
            disk = model.language_model.layers[0].ple_embedding
            assert type(disk).__name__ == "DiskPleNGramEmbedding"
        finally:
            store.close()

    print("OFFICIAL_SNAPSHOT_DISK_PLE_STRUCTURE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
