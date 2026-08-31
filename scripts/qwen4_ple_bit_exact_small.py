#!/usr/bin/env python3
"""Phase B2 bit-exact smoke: frozen official PLE vs DiskPleNGramEmbedding.

This uses a small synthetic PLE table (16 heads, small prime sizes) so it runs
on a normal machine without loading the real multi-GB embeddings.  It compares
the frozen official ``Qwen4ExpTextNGramEmbedding`` against EngramDB's
``DiskPleNGramEmbedding`` for:

* batched input,
* EOS token in the middle of a sequence,
* multi-token n-gram context,
* chunked/streaming calls with per-batch context,
* both memory and disk paths using the same deterministic table.

Usage:

    PYTHONPATH=src:../EngramDB/python \\
    python scripts/qwen4_ple_bit_exact_small.py
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

import engramdb
from engramdb.ple_adapter import (
    DiskPleNGramEmbedding,
    head_offsets,
    head_vocab_sizes,
    padded_vocab_size,
)
from qwen35_ple.official_ple_snapshot import Qwen4ExpTextNGramEmbedding


def main() -> int:
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

    with tempfile.TemporaryDirectory(prefix="engramdb-ple-bit-exact-") as td:
        root = Path(td)
        with open(root / "shard_000.bin", "wb") as f:
            for i in range(total_padded):
                f.write(struct.pack("<f", float(i)))

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
            print(f"[bit-exact] batch shapes {tuple(actual.shape)} maxdiff 0.0")

            # Chunked/streaming path: feed the same sequence in two calls and
            # make sure the adapter's per-batch n-gram context preserves the
            # exact same rowids as a single full-sequence official forward.
            stream_tokens = torch.tensor([[1, 2, 3, 4, eos, 6]], dtype=torch.long)
            with torch.no_grad():
                expected_stream = official(stream_tokens, None)[0]
                disk.reset_history()
                part1 = disk(stream_tokens[:, :2], None)
                part2 = disk(stream_tokens[:, 2:], None)
                actual_stream = torch.cat([part1[0], part2[0]], dim=0)

            torch.testing.assert_close(actual_stream, expected_stream, atol=0, rtol=0)
            print("[bit-exact] streaming maxdiff 0.0")
        finally:
            store.close()

    print("OFFICIAL_DISK_PLE_BIT_EXACT_SMALL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
