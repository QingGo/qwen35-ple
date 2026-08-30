#!/usr/bin/env python3
"""M0 smoke test: EngramDB disk-backed embedding + (optional) TinyLlama e2e.

This script is intentionally a *development harness*, not a benchmark:

* ``--quick`` exercises the EngramDB disk-backed MultiHeadEmbedding against an
  in-memory logical table (no base model required).
* ``--e2e`` additionally tries to build a real engram-peft model with disk
  injection and run a tiny forward/generate pass.

Requirements:
  - engramdb-python (from the sibling EngramDB repo)
  - torch
  - For --e2e: engram-peft + transformers + peft
"""

from __future__ import annotations

import argparse
import struct
import tempfile
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("M0 smoke requires torch") from exc

import engramdb
from engramdb.integrations import DiskMultiHeadEmbedding


def build_synthetic_store(num_rows: int, row_width: int) -> tuple[engramdb.Store, Path]:
    """Create a tiny Store-I with deterministic byte rows."""
    directory = Path(tempfile.mkdtemp(prefix="qwen35-ple-m0-"))
    directory.mkdir(parents=True, exist_ok=True)
    rows = [bytes((j + i) % 251 for j in range(row_width)) for i in range(num_rows)]
    with open(directory / "shard_000.bin", "wb") as f:
        for row in rows:
            f.write(row)
    store = engramdb.Store(str(directory), 1, num_rows, row_width)
    return store, directory


def quick_check() -> None:
    print("[M0] quick disk-backed MultiHeadEmbedding self-check")

    primes = [4, 5, 7]
    per_head = 4  # float32 elements per head
    total = sum(primes)
    row_width = per_head * 4

    store, directory = build_synthetic_store(total, row_width)
    # Write a deterministic logical table, then compare disk fetch to direct index.
    table = torch.arange(total * per_head, dtype=torch.float32).reshape(total, per_head)
    with open(directory / "shard_000.bin", "wb") as f:
        for value in table.reshape(-1).tolist():
            f.write(struct.pack("<f", value))

    disk = DiskMultiHeadEmbedding(primes, per_head, store=store)
    hashes = torch.tensor([[[0, 1, 2], [3, 4, 5]]])
    out = disk(hashes)

    offsets = torch.tensor([0, 4, 9])
    shifted = hashes + offsets
    expected = table[shifted.reshape(-1)].reshape(*shifted.shape, per_head)
    assert torch.equal(out, expected), "disk MultiHeadEmbedding mismatch"
    store.close()
    print("[M0] quick check OK")


def e2e_check(model_name: str, store_dir: str | None, steps: int = 2) -> None:
    print(f"[M0] e2e with base model: {model_name}")
    try:
        import typing
        import typing_extensions
        if not hasattr(typing, "override"):
            typing.override = typing_extensions.override
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from engram_peft import EngramConfig, get_engram_model
    except Exception as exc:  # pragma: no cover - optional e2e dependencies
        raise SystemExit(
            "e2e dependencies unavailable; install engram-peft/transformers/peft "
            f"and retry. ({exc})"
        ) from exc

    if store_dir is None:
        raise SystemExit(
            "--e2e requires --store-dir pointing to a real Store-I directory "
            "(e.g. data/real-rows or an EngramDB Store-I directory). Synthetic tiny stores "
            "cannot cover the real 320M-row PLE rowid space."
        )

    # Real Qwen PLE Store-I: 128 shards x 2,500,012 rows x 160 bytes.
    from engramdb.integrations import install_disk_multi_head_embedding
    store = engramdb.Store(store_dir, 128, 2_500_012, 160)
    install_disk_multi_head_embedding(store)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    config = EngramConfig(
        hidden_size=model.config.hidden_size,
        embedding_dim=2560,
        ngram_sizes=[2, 3],
        n_head_per_ngram=8,
        target_layers=[1],
        engram_vocab_size_per_ngram=[160_000_000, 160_000_000],
        compressed_vocab_size=model.config.vocab_size,
        pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 2,
        tokenizer_name_or_path=model_name,
        engine="qwen_ple",
        table_spec="PLE_QWEN_V1",
        table_source="engramdb:store",
        enable_tokenizer_compression=False,
        hc_mult=1,
        conv_kernel_size=4,
        conv_dilation=3,
        gating_zero_init=True,
        conv_zero_init=True,
    )

    engram_model = get_engram_model(model, config, tokenizer, train_mode="engram_only")
    engram_model.eval()

    ids = tokenizer("Qwen3.5 PLE M0 smoke", return_tensors="pt").input_ids
    with torch.no_grad():
        for _ in range(steps):
            out = engram_model(ids)
            logits = out.logits if hasattr(out, "logits") else out
            assert torch.isfinite(logits).all(), "non-finite logits"
            ids = tokenizer("The meaning of life is", return_tensors="pt").input_ids

    store.close()
    print("[M0] e2e forward/generate smoke OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run only disk embedding check")
    parser.add_argument("--e2e", action="store_true", help="run full TinyLlama/Qwen e2e")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--store-dir", default=None, help="real Store-I dir needed for --e2e")
    args = parser.parse_args()

    if args.e2e:
        e2e_check(args.model, args.store_dir)
    else:
        quick_check()


if __name__ == "__main__":
    main()
