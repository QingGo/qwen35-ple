#!/usr/bin/env python3
"""Generate the official Qwen4-Exp PLE forward golden fixture.

The fixture is produced solely from the frozen upstream snapshot
(``qwen35_ple.official_ple_snapshot``), not from engram-peft.  It stores:

* a 4096-token input sequence with EOS/segment boundaries,
* the backbone hidden states,
* the official PLE contribution (gated value + short conv),
* the expected residual ``hidden + PLE contribution``,
* the deterministic official PLE weights needed to reconstruct the same
  production \"engram-peft\" layer in tests.

Use:
    python scripts/generate_official_ple_forward_golden.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from qwen35_ple.ple_hash import PLE_MULTIPLIERS

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "golden"
NPZ_PATH = OUT_DIR / "official_ple_forward_4096.npz"
JSON_PATH = OUT_DIR / "official_ple_forward_4096.meta.json"

SEQ_LEN = 4096
HIDDEN_SIZE = 8
EMBEDDING_DIM = 160
PER_HEAD = EMBEDDING_DIM // 16
PRIME_SIZES = [
    17, 19, 23, 29, 31, 37, 41, 43,
    47, 53, 59, 61, 67, 71, 73, 79,
]
CONV_KERNEL_SIZE = 2
CONV_DILATION = 2


def _make_official_layer():
    from qwen35_ple.official_ple_snapshot import Qwen4ExpTextPLELayer

    from types import SimpleNamespace

    config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=8,
        vocab_size=248_320,
        # The first 16 primes after 16 are exactly PRIME_SIZES.
        ngram_vocab_size_base=PRIME_SIZES[0],
        seed=0,
        eos_token_id=[248_044],
        make_ngram_vocab_size_divisible_by=1,
        ple_embed_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        hc_count=1,
        ple_conv_kernel_size=CONV_KERNEL_SIZE,
        rms_norm_eps=1e-5,
    )
    layer = Qwen4ExpTextPLELayer(config, 1, 0)
    # Real PLE_QWEN_V1 uses the checkpoint multipliers; the upstream fresh-init
    # path would derive different numbers unless a matching seed is found.
    layer.ple_embedding.layer_multipliers.data.copy_(
        torch.tensor(PLE_MULTIPLIERS, dtype=torch.long)
    )
    return layer


def main() -> int:
    torch.manual_seed(0)
    layer = _make_official_layer()
    layer.eval()

    ids = torch.tensor(
        [[248_044, 5, 100, 999, 42, 7] * (SEQ_LEN // 6 + 1)]
    )[:, :SEQ_LEN]
    hidden = torch.randn(1, SEQ_LEN, HIDDEN_SIZE)

    with torch.no_grad():
        ple_output = layer(hidden, ids, None)
    expected = hidden + ple_output

    # Save the official weights so tests can reconstruct the equivalent
    # engram-peft layer deterministically from this golden fixture alone.
    emb = layer.ple_embedding.ngram_embedding.weight.detach()
    total_rows = sum(PRIME_SIZES)
    emb = emb[:total_rows].contiguous()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        NPZ_PATH,
        input_ids=ids.numpy(),
        hidden_states=hidden.numpy(),
        ple_output=ple_output.numpy(),
        expected=expected.numpy(),
        embedding_weight=emb.numpy(),
        value_weight=layer.value_proj.weight.detach().numpy(),
        key_weight=layer.key_proj.weight.detach().numpy(),
        conv_weight=layer.conv1d.weight.detach().numpy(),
        norm_key_weight=layer.norm_key.weight.detach().numpy(),
        norm_query_weight=layer.norm_query.weight.detach().numpy(),
        norm_conv_weight=layer.norm_conv.weight.detach().numpy(),
    )

    meta = {
        "seq_len": SEQ_LEN,
        "hidden_size": HIDDEN_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "per_head": PER_HEAD,
        "prime_sizes": PRIME_SIZES,
        "total_rows": total_rows,
        "ngram_sizes": [2, 3],
        "n_head_per_ngram": 8,
        "hc_mult": 1,
        "conv_kernel_size": CONV_KERNEL_SIZE,
        "conv_dilation": CONV_DILATION,
        "multipliers": list(PLE_MULTIPLIERS),
        "eos": 248_044,
        "source": "qwen35_ple.official_ple_snapshot (frozen upstream refs/qwen4_exp_modeling.py)",
    }
    JSON_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {NPZ_PATH}")
    print(f"wrote {JSON_PATH}")
    print(f"sequence length={SEQ_LEN}, hidden={HIDDEN_SIZE}, rows={total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
