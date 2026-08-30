"""Cross-repo golden check: engram-peft QwenPleHashMapping vs EngramDB golden.

This test loads the *source* hashing module directly so it can run in a
dependency-light CI environment, while still pinning the production mapping
inside engram-peft to the same row semantics as EngramDB.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from qwen35_ple.ple_hash import real_spec

ROOT = Path(__file__).resolve().parents[2]
ENGRAMDB_GOLDEN = ROOT / "EngramDB" / "crates" / "engramdb-keygen" / "tests" / "golden.json"
ENGRAM_PEFT_HASHING = (
    ROOT / "engram-peft" / "src" / "engram_peft" / "hashing.py"
)


def _load_engram_peft_hashing():
    if not ENGRAM_PEFT_HASHING.exists():
        pytest.skip(f"engram-peft hashing source not found: {ENGRAM_PEFT_HASHING}")
    # hashing.py only needs torch for type annotations in the hash methods;
    # the QwenPleHashMapping path is pure NumPy, so a tiny stub keeps this
    # cross-repo contract test runnable without a multi-GB torch install.
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = type("Tensor", (), {})
        sys.modules["torch"] = torch_stub
    spec = importlib.util.spec_from_file_location("engram_peft_hashing", ENGRAM_PEFT_HASHING)
    if spec is None or spec.loader is None:
        pytest.skip("cannot load engram-peft hashing source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qwen_ple_mapping_matches_engramdb_golden():
    ep = _load_engram_peft_hashing()
    golden = json.loads(ENGRAMDB_GOLDEN.read_text(encoding="utf-8"))

    mapping = ep.QwenPleHashMapping(
        compressed_vocab_size=248_320,
        ngram_sizes=[2, 3],
        n_head_per_ngram=8,
        layer_ids=[1],
    )

    # Local per-head indices are exactly the golden rowids minus head offsets.
    spec = real_spec()
    tokens = np.array([golden["tokens"]], dtype=np.int64)
    actual = mapping.hash(tokens)[1]
    expected = np.array(
        [
            [row[i] - spec.head_offsets[i] for i in range(16)]
            for row in spec.rowids_for_seq(golden["tokens"])
        ],
        dtype=np.int64,
    )[None, :, :]

    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected)

    # The flattened prime order is exactly what MultiHeadEmbedding needs:
    # bigram heads 0..7 then trigram heads 8..15.
    flat_primes = [p for group in mapping.prime_tables[1] for p in group]
    assert flat_primes == list(golden["prime_sizes"])


def test_create_hash_mapping_selects_qwen_engine():
    ep = _load_engram_peft_hashing()
    mapping = ep.create_hash_mapping(
        compressed_vocab_size=248_320,
        engram_vocab_size_per_ngram=[160_000_000, 160_000_000],
        ngram_sizes=[2, 3],
        n_head_per_ngram=8,
        layer_ids=[1],
        pad_id=2,
        seed=0,
        engine="qwen_ple",
        table_spec="PLE_QWEN_V1",
    )
    assert isinstance(mapping, ep.QwenPleHashMapping)
