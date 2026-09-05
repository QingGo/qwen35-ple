"""Tests for the PLE-2 serving router / logit processor."""

from __future__ import annotations

import numpy as np

from qwen35_ple.addressable_memory import AddressableNgramMemory
from qwen35_ple.router import CalibratedNgramLogitProcessor


def _memory() -> AddressableNgramMemory:
    mem = AddressableNgramMemory(min_order=2, max_order=4)
    mem.add_sequence([1, 2, 3, 1, 2, 3, 4])
    return mem


def test_processor_disabled_returns_same() -> None:
    proc = CalibratedNgramLogitProcessor(_memory(), enabled=False)
    logits = np.zeros(10, dtype=np.float32)
    out = proc(logits, [1, 2])
    assert np.array_equal(out, logits)


def test_processor_applies_ngram_bias() -> None:
    proc = CalibratedNgramLogitProcessor(
        _memory(), scale=1.0, bias=3.0, temperature=1.0
    )
    logits = np.zeros(10, dtype=np.float32)
    out = proc(logits, [1, 2])
    # Token 3 is the dominant continuation; with positive bias it should become
    # the argmax among ngram candidates and above untouched tokens.
    assert out[3] > 0.0
    assert int(np.argmax(out)) == 3


def test_processor_state_dict() -> None:
    proc = CalibratedNgramLogitProcessor(_memory(), scale=0.5, bias=-1.0, temperature=2.0)
    state = proc.state_dict()
    assert state["scale"] == 0.5
    assert state["bias"] == -1.0
    assert state["temperature"] == 2.0
    assert state["enabled"] is True
