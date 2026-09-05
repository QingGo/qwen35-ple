"""Tests for PLE-2 addressable n-gram memory."""

from __future__ import annotations

from qwen35_ple.addressable_memory import AddressableNgramMemory


def test_continuation_distribution() -> None:
    mem = AddressableNgramMemory(min_order=2, max_order=4)
    mem.add_sequence([1, 2, 3, 1, 2, 3, 4])
    top = mem.topk([1, 2], k=2)
    assert top
    assert top[0][0] == 3
    assert top[0][1] > 0.5
    # order is at least 2
    assert top[0][2] >= 2


def test_retrieve_values_by_longest_match() -> None:
    mem = AddressableNgramMemory(min_order=2, max_order=4)
    # doc 0 contains the exact 4-gram; doc 1 only contains a shorter overlap.
    mem.add_document([10, 11, 12, 13, 14, 15], value_id=0)
    mem.add_document([99, 11, 12, 13, 14, 15], value_id=1)
    matches = mem.retrieve([10, 11, 12, 13], top_k=2)
    assert matches
    # The exact/longer match should score highest.
    assert matches[0].value_id == 0
    assert matches[0].order >= 3


def test_retrieve_empty_on_miss() -> None:
    mem = AddressableNgramMemory(min_order=2, max_order=4)
    mem.add_sequence([1, 2, 3])
    assert mem.retrieve([9, 9, 9]) == []


def test_control_like_shuffled_sequence_has_value_index() -> None:
    mem = AddressableNgramMemory(min_order=2, max_order=4)
    mem.add_document([1, 2, 3, 4, 5], value_id=7)
    # Every non-trivial context should still find some value because the whole
    # document is indexed under all of its n-grams.
    assert mem.retrieve([1, 2], top_k=1)[0].value_id == 7
