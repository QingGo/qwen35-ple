"""Tests for the P1 exact n-gram PLE bank."""

from __future__ import annotations

import numpy as np

from qwen35_ple.memory.bank import ExactNgramBank


def _toy_bank(max_order: int = 4) -> ExactNgramBank:
    tokens = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
    e_t = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)
    return ExactNgramBank.from_arrays(
        tokens, e_t, min_order=2, max_order=max_order
    )


def test_longest_match_prefers_high_order() -> None:
    bank = _toy_bank(max_order=4)
    # At position 3 (token 40), exact 4-gram is [10,20,30,40], and
    # exact 3-gram is [20,30,40]. The bank must return the 4-gram value.
    mem, orders = bank.lookup(np.array([10, 20, 30, 40], dtype=np.int64))
    assert orders[3] == 4
    np.testing.assert_allclose(mem[3], bank.values[bank.tables[4][(10, 20, 30, 40)]])


def test_miss_falls_back_to_provided_feature() -> None:
    bank = _toy_bank(max_order=3)
    fallback = np.full((3, 3), 7.0, dtype=np.float32)
    mem, orders = bank.lookup(np.array([99, 98, 97], dtype=np.int64), fallback=fallback)
    np.testing.assert_allclose(mem, fallback)
    np.testing.assert_array_equal(orders, np.zeros(3, dtype=np.int64))


def test_control_keeps_keys_but_breaks_association() -> None:
    bank = _toy_bank(max_order=3)
    control = bank.shuffled(seed=0)
    # Same set of n-gram keys must exist.
    for order in range(2, 4):
        assert set(control.tables[order].keys()) == set(bank.tables[order].keys())
    # With more than one entry, at least one key must now point to a different
    # value row.
    if bank.num_entries > 1:
        differ = False
        for order in range(2, 4):
            for key, idx in bank.tables[order].items():
                if not np.array_equal(bank.values[idx], control.values[control.tables[order][key]]):
                    differ = True
                    break
            if differ:
                break
        assert differ


def test_save_load_roundtrip(tmp_path) -> None:
    bank = _toy_bank(max_order=4)
    path = tmp_path / "bank.npz"
    bank.save(path)
    loaded = ExactNgramBank.load(path)
    assert loaded.stats() == bank.stats()
    seq = np.array([10, 20, 30, 40], dtype=np.int64)
    mem_a, ord_a = bank.lookup(seq)
    mem_b, ord_b = loaded.lookup(seq)
    np.testing.assert_allclose(mem_a, mem_b)
    np.testing.assert_array_equal(ord_a, ord_b)


def test_multi_slot_lookup_shape_and_orders() -> None:
    bank = _toy_bank(max_order=4)
    seq = np.array([10, 20, 30, 40], dtype=np.int64)
    mem, orders = bank.lookup_multi(seq)
    assert mem.shape == (4, 3, 3)
    # At position 3, 4-gram, 3-gram and 2-gram are all real matches.
    assert orders[3].tolist() == [4, 3, 2]
    assert mem[3, 0].sum() > 0
