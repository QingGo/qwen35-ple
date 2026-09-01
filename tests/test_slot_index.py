"""Tests for the generic rowid -> Store-P slot semantic index."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from qwen35_ple.live_store import LiveETDataset, LiveETViewStore
from qwen35_ple.slot_index import SlotIndex


def test_slot_index_lookup_and_save_load() -> None:
    rowids = np.array(
        [
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45],
        ],
        dtype=np.int64,
    )
    index = SlotIndex.from_rowids(rowids)
    assert len(index) == 3
    assert index.lookup(rowids[1]) == 1
    assert index.lookup(rowids[2]) == 2
    assert tuple(rowids[0]) in index

    with tempfile.TemporaryDirectory(prefix="slot-index-") as td:
        path = Path(td) / "index.npz"
        index.save(path)
        restored = SlotIndex.load(path)
        assert restored.lookup(rowids[1]) == 1
        assert restored.lookup_all(rowids[2]) == [2]


def test_slot_index_duplicates_pick_representative() -> None:
    row = np.arange(16, dtype=np.int64)
    rowids = np.stack([row, row + 100, row], axis=0)
    index = SlotIndex.from_rowids(rowids, np.array([5, 7, 9], dtype=np.int64))
    assert index.lookup(row) in (5, 9)
    assert sorted(index.lookup_all(row)) == [5, 9]
    mapped = index.to_slots(np.stack([row, row + 100], axis=0))
    assert len(mapped) == 2
    assert mapped[1] == 7


def test_slot_index_from_keys_file() -> None:
    with tempfile.TemporaryDirectory(prefix="slot-keys-") as td:
        path = Path(td) / "keys.txt"
        path.write_text("\n".join(str(i) for i in range(32)) + "\n", encoding="utf-8")
        index = SlotIndex.from_keys_file(path, heads=16)
        assert len(index) == 2
        assert index.lookup(tuple(range(16))) == 0
        assert index.lookup(tuple(range(16, 32))) == 1


def _install_fake_fetch(view_store: LiveETViewStore, rec_len: int, read_log: list[int] | None = None):
    """Replace the torch-backed _fetch with a pure-NumPy stand-in."""

    def fake_fetch(positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.int64)
        if read_log is not None:
            read_log.extend(int(x) for x in positions)
        return np.stack([np.full(rec_len, float(p), dtype=np.float32) for p in positions])

    view_store._fetch = fake_fetch  # type: ignore[method-assign]
    return fake_fetch


def test_access_order_scheduling_reads_sorted_and_returns_same_values() -> None:
    num_heads = 16
    head_dim = 2
    rec_len = num_heads * head_dim
    slots = np.array([5, 1, 3, 0], dtype=np.int64)

    store = LiveETViewStore(
        None,
        slots,
        num_heads=num_heads,
        head_dim=head_dim,
        embedding_dim=rec_len,
        access_order=True,
    )
    read_log: list[int] = []
    _install_fake_fetch(store, rec_len, read_log)
    out = store.get(0, 4)
    assert out.shape == (4, rec_len)
    # get_sorted reads physical slots in sorted order.
    assert read_log == [0, 1, 3, 5]

    plain = LiveETViewStore(
        None,
        slots,
        num_heads=num_heads,
        head_dim=head_dim,
        embedding_dim=rec_len,
    )
    _install_fake_fetch(plain, rec_len)
    np.testing.assert_array_equal(out, plain.get(0, 4))


def test_live_dataset_access_order_option() -> None:
    num_heads = 16
    head_dim = 2
    rec_len = num_heads * head_dim
    slots = np.array([3, 1, 2, 0, 4], dtype=np.int64)

    view_store = LiveETViewStore(
        None,
        slots,
        num_heads=num_heads,
        head_dim=head_dim,
        embedding_dim=rec_len,
        access_order=True,
    )
    read_log: list[int] = []
    _install_fake_fetch(view_store, rec_len, read_log)

    ds = LiveETDataset(
        np.arange(5, dtype=np.int64),
        view_store,
        seq_len=2,
        step=2,
        access_order=True,
    )
    batches = list(ds)
    assert len(batches) == 2
    # access_order also schedules windows by their minimum physical slot:
    # window starting at 2 has slots [2,0] so it is yielded first.
    assert read_log[:2] == [0, 2]
    plain = LiveETViewStore(
        None,
        slots,
        num_heads=num_heads,
        head_dim=head_dim,
        embedding_dim=rec_len,
    )
    _install_fake_fetch(plain, rec_len)
    np.testing.assert_array_equal(batches[0].e_t, plain.get(2, 2))
    assert read_log[2:4] == [1, 3]


def test_live_et_view_store_from_slot_index() -> None:
    num_heads = 16
    head_dim = 2
    rec_len = num_heads * head_dim
    rowids = np.array(
        [
            list(range(16)),
            list(range(16, 32)),
            list(range(32, 48)),
        ],
        dtype=np.int64,
    )
    index = SlotIndex.from_rowids(rowids, np.array([10, 20, 30], dtype=np.int64))
    # A request stream that repeats a row plus one new row.
    query_rows = np.stack([rowids[0], rowids[2], rowids[0]], axis=0)
    store = LiveETViewStore.from_slot_index(
        None,
        query_rows,
        index,
        num_heads=num_heads,
        head_dim=head_dim,
        embedding_dim=rec_len,
    )
    np.testing.assert_array_equal(store.slot_indices, np.array([10, 30, 10], dtype=np.int64))


def test_slot_index_from_view_manifest() -> None:
    with tempfile.TemporaryDirectory(prefix="slot-manifest-") as td:
        root = Path(td)
        view_path = root / "corpus.view"
        keys_path = root / "corpus.keys.txt"
        keys_path.write_text(
            "\n".join(str(i) for i in range(32)) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "grans": 2,
            "heads": 16,
            "slot_bytes": 2560,
            "record_bytes": 2560,
            "build_seconds": 0.0,
            "build_mb_s": 0.0,
            "rows": 32,
            "source": "provided-keys:32",
            "layout": "access-order",
            "keys_out": str(keys_path),
        }
        (root / "corpus.manifest.json").write_text(
            __import__("json").dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        index = SlotIndex.from_view_manifest(view_path)
        assert len(index) == 2
        assert index.lookup(tuple(range(16))) == 0
        assert index.lookup(tuple(range(16, 32))) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
