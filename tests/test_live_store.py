"""Tests for the reusable disk-first live e_t data flow.

These tests deliberately avoid requiring EngramDB or torch unless they are
cheking the real Store-backed reader.  The dataset/window logic is pure
NumPy and must remain runnable in a dependency-light CI.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qwen35_ple.live_store import (
    LiveETDataset,
    LiveETStore,
    LiveETView,
    LiveETViewStore,
)


def test_live_et_dataset_yields_windows() -> None:
    tokens = np.arange(20, dtype=np.int64)
    e_t = np.arange(20 * 4, dtype=np.float32).reshape(20, 4)
    ds = LiveETDataset(tokens, e_t, seq_len=4, step=4)

    batches = list(ds)
    assert len(batches) == 5
    assert [b.start for b in batches] == [0, 4, 8, 12, 16]
    for b in batches:
        assert b.tokens.shape == (4,)
        assert b.e_t.shape == (4, 4)
        assert b.length == 4
        assert b.rows == 64


def test_live_et_dataset_len_matches_iteration() -> None:
    tokens = np.arange(19, dtype=np.int64)
    e_t = np.zeros((19, 4), dtype=np.float32)
    for step in (1, 2, 5, 7):
        ds = LiveETDataset(tokens, e_t, seq_len=4, step=step)
        assert len(ds) == sum(1 for _ in ds)


def test_live_et_dataset_control_permutes_e_t_stream() -> None:
    tokens = np.arange(6, dtype=np.int64)
    e_t = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)

    real = LiveETDataset(tokens, e_t, seq_len=2, step=2)
    control = LiveETDataset(tokens, e_t, seq_len=2, step=2, control=True, seed=7)

    real_first = next(iter(real)).e_t
    control_first = next(iter(control)).e_t

    # The real stream's first window is tokens 0,1 and e_t rows 0,1.
    np.testing.assert_array_equal(real_first, e_t[:2])
    # A control permutation should normally change row order; with a fixed seed
    # this is reproducible.
    perm = np.random.default_rng(7).permutation(6)
    np.testing.assert_array_equal(control_first, e_t[perm[:2]])


def test_live_et_dataset_worker_sharding() -> None:
    tokens = np.arange(32, dtype=np.int64)
    e_t = np.zeros((32, 2), dtype=np.float32)
    ds = LiveETDataset(tokens, e_t, seq_len=4, step=4, worker_id=1, num_workers=2)
    starts = [b.start for b in ds]
    assert starts == [4, 12, 20, 28]


def test_live_et_view_supports_permutation_and_subset() -> None:
    e_t = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)

    class FakeViewStore:
        def __init__(self) -> None:
            self.rows = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)

        def get(self, indices):
            return self.rows[indices]

    real_view = LiveETView(FakeViewStore(), np.array([0, 2, 4, 6]))
    assert len(real_view) == 4
    np.testing.assert_array_equal(real_view.get(0, 2), e_t[[0, 2]])
    permuted = real_view.permuted(np.array([3, 2, 1, 0]))
    np.testing.assert_array_equal(permuted.get(0, 1), e_t[[6]])
    subset = real_view.subset(np.array([1, 2]))
    np.testing.assert_array_equal(subset.get(0, 2), e_t[[2, 4]])


def test_live_et_store_records_fetch_stats() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("engramdb")

    class FakeStore:
        def __init__(self, width: int) -> None:
            self.width = width

        def fetch(self, rowids):
            out = bytearray()
            for r in rowids:
                out += bytes([(int(r) + j) % 256 for j in range(self.width)])
            return bytes(out)

        def close(self) -> None:
            pass

    rowids = np.arange(5 * 16, dtype=np.int64).reshape(5, 16)
    store = LiveETStore(
        FakeStore(width=4),
        rowids,
        scale=1.0,
        num_heads=16,
        head_dim=4,
        embedding_dim=64,
    )
    try:
        out = store.get(np.array([0, 1, 2]))
        assert out.shape == (3, 64)
        stats = store.stats.as_dict()
        assert stats["windows"] == 1
        assert stats["tokens"] == 3
        assert stats["rows"] == 3 * 16
        assert stats["unique_rows"] == 3 * 16
        assert stats["fetch_seconds"] >= 0
    finally:
        store.close()


def test_live_et_dataset_accepts_live_et_store_view() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("engramdb")

    class FakeStore:
        def __init__(self, width: int) -> None:
            self.width = width

        def fetch(self, rowids):
            out = bytearray()
            for r in rowids:
                out += bytes([(int(r) + j) % 256 for j in range(self.width)])
            return bytes(out)

        def close(self) -> None:
            pass

    rowids = np.arange(7 * 16, dtype=np.int64).reshape(7, 16)
    store = LiveETStore(
        FakeStore(width=4),
        rowids,
        scale=1.0,
        num_heads=16,
        head_dim=4,
        embedding_dim=64,
    )
    try:
        ds = LiveETDataset(np.arange(7, dtype=np.int64), store, seq_len=2, step=2)
        batches = list(ds)
        assert len(batches) == 3
        assert all(b.e_t.shape == (2, 64) for b in batches)
        assert store.stats.windows == 3
    finally:
        store.close()


def test_live_et_store_pickle_reopens_own_store() -> None:
    pytest.importorskip("torch")
    engramdb = pytest.importorskip("engramdb")

    with tempfile.TemporaryDirectory(prefix="live-et-pickle-") as td:
        root = Path(td)
        n = 16
        width = 4
        (root / "shard_000.bin").write_bytes(
            bytes([i % 256 for i in range(n * width)])
        )
        store = engramdb.Store(td, shards=1, rows_per_shard=n, width=width)
        rowids = np.arange(3 * 16, dtype=np.int64).reshape(3, 16) % n
        live = LiveETStore(
            store,
            rowids,
            scale=1.0,
            store_path=td,
            shards=1,
            rows_per_shard=n,
            width=width,
            num_heads=16,
            head_dim=width,
            embedding_dim=64,
        )
        try:
            restored = pickle.loads(pickle.dumps(live))
            assert restored._store_path == td
            out = restored.get(np.array([0, 1]))
            assert out.shape == (2, 64)
        finally:
            live.close()


def test_live_et_view_store_reads_padded_or_raw_slots() -> None:
    pytest.importorskip("torch")

    num_heads = 16
    head_dim = 4
    rec_len = num_heads * head_dim

    class FakeView:
        def __init__(self, rec_len: int) -> None:
            self.rec_len = rec_len

        def slot_bytes(self) -> int:
            return self.rec_len

        def read_records(self, indices):
            return b"".join(self.read_record(i) for i in indices)

        def read_record(self, index: int) -> bytes:
            return bytes([(index + j) % 256 for j in range(self.rec_len)])

    view = LiveETViewStore(
        FakeView(rec_len),
        np.array([0, 2, 4], dtype=np.int64),
        scale=1.0,
        num_heads=num_heads,
        head_dim=head_dim,
        embedding_dim=rec_len,
    )
    try:
        out = view.get(0, 2)
        assert out.shape == (2, rec_len)
        assert view.stats.windows == 1
        sub = view.view(1, 2)
        view.reset_stats()
        assert view.stats.windows == 0
        assert sub.stats.windows == 0
        assert len(sub) == 2
        sub_out = sub.get(0, 2)
        assert sub_out.shape == (2, rec_len)
        ds = LiveETDataset(np.arange(3, dtype=np.int64), view, seq_len=1, step=1)
        assert len(ds) == 3
        batches = list(ds)
        assert all(b.e_t.shape == (1, rec_len) for b in batches)
    finally:
        view.close()
