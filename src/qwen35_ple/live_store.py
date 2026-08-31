"""Disk-first live PLE feature streams.

This module turns the one-off ``LiveETStore`` / ``LiveETView`` helpers used by
``scripts/run_phase0.py`` into a reusable data-flow layer:

* :class:`LiveETStore` holds only the compact ``[T, 16]`` rowid matrix, never
  the full ``e_t`` matrix;
* :class:`LiveETView` provides lazy, permutation-capable views over that store;
* :class:`LiveETDataset` exposes an iterable window stream that can be consumed
  directly or through ``torch.utils.data.DataLoader``;
* :class:`LiveETBatch` carries per-batch fetch timing and read-volume metrics.

The design follows the Track A rule: memory is not used as a substitute for
disk.  Full ``e_t`` materialization remains an explicit anti-pattern.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence

import numpy as np

try:
    from torch.utils.data import IterableDataset as _IterableDataset, get_worker_info
except Exception:  # pragma: no cover - torch is optional for pure-Python consumers
    _IterableDataset = object  # type: ignore[assignment,misc]
    get_worker_info = None  # type: ignore[assignment]


@dataclass
class FetchStats:
    """Aggregate fetch telemetry for a live store or view."""

    windows: int = 0
    tokens: int = 0
    rows: int = 0
    unique_rows: int = 0
    fetch_seconds: float = 0.0
    cache_hits: int = 0

    def record(
        self,
        *,
        tokens: int,
        rows: int,
        unique_rows: int,
        seconds: float,
        cache_hits: int = 0,
    ) -> None:
        self.windows += 1
        self.tokens += int(tokens)
        self.rows += int(rows)
        self.unique_rows += int(unique_rows)
        self.fetch_seconds += float(seconds)
        self.cache_hits += int(cache_hits)

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def merge(self, other: "FetchStats") -> None:
        self.windows += other.windows
        self.tokens += other.tokens
        self.rows += other.rows
        self.unique_rows += other.unique_rows
        self.fetch_seconds += other.fetch_seconds
        self.cache_hits += other.cache_hits


@dataclass
class LiveETBatch:
    """One training/eval window in the live data flow."""

    tokens: np.ndarray
    e_t: np.ndarray
    start: int
    length: int
    fetch_seconds: float = 0.0
    rows: int = 0
    unique_rows: int = 0
    cache_hits: int = 0

    @property
    def read_rows(self) -> int:
        return self.rows


class LiveETStore:
    """A disk-first reader over an EngramDB Store.

    The store keeps only ``rowids`` in memory; each :meth:`get` fetches exactly
    the requested token slice from EngramDB and dequantizes it on the fly.
    """

    def __init__(
        self,
        store: Any,
        rowids: np.ndarray,
        scale: float = 1.0,
        *,
        num_heads: int = 16,
        head_dim: int = 160,
        embedding_dim: int | None = None,
        record_stats: bool = True,
        store_path: str | None = None,
        shards: int | None = None,
        rows_per_shard: int | None = None,
        width: int | None = None,
    ) -> None:
        self.store = store
        self.rowids = np.asarray(rowids, dtype=np.int64)
        self.scale = float(scale)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        if embedding_dim is None:
            embedding_dim = self.num_heads * self.head_dim
        self.embedding_dim = int(embedding_dim)
        self.record_stats = bool(record_stats)
        self.stats = FetchStats()
        self._closed = False
        # Reopen metadata for PyTorch DataLoader workers.  The native Store is
        # not picklable, so when workers are used each worker must open its own
        # handle from the directory path.
        self._store_path = str(store_path) if store_path is not None else None
        self._shards = int(shards) if shards is not None else None
        self._rows_per_shard = int(rows_per_shard) if rows_per_shard is not None else None
        self._width = int(width) if width is not None else (
            getattr(store, "width", None) if hasattr(store, "width") else None
        )

    def __len__(self) -> int:
        return len(self.rowids)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rowids), self.embedding_dim)

    def get(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """Fetch e_t for a 1-D array of token indices.

        Returns a float32 ``[n, embedding_dim]`` numpy array.  The call is
        recorded in :attr:`stats` when enabled.
        """
        import engramdb

        if self._closed:
            raise ValueError("LiveETStore is closed")
        indices_arr = np.asarray(indices, dtype=np.int64)
        if indices_arr.size == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        rows = self.rowids[indices_arr]
        flat = rows.reshape(-1).tolist()
        unique_count = len(set(flat)) if self.record_stats else 0
        t0 = time.perf_counter()
        arr = engramdb.fetch_e_t_tensor(
            self.store,
            flat,
            scale=self.scale,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=None,
            out_dtype=None,
        )
        elapsed = time.perf_counter() - t0
        result = arr.reshape(len(indices_arr), self.embedding_dim).numpy()
        if self.record_stats:
            self.stats.record(
                tokens=len(indices_arr),
                rows=len(flat),
                unique_rows=unique_count,
                seconds=elapsed,
                cache_hits=max(0, len(flat) - unique_count),
            )
        return result

    def reset_stats(self) -> None:
        self.stats = FetchStats()

    def view(self, start: int = 0, length: int | None = None) -> "LiveETView":
        n = len(self.rowids)
        if length is None:
            length = n - start
        if start < 0 or length < 0 or start + length > n:
            raise IndexError(f"view slice out of range: [{start}:{start + length}] of {n}")
        return LiveETView(self, np.arange(start, start + length, dtype=np.int64))

    def close(self) -> None:
        if not self._closed:
            self.store.close()
            self._closed = True

    def __enter__(self) -> "LiveETStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        if (
            self._store_path is None
            or self._shards is None
            or self._rows_per_shard is None
            or self._width is None
        ):
            raise TypeError(
                "LiveETStore cannot be pickled for DataLoader workers without "
                "store_path/shards/rows_per_shard/width; pass these to the "
                "constructor or use num_workers=0"
            )
        return {
            "rowids": self.rowids,
            "scale": self.scale,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "embedding_dim": self.embedding_dim,
            "record_stats": self.record_stats,
            "stats": self.stats,
            "store_path": self._store_path,
            "shards": self._shards,
            "rows_per_shard": self._rows_per_shard,
            "width": self._width,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        import engramdb

        self.rowids = np.asarray(state["rowids"], dtype=np.int64)
        self.scale = float(state["scale"])
        self.num_heads = int(state["num_heads"])
        self.head_dim = int(state["head_dim"])
        self.embedding_dim = int(state["embedding_dim"])
        self.record_stats = bool(state["record_stats"])
        self.stats = state["stats"]
        self._store_path = state["store_path"]
        self._shards = state["shards"]
        self._rows_per_shard = state["rows_per_shard"]
        self._width = state["width"]
        if (
            self._store_path is None
            or self._shards is None
            or self._rows_per_shard is None
            or self._width is None
        ):
            raise TypeError("LiveETStore state is missing Store reopen parameters")
        self.store = engramdb.Store(
            self._store_path,
            shards=self._shards,
            rows_per_shard=self._rows_per_shard,
            width=self._width,
        )
        self._closed = False


class LiveETView:
    """A lazy slice/permutation view over a :class:`LiveETStore`."""

    def __init__(self, base: LiveETStore, indices: np.ndarray) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def get(self, offset: int, length: int) -> np.ndarray:
        return self.base.get(self.indices[offset : offset + length])

    def __getitem__(self, item: int | slice) -> np.ndarray:
        if isinstance(item, slice):
            idx = self.indices[item]
            return self.base.get(idx)
        return self.base.get(np.array([self.indices[item]], dtype=np.int64))

    def permuted(self, perm: np.ndarray) -> "LiveETView":
        return LiveETView(self.base, self.indices[np.asarray(perm, dtype=np.int64)])

    def subset(self, indices: np.ndarray) -> "LiveETView":
        return LiveETView(self.base, self.indices[np.asarray(indices, dtype=np.int64)])

    @property
    def stats(self) -> FetchStats:
        return self.base.stats

    @property
    def num_heads(self) -> int:
        return self.base.num_heads


class LiveETViewStore:
    """Disk-first reader over a Store-P materialized View.

    This is the Store-P counterpart of :class:`LiveETStore`.  Instead of
    scattering 16 independent Store-I rows per token, it reads one contiguous
    2560-byte (or padded) slot per token.  ``slot_indices`` maps token positions
    to physical view slots (for example, the access-order index used when the
    view was built with ``view build --keys``).
    """

    def __init__(
        self,
        view: Any,
        slot_indices: np.ndarray,
        scale: float = 1.0,
        *,
        num_heads: int = 16,
        head_dim: int = 160,
        embedding_dim: int | None = None,
        record_stats: bool = True,
        stats: FetchStats | None = None,
        dtype: Any = None,
        out_dtype: Any = None,
    ) -> None:
        self.view = view
        self.slot_indices = np.asarray(slot_indices, dtype=np.int64)
        self.scale = float(scale)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        if embedding_dim is None:
            embedding_dim = self.num_heads * self.head_dim
        self.embedding_dim = int(embedding_dim)
        self.record_stats = bool(record_stats)
        self.stats = stats if stats is not None else FetchStats()
        self.dtype = dtype
        self.out_dtype = out_dtype
        self._closed = False

    def __len__(self) -> int:
        return len(self.slot_indices)

    def get(self, offset: int, length: int) -> np.ndarray:
        """Fetch one window using the view-protocol ``(offset, length)`` shape."""
        if self._closed:
            raise ValueError("LiveETViewStore is closed")
        return self._fetch(self.slot_indices[offset : offset + length])

    def get_indices(self, indices: np.ndarray) -> np.ndarray:
        """Fetch by absolute token-position indices."""
        if self._closed:
            raise ValueError("LiveETViewStore is closed")
        return self._fetch(self.slot_indices[np.asarray(indices, dtype=np.int64)])

    def permuted(self, perm: np.ndarray) -> "LiveETViewStore":
        return LiveETViewStore(
            self.view,
            self.slot_indices[np.asarray(perm, dtype=np.int64)],
            self.scale,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            embedding_dim=self.embedding_dim,
            record_stats=self.record_stats,
            stats=self.stats,
            dtype=self.dtype,
            out_dtype=self.out_dtype,
        )

    def subset(self, indices: np.ndarray) -> "LiveETViewStore":
        return LiveETViewStore(
            self.view,
            self.slot_indices[np.asarray(indices, dtype=np.int64)],
            self.scale,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            embedding_dim=self.embedding_dim,
            record_stats=self.record_stats,
            stats=self.stats,
            dtype=self.dtype,
            out_dtype=self.out_dtype,
        )

    def close(self) -> None:
        if not self._closed:
            close = getattr(self.view, "close", None)
            if callable(close):
                close()
            self._closed = True

    def _fetch(self, slot_positions: np.ndarray) -> np.ndarray:
        import torch

        slot_positions = np.asarray(slot_positions, dtype=np.int64)
        n = len(slot_positions)
        if n == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        rec_len = self.num_heads * self.head_dim
        view_slot = int(getattr(self.view, "slot_bytes", rec_len) or rec_len)

        unique_slots = len(set(slot_positions.tolist())) if self.record_stats else 0
        t0 = time.perf_counter()
        if hasattr(self.view, "read_records"):
            raw = self.view.read_records(slot_positions.tolist())
        else:
            raw = b"".join(self.view.read_record(int(i)) for i in slot_positions)
        elapsed = time.perf_counter() - t0

        dtype = self.dtype
        if dtype is None:
            dtype = torch.float8_e4m3fn
        out_dtype = self.out_dtype
        if out_dtype is None:
            out_dtype = torch.float32

        if view_slot != rec_len:
            buf = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(n, view_slot)
            arr = buf[:, :rec_len].reshape(-1)
        else:
            arr = torch.frombuffer(bytearray(raw), dtype=dtype)
        if arr.dtype != out_dtype:
            arr = arr.to(out_dtype)
        if self.scale != 1.0:
            arr = arr * self.scale
        arr = arr.reshape(n, self.num_heads, self.head_dim).reshape(n, self.embedding_dim)

        if self.record_stats:
            unique_rows = unique_slots * self.num_heads
            self.stats.record(
                tokens=n,
                rows=n * self.num_heads,
                unique_rows=unique_rows,
                seconds=elapsed,
                cache_hits=max(0, n * self.num_heads - unique_rows),
            )
        return arr.numpy()


class LiveETDataset(_IterableDataset):  # type: ignore[misc]
    """Iterable live-store window dataset.

    The dataset yields one :class:`LiveETBatch` at a time and never constructs
    the full ``e_t`` array.  It can be used in a plain ``for`` loop or passed to
    ``torch.utils.data.DataLoader``.

    Parameters mirror the existing experimental harness:

    * ``tokens`` is the full token sequence.
    * ``e_t`` may be a ``LiveETStore``, ``LiveETView``, ``LiveETViewStore``,
      or a precomputed numpy array.  For the numpy case the stream works
      identically but records no fetch time.
    * ``control`` permutes the e_t stream relative to the token stream, matching
      the shuffled-control arm.
    * ``shuffle`` randomly orders windows (not tokens) for the current epoch.
    * ``worker_id`` / ``num_workers`` enable explicit sharding; when used with
      ``DataLoader`` and left as ``None`` the dataset shards itself using the
      standard worker info.
    """

    def __init__(
        self,
        tokens: np.ndarray,
        e_t: LiveETStore | LiveETView | LiveETViewStore | np.ndarray,
        *,
        seq_len: int = 128,
        step: int | None = None,
        shuffle: bool = False,
        control: bool = False,
        seed: int = 0,
        worker_id: int | None = None,
        num_workers: int | None = None,
        drop_last: bool = False,
        max_windows: int | None = None,
    ) -> None:
        self.tokens = np.asarray(tokens, dtype=np.int64)
        if len(self.tokens) != len(e_t):
            raise ValueError(
                f"tokens length {len(self.tokens)} != e_t length {len(e_t)}"
            )
        self.seq_len = int(seq_len)
        if self.seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.step = int(step) if step is not None else self.seq_len
        if self.step < 1:
            raise ValueError("step must be >= 1")
        self.shuffle = bool(shuffle)
        self.control = bool(control)
        self.seed = int(seed)
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.drop_last = bool(drop_last)
        self.max_windows = max_windows

        # Normalize stores/views to the ``get(offset, length)`` view protocol so
        # iterating a bare LiveETStore uses the same slicing semantics as a view.
        base_e_t: LiveETStore | LiveETView | LiveETViewStore | np.ndarray = e_t
        if isinstance(base_e_t, LiveETStore):
            base_e_t = base_e_t.view(0, len(self.tokens))

        if control:
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(len(base_e_t))
            if hasattr(base_e_t, "permuted"):
                self.e_t = base_e_t.permuted(perm)  # type: ignore[attr-defined]
            else:
                self.e_t = np.asarray(base_e_t)[perm]
        else:
            self.e_t = base_e_t

    def __len__(self) -> int:
        n = len(self.tokens)
        if n < self.seq_len + 1:
            # The phase0-style tiny-sequence fallback yields one window even
            # when the sequence is shorter than seq_len.
            count = 1 if n > 0 else 0
        else:
            count = (n - self.seq_len) // self.step + 1
        if self.max_windows is not None:
            count = min(count, int(self.max_windows))
        return max(0, count)

    def _window_starts(self) -> np.ndarray:
        n = len(self.tokens)
        if n < self.seq_len + 1:
            # The phase0 harness still allows tiny sequences for a single window.
            starts = np.array([0], dtype=np.int64)
        else:
            limit = n - self.seq_len
            if self.step == 1:
                starts = np.arange(0, limit + 1, dtype=np.int64)
            else:
                starts = np.arange(0, limit + 1, self.step, dtype=np.int64)
        if self.drop_last and len(starts) and starts[-1] + self.seq_len > n:
            starts = starts[:-1]
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            starts = starts[rng.permutation(len(starts))]
        if self.max_windows is not None:
            starts = starts[: int(self.max_windows)]
        return starts

    def _resolve_workers(self) -> tuple[int, int]:
        if self.worker_id is not None and self.num_workers is not None:
            return int(self.worker_id), int(self.num_workers)
        if get_worker_info is not None:
            info = get_worker_info()
            if info is not None:
                return int(info.id), int(info.num_workers)
        return 0, 1

    def __iter__(self) -> Iterator[LiveETBatch]:
        starts = self._window_starts()
        worker_id, num_workers = self._resolve_workers()
        for i, start in enumerate(starts):
            if i % num_workers != worker_id:
                continue
            length = min(self.seq_len, len(self.tokens) - start)
            ids = self.tokens[start : start + length]
            t0 = time.perf_counter()
            if hasattr(self.e_t, "get"):
                et = self.e_t.get(start, length)  # type: ignore[attr-defined]
            else:
                et = np.asarray(self.e_t[start : start + length], dtype=np.float32)
            elapsed = time.perf_counter() - t0
            rows = length * self._head_count()
            # A precise unique-row count is maintained by the LiveETStore when it
            # exists; the batch-level value is a light placeholder for ndarray
            # inputs and is not used for store-backed streams.
            yield LiveETBatch(
                tokens=ids,
                e_t=et,
                start=int(start),
                length=length,
                fetch_seconds=elapsed,
                rows=rows,
                unique_rows=0,
            )

    def _head_count(self) -> int:
        if hasattr(self.e_t, "num_heads"):
            return int(self.e_t.num_heads)  # type: ignore[attr-defined]
        return 16

    @property
    def stats(self) -> FetchStats:
        if hasattr(self.e_t, "stats"):
            return self.e_t.stats  # type: ignore[attr-defined]
        return FetchStats()
