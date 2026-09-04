"""Exact longest-match n-gram PLE bank.

This is the P1 read interface described in ``docs/round-50-systematic-plan.md``.

The real PLE table is a multi-hash 2/3-gram table: it is fast and huge, but it
does not provide an exact 4-gram lookup and it can suffer from hash collisions.
An exact bank is a small, auditable complement:

* keys are exact token n-grams (2/3/4);
* values are frozen PLE feature vectors observed for those n-grams;
* lookup returns the longest exact match, falling back to the caller-supplied
  direct PLE feature (or to a zero vector) on a miss.

The control variant is obtained by keeping the same exact keys but permuting the
value rows, which preserves the marginal feature distribution while destroying
the exact n-gram -> feature association.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


class ExactNgramBank:
    """A frozen exact n-gram -> PLE feature bank.

    The class is intentionally implemented with plain Python dicts + NumPy so it
    can be audited, saved/loaded, and unit-tested without the PyTorch/EngramDB
    stack.
    """

    def __init__(
        self,
        values: np.ndarray,
        tables: dict[int, dict[tuple[int, ...], int]] | None = None,
        *,
        min_order: int = 2,
        max_order: int = 4,
    ) -> None:
        if values.ndim != 2:
            raise ValueError(f"values must be 2-D, got shape {values.shape}")
        self.values = np.asarray(values, dtype=np.float32)
        self.min_order = int(min_order)
        self.max_order = int(max_order)
        if self.min_order < 2:
            raise ValueError("exact bank requires min_order >= 2")
        if self.max_order < self.min_order:
            raise ValueError("max_order must be >= min_order")
        self.tables: dict[int, dict[tuple[int, ...], int]] = (
            tables if tables is not None else {}
        )
        for order in range(self.min_order, self.max_order + 1):
            self.tables.setdefault(order, {})

    @property
    def num_orders(self) -> int:
        return self.max_order - self.min_order + 1

    @property
    def num_entries(self) -> int:
        return int(self.values.shape[0])

    @property
    def d_mem(self) -> int:
        return int(self.values.shape[1])

    @classmethod
    def from_arrays(
        cls,
        tokens: np.ndarray | Iterable[int],
        e_t: np.ndarray,
        *,
        min_order: int = 2,
        max_order: int = 4,
    ) -> "ExactNgramBank":
        """Build a bank from a precomputed ``tokens`` / ``PLE e_t`` pair."""
        tokens_arr = np.asarray(tokens, dtype=np.int64).reshape(-1)
        e_t_arr = np.asarray(e_t, dtype=np.float32)
        if e_t_arr.ndim != 2:
            raise ValueError(f"e_t must be [T, d_mem], got {e_t_arr.shape}")
        if len(tokens_arr) != len(e_t_arr):
            raise ValueError(
                f"tokens and e_t length mismatch: {len(tokens_arr)} vs {len(e_t_arr)}"
            )

        value_rows: list[np.ndarray] = []
        tables: dict[int, dict[tuple[int, ...], int]] = {}
        for order in range(min_order, max_order + 1):
            table: dict[tuple[int, ...], int] = {}
            for i in range(order - 1, len(tokens_arr)):
                key = tuple(int(x) for x in tokens_arr[i - order + 1 : i + 1])
                if key not in table:
                    table[key] = len(value_rows)
                    value_rows.append(e_t_arr[i])
            tables[order] = table

        if value_rows:
            values = np.stack(value_rows, axis=0).astype(np.float32)
        else:
            values = np.zeros((0, int(e_t_arr.shape[1])), dtype=np.float32)
        return cls(values, tables, min_order=min_order, max_order=max_order)

    def lookup(
        self,
        tokens: np.ndarray | Iterable[int],
        *,
        fallback: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(best_mem, match_order)`` for a 1-D token sequence.

        ``best_mem`` has shape ``[T, d_mem]`` and is the longest exact-match
        value when available.  ``match_order`` is the matched n-gram order, or 0
        for a fallback/miss.
        """
        tokens_arr = np.asarray(tokens, dtype=np.int64).reshape(-1)
        t = len(tokens_arr)
        if t == 0:
            return (
                np.zeros((0, self.d_mem), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        out = np.zeros((t, self.d_mem), dtype=np.float32)
        orders = np.zeros((t,), dtype=np.int64)
        for i in range(t):
            for order in range(self.max_order, self.min_order - 1, -1):
                if i + 1 < order:
                    continue
                key = tuple(int(x) for x in tokens_arr[i - order + 1 : i + 1])
                idx = self.tables[order].get(key)
                if idx is not None:
                    out[i] = self.values[idx]
                    orders[i] = order
                    break
            else:
                if fallback is not None:
                    fb = np.asarray(fallback, dtype=np.float32)
                    if fb.ndim != 2 or fb.shape[0] != t or fb.shape[1] != self.d_mem:
                        raise ValueError(
                            "fallback must have shape [T, d_mem] and match d_mem"
                        )
                    out[i] = fb[i]
        return out, orders

    def lookup_multi(
        self,
        tokens: np.ndarray | Iterable[int],
        *,
        fallback: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-position memory candidates for every supported order.

        Returns ``(mem, orders)`` where:

        * ``mem`` has shape ``[T, num_orders, d_mem]``;
        * slot 0 corresponds to the highest order, slot ``num_orders-1`` to the
          lowest;
        * a slot is zero when there is no exact match and no fallback is given.

        This is convenient for TokenMem-style multi-slot cross-attention.
        """
        tokens_arr = np.asarray(tokens, dtype=np.int64).reshape(-1)
        t = len(tokens_arr)
        k = self.num_orders
        if t == 0:
            return (
                np.zeros((0, k, self.d_mem), dtype=np.float32),
                np.zeros((0, k), dtype=np.int64),
            )
        out = np.zeros((t, k, self.d_mem), dtype=np.float32)
        orders = np.zeros((t, k), dtype=np.int64)
        fb = None
        if fallback is not None:
            fb = np.asarray(fallback, dtype=np.float32)
            if fb.ndim != 2 or fb.shape[0] != t or fb.shape[1] != self.d_mem:
                raise ValueError(
                    "fallback must have shape [T, d_mem] and match d_mem"
                )
        for i in range(t):
            slot = 0
            for order in range(self.max_order, self.min_order - 1, -1):
                if i + 1 >= order:
                    key = tuple(int(x) for x in tokens_arr[i - order + 1 : i + 1])
                    idx = self.tables[order].get(key)
                    if idx is not None:
                        out[i, slot] = self.values[idx]
                        orders[i, slot] = order
                    elif fb is not None:
                        out[i, slot] = fb[i]
                elif fb is not None:
                    out[i, slot] = fb[i]
                slot += 1
        return out, orders

    def shuffled(self, seed: int = 0) -> "ExactNgramBank":
        """Return a control bank with the same keys but permuted values."""
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.num_entries)
        return ExactNgramBank(
            self.values[perm].copy(),
            {order: dict(table) for order, table in self.tables.items()},
            min_order=self.min_order,
            max_order=self.max_order,
        )

    def save(self, path: str | Path) -> None:
        """Save the bank to a single ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "values": self.values,
            "min_order": np.asarray([self.min_order], dtype=np.int64),
            "max_order": np.asarray([self.max_order], dtype=np.int64),
            "num_entries": np.asarray([self.num_entries], dtype=np.int64),
        }
        for order in range(self.min_order, self.max_order + 1):
            table = self.tables[order]
            if not table:
                arrays[f"order_{order}_keys"] = np.zeros((0, order), dtype=np.int64)
                arrays[f"order_{order}_ids"] = np.zeros((0,), dtype=np.int64)
                continue
            keys = np.asarray(list(table.keys()), dtype=np.int64)
            ids = np.asarray(list(table.values()), dtype=np.int64)
            arrays[f"order_{order}_keys"] = keys
            arrays[f"order_{order}_ids"] = ids
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "ExactNgramBank":
        """Load a bank saved by :meth:`save`."""
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        values = data["values"].astype(np.float32)
        min_order = int(data["min_order"][0])
        max_order = int(data["max_order"][0])
        tables: dict[int, dict[tuple[int, ...], int]] = {}
        for order in range(min_order, max_order + 1):
            keys = data[f"order_{order}_keys"]
            ids = data[f"order_{order}_ids"]
            table = {
                tuple(int(x) for x in keys[i, :]): int(ids[i])
                for i in range(len(keys))
            }
            tables[order] = table
        return cls(values, tables, min_order=min_order, max_order=max_order)

    def stats(self) -> dict[str, int | float]:
        return {
            "num_entries": self.num_entries,
            "d_mem": self.d_mem,
            "min_order": self.min_order,
            "max_order": self.max_order,
            **{
                f"order_{order}_unique": len(self.tables[order])
                for order in range(self.min_order, self.max_order + 1)
            },
        }
