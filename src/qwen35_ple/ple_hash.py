"""Pure-Python reference for the Qwen PLE ``PLE_QWEN_V1`` rowid mapping.

This module is deliberately kept dependency-free and does NOT claim to be the
production implementation.  It serves two purposes in this repository:

* a fast, auditable oracle for golden/contract tests;
* a language-readable specification mirror of ``EngramDB``'s
  ``engramdb-keygen`` and the Qwen reference model.

The authoritative production mapping for training/inference lives in
``EngramDB`` / ``engram-peft`` (see ``docs/integration-contract.md`` C1/C2).
"""

from __future__ import annotations

from dataclasses import dataclass

PLE_MULTIPLIERS: tuple[int, int, int] = (
    23_703_573_157_769,
    20_109_073_645_365,
    8_052_911_324_071,
)

PLE_EOS: int = 248_044
PLE_NGRAM_SIZE: int = 3
PLE_HEADS_PER_NGRAM: int = 8
PLE_HEADS: int = 16
PLE_BASE: int = 20_000_000
PLE_DIVISOR: int = 128
PLE_SHARDS: int = 128
PLE_ROWS_PER_SHARD: int = 2_500_012
PLE_PADDED_ROWS: int = 320_001_536
_U64_MASK: int = (1 << 64) - 1


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    d = 3
    while d * d <= value:
        if value % d == 0:
            return False
        d += 2
    return True


def nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not is_prime(p):
            p += 1
    return p


@dataclass(frozen=True)
class PleSpec:
    """Frozen mirror of EngramDB's ``PleSpec::real``."""

    multipliers: tuple[int, int, int]
    prime_sizes: tuple[int, ...]
    head_offsets: tuple[int, ...]
    total: int
    padded: int
    rows_per_shard: int
    shards: int
    eos: int

    @property
    def total_vocab(self) -> int:
        return self.total

    def rowids_for_seq(self, tokens: list[int] | tuple[int, ...]) -> list[tuple[int, ...]]:
        hist = [self.eos, self.eos, *tokens]
        shifted = [self._shift_right_ignore_eos(hist, shift) for shift in range(PLE_NGRAM_SIZE)]
        ids_all: list[list[int]] = []
        for pos in range(len(hist)):
            row = [0] * PLE_HEADS
            for ngram_order, shift_range in ((2, 0), (3, PLE_HEADS_PER_NGRAM)):
                mixed = (shifted[0][pos] * self.multipliers[0]) & _U64_MASK
                for shifted_row, multiplier in zip(
                    shifted[1:ngram_order], self.multipliers[1:]
                ):
                    mixed ^= (shifted_row[pos] * multiplier) & _U64_MASK
                for h in range(PLE_HEADS_PER_NGRAM):
                    global_head = (ngram_order - 2) * PLE_HEADS_PER_NGRAM + h
                    size = self.prime_sizes[global_head]
                    offset = self.head_offsets[global_head]
                    row[shift_range + h] = (mixed % size) + offset
            ids_all.append(row)
        # The first (ngram_size - 1) positions are context filler; output aligns
        # one row per input token.
        return [tuple(r) for r in ids_all[PLE_NGRAM_SIZE - 1 :]]

    @staticmethod
    def _shift_right_ignore_eos(hist: list[int], shift: int) -> list[int]:
        if shift == 0:
            return hist[:]
        n = len(hist)
        prev_incl: list[int] = []
        last = -1
        for x in hist:
            if x == PLE_EOS:
                last = len(prev_incl)
            prev_incl.append(last)

        out: list[int] = []
        for i in range(n):
            seg_start = 0 if i == 0 else prev_incl[i - 1] + 1
            pos_in_seg = i - seg_start
            src = i - shift
            out.append(hist[src] if pos_in_seg >= shift and src >= 0 else PLE_EOS)
        return out


def real_spec() -> PleSpec:
    prime_sizes = tuple(nth_prime_after(PLE_BASE - 1, i + 1) for i in range(PLE_HEADS))
    head_offsets: list[int] = []
    acc = 0
    for size in prime_sizes:
        head_offsets.append(acc)
        acc += size
    total = acc
    padded = (total + PLE_DIVISOR - 1) // PLE_DIVISOR * PLE_DIVISOR
    assert padded == PLE_PADDED_ROWS
    return PleSpec(
        multipliers=PLE_MULTIPLIERS,
        prime_sizes=prime_sizes,
        head_offsets=tuple(head_offsets),
        total=total,
        padded=padded,
        rows_per_shard=PLE_ROWS_PER_SHARD,
        shards=PLE_SHARDS,
        eos=PLE_EOS,
    )
