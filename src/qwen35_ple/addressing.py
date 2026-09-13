"""Stage 1.5f: what the *addressing* step destroys, measured without any reader.

Why this module exists
----------------------
The proven bound (round 167) is

    I(Y ; e_{t-r:t} | h_t)  <=  I(Y ; w_t | h_t) ,      |w_t| <= 3 tokens

and the project's "capability ladder" has four rungs:

    1. the trigram ceiling        I(Y; w_t | h_t)      (round 158, a proxy)
    2. addressing                 trigram -> the 16 row ids
    3. row content                what a probe recovers from the frozen rows
                                                       (round 161: ~59% of top-1)
    4. the read-out               production's reader   (round 167: ~0)

Rung 2 was never isolated, and the natural guess is that it must lose a lot: the
key spaces are enormous (``V^2 = 6.15e10`` bigrams, ``V^3 = 1.53e16`` trigrams)
while the table has 16 heads of ~2.0e7 rows each (``3.200014e8`` in total, 8
bigram-keyed heads and 8 trigram-keyed heads).  A hand probe during round 168
suggested the opposite, and this module is that probe made testable.

The measurement
---------------
Two independent statistics, on the same positions:

* **resolution**: the number of distinct trigrams versus the number of distinct
  16-tuples.  The tuple is a function of the trigram, so
  ``distinct_tuples <= distinct_trigrams`` always; equality means the addressing
  is injective on the sampled trigrams.  Exact, no estimator, no model.
* **continuation top-1**: answer each held-out position from the most frequent
  continuation of its *trigram* key, and again from its *tuple* key, both
  estimated on a training split.  If the addressing is lossless these two
  accuracies must be *identical*; the gap is exactly what the addressing threw
  away, with no entropy estimator to argue about.

Two-way validation (the round-168 TD-14 rule)
--------------------------------------------
An instrument that only ever reports zero proves nothing.  :func:`crushed_spec`
builds a control whose head moduli are divided by a factor, so it collides on
purpose; the same measurements must show a large loss there.  The claim
"addressing is lossless" is only accepted together with "and here is what a
lossy addressing looks like on the same corpus".

Torch-free, deterministic, unit-tested in ``tests/test_addressing.py``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from qwen35_ple.ple_hash import PLE_EOS, PLE_HEADS, PLE_MULTIPLIERS, PleSpec

__all__ = [
    "ResolutionReport",
    "build_examples",
    "crushed_spec",
    "evaluate_addressing",
    "make_spec",
    "most_frequent_continuations",
    "resolution_report",
    "structured_stream",
    "top1_accuracy",
]


def structured_stream(
    *, n_phrases: int = 3000, length: int = 6, vocab: int = 5000,
    repeats: int = 1200, seed: int = 0,
) -> np.ndarray:
    """A stream built from repeated fixed phrases.

    The applicability demonstration for :func:`top1_accuracy`.  On a Zipf-like
    corpus (WikiText: measured trigram top-1 = 8.4%) the modal continuation is
    the same token for nearly every context, so top-1 cannot see a collision at
    all.  On a phrase stream the continuation is genuinely context-determined,
    which is also the regime an exact n-gram memory is supposed to win in.

    Deliberately generated in-process rather than read from ``data/``: the point
    is to be reproducible on any machine without the corpora.
    """
    rng = np.random.default_rng(seed)
    phrases = rng.integers(1, max(vocab, 2), size=(n_phrases, max(length, 2)))
    picked = rng.integers(0, n_phrases, size=max(repeats, 1))
    return phrases[picked].reshape(-1).astype(np.int64)


def make_spec(
    prime_sizes: tuple[int, ...] | list[int],
    *,
    multipliers: tuple[int, int, int] = PLE_MULTIPLIERS,
    eos: int = PLE_EOS,
) -> PleSpec:
    """Build a :class:`PleSpec` from head moduli (offsets/total derived).

    ``PleSpec`` is frozen and its offsets are a running sum, so constructing a
    control by hand is error-prone; deriving them here keeps the control honest.
    """
    sizes = tuple(int(s) for s in prime_sizes)
    if len(sizes) != PLE_HEADS:
        raise ValueError(f"expected {PLE_HEADS} head sizes, got {len(sizes)}")
    if any(s < 2 for s in sizes):
        raise ValueError("every head needs at least two rows")
    offsets: list[int] = []
    total = 0
    for size in sizes:
        offsets.append(total)
        total += size
    return PleSpec(
        multipliers=tuple(multipliers),
        prime_sizes=sizes,
        head_offsets=tuple(offsets),
        total=total,
        padded=total,
        rows_per_shard=max(total, 1),
        shards=1,
        eos=int(eos),
    )


def crushed_spec(factor: int, *, base: PleSpec | None = None) -> PleSpec:
    """A deliberately lossy variant: divide every head's modulus by ``factor``.

    This is the instrument's own control.  If :func:`resolution_report` cannot
    see *this* losing information, it cannot see anything, and a "lossless"
    reading on the real spec would be vacuous.
    """
    if factor < 2:
        raise ValueError("crush factor must be >= 2")
    from qwen35_ple.ple_hash import real_spec

    spec = real_spec() if base is None else base
    return make_spec(
        [max(2, s // int(factor)) for s in spec.prime_sizes],
        multipliers=spec.multipliers,
        eos=spec.eos,
    )


def build_examples(
    tokens: np.ndarray, spec: PleSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(trigram_keys, row_tuples, targets)`` for every usable position.

    Row ``i`` of :meth:`PleSpec.rowids_for_seq` is determined by tokens
    ``t, t-1, t-2`` and is used to predict token ``t+1``, so the trigram key is
    exactly the addressing key and the target is the next token.
    """
    toks = np.asarray(tokens, dtype=np.int64)
    if toks.ndim != 1:
        raise ValueError("tokens must be 1-D")
    if toks.shape[0] < 5:
        raise ValueError("need at least 5 tokens to form one example")
    rows = np.asarray(spec.rowids_for_seq(tuple(int(t) for t in toks)), dtype=np.int64)
    if rows.shape[0] != toks.shape[0]:
        raise ValueError(
            f"rowids_for_seq returned {rows.shape[0]} rows for {toks.shape[0]} tokens"
        )
    # position t runs from 2 (needs t-2) to len-2 (needs t+1 as the target)
    t = np.arange(2, toks.shape[0] - 1, dtype=np.int64)
    keys = (
        (toks[t] * 1_000_003 + toks[t - 1]) * 1_000_003 + toks[t - 2]
    ).astype(np.int64)
    return keys, rows[t], toks[t + 1]


@dataclass(frozen=True)
class ResolutionReport:
    """How much of the trigram the 16-tuple key preserves."""

    n_positions: int
    n_distinct_trigrams: int
    n_distinct_tuples: int
    injective: bool
    per_head_distinct: list[int] = field(default_factory=list)
    per_head_excess: list[int] = field(default_factory=list)

    @property
    def lost_tuples(self) -> int:
        return self.n_distinct_trigrams - self.n_distinct_tuples

    def to_dict(self) -> dict[str, object]:
        return {
            "n_positions": self.n_positions,
            "n_distinct_trigrams": self.n_distinct_trigrams,
            "n_distinct_tuples": self.n_distinct_tuples,
            "lost_tuples": self.lost_tuples,
            "injective": self.injective,
            "per_head_distinct": self.per_head_distinct,
            "per_head_excess": self.per_head_excess,
        }


def resolution_report(trigram_keys: np.ndarray, rows: np.ndarray) -> ResolutionReport:
    """Distinct trigrams vs distinct row tuples, exactly.

    ``per_head_excess[head] = distinct_trigrams - distinct_values_in_that_head``
    is the number of trigrams that head alone failed to keep apart; the tuple
    can still be injective when every individual head collides, which is the
    whole point of having sixteen of them.
    """
    keys = np.asarray(trigram_keys, dtype=np.int64)
    r = np.asarray(rows, dtype=np.int64)
    if keys.ndim != 1:
        raise ValueError("trigram_keys must be 1-D")
    if r.ndim != 2 or r.shape[0] != keys.shape[0]:
        raise ValueError(
            f"rows must be [n, heads] aligned with keys; got {r.shape} vs {keys.shape}"
        )
    distinct_tri = int(np.unique(keys).shape[0])
    distinct_tup = int(np.unique(r, axis=0).shape[0])
    per_head = [int(np.unique(r[:, h]).shape[0]) for h in range(r.shape[1])]
    return ResolutionReport(
        n_positions=int(keys.shape[0]),
        n_distinct_trigrams=distinct_tri,
        n_distinct_tuples=distinct_tup,
        injective=distinct_tup == distinct_tri,
        per_head_distinct=per_head,
        per_head_excess=[distinct_tri - d for d in per_head],
    )


def _group_index(rows: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(rows, axis=0, return_inverse=True)
    return np.asarray(inverse, dtype=np.int64).reshape(-1)


def most_frequent_continuations(
    keys: np.ndarray, targets: np.ndarray,
) -> dict[int, int]:
    """``key -> most frequent next token`` over the training split."""
    k = np.asarray(keys, dtype=np.int64)
    t = np.asarray(targets, dtype=np.int64)
    if k.shape != t.shape:
        raise ValueError("keys and targets must align")
    counter: Counter[tuple[int, int]] = Counter(zip(k.tolist(), t.tolist()))
    best: dict[int, int] = {}
    best_count: dict[int, int] = {}
    for (key, tok), count in counter.items():
        if count > best_count.get(key, 0):
            best_count[key] = count
            best[key] = tok
    return best


def top1_accuracy(
    keys: np.ndarray, targets: np.ndarray, table: dict[int, int],
) -> dict[str, float]:
    """Top-1 accuracy of answering from ``table``; coverage is reported too."""
    k = np.asarray(keys, dtype=np.int64)
    t = np.asarray(targets, dtype=np.int64)
    if k.shape != t.shape:
        raise ValueError("keys and targets must align")
    if k.size == 0:
        return {"accuracy": 0.0, "coverage": 0.0, "n": 0.0}
    picks = np.array([table.get(int(key), -1) for key in k.tolist()], dtype=np.int64)
    hit = picks == t
    covered = picks >= 0
    return {
        "accuracy": float(np.count_nonzero(hit) / k.size),
        "coverage": float(np.count_nonzero(covered) / k.size),
        "n": float(k.size),
    }


def evaluate_addressing(
    tokens: np.ndarray, spec: PleSpec, *, train_fraction: float = 0.6,
) -> dict[str, object]:
    """The full measurement: resolution plus the two top-1 accuracies.

    The train/test split is positional and contiguous, so a repeated trigram
    cannot leak from the training half into the test half through the table
    itself; both tables see exactly the same training half.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    keys, rows, targets = build_examples(tokens, spec)
    n = int(keys.shape[0])
    cut = int(n * train_fraction)
    if cut < 1 or cut >= n:
        raise ValueError(f"split leaves an empty half ({cut} of {n})")
    # The group index must be built over ALL rows and then split.  Indexing the
    # two halves independently would give two unrelated index spaces, and the
    # test half would then be scoring against the wrong table.
    group = _group_index(rows)
    tri_table = most_frequent_continuations(keys[:cut], targets[:cut])
    row_table = most_frequent_continuations(group[:cut], targets[:cut])
    report = resolution_report(keys, rows)
    tri_acc = top1_accuracy(keys[cut:], targets[cut:], tri_table)
    row_acc = top1_accuracy(group[cut:], targets[cut:], row_table)
    return {
        "n_examples": n,
        "train_fraction": train_fraction,
        "resolution": report.to_dict(),
        "trigram_top1": tri_acc,
        "row_tuple_top1": row_acc,
        "top1_gap": tri_acc["accuracy"] - row_acc["accuracy"],
        "spec": {
            "prime_sizes": list(spec.prime_sizes),
            "total_rows": int(spec.total),
            "multipliers": list(spec.multipliers),
        },
    }
