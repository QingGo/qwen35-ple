"""Does the evaluation template itself make the memory content-independent?

Why this module exists
----------------------
Round 167 measured that at the position where an answer is generated, every item
in a task addresses a **byte-identical row**, because each QA template ends in a
fixed suffix ("``Answer:``", "``Answer with one word, Yes or No:``").  Where that
holds, the memory's content cannot distinguish items, so a ``real``/``control``
comparison at that position is a **forced null** -- forced by the evaluation
premise, not by the architecture.

That is a large claim about a whole class of evaluations, and it currently rests
on one case.  This module makes it measurable: given the row-id tuple of the
**generation position** for each item, it reports how many distinct tuples there
are and what share of items fall on the modal one.

The general form is not empirical at all.  A row id is a function of at most
three tokens, so:

    if a template appends a FIXED suffix of >= 3 tokens, then the last three
    tokens at the generation position are the same for every item,
    hence the row id is the same for every item.

Tokenizer merging across the question/suffix boundary is the only escape, and
the script measures it rather than assuming it away.  The counterfactual -- the
row at the last token of the *question* -- is the item-specific addressing that
should have been used, and it is measured alongside.

Torch-free and deterministic: unit-tested in ``tests/test_template_audit.py``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "TEMPLATE_DETERMINED_SHARE",
    "AddressDiversity",
    "address_diversity",
    "per_task_diversity",
    "verdict_from_diversity",
    "verdict_from_tasks",
]

#: Modal share at or above which the addressing is called template-determined.
#: A template with a fixed >= 3-token suffix forces exactly 1.0 up to tokenizer
#: merging, so the interesting question is only how far merging pulls it down.
TEMPLATE_DETERMINED_SHARE: float = 0.90


@dataclass(frozen=True)
class AddressDiversity:
    """How item-specific the addressing is at a set of positions."""

    n: int
    n_distinct: int
    modal_share: float
    entropy_bits: float
    entropy_bits_normalised: float
    modal_tuple: tuple[int, ...] | None = None
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_distinct": self.n_distinct,
            "modal_share": self.modal_share,
            "entropy_bits": self.entropy_bits,
            "entropy_bits_normalised": self.entropy_bits_normalised,
            "modal_tuple": list(self.modal_tuple) if self.modal_tuple else None,
            "n_other_tuples": len(self.counts),
        }


def address_diversity(rows: Any) -> AddressDiversity:
    """Diversity of row-id tuples over items.

    ``rows`` is ``[n_items, n_heads]``.  ``modal_share`` is the quantity the
    argument turns on: 1.0 means every item addressed the same row.
    """
    r = np.asarray(rows, dtype=np.int64)
    if r.ndim != 2:
        raise ValueError(f"rows must be [n_items, n_heads], got {r.shape}")
    n = int(r.shape[0])
    if n == 0:
        raise ValueError("no items")
    tuples = [tuple(int(x) for x in row) for row in r]
    counter = Counter(tuples)
    modal_tuple, modal_count = counter.most_common(1)[0]
    probs = np.array([c / n for c in counter.values()], dtype=np.float64)
    entropy = float(-(probs * np.log2(probs)).sum())
    max_entropy = math.log2(n) if n > 1 else 0.0
    return AddressDiversity(
        n=n,
        n_distinct=len(counter),
        modal_share=modal_count / n,
        entropy_bits=entropy,
        entropy_bits_normalised=(entropy / max_entropy) if max_entropy > 0 else 0.0,
        modal_tuple=modal_tuple,
        counts={str(k): int(v) for k, v in counter.most_common(10)},
    )


def per_task_diversity(rows: Any, tasks: Any) -> dict[str, AddressDiversity]:
    """``address_diversity`` per task: collapsing can happen task by task."""
    r = np.asarray(rows, dtype=np.int64)
    t = np.asarray(tasks)
    if t.shape[0] != r.shape[0]:
        raise ValueError(f"tasks length {t.shape[0]} != items {r.shape[0]}")
    out: dict[str, AddressDiversity] = {}
    for task in sorted({str(x) for x in t}):
        mask = np.array([str(x) == task for x in t])
        out[task] = address_diversity(r[mask])
    return out


def verdict_from_diversity(
    generation: AddressDiversity, content: AddressDiversity,
    *, threshold: float = TEMPLATE_DETERMINED_SHARE,
) -> dict[str, Any]:
    """Pre-registered reading of the two positions.

    ``generation`` is the position the answer is produced at (template-suffixed);
    ``content`` is the last token of the question (item-specific).
    """
    collapsed = generation.modal_share >= threshold
    restored = content.modal_share < threshold
    detail = {
        "generation": generation.to_dict(),
        "content": content.to_dict(),
        "threshold": threshold,
    }
    if collapsed and restored:
        return {
            "label": "TEMPLATE_DETERMINES_ADDRESSING",
            "reason": (
                f"at the generation position {generation.modal_share:.1%} of items "
                f"address the modal row ({generation.n_distinct} distinct tuples "
                f"over {generation.n} items) -- a real/control comparison there is a "
                f"null forced by the template; at the last question token the modal "
                f"share falls to {content.modal_share:.1%} "
                f"({content.n_distinct} tuples), so item-specific addressing is "
                "available and was not used"
            ),
            **detail,
        }
    if collapsed:
        return {
            "label": "TEMPLATE_DETERMINES_ADDRESSING",
            "reason": (
                f"the generation position collapses to {generation.modal_share:.1%} "
                f"modal share; the content position does not restore diversity here "
                f"({content.modal_share:.1%}), so the counterfactual is not "
                "established on this data"
            ),
            **detail,
        }
    return {
        "label": "ADDRESSING_IS_ITEM_SPECIFIC",
        "reason": (
            f"the generation position is already item-specific "
            f"({generation.modal_share:.1%} modal share, "
            f"{generation.n_distinct} distinct tuples over {generation.n} items); "
            "the template does not force a null here"
        ),
        **detail,
    }


def verdict_from_tasks(
    generation: dict[str, AddressDiversity],
    content: dict[str, AddressDiversity],
    *, n_template_variants: int | None = None,
    threshold: float = TEMPLATE_DETERMINED_SHARE,
) -> dict[str, Any]:
    """Per-task version of :func:`verdict_from_diversity`.

    Collapse happens *task by task*, and a single global modal share is the wrong
    statistic when a file mixes templates: on our evaluation file the completion
    form uses two suffixes (BoolQ's differs), so the global modal share is 0.667
    while **every task is at 1.000**.  The per-task view is the honest one.

    ``n_template_variants`` adds the sharper statement: if the number of distinct
    rows equals the number of distinct suffixes, the addressing is explained by
    the *template* rather than by the item.
    """
    if not generation:
        raise ValueError("no generation tasks")
    if set(generation) != set(content):
        raise ValueError("generation and content must cover the same tasks")
    worst = min(d.modal_share for d in generation.values())
    worst_task = min(generation, key=lambda k: generation[k].modal_share)
    collapsed = all(d.modal_share >= threshold for d in generation.values())
    restored = all(d.modal_share < threshold for d in content.values())
    suffix_explains = n_template_variants is not None and all(
        d.n_distinct <= n_template_variants for d in generation.values()
    )
    detail = {
        "per_task_generation": {k: v.to_dict() for k, v in generation.items()},
        "per_task_content": {k: v.to_dict() for k, v in content.items()},
        "worst_task": worst_task,
        "worst_task_modal_share": worst,
        "n_template_variants": n_template_variants,
        "threshold": threshold,
    }
    if collapsed and restored:
        extra = (
            " Distinct rows equal distinct suffixes, so the addressing is set by "
            "the template, not by the item."
            if suffix_explains else ""
        )
        return {
            "label": "TEMPLATE_DETERMINES_ADDRESSING",
            "reason": (
                f"every task's items share one row at the generation position "
                f"(worst task {worst_task!r} at {worst:.1%} modal share); at the "
                f"last question token the modal share drops below {threshold:.0%} "
                f"in every task, so item-specific addressing was available and "
                f"unused.{extra}"
            ),
            **detail,
        }
    if collapsed:
        return {
            "label": "TEMPLATE_DETERMINES_ADDRESSING",
            "reason": (
                f"every task collapses at the generation position (worst "
                f"{worst:.1%}), but the content position does not restore "
                "diversity here"
            ),
            **detail,
        }
    return {
        "label": "ADDRESSING_IS_ITEM_SPECIFIC",
        "reason": (
            f"at least one task is already item-specific (worst task "
            f"{worst_task!r} at {worst:.1%} modal share); the template does not "
            "force a null here"
        ),
        **detail,
    }
