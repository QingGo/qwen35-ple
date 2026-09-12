#!/usr/bin/env python3
"""Round 164: how much of the addressing window is actually about the item?

The graft's contribution ``c_t`` is, by construction, a function of the
addressing window ``w_t`` alone.  This is the premise of the bound proved in
round 163 -- it is why a frozen reader cannot carry more than
``I(future ; w_t | h_t)``.  Round 164 asks the prior question that the bound
makes load-bearing but that nothing has measured:

    **how much item-specific information is in ``w_t`` at all?**

For the production geometry the window is 12 tokens.  The window is taken at
the answer position, so it contains the fixed instruction suffix of the prompt
template.  If the suffix is long relative to the window, then the window -- and
therefore ``c_t`` -- is largely a property of the *template*, not of the query,
and the bound's room ``I(future ; w_t | h_t)`` is correspondingly smaller for
that task.

This script measures that, exactly, from the tokenizer and the item set.  It
needs no model, no table and no GPU.

It also records the feasibility of the *other* natural round-164 design -- a
corpus-count stratifier ("is this window rare in text?") -- by reporting the
count resolution of the reference corpora available locally.  That table is a
negative design result and is reported as such.

Usage::

    python3 scripts/analyze_window_composition.py \
      --items outputs/round162-0.8B-nosft600/arm-wiki.json \
      --tokenizer data/models/Qwen3.5-0.8B \
      --reference data/phase1/PURE_WIKI/tokens.npy \
      --reference data/phase1/PURE_STEM/tokens.npy \
      --reference data/phase1/PURE_CODE/tokens.npy \
      --output outputs/round164/window-composition.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# Production geometry, read off the code rather than assumed:
#   ple_reference.py  pad_len = (conv_kernel_size - 1) * conv_dilation = 9
#   official_ple_snapshot.py  ngram_size = 3  -> each gated_value sees 3 tokens
# so the short-conv output at t depends on tokens t-9-2 .. t = 12 tokens.
WINDOW = 12
NGRAM_SIZE = 3

PROMPT = "Question: {question}\nAnswer:"
BOOLQ_PROMPT = "Question: {question}\nAnswer with one word, Yes or No:"


def _template_for(task: str) -> str:
    return BOOLQ_PROMPT if task == "boolq" else PROMPT


def _suffix_ids(tokenizer, template: str) -> list[int]:
    """Token ids of the fixed part of the prompt, i.e. everything after the item."""
    marker = template.format(question="\x00")
    head, _, tail = marker.partition("\x00")
    assert not head.endswith("\x00")
    return tokenizer.encode(tail, add_special_tokens=False)


def _load_items(path: Path, tokenizer) -> list[dict]:
    """Load the item set that the round-162 arms were actually evaluated on.

    The arm JSONs carry ``qa_gold.answers`` -- the exact 600 items, with task
    and question -- so the window analysis is done on the evaluated set rather
    than on a re-derived one.  ``data/qa-standard/eval.jsonl`` is the original
    source but is not present locally.
    """
    payload = json.loads(path.read_text())
    answers = payload["results"][0]["qa_gold"]["answers"]
    return [{"task": a["task"], "question": a["question"]} for a in answers]


def window_table(tokenizer, items: list[dict]) -> dict:
    per_task: dict[str, dict] = {}
    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for item in items:
        by_task[item["task"]].append(item)

    for task, group in sorted(by_task.items()):
        template = _template_for(task)
        suffix = _suffix_ids(tokenizer, template)
        n_suffix = len(suffix)
        item_slots = max(0, WINDOW - n_suffix)
        prompt_lens = []
        short = 0
        for item in group:
            ids = tokenizer.encode(
                template.format(question=item["question"]), add_special_tokens=False
            )
            prompt_lens.append(len(ids))
            if len(ids) < WINDOW:
                short += 1
        prompt_lens.sort()
        n = len(prompt_lens)
        per_task[task] = {
            "n": n,
            "suffix_tokens": n_suffix,
            "suffix_ids": suffix,
            "window_tokens": WINDOW,
            "item_specific_slots": item_slots,
            "suffix_share": n_suffix / WINDOW,
            "prompt_tokens_min": prompt_lens[0],
            "prompt_tokens_median": prompt_lens[n // 2],
            "prompt_tokens_max": prompt_lens[-1],
            "prompt_shorter_than_window": short,
        }
    return per_task


def stratifier_feasibility(tokenizer, items: list[dict], references: dict[str, list[int]]) -> dict:
    """How much resolution does a corpus-count stratifier actually have?

    For each reference corpus and each n-gram order k, count the fraction of
    items whose *item-specific* window tail (the last k tokens of the window
    that are not the fixed suffix) has a zero count.  A stratifier that puts
    90%+ of items in the zero bucket cannot support a tertile split.
    """
    combined: list[int] = []
    for ids in references.values():
        combined.extend(ids)

    out: dict[str, dict] = {}
    for ref_name, ref_ids in list(references.items()) + [("COMBINED", combined)]:
        ref = list(ref_ids)
        for k in (2, 3, 4):
            if k > len(ref):
                continue
            counts: collections.Counter = collections.Counter()
            for i in range(len(ref) - k + 1):
                counts[tuple(ref[i : i + k])] += 1
            vals = []
            for item in items:
                task = item["task"]
                template = _template_for(task)
                n_suffix = len(_suffix_ids(tokenizer, template))
                ids = tokenizer.encode(
                    template.format(question=item["question"]), add_special_tokens=False
                )
                kk = min(k, max(0, WINDOW - n_suffix))
                if kk <= 0 or len(ids) < n_suffix + kk:
                    vals.append(0)
                    continue
                seg = ids[-(n_suffix + kk) : -n_suffix] if n_suffix else ids[-kk:]
                vals.append(counts.get(tuple(seg), 0))
            zeros = sum(1 for v in vals if v == 0)
            out[f"{ref_name}|k={k}"] = {
                "reference_tokens": len(ref),
                "n": len(vals),
                "zero_count": zeros,
                "zero_fraction": zeros / len(vals),
                "distinct_counts": len(set(vals)),
                "max_count": max(vals) if vals else 0,
            }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True, help="arm JSON carrying qa_gold.answers")
    parser.add_argument("--tokenizer", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--reference", action="append", default=[], help="tokens.npy")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    import numpy as np

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    items = _load_items(Path(args.items), tokenizer)
    print(f"[round164] {len(items)} items from {args.items}", flush=True)

    windows = window_table(tokenizer, items)
    print(f"[round164] window = {WINDOW} tokens; template suffix per task:")
    for task, row in windows.items():
        print(
            f"  {task:10s} n={row['n']:4d} suffix={row['suffix_tokens']:2d}/"
            f"{WINDOW}  item-specific slots={row['item_specific_slots']:2d}  "
            f"prompt len min/med/max = {row['prompt_tokens_min']}/"
            f"{row['prompt_tokens_median']}/{row['prompt_tokens_max']}"
        )

    references: dict[str, list[int]] = {}
    for path in args.reference:
        arr = np.load(path)
        references[Path(path).parent.name or Path(path).stem] = arr.tolist()
    feasibility = {}
    if references:
        feasibility = stratifier_feasibility(tokenizer, items, references)
        print("[round164] corpus-count stratifier resolution:")
        for key, row in feasibility.items():
            print(
                f"  {key:22s} ref={row['reference_tokens']:>9,d} tok  "
                f"zero={row['zero_fraction']*100:5.1f}%  "
                f"distinct={row['distinct_counts']:3d}  max={row['max_count']}"
            )

    payload = {
        "items_file": args.items,
        "tokenizer": args.tokenizer,
        "geometry": {
            "window_tokens": WINDOW,
            "ngram_size": NGRAM_SIZE,
            "source": "ple_reference.py pad_len=(4-1)*3=9 plus ngram_size=3 lookback",
        },
        "windows": windows,
        "stratifier_feasibility": feasibility,
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[round164] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
