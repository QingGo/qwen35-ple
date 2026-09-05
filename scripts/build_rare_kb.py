#!/usr/bin/env python3
"""Build the Phase-A rare-token knowledge benchmark.

This benchmark is intentionally simple and reproducible:

* takes the existing QA set as the core knowledge questions;
* optionally adds short-answer questions extracted from Alpaca;
* scores each answer by how rare its content words are in a reference
  corpus (default: the 1M-token web sample already in the repo);
* writes a JSON list compatible with ``run_phase0.py --qa-file``, with
  additional ``rarity`` fields used by Phase-A reporting.

The key Phase-A question is whether real PLE row features carry task-level
signal on rare answers beyond the backbone hidden state.  This file only builds
the benchmark; the measurement scripts consume the output.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

CONTENT_WORD_RE = re.compile(r"[a-zA-Z0-9']+")
QUESTION_PREFIX_RE = re.compile(
    r"^(What|Who|When|Where|Which|How|Why|What's|Who's)\b", re.IGNORECASE
)


def _load_qa(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"{path}: expected a JSON list")
    out: list[dict] = []
    for i, item in enumerate(data):
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        out.append(
            {
                "id": f"qa-{i:04d}",
                "source": str(item.get("source", "qa-expanded")),
                "task": str(item.get("task", "qa")),
                "question": question,
                "answer": answer,
            }
        )
    return out


def _load_alpaca(path: Path, max_items: int | None = None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict] = []
    for i, item in enumerate(data):
        inst = str(item.get("instruction", "")).strip().rstrip("?")
        output = str(item.get("output", "")).strip()
        if not inst or not output:
            continue
        if not QUESTION_PREFIX_RE.match(inst):
            continue
        # Keep only concise single-answer outputs.  This avoids open-ended
        # generations that cannot be scored by exact-match.
        if "\n" in output or len(output.split()) > 10:
            continue
        out.append(
            {
                "id": f"alpaca-{i:04d}",
                "source": "alpaca",
                "task": "alpaca-knowledge",
                "question": inst + "?",
                "answer": output,
            }
        )
    if max_items is not None:
        out = out[:max_items]
    return out


def _load_word_counts(path: Path | None) -> Counter:
    if path is None or not path.exists():
        return Counter()
    counts: Counter = Counter()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            counts.update(CONTENT_WORD_RE.findall(line.lower()))
    return counts


def _answer_stats(answer: str, counts: Counter) -> dict:
    words = CONTENT_WORD_RE.findall(answer.lower())
    if not words:
        return {
            "answer_word_min_freq": 0,
            "answer_word_mean_freq": 0.0,
            "answer_content_words": [],
        }
    freqs = [counts.get(w, 0) for w in words]
    return {
        "answer_word_min_freq": int(min(freqs)),
        "answer_word_mean_freq": float(sum(freqs) / len(freqs)),
        "answer_content_words": words,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-file", default="data/qa-expanded-150.json")
    parser.add_argument("--alpaca", default="data/sources/alpaca_data_cleaned.json")
    parser.add_argument("--corpus", default="data/wet-1m-one.txt")
    parser.add_argument("--output", default="data/rare-kb-v1.json")
    parser.add_argument("--max-alpaca", type=int, default=120)
    parser.add_argument("--rare-min-freq", type=int, default=5)
    args = parser.parse_args()

    qa_items = _load_qa(Path(args.qa_file))
    alpaca_items = _load_alpaca(Path(args.alpaca), max_items=args.max_alpaca)
    counts = _load_word_counts(Path(args.corpus))

    all_items = qa_items + alpaca_items
    for item in all_items:
        item.update(_answer_stats(item["answer"], counts))
        item["is_rare"] = item["answer_word_min_freq"] <= args.rare_min_freq
        item["is_common"] = not item["is_rare"]

    # Stable ordering: rare first, then by min frequency ascending.
    all_items.sort(
        key=lambda x: (
            0 if x["is_rare"] else 1,
            x["answer_word_min_freq"],
            x["id"],
        )
    )

    rare = [x for x in all_items if x["is_rare"]]
    common = [x for x in all_items if not x["is_rare"]]
    result = {
        "schema": "rare-kb-v1",
        "description": (
            "Rare-token knowledge benchmark. A question is rare when at least "
            "one content word in the reference answer is absent or very rare "
            "in the reference corpus."
        ),
        "corpus": args.corpus,
        "rare_min_freq": args.rare_min_freq,
        "n_items": len(all_items),
        "n_rare": len(rare),
        "n_common": len(common),
        "items": all_items,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[build_rare_kb] wrote {out}: {len(all_items)} items "
          f"({len(rare)} rare, {len(common)} common)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
