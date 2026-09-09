#!/usr/bin/env python3
"""Build an oracle-context QA file for upper-bound probing.

The Phase 2 reader injections are evaluated with a closed-book prompt.  To
separate "knowledge is missing" from "the model cannot use knowledge", we also
need an oracle-context arm: put the gold answer (or a gold passage) directly in
the prompt and measure whether the model can copy/use it.

Usage::

    python scripts/make_oracle_context_qa.py \
      --input data/phase1/kb-wiki/qa.eval.jsonl \
      --output data/qa-oracle-answer-eval62.json \
      --mode answer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    records: list[dict] = []
    if text.lstrip().startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise SystemExit(f"{path}: expected a JSON list")
        records = data
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    out = []
    for item in records:
        if "question" not in item or "answer" not in item:
            raise SystemExit(f"{path}: item missing question/answer")
        out.append(
            {
                "task": str(item.get("task", "qa")),
                "question": str(item["question"]),
                "answer": str(item["answer"]),
                "context": str(item.get("context", item.get("passage", ""))),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=("answer", "passage", "answer_and_question"),
        default="answer",
    )
    args = parser.parse_args()

    items = _load(Path(args.input))
    out: list[dict] = []
    for item in items:
        if args.mode == "answer":
            context = f"The answer to the question is {item['answer']}."
        elif args.mode == "passage":
            context = item["context"] or f"The answer is {item['answer']}."
        else:
            context = (
                f"Relevant fact: {item['answer']}.\n"
                f"Question: {item['question']}"
            )
        question = (
            f"Context: {context}\n"
            f"Question: {item['question']}"
        )
        out.append(
            {
                "task": item["task"],
                "question": question,
                "answer": item["answer"],
                "oracle_context": context,
            }
        )

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(out)} items, mode={args.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
