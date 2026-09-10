#!/usr/bin/env python3
"""Build a larger standard QA SFT / held-out eval split.

The original Phase 1 KB split only has 88 SFT / 62 eval items.  This script
streams standard QA datasets from the HuggingFace mirror and writes:

* ``train.jsonl`` : disjoint training questions (default 2000/task);
* ``eval.jsonl``  : official validation questions (default 500/task);
* ``manifest.json``: counts, sources, seed, and exclusion statistics.

Tasks:

* BoolQ  (``google/boolq``)                  -> ``boolq``
* TriviaQA (``mandarjoshi/trivia_qa`` rc.wikipedia) -> ``triviaqa``
* NQ-open (``google-research-datasets/nq_open``)    -> ``nq``

The BoolQ question is stored in the same ``Passage: ...\\nQuestion: ...`` form
as the existing Phase 1 QA files, so the BoolQ prompt override still works.

Usage::

    HF_ENDPOINT=https://hf-mirror.com \
    PYTHONPATH=src python scripts/build_qa_standard_split.py \
      --output-dir data/qa-standard \
      --train-per-task 2000 --eval-per-task 500 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

TASKS = ("boolq", "triviaqa", "nq")


def _norm_question(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_existing(paths: list[Path]) -> set[str]:
    out: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        records: list[dict] = []
        if text.lstrip().startswith("["):
            records = json.loads(text)
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        for item in records:
            question = str(item.get("question", ""))
            # Keep both the full prompt and the trailing question after the last
            # ``Question:`` marker, so BoolQ passage+question forms match too.
            out.add(_norm_question(question))
            if "Question:" in question:
                out.add(_norm_question(question.rsplit("Question:", 1)[-1]))
    return out


def _iter_stream(name: str, split: str):
    from datasets import load_dataset

    if name == "boolq":
        return iter(load_dataset("google/boolq", split=split, streaming=True))
    if name == "triviaqa":
        return iter(
            load_dataset(
                "mandarjoshi/trivia_qa",
                "rc.wikipedia",
                split=split,
                streaming=True,
            )
        )
    if name == "nq":
        return iter(
            load_dataset(
                "google-research-datasets/nq_open",
                split=split,
                streaming=True,
            )
        )
    raise ValueError(name)


def _extract(name: str, row: dict) -> dict | None:
    if name == "boolq":
        answer = row.get("answer")
        if isinstance(answer, bool):
            answer_text = "yes" if answer else "no"
        else:
            answer_text = "yes" if str(answer).strip().lower() in {"true", "yes", "1"} else "no"
        passage = str(row.get("passage", "")).strip()
        question = str(row.get("question", "")).strip()
        if not question:
            return None
        return {
            "task": "boolq",
            "question": f"Passage: {passage}\nQuestion: {question}",
            "answer": answer_text,
            "source": "google/boolq",
        }

    if name == "triviaqa":
        question = str(row.get("question", "")).strip()
        answer = row.get("answer")
        answer_text = ""
        if isinstance(answer, dict):
            answer_text = str(
                answer.get("value")
                or answer.get("normalized_value")
                or (answer.get("aliases") or [""])[0]
            ).strip()
        else:
            answer_text = str(answer or "").strip()
        if not question or not answer_text:
            return None
        return {
            "task": "triviaqa",
            "question": question,
            "answer": answer_text,
            "source": "mandarjoshi/trivia_qa:rc.wikipedia",
        }

    if name == "nq":
        question = str(row.get("question", "")).strip()
        answer = row.get("answer")
        if isinstance(answer, list):
            answer_text = str(answer[0]).strip() if answer else ""
        else:
            answer_text = str(answer or "").strip()
        if not question or not answer_text:
            return None
        return {
            "task": "nq",
            "question": question,
            "answer": answer_text,
            "source": "google-research-datasets/nq_open",
        }
    raise ValueError(name)


def _collect(
    name: str,
    split: str,
    limit: int,
    exclude: set[str],
    seen: set[str],
) -> tuple[list[dict], dict[str, int]]:
    stats = {"scanned": 0, "kept": 0, "excluded": 0, "duplicate": 0, "skipped": 0}
    out: list[dict] = []
    it = _iter_stream(name, split)
    try:
        while len(out) < limit:
            try:
                row = next(it)
            except StopIteration:
                break
            stats["scanned"] += 1
            item = _extract(name, row)
            if item is None:
                stats["skipped"] += 1
                continue
            norm = _norm_question(item["question"])
            if norm in exclude:
                stats["excluded"] += 1
                continue
            if norm in seen:
                stats["duplicate"] += 1
                continue
            seen.add(norm)
            out.append(item)
            stats["kept"] += 1
    finally:
        # Streaming generators can crash during interpreter finalization; the
        # caller writes files before calling os._exit(0).
        del it
    return out, stats


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/qa-standard")
    parser.add_argument("--train-per-task", type=int, default=2000)
    parser.add_argument("--eval-per-task", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[
            "data/qa-expanded-150.json",
            "data/phase1/kb-wiki/qa.train.jsonl",
            "data/phase1/kb-wiki/qa.eval.jsonl",
        ],
    )
    args = parser.parse_args()

    import random

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    exclude = _load_existing([Path(p) for p in args.exclude])
    print(f"[qa-standard] excluded {len(exclude)} normalized existing questions")

    train: list[dict] = []
    eval_rows: list[dict] = []
    manifest: dict[str, Any] = {
        "schema": "qwen35-ple-qa-standard-v1",
        "seed": args.seed,
        "sources": {},
        "counts": {},
        "excluded_existing_questions": len(exclude),
    }
    seen_train: set[str] = set()
    seen_eval: set[str] = set()
    for name in TASKS:
        train_rows, train_stats = _collect(
            name, "train", args.train_per_task, exclude, seen_train
        )
        eval_split = "validation"
        eval_rows_task, eval_stats = _collect(
            name, eval_split, args.eval_per_task, exclude, seen_eval
        )
        rng.shuffle(train_rows)
        rng.shuffle(eval_rows_task)
        train.extend(train_rows)
        eval_rows.extend(eval_rows_task)
        manifest["sources"][name] = {
            "train_split": "train",
            "eval_split": eval_split,
            "train_stats": train_stats,
            "eval_stats": eval_stats,
        }
        manifest["counts"][name] = {
            "train": len(train_rows),
            "eval": len(eval_rows_task),
        }
        print(
            f"[qa-standard] {name}: train={len(train_rows)} eval={len(eval_rows_task)} "
            f"(scanned train={train_stats['scanned']}, eval={eval_stats['scanned']})"
        )

    train_sha = _write_jsonl(out_dir / "train.jsonl", train)
    eval_sha = _write_jsonl(out_dir / "eval.jsonl", eval_rows)
    manifest["files"] = {
        "train": {"path": str(out_dir / "train.jsonl"), "sha256": train_sha},
        "eval": {"path": str(out_dir / "eval.jsonl"), "sha256": eval_sha},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[qa-standard] wrote {out_dir}/train.jsonl ({len(train)} items)")
    print(f"[qa-standard] wrote {out_dir}/eval.jsonl ({len(eval_rows)} items)")
    # Avoid a datasets streaming finalizer crash after all files are written.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
