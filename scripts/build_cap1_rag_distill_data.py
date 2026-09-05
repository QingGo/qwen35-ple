#!/usr/bin/env python3
"""Build RAG self-distillation training data.

For each source problem, retrieve similar solved examples from a teacher-style
corpus with BM25 and package them as RAG context.  The target answer remains
the original teacher solution.  This is the first CAP-1 data asset.

Output JSONL records:
    {"category": ..., "problem": ..., "context": ..., "solution": ...}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen35_ple.rag import BM25Index


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def corpus_text(obj: dict) -> str:
    problem = str(obj.get("problem") or obj.get("question") or obj.get("instruction") or "")
    solution = str(obj.get("solution") or obj.get("answer") or obj.get("output") or "")
    return f"{problem}\n{solution}".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/teacher-distill-smoke.jsonl")
    parser.add_argument("--corpus", default="data/sources/distilled_corpus_400k_with_cot-filtered.jsonl")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=2000)
    parser.add_argument("--output", default="data/cap1-rag-distill-smoke.jsonl")
    args = parser.parse_args()

    corpus = load_jsonl(Path(args.corpus))[: args.max_docs]
    docs = [corpus_text(o) for o in corpus if corpus_text(o)]
    bm25 = BM25Index(docs)

    source_items = load_jsonl(Path(args.source))
    if args.limit is not None:
        source_items = source_items[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in source_items:
            problem = str(item.get("problem") or item.get("question") or "")
            solution = str(item.get("solution") or item.get("answer") or "")
            if not problem or not solution:
                continue
            hits = bm25.search(problem, top_k=args.top_k)
            contexts = [docs[i] for i in hits]
            context = "\n\n---\n\n".join(contexts)
            record = {
                "category": str(item.get("category") or "general"),
                "problem": problem,
                "context": context,
                "solution": solution,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    print(f"[cap1] wrote {n} records to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
