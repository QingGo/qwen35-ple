#!/usr/bin/env python3
"""Audit whether evaluation answers appear in a retrieval/training corpus.

This is a simple, transparent leakage check for the RAG baseline and any
future memory bank.  It reports how many QA answers (or their normalized word
sequences) occur verbatim in the corpus.

Usage::

    python scripts/audit_eval_contamination.py \
        --qa-file data/rare-kb-v1.json \
        --corpus data/sources/wikitext.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def _load_qa(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("items") if isinstance(data, dict) else data
    return items


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _load_corpus(path: str, max_docs: int | None) -> list[str]:
    path = Path(path)
    docs: list[str] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(obj.get("text") or obj.get("solution") or "")
                if text.strip():
                    docs.append(text.strip())
    else:
        docs = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
    if max_docs is not None:
        docs = docs[:max_docs]
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-file", default="data/rare-kb-v1.json")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--min-answer-tokens", type=int, default=2)
    parser.add_argument("--output", default="outputs/contamination-audit.json")
    args = parser.parse_args()

    items = _load_qa(args.qa_file)
    docs = _load_corpus(args.corpus, args.max_docs)
    corpus_norm = "\n".join(_normalize(d) for d in docs)

    findings = []
    n_checked = 0
    for item in items:
        answer = str(item.get("answer", "")).strip()
        norm_answer = _normalize(answer)
        if len(norm_answer.split()) < args.min_answer_tokens:
            continue
        n_checked += 1
        found = norm_answer in corpus_norm
        if found:
            findings.append(
                {
                    "id": str(item.get("id", "")),
                    "task": str(item.get("task", "")),
                    "is_rare": bool(item.get("is_rare", False)),
                    "answer": answer,
                }
            )

    summary = {
        "qa_items": len(items),
        "answers_checked": n_checked,
        "answers_found_in_corpus": len(findings),
        "contamination_rate": len(findings) / n_checked if n_checked else 0.0,
        "findings": findings,
        "corpus_docs": len(docs),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "findings"}, indent=2))
    print(f"[audit] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
