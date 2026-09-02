#!/usr/bin/env python3
"""Strict QA contamination audit for qwen35-ple mixed corpora.

This tool should be run on every newly built training corpus before any
scientific conclusion is drawn.  The user-level invariant:

    Training data should enable semantic alignment with the frozen PLE table.
    It must NOT contain evaluation questions, answers, or complete QA passages.

Checks performed:

* normalized exact answer substring
* normalized exact question substring
* normalized question + answer substring
* answer n-gram / question n-gram / QA n-gram overlap with the corpus

The report is intentionally a *report*, not a hard gate: common answers such as
``yes`` / ``no`` will naturally have many substring hits.  Use the per-row
output and thresholds to decide whether a source must be dropped or patched.

Usage:

    python scripts/audit_contamination.py \
      --qa data/qa-expanded-150.json \
      --corpus data/mixes/M1/corpus.txt \
      --output outputs/contamination-M1.json \
      --n-gram 8
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or"}


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    words = [w for w in text.split() if w and w not in STOPWORDS]
    return " ".join(words)


def ngrams(text: str, n: int) -> set[str]:
    words = text.split()
    if len(words) < n:
        return {text} if text else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def overlap_ratio(haystack: str, needle: str, n: int) -> float:
    """Fraction of needle n-grams that appear in haystack."""
    needle_grams = ngrams(needle, n)
    if not needle_grams:
        return 0.0
    hay_grams = ngrams(haystack, n)
    hits = len(needle_grams & hay_grams)
    return hits / len(needle_grams)


def load_qa(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    raise SystemExit("QA file must be a JSON list or {\"items\": [...]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default="data/qa-expanded-150.json")
    parser.add_argument("--corpus", required=True, help="corpus.txt or similar text file")
    parser.add_argument("--output", default=None, help="JSON report path")
    parser.add_argument("--n-gram", type=int, default=8)
    parser.add_argument("--min-answer-tokens", type=int, default=2,
                        help="answers shorter than this are still reported but not called critical")
    args = parser.parse_args()

    qa = load_qa(Path(args.qa))
    corpus_text = Path(args.corpus).read_text(encoding="utf-8", errors="ignore")
    corpus_norm = normalize_text(corpus_text)
    corpus_grams = ngrams(corpus_norm, args.n_gram)

    rows: list[dict] = []
    summary: dict[str, dict] = {}
    for i, item in enumerate(qa):
        question = str(item.get("question", ""))
        answer = str(item.get("answer", ""))
        task = str(item.get("task", "qa"))
        q_norm = normalize_text(question)
        a_norm = normalize_text(answer)
        qa_norm = normalize_text(question + " " + answer)

        answer_exact = bool(a_norm and a_norm in corpus_norm)
        question_exact = bool(q_norm and q_norm in corpus_norm)
        qa_exact = bool(qa_norm and qa_norm in corpus_norm)

        answer_ngram_hits = len(ngrams(a_norm, args.n_gram) & corpus_grams)
        question_ngram_hits = len(ngrams(q_norm, args.n_gram) & corpus_grams)
        qa_ngram_hits = len(ngrams(qa_norm, args.n_gram) & corpus_grams)
        answer_ngrams = len(ngrams(a_norm, args.n_gram))
        question_ngrams = len(ngrams(q_norm, args.n_gram))
        qa_ngrams = len(ngrams(qa_norm, args.n_gram))
        answer_ratio = answer_ngram_hits / answer_ngrams if answer_ngrams else 0.0
        question_ratio = question_ngram_hits / question_ngrams if question_ngrams else 0.0
        qa_ratio = qa_ngram_hits / qa_ngrams if qa_ngrams else 0.0

        answer_len = len(a_norm.split())
        if qa_exact or (answer_exact and answer_len >= args.min_answer_tokens):
            severity = "critical"
        elif qa_ratio >= 0.9 or answer_ratio >= 0.9:
            severity = "high"
        elif qa_ratio >= 0.6 or question_ratio >= 0.8:
            severity = "medium"
        else:
            severity = "low"

        row = {
            "index": i,
            "task": task,
            "question": question,
            "answer": answer,
            "answer_norm": a_norm,
            "answer_exact": answer_exact,
            "question_exact": question_exact,
            "qa_exact": qa_exact,
            "answer_ngram_hits": answer_ngram_hits,
            "answer_ngram_total": answer_ngrams,
            "answer_ngram_overlap": round(answer_ratio, 4),
            "question_ngram_hits": question_ngram_hits,
            "question_ngram_total": question_ngrams,
            "question_ngram_overlap": round(question_ratio, 4),
            "qa_ngram_hits": qa_ngram_hits,
            "qa_ngram_total": qa_ngrams,
            "qa_ngram_overlap": round(qa_ratio, 4),
            "severity": severity,
        }
        rows.append(row)
        summary.setdefault(task, Counter())
        summary[task][severity] += 1

    total = len(rows)
    sev_counts = Counter(r["severity"] for r in rows)
    summary_out = {
        "total": total,
        "n_gram": args.n_gram,
        "corpus": str(Path(args.corpus).resolve()),
        "severity_counts": dict(sev_counts),
        "critical_rows": [r["index"] for r in rows if r["severity"] == "critical"],
        "high_rows": [r["index"] for r in rows if r["severity"] == "high"],
        "by_task": {
            task: {
                k: v for k, v in sorted(cnt.items())
            }
            for task, cnt in sorted(summary.items())
        },
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"summary": summary_out, "items": rows}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[audit] report -> {out_path}")

    print("\n=== Contamination summary ===")
    print(f"total={total} n_gram={args.n_gram}")
    for sev, cnt in sorted(sev_counts.items()):
        print(f"  {sev:8s} {cnt}")

    print("\n=== Critical/high rows ===")
    for r in rows:
        if r["severity"] in {"critical", "high"}:
            print(
                f"  [{r['index']}] {r['task']} severity={r['severity']} "
                f"answer_exact={r['answer_exact']} qa_exact={r['qa_exact']} "
                f"qa_overlap={r['qa_ngram_overlap']:.3f} "
                f"answer={r['answer'][:60]!r}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
