#!/usr/bin/env python3
"""Same-protocol RAG baseline for the rare-token QA benchmark.

This is the pivot comparison after P1.  It does **not** use PLE at all.
Instead it retrieves a few text passages from an external corpus and prepends
them to the question, then measures the frozen backbone's answer-token
log-probability with the same metric used by ``eval_p1_memory.py``.

Usage::

    python scripts/run_rag_baseline.py \
        --model data/models/Qwen3.5-0.8B \
        --corpus data/sources/wikitext.jsonl \
        --qa-file data/rare-kb-v1.json \
        --top-k 3 \
        --output outputs/rag-baseline.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _load_model(model_path: str, device: str):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model


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


class BM25Index:
    """Tiny, dependency-free BM25 index."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = [_tokenize(d) for d in docs]
        self.doc_len = np.asarray([len(t) for t in self.doc_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 1.0
        self.df: Counter[str] = Counter()
        self.tf: list[Counter[str]] = []
        for tokens in self.doc_tokens:
            c = Counter(tokens)
            self.tf.append(c)
            for term in c:
                self.df[term] += 1
        self.n = len(docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 3) -> list[int]:
        q_tokens = _tokenize(query)
        scores = np.zeros(self.n, dtype=np.float64)
        for term in q_tokens:
            idf = self._idf(term)
            if idf <= 0.0:
                continue
            for i, c in enumerate(self.tf):
                tf = c.get(term, 0)
                if tf:
                    denom = tf + self.k1 * (
                        1.0 - self.b + self.b * self.doc_len[i] / self.avgdl
                    )
                    scores[i] += idf * tf * (self.k1 + 1.0) / denom
        if len(scores) == 0:
            return []
        k = min(top_k, len(scores))
        return np.argsort(scores)[-k:][::-1].tolist()


def _load_qa(path: str, limit: int | None, offset: int = 0) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("items") if isinstance(data, dict) else data
    if offset:
        items = items[offset:]
    if limit is not None:
        items = items[:limit]
    return items


def _answer_logprob(logits: torch.Tensor, ids: np.ndarray, answer_start: int) -> float:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    total = 0.0
    n = 0
    start = max(0, answer_start - 1)
    for t in range(start, len(ids) - 1):
        total += float(log_probs[t, int(ids[t + 1])])
        n += 1
    return total / max(1, n)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--corpus", default="data/sources/wikitext.jsonl")
    parser.add_argument("--qa-file", default="data/rare-kb-v1.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/rag-baseline.json")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = _load_model(args.model, args.device)
    docs = _load_corpus(args.corpus, args.max_docs)
    print(f"[rag] corpus docs={len(docs)}", flush=True)
    index = BM25Index(docs)
    items = _load_qa(args.qa_file, args.limit, args.offset)
    print(f"[rag] items={len(items)}", flush=True)

    results: list[dict] = []
    for idx, item in enumerate(items):
        question = str(item.get("question", ""))
        answer = str(item.get("answer", ""))
        ctx_ids = index.search(question, args.top_k)
        context = "\n\n".join(docs[i] for i in ctx_ids)
        qids = tokenizer.encode(question, add_special_tokens=False)
        ans_ids = tokenizer.encode(answer, add_special_tokens=False)
        if (not ans_ids) or (not qids):
            continue

        # No-context condition.
        full_ids = qids + ans_ids
        ids_t = torch.tensor([full_ids], dtype=torch.long, device=args.device)
        with torch.no_grad():
            out = model(input_ids=ids_t, use_cache=False)
        no_ctx_lp = _answer_logprob(out.logits[0], np.asarray(full_ids), len(qids))

        # RAG condition: context + question + answer.
        if context:
            ctx_ids_tok = tokenizer.encode(context, add_special_tokens=False)
            rag_ids = ctx_ids_tok + qids + ans_ids
            rag_start = len(ctx_ids_tok) + len(qids)
            rag_t = torch.tensor([rag_ids], dtype=torch.long, device=args.device)
            with torch.no_grad():
                out_rag = model(input_ids=rag_t, use_cache=False)
            rag_lp = _answer_logprob(out_rag.logits[0], np.asarray(rag_ids), rag_start)
        else:
            rag_lp = no_ctx_lp

        results.append(
            {
                "id": str(item.get("id", idx)),
                "task": str(item.get("task", "qa")),
                "is_rare": bool(item.get("is_rare", False)),
                "answer": answer,
                "retrieved": ctx_ids,
                "no_context_answer_logprob": no_ctx_lp,
                "rag_answer_logprob": rag_lp,
                "delta": rag_lp - no_ctx_lp,
            }
        )
        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{len(items)}]", flush=True)

    # Summaries.
    def avg(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else float("nan")

    summary: dict = {}
    rare = [r for r in results if r["is_rare"]]
    common = [r for r in results if not r["is_rare"]]
    for name, group in [("all", results), ("rare", rare), ("common", common)]:
        deltas = [r["delta"] for r in group]
        summary[name] = {
            "n": len(group),
            "no_context_answer_logprob": avg([r["no_context_answer_logprob"] for r in group]),
            "rag_answer_logprob": avg([r["rag_answer_logprob"] for r in group]),
            "delta_mean": avg(deltas),
            "delta_std": float(np.std(deltas)) if deltas else float("nan"),
            "delta_wins": int(sum(1 for d in deltas if d > 0)),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": vars(args),
                "summary": summary,
                "results": results,
                "runtime_seconds": time.time() - t0,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[rag] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
