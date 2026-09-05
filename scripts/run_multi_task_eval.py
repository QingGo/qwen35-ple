#!/usr/bin/env python3
"""Small multi-task evaluation for the 0.8B baseline and future RAG/distillation.

Current task mix:

* knowledge: rare-kb QA;
* arithmetic: generated arithmetic questions;
* code-output: simple Python expression evaluation questions.

The script measures answer-token log-probability and first-token hit.  It also
computes an RAG condition for knowledge questions when a retrieval corpus is
provided.

Usage::

    python scripts/run_multi_task_eval.py \
        --model data/models/Qwen3.5-0.8B \
        --qa-file data/rare-kb-v1.json \
        --corpus data/sources/wikitext.jsonl \
        --device cuda \
        --output outputs/multi-task-eval.json
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


def _load_model(model_path: str, device: str, adapter: str | None = None):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
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


def _build_arithmetic(n: int) -> list[dict]:
    rng = np.random.default_rng(123)
    items = []
    for _ in range(n):
        a = int(rng.integers(2, 100))
        b = int(rng.integers(2, 100))
        op = rng.choice([" + ", " - ", " * "])
        expr = f"{a}{op}{b}"
        ans = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
        items.append({
            "id": f"arith-{len(items)}",
            "task": "arithmetic",
            "question": f"What is {expr}?",
            "answer": str(ans),
        })
    return items


def _build_code_output(n: int) -> list[dict]:
    templates = [
        ("What does `1 + 2 * 3` evaluate to?", "7"),
        ("What does `len([1, 2, 3])` evaluate to?", "3"),
        ("What does `2 ** 3` evaluate to?", "8"),
        ("What does `10 // 3` evaluate to?", "3"),
        ("What does `10 % 3` evaluate to?", "1"),
        ("What does `1 + 2 == 3` evaluate to?", "True"),
        ("What does `3 > 2` evaluate to?", "True"),
        ("What does `'abc'[1]` evaluate to?", "b"),
        ("What does `[1, 2, 3][-1]` evaluate to?", "3"),
        ("What does `max(3, 7, 2)` evaluate to?", "7"),
    ]
    out = []
    for i in range(n):
        q, a = templates[i % len(templates)]
        out.append({
            "id": f"code-{i}",
            "task": "code-output",
            "question": q,
            "answer": a,
        })
    return out


def _load_qa(path: str, limit: int | None) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("items") if isinstance(data, dict) else data
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


def _first_hit(logits: torch.Tensor, ids: np.ndarray, answer_start: int) -> bool:
    t = max(0, answer_start - 1)
    if t >= len(ids) - 1:
        return False
    return int(torch.argmax(logits[t])) == int(ids[t + 1])



def _greedy_generate(
    model,
    tokenizer,
    input_ids: list[int],
    *,
    max_new_tokens: int = 16,
    device: str,
) -> str:
    generated = list(input_ids)
    for _ in range(max_new_tokens):
        ids = torch.tensor([generated], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False).logits[0, -1]
        nxt = int(torch.argmax(logits))
        if nxt == tokenizer.eos_token_id:
            break
        generated.append(nxt)
    return tokenizer.decode(generated[len(input_ids):], skip_special_tokens=True).strip()


def _normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text


def _exact_match(pred: str, gold: str) -> bool:
    return _normalize_answer(pred) == _normalize_answer(gold)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--adapter", default=None, help="optional PEFT adapter directory")
    parser.add_argument("--qa-file", default="data/rare-kb-v1.json")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-docs", type=int, default=20000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-arith", type=int, default=20)
    parser.add_argument("--n-code", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", default="outputs/multi-task-eval.json")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = _load_model(args.model, args.device, adapter=args.adapter)

    items = _load_qa(args.qa_file, args.limit)
    items = [
        {
            "id": f"kb-{i}",
            "task": "knowledge",
            "question": str(x.get("question", "")),
            "answer": str(x.get("answer", "")),
            "is_rare": bool(x.get("is_rare", False)),
        }
        for i, x in enumerate(items)
    ]
    items += _build_arithmetic(args.n_arith)
    items += _build_code_output(args.n_code)

    index = None
    if args.corpus:
        docs = _load_corpus(args.corpus, args.max_docs)
        index = BM25Index(docs)
        print(f"[multi] retrieval docs={len(docs)}", flush=True)

    print(f"[multi] items={len(items)}", flush=True)
    results: list[dict] = []
    for idx, item in enumerate(items):
        question = item["question"]
        answer = item["answer"]
        qids = tokenizer.encode(question, add_special_tokens=False)
        ans_ids = tokenizer.encode(answer, add_special_tokens=False)
        if not qids or not ans_ids:
            continue
        full_ids = qids + ans_ids
        ids = torch.tensor([full_ids], dtype=torch.long, device=args.device)
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
            base_lp = _answer_logprob(out.logits[0], np.asarray(full_ids), len(qids))
            base_hit = _first_hit(out.logits[0], np.asarray(full_ids), len(qids))

        base_greedy = _greedy_generate(
            model, tokenizer, qids, max_new_tokens=args.max_new_tokens, device=args.device
        )
        entry = {
            "id": item["id"],
            "task": item["task"],
            "is_rare": item.get("is_rare", False),
            "no_context_answer_logprob": base_lp,
            "no_context_first_hit": bool(base_hit),
            "no_context_greedy": base_greedy,
            "no_context_exact": _exact_match(base_greedy, answer),
            "rag_answer_logprob": None,
            "rag_first_hit": None,
            "rag_greedy": None,
            "rag_exact": None,
        }

        if index is not None and item["task"] == "knowledge":
            ctx_ids = index.search(question, args.top_k)
            context = "\n\n".join(index.docs[i] for i in ctx_ids)
            if context:
                ctx_tok = tokenizer.encode(context, add_special_tokens=False)
                rag_input = ctx_tok + qids
                rag_start = len(ctx_tok) + len(qids)
                rag_full = rag_input + ans_ids
                rag_t = torch.tensor([rag_full], dtype=torch.long, device=args.device)
                with torch.no_grad():
                    out_rag = model(input_ids=rag_t, use_cache=False)
                entry["rag_answer_logprob"] = _answer_logprob(
                    out_rag.logits[0], np.asarray(rag_full), rag_start
                )
                entry["rag_first_hit"] = _first_hit(
                    out_rag.logits[0], np.asarray(rag_full), rag_start
                )
                rag_greedy = _greedy_generate(
                    model, tokenizer, rag_input, max_new_tokens=args.max_new_tokens, device=args.device
                )
                entry["rag_greedy"] = rag_greedy
                entry["rag_exact"] = _exact_match(rag_greedy, answer)

        results.append(entry)
        if (idx + 1) % 25 == 0:
            print(f"  [{idx + 1}/{len(items)}]", flush=True)

    summary: dict = {}
    for task in ["knowledge", "arithmetic", "code-output"]:
        group = [r for r in results if r["task"] == task]
        if not group:
            continue
        row = {
            "n": len(group),
            "no_context_answer_logprob": float(np.mean([r["no_context_answer_logprob"] for r in group])),
            "no_context_first_hit": float(np.mean([1.0 if r["no_context_first_hit"] else 0.0 for r in group])),
            "no_context_exact": float(np.mean([1.0 if r["no_context_exact"] else 0.0 for r in group])),
        }
        rag_lps = [r["rag_answer_logprob"] for r in group if r["rag_answer_logprob"] is not None]
        if rag_lps:
            row["rag_answer_logprob"] = float(np.mean(rag_lps))
            row["rag_first_hit"] = float(np.mean([1.0 if r["rag_first_hit"] else 0.0 for r in group if r["rag_first_hit"] is not None]))
            row["rag_exact"] = float(np.mean([1.0 if r["rag_exact"] else 0.0 for r in group if r["rag_exact"] is not None]))
            row["rag_minus_no_context"] = row["rag_answer_logprob"] - row["no_context_answer_logprob"]
        summary[task] = row
    # knowledge rare/common if present
    for sub in ["rare", "common"]:
        group = [r for r in results if r["task"] == "knowledge" and r["is_rare"] == (sub == "rare")]
        if not group:
            continue
        row = {
            "n": len(group),
            "no_context_answer_logprob": float(np.mean([r["no_context_answer_logprob"] for r in group])),
            "no_context_first_hit": float(np.mean([1.0 if r["no_context_first_hit"] else 0.0 for r in group])),
            "no_context_exact": float(np.mean([1.0 if r["no_context_exact"] else 0.0 for r in group])),
        }
        rag_lps = [r["rag_answer_logprob"] for r in group if r["rag_answer_logprob"] is not None]
        if rag_lps:
            row["rag_answer_logprob"] = float(np.mean(rag_lps))
            row["rag_exact"] = float(np.mean([1.0 if r["rag_exact"] else 0.0 for r in group if r["rag_exact"] is not None]))
            row["rag_minus_no_context"] = row["rag_answer_logprob"] - row["no_context_answer_logprob"]
        summary[f"knowledge_{sub}"] = row

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
    print(f"[multi] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
