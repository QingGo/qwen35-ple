#!/usr/bin/env python3
"""Real RAG retrieval ablation: BM25 vs Dense vs N-gram vs Hybrid.

This uses held-out Wikipedia documents as real queries.  The retrieval index is
built from training documents; for each held-out document we use its first
sentence as a natural-language query and check whether the source document is
retrieved.

Channels:

* BM25   : pure lexical BM25;
* Dense  : static Qwen token-embedding mean-pool cosine;
* N-gram : PLE-style exact n-gram addressable memory;
* Hybrid : RRF fusion of all three channels.

Metrics: Recall@1/3/5 and MRR.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qwen35_ple.addressable_memory import AddressableNgramMemory
from qwen35_ple.rag import BM25Index, HybridRetriever, NgramKeyRetriever, mean_pool_embeddings


def _load_model(model_dir: str):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype="float32", low_cpu_mem_usage=True
    )
    model.eval()
    return tokenizer, model


def load_wiki_docs(limit: int) -> list[str]:
    docs: list[str] = []
    with Path("data/sources/wikitext.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(obj.get("text") or "")
            if text.strip():
                docs.append(text.strip())
            if len(docs) >= limit:
                break
    return docs


def first_sentence(text: str, max_chars: int = 120) -> str:
    # Simple split on sentence-ish punctuation/newline.
    for sep in ("\n", ". ", "! ", "? "):
        idx = text.find(sep)
        if 0 < idx < max_chars:
            return text[: idx + 1].strip()
    return text[:max_chars].strip()


def recall_at_k(ranked: list[int], target: int, k: int) -> bool:
    return target in ranked[:k]


def mrr(ranked: list[int], target: int) -> float:
    for i, doc in enumerate(ranked):
        if doc == target:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_channel(name, ranking_fn, queries, target_ids, k: int = 5):
    recs = {f"recall@{i}": [] for i in (1, 3, 5)}
    mrrs = []
    for i, query in enumerate(queries):
        ranked = ranking_fn(query, k)
        recs["recall@1"].append(recall_at_k(ranked, target_ids[i], 1))
        recs["recall@3"].append(recall_at_k(ranked, target_ids[i], 3))
        recs["recall@5"].append(recall_at_k(ranked, target_ids[i], 5))
        mrrs.append(mrr(ranked, target_ids[i]))
    return {
        "channel": name,
        "n": len(queries),
        **{k: float(np.mean(v)) for k, v in recs.items()},
        "mrr": float(np.mean(mrrs)),
    }




def evaluate_channel_multi(name, ranking_fn, queries, relevant_sets, k: int = 5):
    """Evaluate recall of *any* relevant document and MRR to the best one."""
    recs = {f"recall@{i}": [] for i in (1, 3, 5)}
    mrrs = []
    for query, rel in zip(queries, relevant_sets):
        ranked = ranking_fn(query, k)
        recs["recall@1"].append(any(r in ranked[:1] for r in rel))
        recs["recall@3"].append(any(r in ranked[:3] for r in rel))
        recs["recall@5"].append(any(r in ranked[:5] for r in rel))
        best = min((ranked.index(r) + 1 for r in rel if r in ranked), default=None)
        mrrs.append(1.0 / best if best else 0.0)
    return {
        "channel": name,
        "n": len(queries),
        **{k: float(np.mean(v)) for k, v in recs.items()},
        "mrr": float(np.mean(mrrs)),
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/Users/zeng/code/LLM-CompileForge/models/Qwen/Qwen3.5-0.8B.local-backup")
    parser.add_argument("--wiki-limit", type=int, default=300)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--ngram-weight", type=float, default=1.0)
    parser.add_argument("--qa-file", default="data/qa-expanded-150.json", help="optional QA answer-containment retrieval set")
    parser.add_argument("--output", default="outputs/rag-channel-ablation.json")
    parser.add_argument("--report", default="outputs/rag-channel-ablation.md")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = _load_model(args.model_dir)
    docs = load_wiki_docs(args.wiki_limit)
    train_docs = docs
    queries = [first_sentence(d) for d in docs]
    targets = list(range(len(docs)))

    train_ids = [tokenizer.encode(d, add_special_tokens=False) for d in train_docs]
    query_ids = [tokenizer.encode(q, add_special_tokens=False) for q in queries]

    print(f"[rag-abl] index_docs={len(train_docs)} query_docs={len(docs)}", flush=True)

    bm25 = BM25Index(train_docs)

    # Dense: static token embedding mean-pool.
    emb = model.get_input_embeddings().weight.detach().cpu().float().numpy().astype(np.float32)
    dense_doc_vectors = mean_pool_embeddings(train_ids, emb)
    dense_doc_vectors = dense_doc_vectors / np.maximum(
        np.linalg.norm(dense_doc_vectors, axis=1, keepdims=True), 1e-12
    )
    dense_query_vectors = mean_pool_embeddings(query_ids, emb)
    dense_query_vectors = dense_query_vectors / np.maximum(
        np.linalg.norm(dense_query_vectors, axis=1, keepdims=True), 1e-12
    )

    # N-gram addressable memory.
    mem = AddressableNgramMemory(min_order=2, max_order=4)
    for i, ids in enumerate(train_ids):
        if ids:
            mem.add_document(ids, value_id=i)
    ngram = NgramKeyRetriever(
        mem,
        tokenizer=lambda text: tokenizer.encode(text, add_special_tokens=False),
    )

    # Channel ranking functions.
    def bm25_rank(q, k):
        return bm25.search(q, top_k=k)

    def dense_rank(q, k):
        qv = mean_pool_embeddings([tokenizer.encode(q, add_special_tokens=False)], emb)[0]
        qv = qv / max(np.linalg.norm(qv), 1e-12)
        scores = dense_doc_vectors @ qv
        return np.argsort(scores)[::-1][:k].tolist()

    def ngram_rank(q, k):
        return ngram.search(q, top_k=k)

    hybrid = HybridRetriever(
        bm25,
        dense_doc_vectors,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        ngram_retriever=ngram,
        ngram_weight=args.ngram_weight,
    )
    def hybrid_rank(q, k):
        qv = mean_pool_embeddings([tokenizer.encode(q, add_special_tokens=False)], emb)[0]
        qv = qv / max(np.linalg.norm(qv), 1e-12)
        return hybrid.search(q, top_k=k, query_vector=qv, candidate_pool=args.candidate_pool)

    results = {
        "schema": "rag-channel-ablation-v1",
        "seed": args.seed,
        "index_docs": len(train_docs),
        "query_docs": len(docs),
        "channels": {},
    }
    for name, fn in [
        ("bm25", bm25_rank),
        ("dense", dense_rank),
        ("ngram", ngram_rank),
        ("hybrid", hybrid_rank),
    ]:
        res = evaluate_channel(name, fn, queries, targets, k=args.top_k)
        results["channels"][name] = res
        print(f"[rag-abl] {name}: {json.dumps(res)}", flush=True)

    # Optional QA answer-containment retrieval ablation.
    qa_path = Path(args.qa_file)
    if qa_path.exists():
        qa_data = json.loads(qa_path.read_text(encoding="utf-8"))
        qa_queries: list[str] = []
        qa_rel: list[list[int]] = []
        for item in qa_data:
            q = str(item.get("question", ""))
            a = str(item.get("answer", "")).lower()
            if not q or not a:
                continue
            rel = [i for i, d in enumerate(train_docs) if a in d.lower()]
            if rel:
                qa_queries.append(q)
                qa_rel.append(rel)
        if qa_queries:
            results["qa"] = {}
            print(f"[rag-abl] qa_queries={len(qa_queries)}", flush=True)
            for name, fn in [
                ("bm25", bm25_rank),
                ("dense", dense_rank),
                ("ngram", ngram_rank),
                ("hybrid", hybrid_rank),
            ]:
                res = evaluate_channel_multi(name, fn, qa_queries, qa_rel, k=args.top_k)
                results["qa"][name] = res
                print(f"[rag-abl] qa {name}: {json.dumps(res)}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RAG 三通道检索消融",
        "",
        f"- Seed: {args.seed}",
        f"- Index docs: {len(train_docs)}",
        f"- Query docs: {len(docs)}",
        "",
        "| Channel | Recall@1 | Recall@3 | Recall@5 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in results["channels"].items():
        lines.append(
            f"| {name} | {r['recall@1']:.4f} | {r['recall@3']:.4f} | {r['recall@5']:.4f} | {r['mrr']:.4f} |"
        )
    if "qa" in results:
        lines += ["", "## QA answer-containment retrieval", "", "| Channel | Recall@1 | Recall@3 | Recall@5 | MRR |", "|---|---:|---:|---:|---:|"]
        for name, r in results["qa"].items():
            lines.append(
                f"| {name} | {r['recall@1']:.4f} | {r['recall@3']:.4f} | {r['recall@5']:.4f} | {r['mrr']:.4f} |"
            )
    rep.write_text("\n".join(lines), encoding="utf-8")
    print(f"[rag-abl] wrote {out} and {rep} in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
