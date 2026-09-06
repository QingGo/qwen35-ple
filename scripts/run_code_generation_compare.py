#!/usr/bin/env python3
"""Compare code generation with BM25-only, BM25+PLE-retrieval, BM25+PLE-fusion.

This is a small end-to-end generation ablation on a same-domain Python code
corpus.  It answers the question: should PLE be used as a retrieval channel, as
a logit fusion prior, or neither for open-ended code generation?
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from qwen35_ple.addressable_memory import AddressableNgramMemory
from qwen35_ple.rag import (
    BM25Index,
    HybridRetriever,
    NgramKeyRetriever,
    chunk_corpus,
    load_corpus,
)
from qwen35_ple.serving.rag import RAGServingAdapter

DEFAULT_PROMPTS = [
    "Write a Python function that returns the sum of two numbers.",
    "Write a Python function that returns the length of a list.",
    "Write a Python function that reverses a string.",
    "Write a Python function that checks whether a number is even.",
    "Write a Python function that returns the maximum of two numbers.",
]


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
    return tokenizer, model


def _build_memory(chunk_texts, tokenizer):
    mem = AddressableNgramMemory(min_order=2, max_order=4)
    for i, text in enumerate(chunk_texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            mem.add_document(ids, value_id=i)
    return mem


def _make_adapter(
    model,
    tokenizer,
    bm25,
    *,
    use_ngram: bool,
    use_fusion: bool,
    mem,
    fusion_config,
    max_new_tokens: int,
    top_k: int,
    device: str,
):
    ngram_retriever = None
    if use_ngram:
        ngram_retriever = NgramKeyRetriever(
            mem,
            tokenizer=lambda text: tokenizer.encode(text, add_special_tokens=False),
        )
    retriever = HybridRetriever(
        bm25,
        None,
        ngram_retriever=ngram_retriever,
        ngram_weight=1.0,
    )
    return RAGServingAdapter(
        model,
        tokenizer,
        retriever,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        concise=True,
        device=device,
        ngram_memory=mem if use_fusion else None,
        fusion_config=fusion_config if use_fusion else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--fusion-config", default="configs/ngram-fusion-router.json")
    parser.add_argument("--max-docs", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/code-generation-compare.json")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = _load_model(args.model, args.device)
    docs = load_corpus(args.corpus, args.max_docs)
    chunks = chunk_corpus(docs, chunk_size=args.chunk_size, overlap=100)
    chunk_texts = [c.text for c in chunks]
    bm25 = BM25Index(chunk_texts)
    mem = _build_memory(chunk_texts, tokenizer)
    print(f"[code-gen] chunks={len(chunk_texts)}", flush=True)

    adapters = {
        "bm25": _make_adapter(
            model, tokenizer, bm25, use_ngram=False, use_fusion=False,
            mem=mem, fusion_config=args.fusion_config,
            max_new_tokens=args.max_new_tokens, top_k=args.top_k, device=args.device,
        ),
        "bm25_ngram_retrieval": _make_adapter(
            model, tokenizer, bm25, use_ngram=True, use_fusion=False,
            mem=mem, fusion_config=args.fusion_config,
            max_new_tokens=args.max_new_tokens, top_k=args.top_k, device=args.device,
        ),
        "bm25_ngram_fusion": _make_adapter(
            model, tokenizer, bm25, use_ngram=True, use_fusion=True,
            mem=mem, fusion_config=args.fusion_config,
            max_new_tokens=args.max_new_tokens, top_k=args.top_k, device=args.device,
        ),
    }

    results: dict[str, dict] = {}
    for name, adapter in adapters.items():
        rows = []
        for q in DEFAULT_PROMPTS:
            out = adapter.answer(q)
            rows.append({
                "question": q,
                "answer": out.get("answer", ""),
                "contexts": [str(c)[:160] for c in out.get("contexts", [])],
            })
        results[name] = {"n": len(rows), "rows": rows}
        print(f"[code-gen] {name}: {len(rows)} answers", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "code-generation-compare-v1",
        "config": vars(args),
        "results": results,
        "runtime_seconds": time.time() - t0,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[code-gen] wrote {out} in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
