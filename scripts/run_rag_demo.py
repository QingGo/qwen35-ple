#!/usr/bin/env python3
"""Hybrid RAG demo: chunked corpus, BM25 + dense embedding, RRF rerank, generate.

Usage::

    python scripts/run_rag_demo.py \
        --model data/models/Qwen3.5-0.8B \
        --corpus data/sources/wikitext.jsonl \
        --question "Who is Nikola Tesla?" \
        --mode hybrid
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from qwen35_ple.addressable_memory import AddressableNgramMemory
from qwen35_ple.rag import (
    BM25Index,
    HybridRetriever,
    NgramKeyRetriever,
    chunk_corpus,
    load_corpus,
    mean_pool_embeddings,
)
from qwen35_ple.router import CalibratedNgramLogitProcessor
from qwen35_ple.serving.rag import RAGServingAdapter


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


def _embed_texts(
    tokenizer,
    embedding_matrix: np.ndarray,
    texts: list[str],
) -> np.ndarray:
    ids = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    return mean_pool_embeddings(ids, embedding_matrix)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--mode", choices=["hybrid", "bm25"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--max-docs", type=int, default=20000)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--use-ngram", action="store_true", help="add PLE-style n-gram key retrieval to hybrid RAG")
    parser.add_argument("--ngram-weight", type=float, default=1.0)
    parser.add_argument("--use-ngram-fusion", action="store_true", help="apply calibrated n-gram logit fusion during generation")
    parser.add_argument("--fusion-scale", type=float, default=1.0)
    parser.add_argument("--fusion-bias", type=float, default=0.0)
    parser.add_argument("--fusion-temperature", type=float, default=1.0)
    parser.add_argument(
        "--fusion-config",
        default="configs/ngram-fusion-router.json",
        help="persisted calibrated fusion/router JSON; used when --use-ngram-fusion",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = _load_model(args.model, args.device)
    docs = load_corpus(args.corpus, args.max_docs)
    chunks = chunk_corpus(docs, chunk_size=args.chunk_size, overlap=args.overlap)
    chunk_texts = [c.text for c in chunks]
    print(f"[demo] chunks={len(chunk_texts)}", flush=True)
    bm25 = BM25Index(chunk_texts)

    dense_vectors = None
    query_vector = None
    if args.mode == "hybrid":
        emb = model.get_input_embeddings().weight.detach().cpu().numpy()
        dense_vectors = _embed_texts(tokenizer, emb, chunk_texts)
        dense_vectors = dense_vectors / np.maximum(
            np.linalg.norm(dense_vectors, axis=1, keepdims=True), 1e-12
        )
        query_vector = _embed_texts(tokenizer, emb, [args.question])[0]
        qn = np.linalg.norm(query_vector)
        if qn > 0:
            query_vector = query_vector / qn

    ngram_retriever = None
    logit_processor = None
    mem = None
    use_adapter_config = False
    if args.use_ngram:
        mem = AddressableNgramMemory(min_order=2, max_order=4)
        for i, text in enumerate(chunk_texts):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                mem.add_document(ids, value_id=i)
        ngram_retriever = NgramKeyRetriever(
            mem,
            tokenizer=lambda text: tokenizer.encode(text, add_special_tokens=False),
        )
        if args.use_ngram_fusion:
            if args.fusion_config:
                # The adapter will load the persisted calibrated parameters and
                # build a task-conditioned processor from the same memory.
                use_adapter_config = True
                logit_processor = None
            else:
                logit_processor = CalibratedNgramLogitProcessor(
                    mem,
                    scale=args.fusion_scale,
                    bias=args.fusion_bias,
                    temperature=args.fusion_temperature,
                )
        print(f"[demo] ngram memory entries={mem.stats()}", flush=True)

    retriever = HybridRetriever(
        bm25,
        dense_vectors,
        ngram_retriever=ngram_retriever,
        ngram_weight=args.ngram_weight,
    )
    adapter = RAGServingAdapter(
        model,
        tokenizer,
        retriever,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        candidate_pool=args.candidate_pool,
        concise=True,
        device=args.device,
        logit_processor=logit_processor,
        ngram_memory=mem if use_adapter_config else None,
        fusion_config=args.fusion_config if use_adapter_config else None,
    )
    result = adapter.answer(args.question)
    elapsed = time.time() - t0

    print("== Query ==")
    print(result["question"])
    print("\n== Retrieved contexts ==")
    for i, d in enumerate(result["contexts"], 1):
        print(f"[{i}] {d[:200]}")
    print("\n== Answer ==")
    print(result["answer"])
    print(f"\n[elapsed {elapsed:.2f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
