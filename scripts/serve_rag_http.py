#!/usr/bin/env python3
"""Minimal HTTP RAG serving demo using only the Python standard library.

This is a pragmatic productization step: it exposes the hybrid RAG path over a
simple HTTP endpoint, so engine-specific integration (vLLM/SGLang/CompileForge)
can replace the transport while keeping the same retriever + adapter.

Usage::

    python scripts/serve_rag_http.py \
        --model data/models/Qwen3.5-0.8B \
        --corpus data/sources/wikitext.jsonl \
        --port 8765
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

from qwen35_ple.rag import (
    BM25Index,
    HybridRetriever,
    chunk_corpus,
    load_corpus,
    mean_pool_embeddings,
)
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


def build_service(args: argparse.Namespace) -> RAGServingAdapter:
    tokenizer, model = _load_model(args.model, args.device)
    docs = load_corpus(args.corpus, args.max_docs)
    chunks = chunk_corpus(docs, chunk_size=args.chunk_size, overlap=args.overlap)
    chunk_texts = [c.text for c in chunks]
    bm25 = BM25Index(chunk_texts)

    dense_vectors = None
    if args.mode == "hybrid":
        emb = model.get_input_embeddings().weight.detach().cpu().numpy()
        dense_vectors = _embed_texts(tokenizer, emb, chunk_texts)
        dense_vectors = dense_vectors / np.maximum(
            np.linalg.norm(dense_vectors, axis=1, keepdims=True), 1e-12
        )

    retriever = HybridRetriever(bm25, dense_vectors)
    return RAGServingAdapter(
        model,
        tokenizer,
        retriever,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        candidate_pool=args.candidate_pool,
        concise=True,
        device=args.device,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--mode", choices=["hybrid", "bm25"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--max-docs", type=int, default=20000)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    service = build_service(args)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/health":
                payload = json.dumps({"ok": True, "chunks": len(service.retriever.bm25.docs)})
                self._send_json(payload)
                return
            if parsed.path != "/answer":
                self.send_error(404, "use /answer?q=...")
                return
            query = urllib.parse.parse_qs(parsed.query).get("q", [""])[0]
            if not query:
                self._send_json(json.dumps({"error": "missing q"}), status=400)
                return
            t0 = time.time()
            try:
                result = service.answer(query)
                result["latency_seconds"] = time.time() - t0
                self._send_json(json.dumps(result, ensure_ascii=False))
            except Exception as exc:  # noqa: BLE001
                self._send_json(json.dumps({"error": str(exc)}), status=500)

        def _send_json(self, body: str, status: int = 200) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"[http] {fmt % args}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] listening on http://{args.host}:{args.port}", flush=True)
    print("[serve] endpoints: /health, /answer?q=...", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
