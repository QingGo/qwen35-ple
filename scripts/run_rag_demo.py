#!/usr/bin/env python3
"""Minimal RAG demo: retrieve context and generate an answer with frozen 0.8B.

This is a small productization step: a single CLI that shows the full RAG path
without any PLE/memory machinery.

Usage::

    python scripts/run_rag_demo.py \
        --model data/models/Qwen3.5-0.8B \
        --corpus data/sources/wikitext.jsonl \
        --question "What is the capital of France?"
"""

from __future__ import annotations

import argparse
import time

import torch

from qwen35_ple.rag import BM25Index, build_rag_prompt, load_corpus


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


def _generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = list(ids)
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
        nxt = int(torch.argmax(logits))
        if nxt == tokenizer.eos_token_id:
            break
        generated.append(nxt)
    return tokenizer.decode(generated[len(ids):], skip_special_tokens=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-docs", type=int, default=20000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = _load_model(args.model, args.device)
    docs = load_corpus(args.corpus, args.max_docs)
    index = BM25Index(docs)
    ctx_docs = index.retrieve(args.question, args.top_k)
    context = "\n\n".join(ctx_docs)
    prompt = build_rag_prompt(context, args.question)
    answer = _generate(model, tokenizer, prompt, args.max_new_tokens, args.device)
    elapsed = time.time() - t0

    print("== Query ==")
    print(args.question)
    print("\n== Retrieved contexts ==")
    for i, d in enumerate(ctx_docs, 1):
        print(f"[{i}] {d[:200]}")
    print("\n== Answer ==")
    print(answer)
    print(f"\n[elapsed {elapsed:.2f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
