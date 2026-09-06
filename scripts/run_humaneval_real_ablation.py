#!/usr/bin/env python3
"""Real HumanEval ablation for the PLE/RAG system.

This script downloads a small subset of the official `openai/openai_humaneval`
dataset and runs:

* base
* BM25-RAG
* PLE n-gram fusion
* BM25 + PLE fusion

on the same HumanEval prompts, reporting pass@1 where possible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from qwen35_ple.addressable_memory import AddressableNgramMemory
from qwen35_ple.rag import BM25Index, chunk_corpus, load_corpus
from qwen35_ple.router import (
    LogDensityRatioGate,
    TaskConditionedNgramLogitProcessor,
)


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


def _load_humaneval(max_problems: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    return [dict(ds[i]) for i in range(min(max_problems, len(ds)))]


def _build_memory(chunks: list[str], tokenizer):
    mem = AddressableNgramMemory(min_order=2, max_order=4)
    for i, text in enumerate(chunks):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            mem.add_document(ids, value_id=i)
    return mem


def _build_ple_processor(memory, tokenizer):
    return TaskConditionedNgramLogitProcessor(
        memory,
        scale=1.0,
        bias=3.0,
        temperature=0.5,
        enabled=True,
        task="code",
        density_gate=LogDensityRatioGate(mode="expected_kl", threshold=0.0),
        context_decoder=lambda ids: tokenizer.decode(ids),
    )


def _generate(
    model,
    tokenizer,
    prompt: str,
    *,
    context: str | None,
    ple: TaskConditionedNgramLogitProcessor | None,
    max_new_tokens: int,
    device: str,
) -> str:
    prefix = ""
    if context:
        prefix = context + "\n\n"
    input_ids = tokenizer.encode(prefix + prompt, add_special_tokens=False)
    generated = list(input_ids)
    for _ in range(max_new_tokens):
        ids = torch.tensor([generated], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(
                input_ids=ids,
                use_cache=False,
                output_hidden_states=ple is not None,
            )
            logits = out.logits[0, -1]
            if ple is not None:
                hidden = None
                if getattr(out, "hidden_states", None) is not None:
                    hidden = out.hidden_states[-1][0, -1]
                try:
                    logits = ple(logits, generated, hidden_state=hidden)
                except TypeError:
                    logits = ple(logits, generated)
        nxt = int(torch.argmax(logits))
        if nxt == tokenizer.eos_token_id:
            break
        generated.append(nxt)
    return tokenizer.decode(generated[len(input_ids):], skip_special_tokens=True)


def _repetition_rate(text: str, n: int = 3) -> float:
    tokens = text.split()
    if len(tokens) < n + 1:
        return 0.0
    grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def _run_test(prompt: str, completion: str, test_code: str) -> bool:
    code = prompt + completion
    script = code + "\n" + test_code
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=15,
            text=True,
            check=False,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:  # noqa: BLE001 - defensive execution guard
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--corpus", default="data/code-corpus.jsonl")
    parser.add_argument("--max-problems", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--conditions", default="base,bm25,base_ple,bm25_ple")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/humaneval-real.json")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = _load_model(args.model, args.device)
    problems = _load_humaneval(args.max_problems)
    docs = load_corpus(args.corpus, max_docs=500)
    chunks = chunk_corpus(docs, chunk_size=args.chunk_size, overlap=100)
    chunk_texts = [c.text for c in chunks]
    bm25 = BM25Index(chunk_texts)
    mem = _build_memory(chunk_texts, tokenizer)
    ple = _build_ple_processor(mem, tokenizer)
    print(f"[humaneval] problems={len(problems)} chunks={len(chunk_texts)}", flush=True)

    results = {}
    for prob in problems:
        tid = prob["task_id"]
        prompt = prob["prompt"]
        test_code = prob["test"]
        hits = bm25.search(prompt, top_k=args.top_k)
        context = "\n\n".join(chunk_texts[i] for i in hits) if hits else None

        row = {"task_id": tid, "entry_point": prob["entry_point"], "prompt": prompt[:80]}
        condition_names = [c.strip() for c in args.conditions.split(",") if c.strip()]
        for name in condition_names:
            ctx = context if name in {"bm25", "bm25_ple"} else None
            use_ple = name.endswith("_ple")
            use_ple = name.endswith("_ple")
            completion = _generate(
                model,
                tokenizer,
                prompt,
                context=ctx,
                ple=ple if use_ple else None,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )
            passed = _run_test(prompt, completion, test_code)
            row[name] = {
                "completion": completion,
                "passed": passed,
                "length": len(completion),
                "repetition_rate": _repetition_rate(completion),
            }
            print(
                f"[humaneval] {tid} {name} pass={passed} len={len(completion)}",
                flush=True,
            )
        results[tid] = row

    summary = {}
    condition_names = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for name in condition_names:
        passed = sum(1 for r in results.values() if r[name]["passed"])
        reps = [r[name]["repetition_rate"] for r in results.values()]
        summary[name] = {
            "n": len(results),
            "pass@1": passed / max(1, len(results)),
            "passed": passed,
            "mean_repetition_rate": float(sum(reps) / len(reps)) if reps else 0.0,
        }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": "humaneval-real-v1",
                "config": vars(args),
                "summary": summary,
                "results": results,
                "runtime_seconds": time.time() - t0,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[humaneval] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
