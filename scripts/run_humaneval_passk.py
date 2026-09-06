#!/usr/bin/env python3
"""HumanEval pass@k with multiple sampled completions.

This script samples ``k`` completions per problem for the base model and for the
BM25+PLE system, and reports the empirical pass@k (at least one pass per
problem) plus repetition rate.
"""

from __future__ import annotations

import argparse
import json
import random
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


def _build_memory(chunks: list[str], tokenizer):
    mem = AddressableNgramMemory(min_order=2, max_order=4)
    for i, text in enumerate(chunks):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            mem.add_document(ids, value_id=i)
    return mem


def _build_ple(memory, tokenizer):
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


def _generate_sample(
    model,
    tokenizer,
    prompt: str,
    *,
    context: str | None,
    ple,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
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
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=0)
            keep = cum <= float(top_p)
            keep[0] = True
            filtered = torch.zeros_like(probs)
            filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
            if filtered.sum() <= 0:
                filtered = probs
            filtered = filtered / filtered.sum()
            nxt = int(torch.multinomial(filtered, 1)[0])
        else:
            nxt = int(torch.argmax(logits))
        if nxt == tokenizer.eos_token_id:
            break
        generated.append(nxt)
    return tokenizer.decode(generated[len(input_ids):], skip_special_tokens=True)


def _run_test(prompt: str, completion: str, test_code: str) -> bool:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", prompt + completion + "\n" + test_code],
            capture_output=True,
            timeout=10,
            text=True,
            check=False,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:  # noqa: BLE001 - defensive execution guard
        return False


def _repetition(text: str, n: int = 3) -> float:
    tokens = text.split()
    if len(tokens) < n + 1:
        return 0.0
    grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--corpus", default="data/code-corpus.jsonl")
    parser.add_argument("--max-problems", type=int, default=10)
    parser.add_argument("--samples-per-problem", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/humaneval-passk.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    t0 = time.time()
    tokenizer, model = _load_model(args.model, args.device)
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = [dict(ds[i]) for i in range(min(args.max_problems, len(ds)))]
    docs = load_corpus(args.corpus, max_docs=500)
    chunks = chunk_corpus(docs, chunk_size=600, overlap=100)
    chunk_texts = [c.text for c in chunks]
    bm25 = BM25Index(chunk_texts)
    mem = _build_memory(chunk_texts, tokenizer)
    ple = _build_ple(mem, tokenizer)

    per_condition = {}
    for cond in ["base", "bm25_ple"]:
        passed_any = 0
        reps = []
        for prob in problems:
            prompt = prob["prompt"]
            test_code = prob["test"]
            context = None
            if cond == "bm25_ple":
                hits = bm25.search(prompt, top_k=3)
                context = "\n\n".join(chunk_texts[i] for i in hits) if hits else None
            any_pass = False
            for _ in range(args.samples_per_problem):
                completion = _generate_sample(
                    model,
                    tokenizer,
                    prompt,
                    context=context,
                    ple=ple if cond == "bm25_ple" else None,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=args.device,
                )
                reps.append(_repetition(completion))
                if _run_test(prompt, completion, test_code):
                    any_pass = True
            if any_pass:
                passed_any += 1
        per_condition[cond] = {
            "n_problems": len(problems),
            "samples_per_problem": args.samples_per_problem,
            "pass@k": passed_any / max(1, len(problems)),
            "passed": passed_any,
            "mean_repetition_rate": sum(reps) / len(reps) if reps else 0.0,
        }
        print(per_condition[cond], flush=True)

    result = {
        "schema": "humaneval-passk-v1",
        "config": vars(args),
        "results": per_condition,
        "runtime_seconds": time.time() - t0,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
