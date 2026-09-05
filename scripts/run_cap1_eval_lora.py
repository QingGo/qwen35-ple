#!/usr/bin/env python3
"""Small CAP-1 eval: base vs LoRA adapter on RAG self-distill examples.

This is a quick CPU/GPU evaluator for the smoke CAP-1 LoRA run.  It computes
answer-token average log-probability and first-token hit for the same RAG
context prompts, with and without the trained LoRA adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _load_model(model_path: str, adapter_path: str | None, device: str):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def _make_prompt(item: dict) -> str:
    problem = item["problem"]
    context = item.get("context", "")
    if context:
        return f"Question: {problem}\n\nContext:\n{context}\n\nAnswer:"
    return f"Question: {problem}\n\nAnswer:"


def evaluate(model, tokenizer, items, device):
    logprobs = []
    hits = []
    for item in items:
        prompt = _make_prompt(item)
        answer = item["solution"]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        if not answer_ids:
            continue
        full = prompt_ids + answer_ids
        ids = torch.tensor([full], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
            logits = out.logits[0]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        total = 0.0
        n = 0
        for t in range(max(0, len(prompt_ids) - 1), len(full) - 1):
            total += float(log_probs[t, full[t + 1]])
            n += 1
        logprobs.append(total / max(n, 1))
        first_t = max(0, len(prompt_ids) - 1)
        hits.append(int(torch.argmax(logits[first_t]).item()) == full[first_t + 1] if first_t < len(full) - 1 else False)
    return {
        "n": len(logprobs),
        "mean_answer_logprob": float(np.mean(logprobs)) if logprobs else None,
        "first_token_hit": float(np.mean(hits)) if hits else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/cap1-lora-eval.json")
    args = parser.parse_args()

    items = []
    with Path(args.data).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            items.append(obj)
            if len(items) >= args.limit:
                break

    tokenizer, model = _load_model(args.model, args.adapter, args.device)
    result = evaluate(model, tokenizer, items, args.device)
    result["model"] = args.model
    result["adapter"] = args.adapter
    result["items"] = len(items)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
