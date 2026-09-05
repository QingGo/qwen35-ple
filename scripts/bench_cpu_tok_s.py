#!/usr/bin/env python3
"""Minimal CPU tokens/sec benchmark for the 0.8B serving path.

This is a simple, honest throughput probe:

- Load the model on CPU in float32;
- Run greedy generation for a fixed prompt and a fixed number of new tokens;
- Report wall time and tokens/sec.

It does not measure RAG/PLE serving overhead yet; use it as the baseline for
the CPU 100 tok/s product goal.
"""

from __future__ import annotations

import argparse
import time

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32
    )
    model.to(args.device)
    model.eval()

    ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    generated = list(ids)
    t0 = time.time()
    for _ in range(args.new_tokens):
        input_ids = torch.tensor([generated], dtype=torch.long, device=args.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
        nxt = int(torch.argmax(logits))
        if nxt == tokenizer.eos_token_id:
            break
        generated.append(nxt)
    elapsed = time.time() - t0
    n = len(generated) - len(ids)
    tok_s = n / elapsed if elapsed > 0 else 0.0
    result = {
        "model": args.model,
        "device": args.device,
        "prompt_tokens": len(ids),
        "generated_tokens": n,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tok_s,
        "target": 100.0,
    }
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
