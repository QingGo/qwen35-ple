#!/usr/bin/env python3
"""Minimal A0/A1 knowledge-recall benchmark runner.

This is the first real evaluation harness for qwen35-ple.  It is intentionally
small and deterministic:

* loads a HuggingFace causal LM;
* runs a tiny built-in knowledge QA set (or a user-provided JSON list);
* uses greedy generation and checks whether the expected answer appears in the
  generated continuation;
* writes an ``EvalResult``-compatible JSON that can be fed to
  ``scripts/run_eval.py``.

The harness is not a scientific benchmark yet: it only provides an executable
A0/A1 evaluation entrypoint and a reproducible JSON contract.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_BENCHMARK = [
    {"prompt": "What is the capital of France?", "answer": "Paris"},
    {"prompt": "What is the largest planet in the Solar System?", "answer": "Jupiter"},
    {"prompt": "Who wrote 'Romeo and Juliet'?", "answer": "Shakespeare"},
    {"prompt": "What is the chemical symbol for gold?", "answer": "Au"},
    {"prompt": "How many continents are there on Earth?", "answer": "Seven"},
]


def load_prompts(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return DEFAULT_BENCHMARK
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("prompt file must be a JSON list of {prompt, answer}")
    out: list[dict[str, str]] = []
    for item in data:
        out.append(
            {
                "prompt": str(item["prompt"]),
                "answer": str(item["answer"]),
            }
        )
    return out


def evaluate_model(
    model_name: str,
    prompts: list[dict[str, str]] | None = None,
    max_new_tokens: int = 16,
) -> dict:
    prompts = prompts if prompts is not None else DEFAULT_BENCHMARK
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()

    eos_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_id

    examples: list[dict[str, object]] = []
    correct = 0
    for item in prompts:
        prompt = item["prompt"]
        answer = item["answer"]
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuation = generated[0][input_ids.shape[-1] :]
        text = tokenizer.decode(continuation, skip_special_tokens=True)
        hit = answer.lower() in text.lower()
        correct += int(hit)
        examples.append(
            {
                "prompt": prompt,
                "answer": answer,
                "generated": text,
                "correct": hit,
            }
        )

    n = len(prompts)
    return {
        "model": model_name,
        "metrics": {"knowledge_recall": correct / n if n else 0.0},
        "metadata": {
            "n_questions": n,
            "max_new_tokens": max_new_tokens,
            "benchmark": "qwen35-ple-mini-knowledge-v1",
            "examples": examples,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--prompts", default=None, help="optional JSON prompt list")
    parser.add_argument("--output", required=True, help="output EvalResult JSON path")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    result = evaluate_model(args.model, prompts, args.max_new_tokens)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"[run_ablation_eval] result written to {out}")


if __name__ == "__main__":
    main()
