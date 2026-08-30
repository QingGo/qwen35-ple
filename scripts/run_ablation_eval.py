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
    {"category": "knowledge", "prompt": "What is the capital of France?", "answer": "Paris"},
    {"category": "knowledge", "prompt": "What is the largest planet in the Solar System?", "answer": "Jupiter"},
    {"category": "knowledge", "prompt": "Who wrote 'Romeo and Juliet'?", "answer": "Shakespeare"},
    {"category": "knowledge", "prompt": "What is the chemical symbol for gold?", "answer": "Au"},
    {"category": "knowledge", "prompt": "How many continents are there on Earth?", "answer": "Seven"},
    {
        "category": "long_context",
        "prompt": (
            "Context: Alice has three apples. Bob gives her two more apples. "
            "Then Alice eats one apple. Question: How many apples does Alice have now?"
        ),
        "answer": "four",
    },
    {
        "category": "long_context",
        "prompt": (
            "Context: The library is on the third floor. The cafeteria is on the "
            "ground floor. The classroom is two floors above the cafeteria. "
            "Question: Which floor is the classroom on?"
        ),
        "answer": "second",
    },
    {"category": "reasoning", "prompt": "What is 12 + 15?", "answer": "27"},
    {"category": "reasoning", "prompt": "What is 7 * 6?", "answer": "42"},
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
                "category": str(item.get("category", "knowledge")),
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
    metric_names = {
        "knowledge": "knowledge_recall",
        "long_context": "long_context_score",
        "reasoning": "reasoning_score",
    }
    counts: dict[str, int] = {}
    correct_counts: dict[str, int] = {}

    for item in prompts:
        prompt = item["prompt"]
        answer = item["answer"]
        category = item.get("category", "knowledge")
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
        counts[category] = counts.get(category, 0) + 1
        correct_counts[category] = correct_counts.get(category, 0) + int(hit)
        examples.append(
            {
                "category": category,
                "prompt": prompt,
                "answer": answer,
                "generated": text,
                "correct": hit,
            }
        )

    metrics: dict[str, float] = {}
    for category in sorted(counts):
        name = metric_names.get(category, f"{category}_score")
        metrics[name] = correct_counts[category] / counts[category]
    n = len(prompts)
    return {
        "model": model_name,
        "metrics": metrics,
        "metadata": {
            "n_questions": n,
            "max_new_tokens": max_new_tokens,
            "benchmark": "qwen35-ple-mini-ablation-v1",
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
