#!/usr/bin/env python3
"""Real TriviaQA exact-match evaluation for the base model.

Loads the official `mandarjoshi/trivia_qa` RC subset, samples N questions, and
evaluates short-form exact match after lightweight normalization.  The script
also records simple repetition statistics for generated answers.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import time
from pathlib import Path

import torch


def _load_model(model_path: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove common English articles for TriviaQA-style normalized match.
    words = [w for w in text.split() if w not in {"the", "a", "an"}]
    return " ".join(words)


def _exact_match(answer: str, aliases: list[str]) -> bool:
    normalized = _normalize(answer)
    if not normalized:
        return False
    for alias in aliases:
        if _normalize(alias) == normalized:
            return True
    return False


def _repetition_rate(text: str, n: int = 3) -> float:
    tokens = text.split()
    if len(tokens) < n + 1:
        return 0.0
    grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/triviaqa-real.json")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = _load_model(args.model, args.device)
    import os

    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split=args.split, streaming=True)
    selected = []
    for i, item in enumerate(ds):
        selected.append(item)
        if len(selected) >= args.max_questions:
            break
    rng = random.Random(args.seed)
    rng.shuffle(selected)
    print(f"[triviaqa] loaded n={len(selected)}", flush=True)

    rows = []
    em = 0.0
    rep_sum = 0.0
    for n, item in enumerate(selected):
        question = item["question"]
        aliases = []
        answer = item.get("answer")
        if isinstance(answer, dict):
            aliases = list(answer.get("aliases", []))
            if answer.get("value"):
                aliases.append(answer["value"])
        else:
            aliases = [str(answer)]
        prompt = f"Question: {question}\nAnswer:"
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        generated = list(input_ids)
        for _ in range(args.max_new_tokens):
            ids = torch.tensor([generated], dtype=torch.long, device=args.device)
            with torch.no_grad():
                logits = model(input_ids=ids, use_cache=False).logits[0, -1]
            nxt = int(torch.argmax(logits))
            if nxt == tokenizer.eos_token_id:
                break
            generated.append(nxt)
        answer_text = tokenizer.decode(
            generated[len(input_ids):], skip_special_tokens=True
        ).strip()
        hit = _exact_match(answer_text, aliases)
        rep = _repetition_rate(answer_text)
        em += 1.0 if hit else 0.0
        rep_sum += rep
        rows.append(
            {
                "question": question,
                "aliases": aliases[:5],
                "answer": answer_text,
                "exact_match": hit,
                "repetition_rate": rep,
            }
        )
        if (n + 1) % 10 == 0:
            print(
                f"[triviaqa] {n + 1}/{len(selected)} em={em / (n + 1):.3f} "
                f"rep={rep_sum / (n + 1):.3f}",
                flush=True,
            )

    result = {
        "schema": "triviaqa-real-v1",
        "config": vars(args),
        "n": len(rows),
        "exact_match": em / max(1, len(rows)),
        "mean_repetition_rate": rep_sum / max(1, len(rows)),
        "rows": rows,
        "runtime_seconds": time.time() - t0,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n": result["n"],
                "exact_match": result["exact_match"],
                "mean_repetition_rate": result["mean_repetition_rate"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
