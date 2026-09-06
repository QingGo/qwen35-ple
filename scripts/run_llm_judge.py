#!/usr/bin/env python3
"""LLM-as-judge scoring scaffold for generated answers.

Reads a JSON file with ``rows`` (each containing ``question``, ``answer`` and
optionally ``reference``), loads a local judge model, and produces numeric
correctness / faithfulness scores.

The judge model is passed via ``--judge-model``; for a paper-grade result this
should be a stronger model and should be validated against human annotations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def _api_judge(
    question: str,
    answer: str,
    reference: str,
    *,
    api_key: str,
    api_model: str,
    api_base: str,
) -> float:
    import requests

    prompt = (
        "Judge whether the following answer is correct and faithful to the "
        "question. Output only an integer 0-5.\n\n"
        f"Question: {question}\n"
        f"Reference: {reference}\n"
        f"Answer: {answer}\n"
        "Score:"
    )
    resp = requests.post(
        f"{api_base}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": api_model,
            "messages": [
                {"role": "system", "content": "You are a strict and concise judge."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    try:
        return float(text.split()[0])
    except (ValueError, IndexError):
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/llm-judge.json")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default="deepseek-v4-flash")
    parser.add_argument("--api-base", default="https://api.deepseek.com/v1")
    args = parser.parse_args()
    t0 = time.time()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = data.get("rows", data if isinstance(data, list) else [])
    rows = rows[: args.max_rows]

    use_api = bool(args.api_key)
    if not use_api:
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.judge_model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.judge_model, local_files_only=True, dtype=torch.float32
        )
        model.to(args.device)
        model.eval()

    scored = []
    for i, row in enumerate(rows):
        reference = row.get("reference") or ""
        if not reference and row.get("aliases"):
            reference = row["aliases"][0]

        if use_api:
            score = _api_judge(
                row.get("question", ""),
                row.get("answer", ""),
                reference,
                api_key=args.api_key,
                api_model=args.api_model,
                api_base=args.api_base,
            )
        else:
            prompt = (
                "Judge whether the following answer is correct and faithful to the "
                "question. Output only an integer 0-5.\n\n"
                f"Question: {row.get('question', '')}\n"
                f"Reference: {reference}\n"
                f"Answer: {row.get('answer', '')}\n"
                "Score:"
            )
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            generated = list(ids)
            for _ in range(4):
                input_ids = torch.tensor([generated], dtype=torch.long, device=args.device)
                with torch.no_grad():
                    logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
                nxt = int(torch.argmax(logits))
                if nxt == tokenizer.eos_token_id:
                    break
                generated.append(nxt)
            text = tokenizer.decode(generated[len(ids):], skip_special_tokens=True).strip()
            try:
                score = float(text.split()[0])
            except (ValueError, IndexError):
                score = 0.0
        scored.append({**row, "judge_score": score})
        if (i + 1) % 10 == 0:
            print(f"[judge] {i + 1}/{len(rows)}", flush=True)

    mean = sum(r["judge_score"] for r in scored) / max(1, len(scored))
    result = {
        "schema": "llm-judge-v1",
        "config": vars(args),
        "n": len(scored),
        "mean_judge_score": mean,
        "rows": scored,
        "runtime_seconds": time.time() - t0,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[judge] mean={mean:.3f} n={len(scored)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
