#!/usr/bin/env python3
"""Evaluate a model on locally generated formal-style benchmark families.

Reads JSONL families produced by scripts/build_formal_benchmarks.py and reports
teacher-forced answer log-probability and first-token hit per family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _load_model(model_path: str, adapter: str | None, device: str):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model


def _load_family(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            items.append(obj)
    return items


def _eval_item(model, tokenizer, item, device):
    problem = str(item.get("problem") or item.get("question") or "")
    answer = str(item.get("answer") or item.get("solution") or "")
    qids = tokenizer.encode(problem, add_special_tokens=False)
    ans_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not qids or not ans_ids:
        return None
    full = qids + ans_ids
    ids = torch.tensor([full], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    start = max(0, len(qids) - 1)
    total = 0.0
    n = 0
    for t in range(start, len(full) - 1):
        total += float(log_probs[t, full[t + 1]])
        n += 1
    first_hit = bool(int(torch.argmax(logits[start])) == full[start + 1]) if start < len(full) - 1 else False
    return {
        "id": item.get("id", ""),
        "category": item.get("category", ""),
        "answer_logprob": total / max(1, n),
        "first_hit": first_hit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--benchmark-dir", default="data/formal-benchmarks")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="outputs/formal-benchmark-eval.json")
    args = parser.parse_args()

    tokenizer, model = _load_model(args.model, args.adapter, args.device)
    out_dir = Path(args.benchmark_dir)
    results = {}
    for path in sorted(out_dir.glob("*.jsonl")):
        if path.name == "manifest.json":
            continue
        items = _load_family(path)
        if args.limit is not None:
            items = items[: args.limit]
        rows = []
        for item in items:
            r = _eval_item(model, tokenizer, item, args.device)
            if r is not None:
                rows.append(r)
        if not rows:
            continue
        family = path.stem
        results[family] = {
            "n": len(rows),
            "answer_logprob": float(np.mean([r["answer_logprob"] for r in rows])),
            "first_hit": float(np.mean([1.0 if r["first_hit"] else 0.0 for r in rows])),
        }
        print(f"[formal-eval] {family}: n={len(rows)} lp={results[family]['answer_logprob']:.4f} hit={results[family]['first_hit']:.4f}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"model": args.model, "adapter": args.adapter, "results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[formal-eval] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
