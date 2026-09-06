#!/usr/bin/env python3
"""Per-item CAP-1 heldout + formal-style benchmark evaluator for multi-seed stats.

This is a companion to the aggregate eval scripts.  It writes the same summary
metrics but also includes per-item rows so downstream scripts can run paired
tests, bootstrap CIs, and per-category error analyses.
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


def _read_jsonl(path: Path) -> list[dict]:
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


def _eval_pair(model, tokenizer, prompt: str, answer: str, device: str) -> dict:
    qids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not qids or not ans_ids:
        return {}
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
    first_hit = (
        bool(int(torch.argmax(logits[start]).item()) == full[start + 1])
        if start < len(full) - 1
        else False
    )
    return {
        "answer_logprob": total / max(1, n),
        "first_hit": first_hit,
        "n_answer_tokens": n,
    }


def eval_cap1(model, tokenizer, items: list[dict], device: str) -> dict:
    rows = []
    for idx, item in enumerate(items):
        problem = str(item.get("problem") or item.get("question") or "")
        context = str(item.get("context") or "")
        answer = str(item.get("solution") or item.get("answer") or "")
        if context:
            prompt = f"Question: {problem}\n\nContext:\n{context}\n\nAnswer:"
        else:
            prompt = f"Question: {problem}\n\nAnswer:"
        r = _eval_pair(model, tokenizer, prompt, answer, device)
        if not r:
            continue
        r["id"] = str(item.get("id", idx))
        r["category"] = str(item.get("category", "")).lower()
        rows.append(r)
    return {
        "n": len(rows),
        "mean_answer_logprob": float(np.mean([r["answer_logprob"] for r in rows])) if rows else None,
        "first_token_hit": float(np.mean([1.0 if r["first_hit"] else 0.0 for r in rows])) if rows else None,
        "per_item": rows,
    }


def eval_formal(model, tokenizer, benchmark_dir: Path, device: str, limit: int | None) -> dict:
    results = {}
    for path in sorted(benchmark_dir.glob("*.jsonl")):
        if path.name == "manifest.json":
            continue
        items = _read_jsonl(path)
        if limit is not None:
            items = items[:limit]
        rows = []
        for item in items:
            problem = str(item.get("problem") or item.get("question") or "")
            answer = str(item.get("answer") or item.get("solution") or "")
            r = _eval_pair(model, tokenizer, problem, answer, device)
            if not r:
                continue
            r["id"] = str(item.get("id", ""))
            r["category"] = str(item.get("category", ""))
            rows.append(r)
        if not rows:
            continue
        family = path.stem
        results[family] = {
            "n": len(rows),
            "answer_logprob": float(np.mean([r["answer_logprob"] for r in rows])) if rows else None,
            "first_hit": float(np.mean([1.0 if r["first_hit"] else 0.0 for r in rows])) if rows else None,
            "per_item": rows,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data", default="data/cap1-rag-distill-eval39.jsonl")
    parser.add_argument("--benchmark-dir", default="data/formal-benchmarks")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/cap1-formal-peritem.json")
    args = parser.parse_args()

    tokenizer, model = _load_model(args.model, args.adapter, args.device)
    cap_items = _read_jsonl(Path(args.data))
    cap = eval_cap1(model, tokenizer, cap_items, args.device)
    formal = eval_formal(model, tokenizer, Path(args.benchmark_dir), args.device, args.limit)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "cap1-formal-peritem-v1",
        "model": args.model,
        "adapter": args.adapter,
        "cap1": {k: v for k, v in cap.items() if k != "per_item"},
        "cap1_per_item": cap.get("per_item", []),
        "formal": {k: {kk: vv for kk, vv in v.items() if kk != "per_item"} for k, v in formal.items()},
        "formal_per_item": {k: v.get("per_item", []) for k, v in formal.items()},
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[cap1-formal] wrote {out} cap1_n={cap.get('n')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
