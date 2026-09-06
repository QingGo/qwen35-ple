#!/usr/bin/env python3
"""Small-scale kNN-LM baseline for PLE comparison.

This implements a lightweight hidden-state kNN-LM:
* a datastore is built from projector dataset entries (context hidden state, next token);
* for each eval context we retrieve top-k nearest datastore keys;
* the next-token distribution is a kernel-weighted mixture with the base LM.

It is intentionally small-scale and transparent, not a full reproduction of
large kNN-LM systems.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from qwen35_ple.fusion import softmax


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
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model


def _load_dataset(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _forward(model, context: list[int], device: str):
    ids = torch.tensor([context], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
        logits = out.logits[0, -1].float().cpu().numpy()
        hidden = out.hidden_states[-1][0, -1].float().cpu().numpy()
    return logits, hidden


def _logprob(p: np.ndarray, target: int) -> float:
    return float(math.log(max(float(p[target]), 1e-12)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--dataset", default="data/ple-projector-dataset-1k.jsonl")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--max-train", type=int, default=500)
    parser.add_argument("--max-eval", type=int, default=200)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--lambda-knn", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/knn-lm-baseline.json")
    args = parser.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    _tokenizer, model = _load_model(args.model, args.device)
    rows = _load_dataset(args.dataset)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_train = int(len(rows) * args.train_frac)
    train_rows = rows[: min(n_train, args.max_train)]
    eval_rows = rows[n_train : n_train + args.max_eval]
    print(
        f"[knn] train={len(train_rows)} eval={len(eval_rows)}",
        flush=True,
    )

    train_hidden = []
    train_targets = []
    train_tasks = []
    for i, row in enumerate(train_rows):
        _, hidden = _forward(model, row["context"], args.device)
        train_hidden.append(hidden)
        train_targets.append(row["target"])
        train_tasks.append(row["task"])
        if (i + 1) % 100 == 0:
            print(f"[knn] keys {i + 1}/{len(train_rows)}", flush=True)
    keys = np.stack(train_hidden).astype(np.float32)
    knorms = np.linalg.norm(keys, axis=1) + 1e-8

    base_nll = 0.0
    knn_nll = 0.0
    base_hit = 0.0
    knn_hit = 0.0
    per_task: dict[str, dict[str, float]] = {}

    for i, row in enumerate(eval_rows):
        logits, hidden = _forward(model, row["context"], args.device)
        target = row["target"]
        pb = softmax(logits)
        base_nll += -_logprob(pb, target)
        base_hit += 1.0 if int(np.argmax(logits)) == target else 0.0

        sims = (keys @ hidden) / (knorms * (np.linalg.norm(hidden) + 1e-8))
        top_idx = np.argsort(-sims)[: args.k]
        weights = np.exp(np.clip(sims[top_idx] / max(args.temperature, 1e-6), -30, 30))
        weights = weights / weights.sum()
        p_knn = np.zeros_like(pb)
        for idx, w in zip(top_idx, weights):
            p_knn[train_targets[idx]] += w
        p_mix = (1.0 - args.lambda_knn) * pb + args.lambda_knn * p_knn
        knn_nll += -_logprob(p_mix, target)
        knn_hit += 1.0 if int(np.argmax(p_mix)) == target else 0.0

        task = row["task"]
        b = per_task.setdefault(task, {"n": 0.0, "base_nll": 0.0, "knn_nll": 0.0, "base_hit": 0.0, "knn_hit": 0.0})
        b["n"] += 1
        b["base_nll"] += -_logprob(pb, target)
        b["knn_nll"] += -_logprob(p_mix, target)
        b["base_hit"] += 1.0 if int(np.argmax(logits)) == target else 0.0
        b["knn_hit"] += 1.0 if int(np.argmax(p_mix)) == target else 0.0
        if (i + 1) % 25 == 0:
            print(f"[knn] eval {i + 1}/{len(eval_rows)}", flush=True)

    n = max(1, len(eval_rows))
    result = {
        "schema": "knn-lm-baseline-v1",
        "config": vars(args),
        "n_eval": len(eval_rows),
        "n_train_keys": len(train_rows),
        "base_nll": base_nll / n,
        "knn_nll": knn_nll / n,
        "base_hit": base_hit / n,
        "knn_hit": knn_hit / n,
        "per_task": {
            k: {
                "n": int(v["n"]),
                "base_nll": v["base_nll"] / v["n"],
                "knn_nll": v["knn_nll"] / v["n"],
                "base_hit": v["base_hit"] / v["n"],
                "knn_hit": v["knn_hit"] / v["n"],
            }
            for k, v in per_task.items()
        },
        "runtime_seconds": time.time() - t0,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "per_task"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
