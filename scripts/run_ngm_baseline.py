#!/usr/bin/env python3
"""NGM baseline using the official NGM n-gram memory hook.

This script uses the NGM implementation from `PioneerQyw/NGM` as a training-free
external memory baseline, without modifying the base model weights.  It compares
base next-token NLL/hit against NGM-augmented forward passes on the local PLE
projector dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from qwen35_ple.fusion import softmax


def _load_ngm(ngm_root: str):
    src = Path(ngm_root) / "NGM_Text" / "src"
    sys.path.insert(0, str(src))
    from memory import NgramMemoryConfig, NgramMemoryHook

    return NgramMemoryConfig, NgramMemoryHook


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


def _nll(logits: np.ndarray, target: int) -> float:
    return -math.log(max(float(softmax(logits)[target]), 1e-12))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--ngm-root", default="/tmp/NGM")
    parser.add_argument("--dataset", default="data/ple-projector-dataset-1k.jsonl")
    parser.add_argument("--max-eval", type=int, default=200)
    parser.add_argument("--ngram-layers", default="1,10")
    parser.add_argument("--ngram-sizes", default="2,3")
    parser.add_argument("--output-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/ngm-baseline.json")
    args = parser.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    _tokenizer, model = _load_model(args.model, args.device)
    rows = _load_dataset(args.dataset)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    eval_rows = rows[: args.max_eval]

    NgramMemoryConfig, NgramMemoryHook = _load_ngm(args.ngm_root)
    layer_ids = [int(x) for x in args.ngram_layers.split(",")]
    ngram_sizes = [int(x) for x in args.ngram_sizes.split(",")]
    config = NgramMemoryConfig(
        ngram_sizes=ngram_sizes,
        layer_ids=layer_ids,
        use_relu=True,
        output_scale=args.output_scale,
    )
    hook = NgramMemoryHook(model, config)
    hook.register_hooks()

    base_nll = 0.0
    ngm_nll = 0.0
    base_hit = 0.0
    ngm_hit = 0.0
    per_task: dict[str, dict[str, float]] = {}

    for i, row in enumerate(eval_rows):
        context = row["context"]
        target = row["target"]
        ids = torch.tensor([context], dtype=torch.long, device=args.device)

        # Base (without hook)
        with torch.no_grad():
            logits_base = model(input_ids=ids, use_cache=False).logits[0, -1].cpu().numpy()
        base_nll += _nll(logits_base, target)
        base_hit += 1.0 if int(np.argmax(logits_base)) == target else 0.0

        # NGM-augmented
        hook.set_input_ids(ids)
        with torch.no_grad():
            logits_ngm = model(input_ids=ids, use_cache=False).logits[0, -1].cpu().numpy()
        ngm_nll += _nll(logits_ngm, target)
        ngm_hit += 1.0 if int(np.argmax(logits_ngm)) == target else 0.0

        task = row["task"]
        b = per_task.setdefault(task, {"n": 0.0, "base_nll": 0.0, "ngm_nll": 0.0, "base_hit": 0.0, "ngm_hit": 0.0})
        b["n"] += 1
        b["base_nll"] += _nll(logits_base, target)
        b["ngm_nll"] += _nll(logits_ngm, target)
        b["base_hit"] += 1.0 if int(np.argmax(logits_base)) == target else 0.0
        b["ngm_hit"] += 1.0 if int(np.argmax(logits_ngm)) == target else 0.0
        if (i + 1) % 25 == 0:
            print(
                f"[ngm] {i + 1}/{len(eval_rows)} base_nll={base_nll / (i + 1):.4f} "
                f"ngm_nll={ngm_nll / (i + 1):.4f}",
                flush=True,
            )

    hook.remove_hooks()
    n = max(1, len(eval_rows))
    result = {
        "schema": "ngm-baseline-v1",
        "config": vars(args),
        "n_eval": len(eval_rows),
        "base_nll": base_nll / n,
        "ngm_nll": ngm_nll / n,
        "base_hit": base_hit / n,
        "ngm_hit": ngm_hit / n,
        "per_task": {
            k: {
                "n": int(v["n"]),
                "base_nll": v["base_nll"] / v["n"],
                "ngm_nll": v["ngm_nll"] / v["n"],
                "base_hit": v["base_hit"] / v["n"],
                "ngm_hit": v["ngm_hit"] / v["n"],
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
