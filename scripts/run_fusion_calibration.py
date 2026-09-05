#!/usr/bin/env python3
"""Real base-logit n-gram fusion calibration.

Compares on a small CPU sample:

1. single lambda (scale only);
2. scale + bias;
3. scale + bias + temperature.

Uses real n-gram memory and shuffled control memory from the same training
corpus, plus the actual Qwen3.5-0.8B base model logits.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from qwen35_ple.fusion import calibrate_ngram_fusion, fuse_ngram_logits, softmax
from qwen35_ple.ngram_lm import NgramLM


def _load_model(model_dir: str):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype="float32", low_cpu_mem_usage=True
    )
    model.eval()
    return tokenizer, model


def load_wiki_docs(limit: int) -> list[str]:
    docs: list[str] = []
    with Path("data/sources/wikitext.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(obj.get("text") or "")
            if text.strip():
                docs.append(text.strip())
            if len(docs) >= limit:
                break
    return docs


def split_sequences(seqs, train_frac, seed):
    rng = random.Random(seed)
    indices = list(range(len(seqs)))
    rng.shuffle(indices)
    n_train = max(1, round(len(seqs) * train_frac))
    return [seqs[i] for i in indices[:n_train]], [seqs[i] for i in indices[n_train:]]


def build_ngram(seqs, max_order, shuffle, seed):
    lm = NgramLM(max_order=max_order)
    rng = random.Random(seed) if shuffle else None
    for seq in seqs:
        if shuffle:
            seq = list(seq)
            rng.shuffle(seq)
        lm.add_sequence(seq)
    return lm


def compute_sample(
    model,
    tokenizer,
    eval_seqs,
    real_lm,
    control_lm,
    *,
    sample_size,
    context_len,
    seed,
):
    import torch

    positions = []
    for si, seq in enumerate(eval_seqs):
        for i in range(1, len(seq)):
            positions.append((si, i))
    rng = random.Random(seed)
    if len(positions) > sample_size:
        positions = rng.sample(positions, sample_size)

    base_logits = []
    targets = []
    real_dists = []
    control_dists = []
    for si, i in positions:
        seq = eval_seqs[si]
        ctx = seq[max(0, i - context_len) : i]
        y = seq[i]
        ids = torch.tensor([ctx], dtype=torch.long)
        with torch.no_grad():
            out = model(ids)
            logits = out.logits[0, -1].float().cpu().numpy()
        base_logits.append(logits)
        targets.append(y)
        real_dists.append(real_lm.distribution(seq[:i]))
        control_dists.append(control_lm.distribution(seq[:i]))
    return base_logits, targets, real_dists, control_dists


def single_lambda_search(base_logits, targets, dists, grid=None):
    if grid is None:
        grid = np.linspace(-5.0, 5.0, 101)
    best_lam = 0.0
    best_nll = float("inf")
    for lam in grid:
        total = 0.0
        for logits, target, dist in zip(base_logits, targets, dists):
            fused = fuse_ngram_logits(logits, dist, scale=float(lam), bias=0.0)
            total -= math.log(max(float(softmax(fused)[target]), 1e-12))
        nll = float(total / len(base_logits))
        if nll < best_nll:
            best_nll = nll
            best_lam = float(lam)
    return {"best_lambda": best_lam, "best_mean_nll": best_nll}


def temperature_search(base_logits, targets, dists):
    """Coarse 3D search: temperature × scale × bias."""
    best = {"temperature": 1.0, "scale": 0.0, "bias": 0.0, "best_mean_nll": float("inf")}
    temps = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    scales = np.linspace(-2.0, 8.0, 21)
    biases = np.linspace(-5.0, 5.0, 11)
    for temp in temps:
        for scale in scales:
            for bias in biases:
                total = 0.0
                for logits, target, dist in zip(base_logits, targets, dists):
                    fused = fuse_ngram_logits(
                        logits, dist, scale=float(scale), bias=float(bias), temperature=float(temp)
                    )
                    total -= math.log(max(float(softmax(fused)[target]), 1e-12))
                nll = float(total / len(base_logits))
                if nll < best["best_mean_nll"]:
                    best = {
                        "temperature": float(temp),
                        "scale": float(scale),
                        "bias": float(bias),
                        "best_mean_nll": nll,
                    }
    return best


def nll_for_params(base_logits, targets, dists, *, scale, bias, temperature):
    total = 0.0
    for logits, target, dist in zip(base_logits, targets, dists):
        fused = fuse_ngram_logits(
            logits, dist, scale=scale, bias=bias, temperature=temperature
        )
        total -= math.log(max(float(softmax(fused)[target]), 1e-12))
    return float(total / len(base_logits))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/Users/zeng/code/LLM-CompileForge/models/Qwen/Qwen3.5-0.8B.local-backup")
    parser.add_argument("--wiki-limit", type=int, default=100)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--sample-size", type=int, default=6)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/fusion-calibration.json")
    parser.add_argument("--report", default="outputs/fusion-calibration.md")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = _load_model(args.model_dir)
    docs = load_wiki_docs(args.wiki_limit)
    seqs = [tokenizer.encode(d, add_special_tokens=False) for d in docs if d.strip()]
    train, eval_ = split_sequences(seqs, args.train_frac, args.seed)
    real_lm = build_ngram(train, args.max_order, shuffle=False, seed=args.seed)
    control_lm = build_ngram(train, args.max_order, shuffle=True, seed=args.seed)

    base_logits, targets, real_dists, control_dists = compute_sample(
        model, tokenizer, eval_, real_lm, control_lm,
        sample_size=args.sample_size, context_len=args.context_len, seed=args.seed,
    )

    base_nll = float(
        -np.mean(
            [math.log(max(softmax(logits)[target], 1e-12)) for logits, target in zip(base_logits, targets)]
        )
    )

    def summarize(label, dists):
        single = single_lambda_search(base_logits, targets, dists)
        pair = calibrate_ngram_fusion(
            base_logits, targets, dists,
            scale_grid=np.linspace(-5.0, 5.0, 41),
            bias_grid=np.linspace(-5.0, 5.0, 41),
        )
        triple = temperature_search(base_logits, targets, dists)
        return {
            "label": label,
            "base_nll": base_nll,
            "single_lambda": single,
            "single_lambda_delta_bits": (base_nll - single["best_mean_nll"]) / math.log(2.0),
            "scale_bias": pair,
            "scale_bias_delta_bits": pair["delta_nll_bits"],
            "temp_scale_bias": triple,
            "temp_scale_bias_nll": triple["best_mean_nll"],
            "temp_scale_bias_delta_bits": (base_nll - triple["best_mean_nll"]) / math.log(2.0),
        }

    real_res = summarize("real", real_dists)
    control_res = summarize("control", control_dists)
    results = {
        "schema": "fusion-calibration-v1",
        "sample_size": len(base_logits),
        "context_len": args.context_len,
        "base_nll": base_nll,
        "real": real_res,
        "control": control_res,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 真实 base logits 上的 n-gram 融合校准",
        "",
        f"- Sample size: {len(base_logits)}",
        f"- Context length: {args.context_len}",
        f"- Base NLL: {base_nll:.4f}",
        "",
        "## Real n-gram",
        "",
        "| Method | Best params | NLL | Δ bits |",
        "|---|---:|---:|---:|",
        f"| single λ | λ={real_res['single_lambda']['best_lambda']:.3f} | {real_res['single_lambda']['best_mean_nll']:.4f} | {real_res['single_lambda_delta_bits']:.4f} |",
        f"| scale+bias | scale={real_res['scale_bias']['best_scale']:.3f}, bias={real_res['scale_bias']['best_bias']:.3f} | {real_res['scale_bias']['best_mean_nll']:.4f} | {real_res['scale_bias_delta_bits']:.4f} |",
        f"| temp+scale+bias | T={real_res['temp_scale_bias']['temperature']:.2f}, scale={real_res['temp_scale_bias']['scale']:.3f}, bias={real_res['temp_scale_bias']['bias']:.3f} | {real_res['temp_scale_bias']['best_mean_nll']:.4f} | {real_res['temp_scale_bias_delta_bits']:.4f} |",
        "",
        "## Control n-gram",
        "",
        "| Method | Best params | NLL | Δ bits |",
        "|---|---:|---:|---:|",
        f"| single λ | λ={control_res['single_lambda']['best_lambda']:.3f} | {control_res['single_lambda']['best_mean_nll']:.4f} | {control_res['single_lambda_delta_bits']:.4f} |",
        f"| scale+bias | scale={control_res['scale_bias']['best_scale']:.3f}, bias={control_res['scale_bias']['best_bias']:.3f} | {control_res['scale_bias']['best_mean_nll']:.4f} | {control_res['scale_bias_delta_bits']:.4f} |",
        f"| temp+scale+bias | T={control_res['temp_scale_bias']['temperature']:.2f}, scale={control_res['temp_scale_bias']['scale']:.3f}, bias={control_res['temp_scale_bias']['bias']:.3f} | {control_res['temp_scale_bias']['best_mean_nll']:.4f} | {control_res['temp_scale_bias_delta_bits']:.4f} |",
    ]
    rep.write_text("\n".join(lines), encoding="utf-8")
    print(f"[fusion-cal] wrote {out} and {rep} in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
