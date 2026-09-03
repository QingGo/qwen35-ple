#!/usr/bin/env python3
"""Diagnose whether MLPValueReader output depends on E content.

For each token we compare the value path output under:
- real E
- shuffled/control E
- random same-norm E
- zero E

If all cosines are ~1, the value path is effectively ignoring E content.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from qwen35_ple.reader_registry import load_reader_with_extra


def _load_model(model_path: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    if next(model.parameters()).dtype != torch.float32:
        model = model.to(torch.float32)
    model.eval()
    return tokenizer, model


def _load_features(feature_dir: str, max_tokens: int):
    tokens = np.load(Path(feature_dir) / "tokens.npy", mmap_mode="r")
    e_t = np.load(Path(feature_dir) / "e_t.npy", mmap_mode="r")
    n = min(len(tokens), max_tokens)
    stride = max(1, len(tokens) // n)
    idx = np.arange(0, len(tokens), stride)[:n]
    return tokens[idx].astype(np.int64), e_t[idx].astype(np.float32)


def _get_hidden(model, tokens: np.ndarray, layer: int, batch_size: int = 128):
    device = next(model.parameters()).device
    all_hidden = []
    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            ids = torch.from_numpy(tokens[i : i + batch_size]).long().unsqueeze(0).to(device)
            out = model(input_ids=ids, output_hidden_states=True)
            h = out.hidden_states[layer]
            all_hidden.append(h[0].cpu().float().numpy())
    return np.concatenate(all_hidden, axis=0)


def _random_like(real_et: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=real_et.shape).astype(np.float32)
    real_norms = np.linalg.norm(real_et, axis=1, keepdims=True)
    noise_norms = np.linalg.norm(noise, axis=1, keepdims=True)
    noise_norms[noise_norms == 0] = 1.0
    return noise * (real_norms / noise_norms)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--features", default="data/ple-books-160k")
    parser.add_argument("--reader", default="outputs/reader-mlp-residual.pt")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/mech-mlp-dependence.json")
    args = parser.parse_args()

    t0 = time.time()
    _, model = _load_model(args.model)
    reader, _ = load_reader_with_extra(args.reader, device="cpu")
    reader.eval()

    tokens, e_t = _load_features(args.features, args.max_tokens)
    H = _get_hidden(model, tokens, args.layer, batch_size=args.batch_size)
    print(f"collected H {H.shape} in {time.time()-t0:.1f}s", flush=True)

    H_t = torch.from_numpy(H).float()
    E_t = torch.from_numpy(e_t).float()

    # control permutation
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(E_t))
    E_control = E_t[perm]
    E_random = torch.from_numpy(_random_like(e_t, args.seed)).float()
    E_zero = torch.zeros_like(E_t)

    with torch.no_grad():
        e_perp_real = E_t - reader.h_to_e(H_t)
        v_real = reader.value_mlp(torch.cat([H_t, e_perp_real], dim=-1))

        e_perp_control = E_control - reader.h_to_e(H_t)
        v_control = reader.value_mlp(torch.cat([H_t, e_perp_control], dim=-1))

        e_perp_random = E_random - reader.h_to_e(H_t)
        v_random = reader.value_mlp(torch.cat([H_t, e_perp_random], dim=-1))

        e_perp_zero = E_zero - reader.h_to_e(H_t)
        v_zero = reader.value_mlp(torch.cat([H_t, e_perp_zero], dim=-1))

        # gate for real
        k = reader.key_proj(E_t)
        norm_h = reader.norm_h(H_t)
        norm_k = reader.norm_k(k)
        gate_logit = (norm_h * norm_k).sum(-1, keepdim=True) / (reader.d_model ** 0.5)
        gate = torch.sigmoid(gate_logit + reader.gate_bias)

    def cos(a, b):
        a = a.numpy()
        b = b.numpy()
        num = (a * b).sum(axis=1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        return float(np.mean(num / (den + 1e-9)))

    result = {
        "config": vars(args),
        "cos_real_control": cos(v_real, v_control),
        "cos_real_random": cos(v_real, v_random),
        "cos_real_zero": cos(v_real, v_zero),
        "norm_real_mean": float(np.linalg.norm(v_real.numpy(), axis=1).mean()),
        "norm_control_mean": float(np.linalg.norm(v_control.numpy(), axis=1).mean()),
        "norm_random_mean": float(np.linalg.norm(v_random.numpy(), axis=1).mean()),
        "norm_zero_mean": float(np.linalg.norm(v_zero.numpy(), axis=1).mean()),
        "gate_mean": float(gate.numpy().mean()),
        "runtime_seconds": time.time() - t0,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
