#!/usr/bin/env python3
"""Incremental R² diagnostic: does PLE E add next-token information beyond H?

This implements the theoretical quantity:

    ΔR²(Y; E | H) = R²(Y; [H,E]) - R²(Y; H)

and also the orthogonalized version:

    ΔR²(Y; E⊥ | H),   E⊥ = E - Π_H E

The target Y is the next-token input embedding, a low-dimensional proxy for
the next-token prediction task.  Ridge regression is used so high-dimensional
E does not simply overfit in-sample.

Example (WSL):
    .venv/bin/python scripts/mechanism_incremental_r2.py \
        --features data/ple-books-160k --layer 8 \
        --max-tokens 2048 --test-frac 0.3 --output outputs/mech-incr-r2.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


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


def _center_train_test(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    return train - mean, test - mean


def _ridge_fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, lam: float) -> np.ndarray:
    """Solve (X'X + λI) β = X'Y and predict."""
    d = train_x.shape[1]
    xtx = train_x.T @ train_x + lam * np.eye(d)
    xty = train_x.T @ train_y
    beta = np.linalg.solve(xtx, xty)
    return test_x @ beta


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum(y_true ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _residualize_features(train_h: np.ndarray, test_h: np.ndarray, train_e: np.ndarray, test_e: np.ndarray, lam: float = 1.0):
    """Remove the linear H-predictable part from E, using train-side projection."""
    d = train_h.shape[1]
    beta = np.linalg.solve(
        train_h.T @ train_h + lam * np.eye(d),
        train_h.T @ train_e,
    )
    return train_e - train_h @ beta, test_e - test_h @ beta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--features", default="data/ple-books-160k")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--output", default="outputs/mechanism-incremental-r2.json")
    args = parser.parse_args()

    t0 = time.time()
    _, model = _load_model(args.model)
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

    raw_tokens, e_t = _load_features(args.features, args.max_tokens)
    # Use positions 0..N-2 for predicting next token.
    tokens = raw_tokens[:-1]
    e_t = e_t[:-1]
    next_tokens = raw_tokens[1:]
    print(f"loaded {len(tokens)} samples, e_t {e_t.shape}", flush=True)

    hidden = _get_hidden(model, tokens, args.layer, batch_size=args.batch_size)
    print(f"hidden captured {hidden.shape} in {time.time()-t0:.1f}s", flush=True)

    # Target: next-token embedding.
    embed_weight = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    target = embed_weight[next_tokens]
    print(f"target {target.shape}", flush=True)

    # Train/test split (sequential).
    rng = np.random.default_rng(args.seed)
    n = len(hidden)
    perm = rng.permutation(n)
    cut = int(n * (1.0 - args.test_frac))
    tr_idx = perm[:cut]
    te_idx = perm[cut:]

    H_tr, H_te = _center_train_test(hidden[tr_idx], hidden[te_idx])
    E_tr, E_te = _center_train_test(e_t[tr_idx], e_t[te_idx])
    Y_tr, Y_te = _center_train_test(target[tr_idx], target[te_idx])

    # H only
    H_pred = _ridge_fit_predict(H_tr, Y_tr, H_te, args.lam)
    r2_H = _r2(Y_te, H_pred)

    # [H, E]
    HE_tr = np.concatenate([H_tr, E_tr], axis=1)
    HE_te = np.concatenate([H_te, E_te], axis=1)
    HE_pred = _ridge_fit_predict(HE_tr, Y_tr, HE_te, args.lam)
    r2_HE = _r2(Y_te, HE_pred)

    # [H, E_perp]
    Ep_tr, Ep_te = _residualize_features(H_tr, H_te, E_tr, E_te, lam=args.lam)
    HEp_tr = np.concatenate([H_tr, Ep_tr], axis=1)
    HEp_te = np.concatenate([H_te, Ep_te], axis=1)
    HEp_pred = _ridge_fit_predict(HEp_tr, Y_tr, HEp_te, args.lam)
    r2_HEperp = _r2(Y_te, HEp_pred)

    delta_r2 = r2_HE - r2_H
    delta_r2_perp = r2_HEperp - r2_H

    result = {
        "config": vars(args),
        "n_train": len(tr_idx),
        "n_test": len(te_idx),
        "r2_H": r2_H,
        "r2_HE": r2_HE,
        "r2_HEperp": r2_HEperp,
        "delta_r2_E_given_H": delta_r2,
        "delta_r2_Eperp_given_H": delta_r2_perp,
        "lambda": args.lam,
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
