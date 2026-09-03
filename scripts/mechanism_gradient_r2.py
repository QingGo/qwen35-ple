#!/usr/bin/env python3
"""Gradient-residual incremental R² diagnostic.

This version uses the actual LM backprop signal:

    R_t = - ∂L / ∂h_t

at the injection layer.  It then measures whether E (or E_perp) linearly
predicts R after controlling for H.  This is closer to the theory than
predicting next-token embeddings directly.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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


def _collect_hidden_grad(model, tokens: np.ndarray, layer: int, batch_size: int = 64):
    device = next(model.parameters()).device
    layer_module = model.model.layers[layer]
    chunk_hiddens = []

    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h.retain_grad()
        chunk_hiddens.append(h)

    handle = layer_module.register_forward_hook(hook)
    hs: list[np.ndarray] = []
    gs: list[np.ndarray] = []
    vocab = model.config.vocab_size

    try:
        for i in range(0, len(tokens), batch_size):
            ids = torch.from_numpy(tokens[i : i + batch_size]).long().unsqueeze(0).to(device)
            model.zero_grad(set_to_none=True)
            out = model(input_ids=ids)
            logits = out.logits[:, :-1, :]
            targets = ids[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, vocab),
                targets.reshape(-1),
            )
            loss.backward()
            h = chunk_hiddens[-1]
            grad = h.grad
            if grad is None:
                raise RuntimeError("hidden grad is None; gradient path missing")
            hs.append(h.detach().cpu().float().numpy()[0])
            gs.append((-grad).detach().cpu().float().numpy()[0])
            print(f"  chunk {i//batch_size}: loss={loss.item():.4f}", flush=True)
    finally:
        handle.remove()

    return np.concatenate(hs, axis=0), np.concatenate(gs, axis=0)


def _center_train_test(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    return train - mean, test - mean


def _ridge_fit_predict(train_x, train_y, test_x, lam):
    d = train_x.shape[1]
    xtx = train_x.T @ train_x + lam * np.eye(d)
    xty = train_x.T @ train_y
    beta = np.linalg.solve(xtx, xty)
    return test_x @ beta


def _r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum(y_true ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _residualize_features(train_h, test_h, train_e, test_e, lam):
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
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--output", default="outputs/mechanism-gradient-r2.json")
    args = parser.parse_args()

    t0 = time.time()
    _, model = _load_model(args.model)
    print(f"model loaded {time.time()-t0:.1f}s", flush=True)

    tokens, e_t = _load_features(args.features, args.max_tokens)
    # Need one extra token for next-token target positions, but gradient uses
    # the full sequence with shifting, so keep one extra.
    raw_tokens = tokens
    seq_tokens = raw_tokens[:-1]
    e_t = e_t[:-1]
    print(f"collecting hidden/grad for {len(seq_tokens)} tokens", flush=True)

    H, G = _collect_hidden_grad(model, seq_tokens, args.layer, batch_size=args.batch_size)
    print(f"collected H {H.shape}, gradient-signal {G.shape} in {time.time()-t0:.1f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    n = len(H)
    perm = rng.permutation(n)
    cut = int(n * (1.0 - args.test_frac))
    tr_idx = perm[:cut]
    te_idx = perm[cut:]

    H_tr, H_te = _center_train_test(H[tr_idx], H[te_idx])
    G_tr, G_te = _center_train_test(G[tr_idx], G[te_idx])
    E_tr, E_te = _center_train_test(e_t[tr_idx], e_t[te_idx])

    pred_H = _ridge_fit_predict(H_tr, G_tr, H_te, args.lam)
    r2_H = _r2(G_te, pred_H)

    HE_tr = np.concatenate([H_tr, E_tr], axis=1)
    HE_te = np.concatenate([H_te, E_te], axis=1)
    pred_HE = _ridge_fit_predict(HE_tr, G_tr, HE_te, args.lam)
    r2_HE = _r2(G_te, pred_HE)

    Ep_tr, Ep_te = _residualize_features(H_tr, H_te, E_tr, E_te, lam=args.lam)
    HEp_tr = np.concatenate([H_tr, Ep_tr], axis=1)
    HEp_te = np.concatenate([H_te, Ep_te], axis=1)
    pred_HEp = _ridge_fit_predict(HEp_tr, G_tr, HEp_te, args.lam)
    r2_HEperp = _r2(G_te, pred_HEp)

    result = {
        "config": vars(args),
        "n_train": len(tr_idx),
        "n_test": len(te_idx),
        "r2_H": r2_H,
        "r2_HE": r2_HE,
        "r2_HEperp": r2_HEperp,
        "delta_r2_E_given_H": r2_HE - r2_H,
        "delta_r2_Eperp_given_H": r2_HEperp - r2_H,
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
