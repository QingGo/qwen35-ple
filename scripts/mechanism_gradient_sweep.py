#!/usr/bin/env python3
"""Gradient-residual sweep: PCA compression and high/low residual subsets.

This script pre-registers one set of experiments around the theoretical
quantity ΔR²(Y;E|H) using the LM gradient residual as Y:

    R_t = - ∂L / ∂h_t

It computes:

1. PCA-compressed raw E at r = 16/32/64/128/256;
2. PCA-compressed E_perp at the same r;
3. ΔR² on high-gradient and low-gradient token subsets.

Predictions (pre-registered in docs/round-35):
- If useful memory signal is low-rank, smaller r should retain most ΔR²;
- If signal is diffuse/noisy, ΔR² should drop sharply as r decreases;
- If memory mainly helps hard tokens, high-residual subset should show larger ΔR².
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
                raise RuntimeError("hidden grad is None")
            hs.append(h.detach().cpu().float().numpy()[0])
            gs.append((-grad).detach().cpu().float().numpy()[0])
    finally:
        handle.remove()

    return np.concatenate(hs, axis=0), np.concatenate(gs, axis=0)


def _center(train, test):
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


def _fit_pca(train_x, n_components):
    """Return projection matrix P (n_components x d) learned on train_x."""
    _, _, vt = np.linalg.svd(train_x, full_matrices=False)
    return vt[:n_components]


def _residualize_features(train_h, test_h, train_e, test_e, lam):
    d = train_h.shape[1]
    beta = np.linalg.solve(
        train_h.T @ train_h + lam * np.eye(d),
        train_h.T @ train_e,
    )
    return train_e - train_h @ beta, test_e - test_h @ beta


def _delta_r2(H_tr, H_te, E_tr, E_te, G_tr, G_te, lam):
    """Return (r2_H, r2_HE, delta_r2) using [H,E]."""
    pred_h = _ridge_fit_predict(H_tr, G_tr, H_te, lam)
    r2_h = _r2(G_te, pred_h)
    HE_tr = np.concatenate([H_tr, E_tr], axis=1)
    HE_te = np.concatenate([H_te, E_te], axis=1)
    pred_he = _ridge_fit_predict(HE_tr, G_tr, HE_te, lam)
    r2_he = _r2(G_te, pred_he)
    return r2_h, r2_he, r2_he - r2_h


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--features", default="data/ple-books-160k")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--pca-rs", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--output", default="outputs/mechanism-gradient-sweep.json")
    args = parser.parse_args()

    t0 = time.time()
    _, model = _load_model(args.model)
    print(f"model loaded {time.time()-t0:.1f}s", flush=True)

    raw_tokens, e_t = _load_features(args.features, args.max_tokens)
    seq_tokens = raw_tokens[:-1]
    e_t = e_t[:-1]
    print(f"collecting hidden/grad for {len(seq_tokens)} tokens", flush=True)

    H, G = _collect_hidden_grad(model, seq_tokens, args.layer, batch_size=args.batch_size)
    print(f"collected H {H.shape}, G {G.shape} in {time.time()-t0:.1f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    n = len(H)
    perm = rng.permutation(n)
    cut = int(n * (1.0 - args.test_frac))
    tr_idx = perm[:cut]
    te_idx = perm[cut:]

    H_tr, H_te = _center(H[tr_idx], H[te_idx])
    G_tr, G_te = _center(G[tr_idx], G[te_idx])
    E_tr, E_te = _center(e_t[tr_idx], e_t[te_idx])

    # Baseline: raw E and E_perp without PCA.
    r2_h_base, r2_he_base, delta_e_base = _delta_r2(H_tr, H_te, E_tr, E_te, G_tr, G_te, args.lam)
    Ep_tr, Ep_te = _residualize_features(H_tr, H_te, E_tr, E_te, lam=args.lam)
    _, r2_heperp_base, delta_eperp_base = _delta_r2(H_tr, H_te, Ep_tr, Ep_te, G_tr, G_te, args.lam)

    pca_results = []
    for r in args.pca_rs:
        if r > E_tr.shape[0] or r > E_tr.shape[1]:
            continue
        P = _fit_pca(E_tr, r)
        Epca_tr = E_tr @ P.T
        Epca_te = E_te @ P.T
        _, _, delta_pca = _delta_r2(H_tr, H_te, Epca_tr, Epca_te, G_tr, G_te, args.lam)

        Pperp = _fit_pca(Ep_tr, r)
        Eppca_tr = Ep_tr @ Pperp.T
        Eppca_te = Ep_te @ Pperp.T
        _, _, delta_pca_perp = _delta_r2(H_tr, H_te, Eppca_tr, Eppca_te, G_tr, G_te, args.lam)

        pca_results.append(
            {
                "r": r,
                "delta_r2_raw_pca": delta_pca,
                "delta_r2_eperp_pca": delta_pca_perp,
            }
        )
        print(f"r={r}: raw_pca_delta={delta_pca:.5f}, eperp_pca_delta={delta_pca_perp:.5f}", flush=True)

    # High / low gradient-norm subsets.
    g_norm = np.linalg.norm(G, axis=1)
    q_high = float(np.quantile(g_norm, 0.7))
    q_low = float(np.quantile(g_norm, 0.3))
    high_idx = np.where(g_norm >= q_high)[0]
    low_idx = np.where(g_norm <= q_low)[0]

    subset_results = {}
    for name, idx in [("high", high_idx), ("low", low_idx)]:
        if len(idx) < 60:
            subset_results[name] = {"n": len(idx), "note": "too few samples"}
            continue
        sub_rng = np.random.default_rng(args.seed + 1)
        sub_perm = sub_rng.permutation(len(idx))
        sub_cut = int(len(idx) * (1.0 - args.test_frac))
        s_tr = idx[sub_perm[:sub_cut]]
        s_te = idx[sub_perm[sub_cut:]]
        sH_tr, sH_te = _center(H[s_tr], H[s_te])
        sG_tr, sG_te = _center(G[s_tr], G[s_te])
        sE_tr, sE_te = _center(e_t[s_tr], e_t[s_te])
        _, _, s_delta = _delta_r2(sH_tr, sH_te, sE_tr, sE_te, sG_tr, sG_te, args.lam)
        # orthogonalized
        sEp_tr, sEp_te = _residualize_features(sH_tr, sH_te, sE_tr, sE_te, lam=args.lam)
        _, _, s_delta_perp = _delta_r2(sH_tr, sH_te, sEp_tr, sEp_te, sG_tr, sG_te, args.lam)
        subset_results[name] = {
            "n": len(idx),
            "q": float(np.quantile(g_norm[idx], 0.5)),
            "delta_r2": s_delta,
            "delta_r2_eperp": s_delta_perp,
        }
        print(f"subset {name}: n={len(idx)}, delta={s_delta:.5f}, eperp_delta={s_delta_perp:.5f}", flush=True)

    result = {
        "config": vars(args),
        "baseline": {
            "r2_H": r2_h_base,
            "r2_HE": r2_he_base,
            "r2_HEperp": r2_heperp_base,
            "delta_r2_E": delta_e_base,
            "delta_r2_Eperp": delta_eperp_base,
        },
        "pca": pca_results,
        "subset": subset_results,
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
