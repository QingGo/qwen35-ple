#!/usr/bin/env python3
"""Oracle MLP upper-bound diagnostic for gradient-residual prediction.

Pre-registered hypotheses:
- If MLP ΔR² is much larger than linear ΔR²: nonlinear reader is needed.
- If MLP ≈ linear: function capacity is not the bottleneck.
- If MLP on high-gradient tokens improves but linear does not: nonlinearity may
  unlock hard/rare-token information.

This script uses the LM gradient residual as target:
    R_t = -∂L/∂h_t
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
    all_tokens = np.load(Path(feature_dir) / "tokens.npy", mmap_mode="r")
    e_t = np.load(Path(feature_dir) / "e_t.npy", mmap_mode="r")
    n = min(len(all_tokens), max_tokens)
    stride = max(1, len(all_tokens) // n)
    idx = np.arange(0, len(all_tokens), stride)[:n]
    return all_tokens, all_tokens[idx].astype(np.int64), e_t[idx].astype(np.float32)


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


def _residualize_features(train_h, test_h, train_e, test_e, lam):
    d = train_h.shape[1]
    beta = np.linalg.solve(
        train_h.T @ train_h + lam * np.eye(d),
        train_h.T @ train_e,
    )
    return train_e - train_h @ beta, test_e - test_h @ beta


def _supervised_components(train_x, train_y, r):
    cov = train_x.T @ train_y
    u, _, _ = np.linalg.svd(cov, full_matrices=False)
    return u[:, :r].T


def _r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum(y_true ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _train_mlp(train_x, train_y, test_x, test_y, hidden=128, epochs=120, lr=1e-3, seed=0, lam=1e-4):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    x_tr = torch.from_numpy(train_x.astype(np.float32)).to(device)
    y_tr = torch.from_numpy(train_y.astype(np.float32)).to(device)
    x_te = torch.from_numpy(test_x.astype(np.float32)).to(device)

    d_in = x_tr.shape[1]
    d_out = y_tr.shape[1]
    mlp = torch.nn.Sequential(
        torch.nn.Linear(d_in, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, d_out),
    )
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=lam)
    n = x_tr.shape[0]
    best_r2 = float("-inf")
    best_state = None
    for epoch in range(epochs):
        mlp.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i : i + 64]
            opt.zero_grad()
            pred = mlp(x_tr[idx])
            loss = F.mse_loss(pred, y_tr[idx])
            loss.backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            pred_te = mlp(x_te).numpy()
        r2 = _r2(test_y, pred_te)
        if r2 > best_r2:
            best_r2 = r2
            best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
    if best_state is not None:
        mlp.load_state_dict(best_state)
    mlp.eval()
    with torch.no_grad():
        pred_te = mlp(x_te).numpy()
    return _r2(test_y, pred_te)


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
    parser.add_argument("--pls-r", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--output", default="outputs/mechanism-oracle-mlp.json")
    args = parser.parse_args()

    t0 = time.time()
    _, model = _load_model(args.model)
    print(f"model loaded {time.time()-t0:.1f}s", flush=True)

    _, raw_tokens, e_t = _load_features(args.features, args.max_tokens)
    seq_tokens = raw_tokens[:-1]
    e_t = e_t[:-1]

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
    Ep_tr, Ep_te = _residualize_features(H_tr, H_te, E_tr, E_te, lam=args.lam)

    # Linear baselines
    def _lin_r2(X_tr, X_te):
        d = X_tr.shape[1]
        beta = np.linalg.solve(X_tr.T @ X_tr + args.lam * np.eye(d), X_tr.T @ G_tr)
        return _r2(G_te, X_te @ beta)

    r2_h_lin = _lin_r2(H_tr, H_te)
    HE_tr = np.concatenate([H_tr, E_tr], axis=1)
    HE_te = np.concatenate([H_te, E_te], axis=1)
    r2_he_lin = _lin_r2(HE_tr, HE_te)
    HEp_tr = np.concatenate([H_tr, Ep_tr], axis=1)
    HEp_te = np.concatenate([H_te, Ep_te], axis=1)
    r2_heperp_lin = _lin_r2(HEp_tr, HEp_te)

    P = _supervised_components(E_tr, G_tr, args.pls_r)
    Epls_tr = E_tr @ P.T
    Epls_te = E_te @ P.T
    Hpls_tr = np.concatenate([H_tr, Epls_tr], axis=1)
    Hpls_te = np.concatenate([H_te, Epls_te], axis=1)
    r2_hpls_lin = _lin_r2(Hpls_tr, Hpls_te)

    # MLP upper bounds
    print("training MLP: H only", flush=True)
    r2_h_mlp = _train_mlp(H_tr, G_tr, H_te, G_te, hidden=args.hidden, epochs=args.epochs, seed=args.seed)
    print("training MLP: H+E", flush=True)
    r2_he_mlp = _train_mlp(HE_tr, G_tr, HE_te, G_te, hidden=args.hidden, epochs=args.epochs, seed=args.seed + 1)
    print("training MLP: H+Eperp", flush=True)
    r2_heperp_mlp = _train_mlp(HEp_tr, G_tr, HEp_te, G_te, hidden=args.hidden, epochs=args.epochs, seed=args.seed + 2)
    print("training MLP: H+PLS64", flush=True)
    r2_hpls_mlp = _train_mlp(Hpls_tr, G_tr, Hpls_te, G_te, hidden=args.hidden, epochs=args.epochs, seed=args.seed + 3)

    result = {
        "config": vars(args),
        "linear": {
            "r2_H": r2_h_lin,
            "r2_HE": r2_he_lin,
            "r2_HEperp": r2_heperp_lin,
            "r2_HPLS": r2_hpls_lin,
            "delta_HE": r2_he_lin - r2_h_lin,
            "delta_HEperp": r2_heperp_lin - r2_h_lin,
            "delta_HPLS": r2_hpls_lin - r2_h_lin,
        },
        "mlp": {
            "r2_H": r2_h_mlp,
            "r2_HE": r2_he_mlp,
            "r2_HEperp": r2_heperp_mlp,
            "r2_HPLS": r2_hpls_mlp,
            "delta_HE": r2_he_mlp - r2_h_mlp,
            "delta_HEperp": r2_heperp_mlp - r2_h_mlp,
            "delta_HPLS": r2_hpls_mlp - r2_h_mlp,
        },
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
