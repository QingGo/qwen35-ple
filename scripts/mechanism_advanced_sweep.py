#!/usr/bin/env python3
"""Advanced gradient-residual sweep: PLS directions and rare/common subsets.

Pre-registered in docs/round-37-preregistered.md.

Metrics:
    ΔR²(Y; E | H), Y = -∂L/∂h_t

Experiments:
1. Supervised/PLS compression of E and E_perp at r = 16/32/64/128/256.
2. Rare-token vs common-token subsets.
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


def _supervised_components(train_x, train_y, r):
    """Top-r right singular directions of X^T Y (max covariance with Y)."""
    cov = train_x.T @ train_y
    u, _, _ = np.linalg.svd(cov, full_matrices=False)
    return u[:, :r].T  # (r, d)


def _delta_r2(H_tr, H_te, E_tr, E_te, G_tr, G_te, lam):
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
    parser.add_argument("--pls-rs", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--output", default="outputs/mechanism-advanced-sweep.json")
    args = parser.parse_args()

    t0 = time.time()
    _, model = _load_model(args.model)
    print(f"model loaded {time.time()-t0:.1f}s", flush=True)

    all_tokens, raw_tokens, e_t = _load_features(args.features, args.max_tokens)
    seq_tokens = raw_tokens[:-1]
    e_t = e_t[:-1]
    print(f"collecting hidden/grad for {len(seq_tokens)} tokens", flush=True)

    H, G = _collect_hidden_grad(model, seq_tokens, args.layer, batch_size=args.batch_size)
    print(f"collected H {H.shape}, G {G.shape} in {time.time()-t0:.1f}s", flush=True)

    # Token frequency from full corpus.
    freq = np.bincount(all_tokens, minlength=model.config.vocab_size)
    token_freq = freq[seq_tokens].astype(np.float64)
    print("token frequency stats:", {
        "median": float(np.median(token_freq)),
        "p20": float(np.quantile(token_freq, 0.2)),
        "p80": float(np.quantile(token_freq, 0.8)),
    }, flush=True)

    rng = np.random.default_rng(args.seed)
    n = len(H)
    perm = rng.permutation(n)
    cut = int(n * (1.0 - args.test_frac))
    tr_idx = perm[:cut]
    te_idx = perm[cut:]

    H_tr, H_te = _center(H[tr_idx], H[te_idx])
    G_tr, G_te = _center(G[tr_idx], G[te_idx])
    E_tr, E_te = _center(e_t[tr_idx], e_t[te_idx])

    _, _, delta_e_base = _delta_r2(H_tr, H_te, E_tr, E_te, G_tr, G_te, args.lam)
    Ep_tr, Ep_te = _residualize_features(H_tr, H_te, E_tr, E_te, lam=args.lam)
    _, _, delta_eperp_base = _delta_r2(H_tr, H_te, Ep_tr, Ep_te, G_tr, G_te, args.lam)

    pls_results = []
    for r in args.pls_rs:
        if r > min(E_tr.shape):
            continue
        P = _supervised_components(E_tr, G_tr, r)
        Eps_tr = E_tr @ P.T
        Eps_te = E_te @ P.T
        _, _, delta_pls = _delta_r2(H_tr, H_te, Eps_tr, Eps_te, G_tr, G_te, args.lam)

        Pp = _supervised_components(Ep_tr, G_tr, r)
        Epps_tr = Ep_tr @ Pp.T
        Epps_te = Ep_te @ Pp.T
        _, _, delta_pls_perp = _delta_r2(H_tr, H_te, Epps_tr, Epps_te, G_tr, G_te, args.lam)

        pls_results.append(
            {
                "r": r,
                "delta_r2_pls_raw": delta_pls,
                "delta_r2_pls_eperp": delta_pls_perp,
            }
        )
        print(f"PLS r={r}: raw={delta_pls:.5f}, eperp={delta_pls_perp:.5f}", flush=True)

    # Rare/common subsets based on current-token frequency.
    q_rare = float(np.quantile(token_freq, 0.2))
    q_common = float(np.quantile(token_freq, 0.8))
    rare_idx = np.where(token_freq <= q_rare)[0]
    common_idx = np.where(token_freq >= q_common)[0]

    subset_results = {}
    for name, idx in [("rare", rare_idx), ("common", common_idx)]:
        if len(idx) < 60:
            subset_results[name] = {"n": len(idx), "note": "too few"}
            continue
        sub_rng = np.random.default_rng(args.seed + 2)
        sub_perm = sub_rng.permutation(len(idx))
        sub_cut = int(len(idx) * (1.0 - args.test_frac))
        s_tr = idx[sub_perm[:sub_cut]]
        s_te = idx[sub_perm[sub_cut:]]
        sH_tr, sH_te = _center(H[s_tr], H[s_te])
        sG_tr, sG_te = _center(G[s_tr], G[s_te])
        sE_tr, sE_te = _center(e_t[s_tr], e_t[s_te])
        _, _, s_delta = _delta_r2(sH_tr, sH_te, sE_tr, sE_te, sG_tr, sG_te, args.lam)
        sEp_tr, sEp_te = _residualize_features(sH_tr, sH_te, sE_tr, sE_te, lam=args.lam)
        _, _, s_delta_perp = _delta_r2(sH_tr, sH_te, sEp_tr, sEp_te, sG_tr, sG_te, args.lam)
        subset_results[name] = {
            "n": len(idx),
            "median_freq": float(np.median(token_freq[idx])),
            "delta_r2": s_delta,
            "delta_r2_eperp": s_delta_perp,
        }
        print(f"subset {name}: n={len(idx)}, delta={s_delta:.5f}, eperp={s_delta_perp:.5f}", flush=True)

    result = {
        "config": vars(args),
        "baseline": {
            "delta_r2_E": delta_e_base,
            "delta_r2_Eperp": delta_eperp_base,
        },
        "pls": pls_results,
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
