#!/usr/bin/env python3
"""Train an MLPValueReader on the LM gradient residual (supervised value path).

The trained checkpoint can be loaded by mechanism_logit_patch.py and used as a
real reader, to test whether residual-supervised nonlinear value injection
improves over LM-loss-only training.

Pre-registered: if residual-supervised MLP reader improves real-vs-control on
BoolQ or rare tokens, nonlinear value path should be pursued.
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

from qwen35_ple.reader import MLPValueReader
from qwen35_ple.reader_registry import (
    MLP_VALUE_V1,
    save_reader,
)


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
    return all_tokens[idx].astype(np.int64), e_t[idx].astype(np.float32)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--features", default="data/ple-books-160k")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-reader", default="outputs/reader-mlp-residual.pt")
    parser.add_argument("--output-metrics", default="outputs/train-mlp-residual.json")
    args = parser.parse_args()

    t0 = time.time()
    _, model = _load_model(args.model)
    print(f"model loaded {time.time()-t0:.1f}s", flush=True)

    raw_tokens, e_t = _load_features(args.features, args.max_tokens)
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

    H_tr = torch.from_numpy(H[tr_idx].astype(np.float32))
    G_tr = torch.from_numpy(G[tr_idx].astype(np.float32))
    E_tr = torch.from_numpy(e_t[tr_idx].astype(np.float32))
    H_te = torch.from_numpy(H[te_idx].astype(np.float32))
    G_te = torch.from_numpy(G[te_idx].astype(np.float32))
    E_te = torch.from_numpy(e_t[te_idx].astype(np.float32))

    reader = MLPValueReader(
        d_model=model.config.hidden_size,
        d_mem=2560,
        hidden=args.hidden,
        zero_init_v=False,
    )
    # Force gate to be wide open for this residual-value experiment.
    with torch.no_grad():
        reader.gate_bias.fill_(10.0)

    opt = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=1e-4)

    def eval_r2():
        reader.eval()
        with torch.no_grad():
            e_perp_te = E_te - reader.h_to_e(H_te)
            pred = reader.value_mlp(e_perp_te).numpy()
        ss_res = float(np.sum((G_te.numpy() - pred) ** 2))
        ss_tot = float(np.sum(G_te.numpy() ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    best = float("-inf")
    best_state = None
    for epoch in range(args.epochs):
        reader.train()
        perm_epoch = torch.randperm(len(H_tr))
        for i in range(0, len(H_tr), args.batch_size):
            idx = perm_epoch[i : i + args.batch_size]
            h = H_tr[idx]
            g = G_tr[idx]
            e = E_tr[idx]
            e_perp = e - reader.h_to_e(h)
            pred = reader.value_mlp(e_perp)
            loss = F.mse_loss(pred, g)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            r2 = eval_r2()
            print(f"epoch {epoch+1}/{args.epochs}: loss={loss.item():.4f} val_r2={r2:.4f}", flush=True)
            if r2 > best:
                best = r2
                best_state = {k: v.clone() for k, v in reader.state_dict().items()}

    final_r2 = eval_r2()
    if best_state is not None:
        reader.load_state_dict(best_state)
    # Save checkpoint with best validation weights.
    cfg = {
        "d_model": model.config.hidden_size,
        "d_mem": 2560,
        "hidden": args.hidden,
        "gate_bias_init": -2.0,
        "zero_init_v": False,
    }
    save_reader(reader, args.output_reader, name=MLP_VALUE_V1, version="1", config=cfg)

    result = {
        "config": vars(args),
        "best_val_r2": best,
        "final_val_r2": final_r2,
        "reader_path": args.output_reader,
        "runtime_seconds": time.time() - t0,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    Path(args.output_metrics).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
