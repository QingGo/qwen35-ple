#!/usr/bin/env python3
"""Mechanism diagnostics: PLE e_t vs Qwen hidden alignment + reader stats.

This script computes, on a small reproducible sample of a precomputed corpus:

* Linear CKA between PLE ``e_t`` and Qwen hidden states at a selected layer.
* Procrustes-style linear alignability (PCA-reduced orthogonal fit).
* kNN local-neighborhood overlap between the two representation spaces.
* Intrinsic-dimension participation ratio for both spaces.
* Reader parameter statistics (trained vs untrained/zero-init).
* Reader gate and output-contribution statistics on the same sample.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
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


def _load_reader(reader_path: str):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from qwen35_ple.reader_registry import load_reader_with_extra
    return load_reader_with_extra(reader_path, device="cpu")


def _load_features(feature_dir: str, max_tokens: int):
    tokens = np.load(Path(feature_dir) / "tokens.npy", mmap_mode="r")
    e_t = np.load(Path(feature_dir) / "e_t.npy", mmap_mode="r")
    n = min(len(tokens), max_tokens)
    stride = max(1, len(tokens) // n)
    idx = np.arange(0, len(tokens), stride)[:n]
    return tokens[idx].astype(np.int64), e_t[idx].astype(np.float32)


def _get_hidden(model, tokens: np.ndarray, layers: list[int], batch_size: int = 256):
    """Run the untouched Qwen model and return hidden states for ``layers``."""
    device = next(model.parameters()).device
    collected = {layer: [] for layer in layers}
    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            ids = torch.from_numpy(tokens[i : i + batch_size]).long().unsqueeze(0).to(device)
            out = model(input_ids=ids, output_hidden_states=True)
            for layer in layers:
                h = out.hidden_states[layer]
                collected[layer].append(h[0].cpu().float().numpy())
    return {layer: np.concatenate(collected[layer], axis=0) for layer in layers}


def _center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = _center(x.astype(np.float64))
    y = _center(y.astype(np.float64))
    kx = x @ x.T
    ky = y @ y.T
    n = kx.shape[0]
    unit = np.ones((n, n)) / n
    kx = kx - unit @ kx - kx @ unit + unit @ kx @ unit
    ky = ky - unit @ ky - ky @ unit + unit @ ky @ unit
    num = float((kx * ky).sum())
    den = float(math.sqrt((kx * kx).sum() * (ky * ky).sum()))
    return num / den if den else float("nan")


def _pca_project(x: np.ndarray, k: int) -> np.ndarray:
    x = _center(x.astype(np.float64))
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :k] * s[:k]


def _procrustes_score(x: np.ndarray, y: np.ndarray, k: int | None = None) -> dict:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    k = k or min(x.shape[1], y.shape[1], x.shape[0])
    xp = _pca_project(x, k)
    yp = _pca_project(y, k)
    u, _, vt = np.linalg.svd(xp.T @ yp, full_matrices=False)
    r = u @ vt
    residual = float(np.linalg.norm(xp @ r - yp) / np.linalg.norm(yp))

    def _explained(a: np.ndarray) -> float:
        a = _center(a)
        ev = np.linalg.svd(a, compute_uv=False) ** 2
        return float(ev[:k].sum() / ev.sum()) if ev.sum() else float("nan")

    return {
        "residual": residual,
        "alignment": 1.0 - residual,
        "explained_x": _explained(x),
        "explained_y": _explained(y),
        "k": k,
    }


def _knn_overlap(x: np.ndarray, y: np.ndarray, k: int = 10, sample_size: int | None = None, seed: int = 0) -> dict:
    n = x.shape[0]
    if sample_size is not None and sample_size < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample_size, replace=False)
        x = x[idx]
        y = y[idx]
        n = sample_size

    def _norm(a):
        a = a.astype(np.float64)
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return a / norms

    xn = _norm(x)
    yn = _norm(y)
    sim_x = xn @ xn.T
    sim_y = yn @ yn.T
    overlaps = []
    for i in range(n):
        sx = np.argsort(-sim_x[i])[1 : k + 1]
        sy = np.argsort(-sim_y[i])[1 : k + 1]
        overlaps.append(len(set(sx.tolist()) & set(sy.tolist())) / k)
    return {
        "k": k,
        "n": n,
        "mean_overlap": float(np.mean(overlaps)),
        "std_overlap": float(np.std(overlaps)),
        "random_baseline": float(k / max(1, n - 1)),
    }


def _participation_ratio(x: np.ndarray) -> float:
    x = _center(x.astype(np.float64))
    ev = np.linalg.svd(x, compute_uv=False) ** 2
    return float((ev.sum() ** 2) / (ev * ev).sum()) if ev.sum() else float("nan")


def _create_fresh_official_reader(checkpoint: dict, source_state_path: str):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from qwen35_ple.reader import OfficialSourceQwenReader

    cfg = checkpoint.get("config", {})
    source = torch.load(source_state_path, map_location="cpu", weights_only=False)
    return OfficialSourceQwenReader(
        d_target=cfg["d_target"],
        d_source=cfg.get("d_source", 2560),
        d_mem=cfg.get("d_mem", 2560),
        hc=cfg.get("hc", 4),
        kernel_size=cfg.get("kernel_size", 4),
        dilation=cfg.get("dilation", 3),
        source_state=source,
        freeze_source=cfg.get("freeze_source", True),
        zero_init_out=cfg.get("zero_init_out", True),
        bridge_mlp=cfg.get("bridge_mlp", False),
        bridge_hidden=cfg.get("bridge_hidden", None),
        out_mlp=cfg.get("out_mlp", False),
        out_hidden=cfg.get("out_hidden", None),
    )


def _param_stats(t: torch.Tensor) -> dict:
    t = t.float()
    return {
        "shape": list(t.shape),
        "norm": float(t.norm().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "abs_mean": float(t.abs().mean().item()),
    }


def _reader_forward_stats(reader, h: np.ndarray, e_t: np.ndarray) -> dict:
    h_t = torch.from_numpy(h).float().unsqueeze(0)
    e_t_t = torch.from_numpy(e_t).float().unsqueeze(0)
    with torch.no_grad():
        b, t, _ = h_t.shape
        key = reader.key_proj(e_t_t)
        key_normed = reader.norm_key(key).view(b, t, reader.hc, reader.d_source)
        query = reader.query_bridge(h_t)
        query_normed = reader.norm_query(query).view(b, t, reader.hc, reader.d_source)
        score = (key_normed * query_normed).sum(-1, keepdim=True) / math.sqrt(reader.d_source)
        score = score.abs().clamp_min(1e-6).sqrt() * score.sign()
        gate = torch.sigmoid(score)
        value = reader.value_proj(e_t_t)
        gated = gate * value.unsqueeze(2)
        gated_flat = gated.reshape(b, t, reader.src_dim)
        gated_normed = reader.norm_conv(gated_flat)
        conv_in = gated_normed.transpose(1, 2)
        pad_len = (reader.kernel_size - 1) * reader.dilation
        conv_in = F.pad(conv_in, (pad_len, 0))
        conv_out = F.silu(reader.conv1d(conv_in)).transpose(1, 2)
        source_output = gated_flat + conv_out
        branch_sum = source_output.view(b, t, reader.hc, reader.d_source).sum(dim=2)
        contribution = reader.out_proj(branch_sum)
    gate_np = gate[0].numpy()
    value_np = value[0].float().numpy()
    contrib_np = contribution[0].float().numpy()
    return {
        "gate_mean": float(gate_np.mean()),
        "gate_std": float(gate_np.std()),
        "gate_min": float(gate_np.min()),
        "gate_max": float(gate_np.max()),
        "gate_by_branch_mean": [float(gate_np[:, i, 0].mean()) for i in range(reader.hc)],
        "gate_fraction_gt_0_1": float((gate_np > 0.1).mean()),
        "gate_fraction_gt_0_5": float((gate_np > 0.5).mean()),
        "value_norm_mean": float(np.linalg.norm(value_np, axis=1).mean()),
        "contribution_norm_mean": float(np.linalg.norm(contrib_np, axis=1).mean()),
        "contribution_norm_std": float(np.linalg.norm(contrib_np, axis=1).std()),
        "hidden_norm_mean": float(np.linalg.norm(h, axis=1).mean()),
        "contribution_over_hidden_mean": float(
            (np.linalg.norm(contrib_np, axis=1) / np.linalg.norm(h, axis=1)).mean()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--features", default="data/ple-books-160k")
    parser.add_argument("--reader", default="outputs/reader-real-seed0.pt")
    parser.add_argument("--source-reader", default="data/official_ple_reader.pt")
    parser.add_argument("--layer", type=int, default=8, help="reader injection layer (kept for compatibility)")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="hidden layers to diagnose; default is [--layer]")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--sample-size", type=int, default=256, help="subsample for kNN/PR")
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/mechanism-alignment.json")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = _load_model(args.model)
    print(f"loaded model in {time.time()-t0:.1f}s", flush=True)

    tokens, e_t = _load_features(args.features, args.max_tokens)
    print(f"loaded {len(tokens)} tokens / e_t {e_t.shape}", flush=True)

    layers = args.layers if args.layers is not None else [args.layer]
    hidden_map = _get_hidden(model, tokens, layers, batch_size=args.batch_size)
    print(f"captured hidden for layers {layers} in {time.time()-t0:.1f}s", flush=True)

    alignment_per_layer = {}
    for layer in layers:
        hidden = hidden_map[layer]
        align_entry = {
            "n_tokens": int(len(tokens)),
            "layer": layer,
            "linear_cka": _linear_cka(e_t, hidden),
            "procrustes": _procrustes_score(e_t, hidden, k=min(256, e_t.shape[1], hidden.shape[1], len(hidden))),
            "knn_overlap": _knn_overlap(e_t, hidden, k=args.knn_k, sample_size=args.sample_size, seed=args.seed),
            "intrinsic_dimension": {
                "ple_participation_ratio": _participation_ratio(e_t),
                "hidden_participation_ratio": _participation_ratio(hidden),
            },
        }
        alignment_per_layer[layer] = align_entry
        print(f"layer {layer} alignment:", align_entry, flush=True)

    # Reader gate/output stats use the reader-layer hidden state.
    reader_hidden = hidden_map[args.layer]

    reader, _extra = _load_reader(args.reader)
    checkpoint = torch.load(args.reader, map_location="cpu", weights_only=False)
    fresh_reader = _create_fresh_official_reader(checkpoint, args.source_reader)

    trainable_params = {}
    for name in ["query_bridge.weight", "out_proj.weight"]:
        trained_t = checkpoint["state_dict"][name]
        fresh_t = getattr(fresh_reader, name.split(".")[0]).weight.detach()
        trainable_params[name] = _param_stats(trained_t)
        trainable_params[name]["fresh_norm"] = float(fresh_t.float().norm().item())
        trainable_params[name]["fresh_mean"] = float(fresh_t.float().mean().item())
        trainable_params[name]["fresh_std"] = float(fresh_t.float().std().item())
        trainable_params[name]["delta_norm_from_fresh"] = float(
            trained_t.float().norm().item() - fresh_t.float().norm().item()
        )
        trainable_params[name]["cosine_to_fresh"] = float(
            F.cosine_similarity(trained_t.float().reshape(-1), fresh_t.float().reshape(-1), dim=0).item()
        )
    frozen_params = {}
    for name in ["key_proj.weight", "value_proj.weight", "norm_key.weight", "norm_query.weight", "norm_conv.weight", "conv1d.weight"]:
        frozen_params[name] = _param_stats(checkpoint["state_dict"][name])

    reader_stats = _reader_forward_stats(reader, reader_hidden, e_t)
    print("reader stats:", {k: v for k, v in reader_stats.items() if isinstance(v, (int, float))}, flush=True)

    result = {
        "config": vars(args),
        "alignment": alignment_per_layer,
        "reader": {
            "checkpoint": args.reader,
            "reader_name": checkpoint.get("reader"),
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "forward_stats": reader_stats,
        },
        "runtime_seconds": time.time() - t0,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
