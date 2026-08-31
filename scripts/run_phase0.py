#!/usr/bin/env python3
"""Phase 0 experiment harness: formal PPL three-arm comparison.

This script establishes the reproducible Phase 0 protocol:

* fixed train/val split (no validation leakage)
* three arms: no-reader baseline, real PLE reader, shuffled control reader
* 3+ seeds with aggregate mean/std
* optional minimal QA log-likelihood probes (TriviaQA / NQ / BoolQ style)
* one command can run the whole matrix

Usage:

    PYTHONPATH=src:../EngramDB/python \
    python scripts/run_phase0.py \
        --features data/ple-adapter-features-20k \
        --steps 20 --seq-len 128 --seeds 0 1 2 \
        --modes no-reader real control \
        --qa --output outputs/phase0.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from qwen35_ple.reader import (
    EngramReader,
    OfficialSourceQwenReader,
    QwenEngramReader,
    ShortConv,
    install_reader_hook,
)
from qwen35_ple.real_ple import resolve_ple_weight_scale
from qwen35_ple.live_store import LiveETStore


DEFAULT_QA = [
    {"task": "triviaqa", "question": "What is the capital of France?", "answer": "Paris"},
    {"task": "triviaqa", "question": "What is the largest planet in the Solar System?", "answer": "Jupiter"},
    {"task": "triviaqa", "question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"task": "nq", "question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    {"task": "nq", "question": "In which country is the city of Kyoto?", "answer": "Japan"},
    {"task": "nq", "question": "What is the currency of Japan?", "answer": "yen"},
    {"task": "boolq", "question": "Is the sky blue?", "answer": "yes"},
    {"task": "boolq", "question": "Can fish fly?", "answer": "no"},
    {"task": "boolq", "question": "Is water wet?", "answer": "yes"},
]


def _install_torch_compat() -> None:
    for name, alias in [
        ("uint16", "int16"),
        ("uint32", "int32"),
        ("uint64", "int64"),
    ]:
        if not hasattr(torch, name):
            setattr(torch, name, getattr(torch, alias))
    if not hasattr(torch, "get_default_device"):
        torch.get_default_device = lambda: torch.device("cpu")  # noqa: E731
    if not hasattr(torch, "set_default_device"):
        torch.set_default_device = lambda device: None  # noqa: E731
    _orig = torch.is_autocast_enabled

    def _autocast(device_type=None):  # noqa: ANN001, ANN202
        return _orig()

    torch.is_autocast_enabled = _autocast
    if not hasattr(torch.nn, "RMSNorm"):
        class _RMSNorm(torch.nn.Module):
            def __init__(self, dim: int, eps: float = 1e-6) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(dim))
                self.eps = eps

            def forward(self, x):
                return (
                    x
                    * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                    * self.weight
                )

        torch.nn.RMSNorm = _RMSNorm

    import typing
    import typing_extensions
    if not hasattr(typing, "override"):
        typing.override = typing_extensions.override

def _load_model(model_path: str, device: str = "cpu"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    model.eval()
    if device != "cpu":
        model = model.to(device)
    return tokenizer, model


def _load_features(feature_dir: Path, model_dir: str, scale: float | None):
    tokens = np.load(feature_dir / "tokens.npy")
    e_t = np.load(feature_dir / "e_t.npy")
    meta_path = feature_dir / "meta.json"
    applied_scale = 1.0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "weight_scale" in meta:
            applied_scale = float(meta["weight_scale"])
        else:
            applied_scale = resolve_ple_weight_scale(model_dir=model_dir, scale=scale)
            e_t = e_t * applied_scale
    elif scale is not None or model_dir:
        applied_scale = resolve_ple_weight_scale(model_dir=model_dir, scale=scale)
        e_t = e_t * applied_scale
    return tokens, e_t, applied_scale


def _split(tokens: np.ndarray, e_t: Any, val_frac: float):
    cut = int(len(tokens) * (1.0 - val_frac))
    if hasattr(e_t, "view"):
        return (tokens[:cut], e_t.view(0, cut)), (tokens[cut:], e_t.view(cut, len(tokens) - cut))
    train = (tokens[:cut], e_t[:cut])
    val = (tokens[cut:], e_t[cut:])
    return train, val

def _e_t_slice(e_t: Any, start: int, length: int) -> np.ndarray:
    """Return e_t[start:start+length], fetching lazily when in live-store mode."""
    if hasattr(e_t, "get"):
        return e_t.get(start, length)
    return e_t[start:start + length]



def _window_loss(
    model,
    tokens: np.ndarray,
    e_t: np.ndarray,
    seq_len: int,
    max_windows: int = 8,
) -> float:
    """Average next-token loss over sampled non-overlapping windows."""
    if len(tokens) < seq_len + 1:
        seq_len = max(1, len(tokens) - 1)
    starts = list(range(0, max(1, len(tokens) - seq_len), seq_len))
    if len(starts) > max_windows:
        idx = np.linspace(0, len(starts) - 1, max_windows).astype(int)
        starts = [starts[i] for i in idx]
    device = next(model.parameters()).device
    losses = []
    with torch.no_grad():
        for start in starts:
            ids = torch.from_numpy(tokens[start : start + seq_len][None, :]).long().to(device)
            ets_np = _e_t_slice(e_t, start, seq_len)
            ets = torch.from_numpy(ets_np[None, :]).float().to(device)
            model._current_ple_e_t = ets
            out = model(input_ids=ids)
            logits = out.logits
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1),
            )
            losses.append(float(loss.item()))
    if not losses:
        return float("nan")
    return float(np.mean(losses))


def _train_reader(
    model,
    reader: EngramReader,
    short_conv: ShortConv | None,
    layer_index: int,
    tokens: np.ndarray,
    e_t: np.ndarray,
    steps: int,
    seq_len: int,
    lr: float,
    seed: int,
) -> list[float]:
    assert len(tokens) > seq_len + 1
    params = [reader] + ([short_conv] if short_conv is not None else [])
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(torch.nn.ModuleList(params).parameters(), lr=lr)
    rng = random.Random(seed)
    losses = []
    for step in range(steps):
        start = rng.randint(0, len(tokens) - seq_len - 1)
        ids = torch.from_numpy(tokens[start : start + seq_len][None, :]).long().to(device)
        ets_np = _e_t_slice(e_t, start, seq_len)
        ets = torch.from_numpy(ets_np[None, :]).float().to(device)
        model._current_ple_e_t = ets
        optimizer.zero_grad()
        out = model(input_ids=ids)
        logits = out.logits
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            ids[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if (step + 1) % 5 == 0 or step == 0:
            print(f"    step {step + 1}/{steps}: loss={loss.item():.4f}")
    return losses


def _qa_inputs(
    tokenizer,
    items: list[dict],
    rows_dir: str,
    tokenizer_path: str,
    model_dir: str,
    scale: float | None,
    control: bool,
    seed: int,
):
    """Build (input_ids, e_t, answer_start) tuples for QA log-likelihood probes."""
    from qwen35_ple.real_ple import precompute_e_t

    out = []
    for idx, item in enumerate(items):
        text = item["question"] + " " + item["answer"]
        tokens, _, et, _meta = precompute_e_t(
            rows_dir=rows_dir,
            tokenizer_path=tokenizer_path,
            texts=[text],
            model_dir=model_dir,
            scale=scale,
        )
        ids = np.asarray(tokens, dtype=np.int64)
        if control:
            rng = np.random.default_rng(seed * 1000 + idx)
            et = et[rng.permutation(len(et))]
        ans_tokens = tokenizer.encode(
            item["answer"], add_special_tokens=False
        )
        answer_start = len(tokenizer.encode(item["question"], add_special_tokens=False))
        out.append(
            {
                "task": item["task"],
                "question": item["question"],
                "answer": item["answer"],
                "ids": ids,
                "e_t": et,
                "answer_start": answer_start,
                "answer_len": len(ans_tokens),
            }
        )
    return out


def _qa_loglik(model, tokenizer, items: list[dict], control: bool, seed: int) -> dict:
    """Return answer-token average log-likelihood per task.

    This is a lightweight Phase 0 signal (not exact-match generation).
    Higher is better; lower loss is better.
    """
    answers = []
    per_task_loss: dict[str, list[float]] = {}
    for idx, item in enumerate(items):
        device = next(model.parameters()).device
        ids = torch.from_numpy(item["ids"]).long().unsqueeze(0).to(device)
        ets = torch.from_numpy(item["e_t"]).float().unsqueeze(0).to(device)
        model._current_ple_e_t = ets
        with torch.no_grad():
            out = model(input_ids=ids)
        logits = out.logits
        start = item["answer_start"]
        end = start + item["answer_len"]
        loss = F.cross_entropy(
            logits[0, start:end].reshape(-1, logits.size(-1)),
            ids[0, start:end],
        )
        val = float(loss.item())
        answers.append({"answer": item["answer"], "loss": val})
        per_task_loss.setdefault(item["task"], []).append(val)

    metrics = {
        f"qa_{task}_loss": float(np.mean(vals)) for task, vals in per_task_loss.items()
    }
    metrics["qa_mean_loss"] = float(np.mean([a["loss"] for a in answers]))
    return {"metrics": metrics, "answers": answers}


def _run_mode(
    args,
    model,
    tokenizer,
    train_tokens,
    train_e_t,
    val_tokens,
    val_e_t,
    mode: str,
    seed: int,
    qa_items,
):
    if mode == "no-reader":
        val_loss = _window_loss(model, val_tokens, val_e_t, args.seq_len)
        qa = _qa_loglik(model, tokenizer, qa_items, control=False, seed=seed) if args.qa else None
        return {
            "mode": mode,
            "seed": seed,
            "val_loss": val_loss,
            "val_ppl": math.exp(val_loss) if val_loss == val_loss else None,
            "qa": qa,
        }

    torch.manual_seed(seed)
    random.seed(seed)
    if args.reader == "official":
        reader = OfficialSourceQwenReader.from_official_checkpoint(
            args.official_reader_path,
            d_target=model.config.hidden_size,
            bridge_mlp=args.bridge_mlp,
            bridge_hidden=args.bridge_hidden,
            out_mlp=args.out_mlp,
            out_hidden=args.out_hidden,
        )
        short_conv = None
    elif args.reader == "engram":
        reader = QwenEngramReader(
            model.config.hidden_size,
            d_mem=2560,
            hc_mult=args.hc_mult,
            kernel_size=args.kernel_size,
            dilation=args.dilation,
            zero_init=args.zero_init_v,
        )
        short_conv = None
    else:
        reader = EngramReader(
            model.config.hidden_size,
            num_branches=args.branches,
            zero_init_v=args.zero_init_v,
        )
        short_conv = ShortConv(model.config.hidden_size) if args.short_conv else None
    if args.device != "cpu":
        reader = reader.to(args.device)
        if short_conv is not None:
            short_conv = short_conv.to(args.device)
    handle = install_reader_hook(model, args.layer, reader, short_conv)

    e_t = train_e_t
    if mode == "control":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(e_t))
        e_t = e_t.permuted(perm) if hasattr(e_t, "permuted") else e_t[perm]

    print(f"  [{mode}] seed={seed} training ...")
    train_losses = _train_reader(
        model,
        reader,
        short_conv,
        args.layer,
        train_tokens,
        e_t,
        steps=args.steps,
        seq_len=args.seq_len,
        lr=args.lr,
        seed=seed,
    )

    val_eval_e_t = val_e_t
    if mode == "control":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(val_e_t))
        val_eval_e_t = val_e_t.permuted(perm) if hasattr(val_e_t, "permuted") else val_e_t[perm]
    val_loss = _window_loss(model, val_tokens, val_eval_e_t, args.seq_len)
    qa = None
    if args.qa:
        # For QA, reuse the same control semantics as training: row permutation.
        qa = _qa_loglik(
            model,
            tokenizer,
            qa_items,
            control=(mode == "control"),
            seed=seed,
        )

    handle.remove()
    return {
        "mode": mode,
        "seed": seed,
        "val_loss": val_loss,
        "val_ppl": math.exp(val_loss) if val_loss == val_loss else None,
        "train_losses": train_losses,
        "train_final_loss": train_losses[-1] if train_losses else None,
        "qa": qa,
    }


def _summarize(results: list[dict], modes: list[str]) -> dict:
    summary = {}
    for mode in modes:
        vals = [r["val_loss"] for r in results if r["mode"] == mode and np.isfinite(r["val_loss"])]
        summary[mode] = {
            "n_seeds": len(vals),
            "val_loss_mean": float(np.mean(vals)) if vals else None,
            "val_loss_std": float(np.std(vals)) if vals else None,
            "val_ppl_mean": float(np.exp(np.mean(vals))) if vals else None,
            "details": [r for r in results if r["mode"] == mode],
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--features", default="data/ple-adapter-features-20k")
    parser.add_argument("--rows-dir", default="/Volumes/My Passport/qwen38-rows")
    parser.add_argument("--model-dir", default="/Volumes/My Passport/qwen38-ple")
    parser.add_argument("--live-store", action="store_true")
    parser.add_argument("--tokens-npy", default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--branches", type=int, default=1)
    parser.add_argument("--reader", choices=["simple", "engram", "official"], default="simple")
    parser.add_argument("--official-reader-path", default="data/official_ple_reader.pt")
    parser.add_argument("--bridge-mlp", action="store_true")
    parser.add_argument("--bridge-hidden", type=int, default=None)
    parser.add_argument("--out-mlp", action="store_true")
    parser.add_argument("--out-hidden", type=int, default=None)
    parser.add_argument("--short-conv", action="store_true")
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=4)
    parser.add_argument("--dilation", type=int, default=3)
    parser.add_argument("--zero-init-v", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["no-reader", "real", "control"],
        default=["no-reader", "real", "control"],
    )
    parser.add_argument("--qa", action="store_true", help="run minimal QA log-likelihood probes")
    parser.add_argument("--output", default="outputs/phase0.json")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    _install_torch_compat()
    feature_dir = Path(args.features)
    live_store_handle = None
    if args.live_store:
        # Live path: read PLE rows directly from EngramDB instead of loading a
        # precomputed e_t.npy.  This uses the fast Store.fetch + torch tensor
        # path (fetch_e_t_tensor) and avoids the slow Python byte expansion.
        if args.tokens_npy:
            tokens = np.load(args.tokens_npy).astype(np.int64)
        elif (feature_dir / "tokens.npy").exists():
            tokens = np.load(feature_dir / "tokens.npy").astype(np.int64)
        else:
            raise SystemExit(
                "--live-store requires --tokens-npy or a --features dir with tokens.npy"
            )

        from qwen35_ple.real_ple import resolve_ple_weight_scale, rowids_from_tokens

        applied_scale = resolve_ple_weight_scale(
            model_dir=args.model_dir, scale=args.scale
        )
        print(
            f"[phase0] live-store: {len(tokens)} tokens, rowids from "
            f"{args.rows_dir} (scale={applied_scale:.6g}) ..."
        )
        t0 = time.time()
        rowids = rowids_from_tokens(tokens)
        rowid_s = time.time() - t0
        import engramdb

        live_store = engramdb.Store(
            args.rows_dir,
            shards=128,
            rows_per_shard=2_500_012,
            width=160,
        )
        live_store_handle = live_store
        # Keep only rowids in memory; e_t is fetched lazily per training/eval
        # window.  This avoids materializing a full 10GB e_t array on WSL.
        e_t = LiveETStore(
            live_store,
            rowids,
            applied_scale,
            store_path=args.rows_dir,
            shards=128,
            rows_per_shard=2_500_012,
            width=160,
        )
        print(
            f"[phase0] live-store ready: {len(tokens)} tokens, "
            f"rowids={len(rowids)}x{len(rowids[0])}, rowid_s={rowid_s:.2f}s, "
            f"no full e_t allocated"
        )
    else:
        print(f"[phase0] loading features from {feature_dir}")
        tokens, e_t, applied_scale = _load_features(feature_dir, args.model_dir, args.scale)
        print(f"[phase0] tokens={len(tokens)} e_t={e_t.shape} scale={applied_scale:.6g}")

    (train_tokens, train_e_t), (val_tokens, val_e_t) = _split(
        tokens, e_t, args.val_frac
    )
    print(
        f"[phase0] split train={len(train_tokens)} val={len(val_tokens)} "
        f"val_frac={args.val_frac}"
    )

    tokenizer, model = _load_model(args.model, args.device)
    for p in model.parameters():
        p.requires_grad_(False)

    qa_items = None
    if args.qa:
        print("[phase0] preparing minimal QA inputs ...")
        qa_items = _qa_inputs(
            tokenizer,
            DEFAULT_QA,
            args.rows_dir,
            args.model,
            args.model_dir,
            args.scale,
            control=False,
            seed=0,
        )

    all_results = []
    for seed in args.seeds:
        print(f"=== seed {seed} ===")
        for mode in args.modes:
            print(f"  mode={mode}")
            res = _run_mode(
                args,
                model,
                tokenizer,
                train_tokens,
                train_e_t,
                val_tokens,
                val_e_t,
                mode,
                seed,
                qa_items,
            )
            all_results.append(res)

    summary = _summarize(all_results, args.modes)
    result = {
        "config": {
            "model": args.model,
            "features": args.features,
            "rows_dir": args.rows_dir,
            "live_store": bool(args.live_store),
            "layer": args.layer,
            "branches": args.branches,
            "reader": args.reader,
            "hc_mult": args.hc_mult,
            "kernel_size": args.kernel_size,
            "dilation": args.dilation,
            "short_conv": bool(args.short_conv),
            "zero_init_v": bool(args.zero_init_v),
            "steps": args.steps,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "val_frac": args.val_frac,
            "seeds": args.seeds,
            "modes": args.modes,
            "weight_scale": applied_scale,
        },
        "summary": summary,
        "results": all_results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if live_store_handle is not None:
        live_store_handle.close()
        print("[phase0] live-store closed")
    print(f"[phase0] saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
