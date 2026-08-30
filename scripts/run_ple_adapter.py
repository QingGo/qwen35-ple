#!/usr/bin/env python3
"""Train a thin PLE-feature adapter on a frozen Qwen3.5 backbone.

This is the next step after the real-PLE knowledge probe.  The experiment:

* Loads a precomputed real ``e_t`` feature file (``data/ple-adapter-features``).
* Freezes all Qwen3.5 backbone parameters.
* Inserts one tiny trainable adapter into an early transformer layer:
      hidden = hidden + MLP(e_t)
* Trains only the adapter on the next-token LM task.
* Compares real e_t against a shuffled e_t control to test whether the
  *content* of PLE features helps, not just the extra parameters.

Usage:
    PYTHONPATH=/tmp/tf53:/tmp/extra:src:../EngramDB/python \
    python scripts/run_ple_adapter.py --mode real
    python scripts/run_ple_adapter.py --mode control
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


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


def _load_model(model_path: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    model.eval()
    return tokenizer, model


class PleAdapter(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(2560, hidden_size)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, e_t):
        return self.fc2(self.act(self.fc1(e_t)))


def _train(
    model,
    adapter,
    tokens: np.ndarray,
    e_t: np.ndarray,
    steps: int,
    seq_len: int,
    lr: float,
    seed: int,
) -> list[float]:
    import torch.nn.functional as F

    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
    losses: list[float] = []
    total = len(tokens)
    for step in range(steps):
        start = rng.randint(0, max(0, total - seq_len - 1))
        ids = torch.from_numpy(tokens[start : start + seq_len][None, :]).long()
        ets = torch.from_numpy(e_t[start : start + seq_len][None, :]).float()
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
            print(f"  step {step + 1}/{steps}: loss={loss.item():.4f}")
    return losses


def _held_out_loss(model, tokens: np.ndarray, e_t: np.ndarray, seq_len: int) -> float:
    import torch.nn.functional as F

    start = max(0, len(tokens) - seq_len - 1)
    ids = torch.from_numpy(tokens[start : start + seq_len][None, :]).long()
    ets = torch.from_numpy(e_t[start : start + seq_len][None, :]).float()
    model._current_ple_e_t = ets
    with torch.no_grad():
        out = model(input_ids=ids)
        logits = out.logits
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            ids[:, 1:].reshape(-1),
        )
    return float(loss.item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--features", default="data/ple-adapter-features")
    parser.add_argument("--mode", choices=["real", "control"], default="real")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/ple-adapter.json")
    args = parser.parse_args()

    _install_torch_compat()

    tokens = np.load(Path(args.features) / "tokens.npy")
    e_t = np.load(Path(args.features) / "e_t.npy")

    if args.mode == "control":
        rng = np.random.default_rng(args.seed)
        e_t = e_t[rng.permutation(len(e_t))]

    tokenizer, model = _load_model(args.model)
    for p in model.parameters():
        p.requires_grad_(False)

    adapter = PleAdapter(model.config.hidden_size)
    layer = model.model.layers[args.layer]

    def pre_hook(module, args):
        hidden = args[0]
        current = getattr(model, "_current_ple_e_t", None)
        if current is not None and current.shape[1] == hidden.shape[1]:
            delta = adapter(current)
            return (hidden + delta,) + args[1:]
        return args

    handle = layer.register_forward_pre_hook(pre_hook)

    print(f"[adapter] mode={args.mode} tokens={len(tokens)} layer={args.layer}")
    before_loss = _held_out_loss(model, tokens, e_t, args.seq_len)
    losses = _train(
        model,
        adapter,
        tokens,
        e_t,
        steps=args.steps,
        seq_len=args.seq_len,
        lr=args.lr,
        seed=args.seed,
    )
    after_loss = _held_out_loss(model, tokens, e_t, args.seq_len)

    handle.remove()
    result = {
        "mode": args.mode,
        "model": args.model,
        "features": args.features,
        "layer": args.layer,
        "steps": args.steps,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "seed": args.seed,
        "train_losses": losses,
        "held_out_loss_before": before_loss,
        "held_out_loss_after": after_loss,
        "held_out_loss_delta": after_loss - before_loss,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[adapter] saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
