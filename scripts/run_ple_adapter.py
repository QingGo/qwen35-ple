#!/usr/bin/env python3
"""Train a thin PLE-feature adapter on a frozen Qwen3.5 backbone.

This is the next step after the real-PLE knowledge probe.  The experiment:

* Loads a precomputed real ``e_t`` feature file (``data/ple-adapter-features``).
* Freezes all Qwen3.5 backbone parameters.
* Inserts an Engram-style target-side reader after a deep transformer layer:
      hidden = hidden + gate * W_V(e_t)
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
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class ShortConv(torch.nn.Module):
    """Depthwise causal short conv used by DeepSeek Engram / Qwen PLE."""

    def __init__(self, hidden_size: int, kernel_size: int = 2, dilation: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.norm = RMSNorm(hidden_size)
        self.conv = torch.nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            dilation=dilation,
            bias=False,
        )
        torch.nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        normed = self.norm(x)
        conv_in = normed.transpose(1, 2)
        pad_len = (self.kernel_size - 1) * self.dilation
        conv_in = F.pad(conv_in, (pad_len, 0))
        out = F.silu(self.conv(conv_in)).transpose(1, 2)
        return out + x


class EngramReader(torch.nn.Module):
    """Engram-style target-side reader with optional multi-branch keys.

    Follows XMemTransfer's main design:

      shared v = W_V(e_t)
      branch i: k_i = W_K_i(e_t)
                gate_i = sigmoid( (RMSNorm(h) . RMSNorm(k_i)) / sqrt(d_model) + gate_bias_i )
      output = mean_i( gate_i * v )
    """

    def __init__(
        self,
        d_model: int,
        d_mem: int = 2560,
        gate_bias_init: float = -2.0,
        num_branches: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_branches = num_branches
        self.w_v = torch.nn.Linear(d_mem, d_model, bias=False)
        if num_branches == 1:
            self.w_k = torch.nn.Linear(d_mem, d_model, bias=False)
            self.norm_k = RMSNorm(d_model)
        else:
            self.w_k = torch.nn.ModuleList(
                [torch.nn.Linear(d_mem, d_model, bias=False) for _ in range(num_branches)]
            )
            self.norm_k = torch.nn.ModuleList(
                [RMSNorm(d_model) for _ in range(num_branches)]
            )
        self.norm_h = RMSNorm(d_model)
        self.gate_bias = torch.nn.Parameter(
            torch.full((num_branches,), gate_bias_init)
        )
        torch.nn.init.normal_(self.w_v.weight, mean=0.0, std=0.02)
        if num_branches == 1:
            torch.nn.init.normal_(self.w_k.weight, mean=0.0, std=0.02)
        else:
            for proj in self.w_k:
                torch.nn.init.normal_(proj.weight, mean=0.0, std=0.02)

    def forward(self, h, e_t):
        v = self.w_v(e_t)
        norm_h = self.norm_h(h)
        if self.num_branches == 1:
            k = self.w_k(e_t)
            gate_logit = (norm_h * self.norm_k(k)).sum(-1) / math.sqrt(self.d_model)
            gate = torch.sigmoid(gate_logit + self.gate_bias[0])
            return gate.unsqueeze(-1) * v
        contributions = []
        for branch_idx, (proj_k, norm_k) in enumerate(
            zip(self.w_k, self.norm_k)
        ):
            k = proj_k(e_t)
            gate_logit = (norm_h * norm_k(k)).sum(-1) / math.sqrt(self.d_model)
            gate = torch.sigmoid(gate_logit + self.gate_bias[branch_idx])
            contributions.append(gate.unsqueeze(-1) * v)
        return torch.stack(contributions, dim=0).mean(dim=0)


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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--branches", type=int, default=1)
    parser.add_argument("--short-conv", action="store_true")
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

    reader = EngramReader(
        model.config.hidden_size,
        num_branches=args.branches,
    )
    short_conv = ShortConv(model.config.hidden_size) if args.short_conv else None
    layer = model.model.layers[args.layer]

    def post_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        current = getattr(model, "_current_ple_e_t", None)
        if current is not None and current.shape[1] == hidden.shape[1]:
            contribution = reader(hidden, current)
            if short_conv is not None:
                contribution = short_conv(contribution)
            new_hidden = hidden + contribution
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden
        return output

    handle = layer.register_forward_hook(post_hook)

    print(f"[adapter] mode={args.mode} tokens={len(tokens)} layer={args.layer}")
    before_loss = _held_out_loss(model, tokens, e_t, args.seq_len)
    losses = _train(
        model,
        torch.nn.ModuleList([reader] + ([short_conv] if short_conv is not None else [])),
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
        "branches": args.branches,
        "short_conv": bool(args.short_conv),
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
