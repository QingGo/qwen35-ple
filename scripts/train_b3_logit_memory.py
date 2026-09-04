#!/usr/bin/env python3
"""Train the B3 lower-bound logit-space memory head.

This prototype deliberately avoids hidden injection.  It only learns:

    final_logits = base_logits + scale * MemoryLogitHead(memory_feature)

The goal is to measure whether PLE can help at all when the memory head is
allowed to touch the output distribution directly.  If this also fails, the
bottleneck is the PLE information itself, not the hidden channel.

Usage::

    python scripts/train_b3_logit_memory.py \
        --model data/models/Qwen3.5-0.8B \
        --features data/ple-books-160k \
        --bank data/exact-ple-bank.npz \
        --steps 200 \
        --output outputs/b3-logit-real.pt
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qwen35_ple.memory.bank import ExactNgramBank
from qwen35_ple.memory.token_mem import PureLogitMemoryModule


def _load_model(model_path: str, device: str):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model


def _load_features(feature_dir: str, max_tokens: int | None):
    path = Path(feature_dir)
    tokens = np.load(path / "tokens.npy", mmap_mode="r").astype(np.int64)
    e_t = np.load(path / "e_t.npy", mmap_mode="r").astype(np.float32)
    if max_tokens is not None and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        e_t = e_t[:max_tokens]
    return tokens, e_t


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--features", default="data/ple-books-160k")
    parser.add_argument("--bank", default="data/exact-ple-bank.npz")
    parser.add_argument("--output", default="outputs/b3-logit-real.pt")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=256)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()

    tokens, e_t = _load_features(args.features, args.max_tokens)
    bank = ExactNgramBank.load(args.bank)
    _tokenizer, model = _load_model(args.model, args.device)
    print(
        f"[train-b3] model={args.model} tokens={len(tokens)} "
        f"bank_entries={bank.num_entries}",
        flush=True,
    )

    module = PureLogitMemoryModule(
        d_mem=bank.d_mem,
        vocab_size=int(model.config.vocab_size),
        hidden=args.hidden,
    ).to(args.device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr)

    total = len(tokens)
    losses: list[float] = []
    for step in range(args.steps):
        start = rng.randint(0, max(0, total - args.seq_len - 1))
        ids_np = np.asarray(tokens[start : start + args.seq_len], dtype=np.int64)
        e_t_np = np.asarray(e_t[start : start + args.seq_len], dtype=np.float32)
        mem_np, _orders = bank.lookup(ids_np, fallback=e_t_np)

        ids = torch.from_numpy(ids_np).unsqueeze(0).to(args.device)
        mem = torch.from_numpy(mem_np).unsqueeze(0).to(args.device)

        optimizer.zero_grad()
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
            base_logits = out.logits.float()

        final_logits = module(mem, base_logits)
        targets = ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(
            final_logits[:, :-1].reshape(-1, final_logits.size(-1)),
            targets,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"  step {step + 1}/{args.steps}: loss={loss.item():.4f} "
                f"scale={float(module.scale):.4f}",
                flush=True,
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "qwen35-ple-b3-logit-v1",
            "config": vars(args),
            "state_dict": module.state_dict(),
            "losses": losses,
        },
        out,
    )
    meta = {
        "config": vars(args),
        "runtime_seconds": time.time() - t0,
        "final_loss": losses[-1] if losses else None,
        "losses": losses,
        "output": str(out),
    }
    (out.with_suffix(".json")).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[train-b3] saved module to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
