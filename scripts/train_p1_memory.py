#!/usr/bin/env python3
"""Train the P1 memory interface (TokenMem channel + distribution head + router).

This script keeps the Qwen3.5 backbone completely frozen and trains only the
small P1 memory module on next-token prediction.  The memory inputs come from
the exact longest-match PLE bank; on misses the original PLE ``e_t`` feature is
used as fallback.

Usage::

    python scripts/train_p1_memory.py \
        --model data/models/Qwen3.5-0.8B \
        --features data/ple-books-160k \
        --bank data/exact-ple-bank.npz \
        --steps 200 \
        --output outputs/p1-memory-real.pt
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
from qwen35_ple.memory.token_mem import P1MemoryModule


def _load_model(model_path: str, device: str) -> tuple[object, object]:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    if next(model.parameters()).dtype != torch.float32:
        model = model.to(torch.float32)
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


def _bank_mem(
    bank: ExactNgramBank,
    ids: np.ndarray,
    e_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mem, orders = bank.lookup_multi(ids, fallback=e_t)
    # If no exact match anywhere, fallback already fills the slots with e_t.
    return mem.astype(np.float32), orders


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--features", default="data/ple-books-160k")
    parser.add_argument("--bank", default="data/exact-ple-bank.npz")
    parser.add_argument("--output", default="outputs/p1-memory-real.pt")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--router-hidden", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()

    tokens, e_t = _load_features(args.features, args.max_tokens)
    bank = ExactNgramBank.load(args.bank)
    _tokenizer, model = _load_model(args.model, args.device)
    print(
        f"[train-p1] model={args.model} tokens={len(tokens)} "
        f"bank_entries={bank.num_entries}",
        flush=True,
    )

    module = P1MemoryModule(
        d_model=int(model.config.hidden_size),
        d_mem=bank.d_mem,
        vocab_size=int(model.config.vocab_size),
        n_heads=args.n_heads,
        head_hidden=args.head_hidden,
        router_hidden=args.router_hidden,
    ).to(args.device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr)

    total = len(tokens)
    losses: list[float] = []
    for step in range(args.steps):
        start = rng.randint(0, max(0, total - args.seq_len - 1))
        ids_np = np.asarray(tokens[start : start + args.seq_len], dtype=np.int64)
        e_t_np = np.asarray(e_t[start : start + args.seq_len], dtype=np.float32)
        mem_np, _orders = _bank_mem(bank, ids_np, e_t_np)

        ids = torch.from_numpy(ids_np).unsqueeze(0).to(args.device)
        mem = torch.from_numpy(mem_np).unsqueeze(0).to(args.device)

        optimizer.zero_grad()
        with torch.no_grad():
            out = model(
                input_ids=ids,
                output_hidden_states=True,
                use_cache=not args.no_cache,
            )
            h = out.hidden_states[args.layer][0].float()  # [T,d_model]
            base_logits = out.logits.float()

        fused, _mem_logits, alpha = module(h.unsqueeze(0), mem, base_logits)
        targets = ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(
            fused[:, :-1].reshape(-1, fused.size(-1)),
            targets,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"  step {step + 1}/{args.steps}: loss={loss.item():.4f} "
                f"alpha={float(alpha.mean()):.4f}",
                flush=True,
            )
        if args.save_every and (step + 1) % args.save_every == 0:
            save_path = Path(args.output).with_name(
                f"{Path(args.output).stem}-step{step + 1}.pt"
            )
            torch.save(
                {
                    "format": "qwen35-ple-p1-memory-v1",
                    "config": vars(args),
                    "state_dict": module.state_dict(),
                },
                save_path,
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "qwen35-ple-p1-memory-v1",
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
    meta_path = out.with_suffix(".json")
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[train-p1] saved module to {out}", flush=True)
    print(f"[train-p1] saved meta to {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
