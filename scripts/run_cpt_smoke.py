#!/usr/bin/env python3
"""M2 CPT training smoke: verify A0 baseline and A1 PLE treatment can train.

This script is intentionally a *smoke*, not an ablation:

* A0 runs the tiny base model directly.
* A1 wraps the same base model with a synthetic PLE_QWEN_V1 layer
  (small ``prime_sizes``, memory table) and trains only the PLE/Engram params.
* Both run a few AdamW steps on one short text and record the LM loss curve.

The output JSON is not an ablation result; it is a trainability gate for the
later A0/A1 CPT experiment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PRIME_SIZES = [
    17, 19, 23, 29, 31, 37, 41, 43,
    47, 53, 59, 61, 67, 71, 73, 79,
]


def _install_compat_shims() -> None:
    import torch

    if not hasattr(torch.nn, "RMSNorm"):
        class _RMSNorm(torch.nn.Module):
            def __init__(self, dim: int, eps: float = 1e-6) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(dim))
                self.eps = eps

            def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def run_smoke(
    model_name: str,
    *,
    ple: bool,
    steps: int,
    text: str,
    lr: float,
    seed: int,
) -> dict:
    _install_compat_shims()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)

    if ple:
        from engram_peft import EngramConfig, get_engram_model

        config = EngramConfig(
            hidden_size=model.config.hidden_size,
            embedding_dim=160,
            ngram_sizes=[2, 3],
            n_head_per_ngram=8,
            target_layers=[1],
            engram_vocab_size_per_ngram=[64, 64],
            compressed_vocab_size=model.config.vocab_size,
            pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 2,
            tokenizer_name_or_path=model_name,
            engine="qwen_ple",
            table_spec="PLE_QWEN_V1",
            table_source="memory",
            enable_tokenizer_compression=False,
            hc_mult=1,
            conv_kernel_size=2,
            conv_dilation=2,
            gating_zero_init=True,
            conv_zero_init=True,
            use_sparse_embeddings=False,
            prime_sizes=PRIME_SIZES,
        )
        trainable_model = get_engram_model(
            model, config, tokenizer, train_mode="engram_only"
        )
        treatment = "A1-ple"
    else:
        trainable_model = model
        treatment = "A0-baseline"

    trainable_model.train()
    params = [p for p in trainable_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)

    ids = tokenizer(text, return_tensors="pt").input_ids
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        out = trainable_model(ids)
        logits = out.logits if hasattr(out, "logits") else out
        loss = F.cross_entropy(logits[0, :-1], ids[0, 1:])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return {
        "model": model_name,
        "treatment": treatment,
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "text": text,
        "trainable_params": sum(p.numel() for p in params),
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    parser.add_argument("--ple", action="store_true", help="run A1 PLE treatment instead of A0 baseline")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/cpt-smoke.json")
    args = parser.parse_args()

    result = run_smoke(
        args.model,
        ple=args.ple,
        steps=args.steps,
        text=args.text,
        lr=args.lr,
        seed=args.seed,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[run_cpt_smoke] result written to {out}")


if __name__ == "__main__":
    main()
