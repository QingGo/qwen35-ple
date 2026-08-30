#!/usr/bin/env python3
"""Small A0/A1 ablation for Qwen3.5-0.8B + PLE-lite on CPU.

This is the first real (still tiny) scientific gate:

* A0 = original Qwen3.5-0.8B, full fine-tuning on a tiny fixed corpus.
* A1 = same Qwen3.5-0.8B plus a synthetic PLE_QWEN_V1 layer, full fine-tuning
  on the same tiny fixed corpus.
* Both use the same optimizer settings and step count.
* We record train loss, held-out loss, and a tiny knowledge/reasoning probe.

The script is intentionally small: it is a go/no-go *smoke*, not a final
ablation report.  If A1 does not show any signal on this mini setup, the next
step is to reconsider the table strategy before scaling.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

PRIME_SIZES = [
    17, 19, 23, 29, 31, 37, 41, 43,
    47, 53, 59, 61, 67, 71, 73, 79,
]

TRAIN_TEXTS = [
    "The capital of France is Paris.",
    "The largest planet in the Solar System is Jupiter.",
    "The chemical symbol for gold is Au.",
    "Alice has three apples and Bob gives her two more, so she has five apples.",
    "Twelve plus fifteen equals twenty-seven.",
]

EVAL_PROMPTS = [
    {
        "category": "knowledge",
        "prompt": "What is the capital of France?",
        "answer": "Paris",
    },
    {
        "category": "knowledge",
        "prompt": "What is the chemical symbol for gold?",
        "answer": "Au",
    },
    {
        "category": "reasoning",
        "prompt": "What is 12 + 15?",
        "answer": "27",
    },
]


def _install_torch_compat() -> None:
    import torch

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

    _original_autocast = torch.is_autocast_enabled

    def _autocast(device_type=None):  # noqa: ANN001, ANN202
        return _original_autocast()

    torch.is_autocast_enabled = _autocast

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


def _load_engram_peft_submodules():
    import importlib.machinery

    if "engram_peft" not in sys.modules:
        pkg = types.ModuleType("engram_peft")
        root = Path(__file__).resolve().parents[1].parent / "engram-peft"
        pkg.__path__ = [str(root / "src" / "engram_peft")]
        pkg.__package__ = "engram_peft"
        pkg.__spec__ = importlib.machinery.ModuleSpec("engram_peft", loader=None)
        sys.modules["engram_peft"] = pkg
        try:
            import peft  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dev dependency
            raise SystemExit(
                "engram-peft requires peft; install it first: "
                "python -m pip install peft"
            ) from exc

    import engram_peft.config as ecfg  # noqa: PLC0415
    import engram_peft.model as emodel  # noqa: PLC0415
    return ecfg, emodel


def _load_base(model_path: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    model.eval()
    return tokenizer, model


def _build_ple_model(emodel, ecfg, model, tokenizer, model_path: str):
    config = ecfg.EngramConfig(
        hidden_size=model.config.hidden_size,
        embedding_dim=160,
        ngram_sizes=[2, 3],
        n_head_per_ngram=8,
        target_layers=[1],
        engram_vocab_size_per_ngram=[64, 64],
        compressed_vocab_size=model.config.vocab_size,
        pad_id=tokenizer.eos_token_id,
        tokenizer_name_or_path=model_path,
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
    return emodel.get_engram_model(
        model, config, tokenizer, train_mode="full_finetune"
    )


def _train_steps(torch, model, tokenizer, texts, steps: int, lr: float, seed: int):
    import torch.nn.functional as F

    torch.manual_seed(seed)
    if hasattr(model, "train"):
        model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    losses: list[float] = []
    for step in range(steps):
        text = texts[step % len(texts)]
        ids = tokenizer(text, return_tensors="pt").input_ids
        optimizer.zero_grad()
        out = model(ids)
        logits = out.logits if hasattr(out, "logits") else out
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            ids[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        print(f"  step {step + 1}/{steps}: loss={loss.item():.4f}")
    if hasattr(model, "eval"):
        model.eval()
    return losses


def _held_out_loss(torch, model, tokenizer, text: str):
    import torch.nn.functional as F

    ids = tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids)
        logits = out.logits if hasattr(out, "logits") else out
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            ids[:, 1:].reshape(-1),
        )
    return float(loss.item())


def _mini_eval(torch, model, tokenizer):
    results = []
    for item in EVAL_PROMPTS:
        ids = tokenizer(item["prompt"], return_tensors="pt").input_ids
        with torch.no_grad():
            generated = model.generate(
                ids,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0][ids.shape[-1]:], skip_special_tokens=True)
        hit = item["answer"].lower() in text.lower()
        results.append(
            {
                "category": item["category"],
                "prompt": item["prompt"],
                "answer": item["answer"],
                "generated": text,
                "correct": hit,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--mode", choices=["a0", "a1"], required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/qwen35-ablation.json")
    args = parser.parse_args()

    _install_torch_compat()
    import torch

    ecfg, emodel = _load_engram_peft_submodules()
    tokenizer, model = _load_base(args.model)

    if args.mode == "a1":
        trainable = _build_ple_model(emodel, ecfg, model, tokenizer, args.model)
    else:
        trainable = model

    held_out = (
        "The library is on the third floor and the classroom is two floors above "
        "the cafeteria, which is on the ground floor."
    )

    before_loss = _held_out_loss(torch, trainable, tokenizer, held_out)
    before_eval = _mini_eval(torch, trainable, tokenizer)

    print(f"[{args.mode}] training {args.steps} steps...")
    losses = _train_steps(
        torch,
        trainable,
        tokenizer,
        TRAIN_TEXTS,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
    )

    after_loss = _held_out_loss(torch, trainable, tokenizer, held_out)
    after_eval = _mini_eval(torch, trainable, tokenizer)

    result = {
        "mode": args.mode,
        "model": args.model,
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "train_losses": losses,
        "held_out_loss_before": before_loss,
        "held_out_loss_after": after_loss,
        "held_out_loss_delta": after_loss - before_loss,
        "eval_before": before_eval,
        "eval_after": after_eval,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[run_qwen35_ablation] result written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
