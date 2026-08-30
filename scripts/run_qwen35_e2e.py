#!/usr/bin/env python3
"""M0 e2e: Qwen3.5-0.8B + engram-peft PLE-lite on CPU.

This script wraps a locally downloaded Qwen3.5-0.8B with a small synthetic
PLE_QWEN_V1 memory table and runs a forward + generate smoke test.  It is the
fastest end-to-end path available on this Intel Mac because it uses:

* transformers 5.3+ (needed for Qwen3.5) with a small torch-2.2 compatibility
  shim (the machine cannot install torch>=2.4 for macOS x86_64);
* engram-peft submodule loading without the full package ``__init__``, so TRL
  and optional training dependencies are not required for inference.

Usage (after installing a Qwen3.5 checkpoint under data/models and a
``peft`` package):

    PYTHONPATH=/tmp/tf53:/tmp/extra \
    python scripts/run_qwen35_e2e.py \
        --model data/models/Qwen3.5-0.8B

The model directory is intentionally ignored by git (``/data/``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

PRIME_SIZES = [
    17, 19, 23, 29, 31, 37, 41, 43,
    47, 53, 59, 61, 67, 71, 73, 79,
]


def _install_torch_compat() -> None:
    """Shims needed to run transformers 5.3 with torch 2.2.2 on this machine."""
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
    """Load engram-peft without executing its full package __init__."""
    import importlib.machinery
    import importlib.util

    if "engram_peft" in sys.modules:
        # Already loaded by the user; keep it.
        pass
    else:
        pkg = types.ModuleType("engram_peft")
        root = Path(__file__).resolve().parents[1].parent / "engram-peft"
        pkg.__path__ = [str(root / "src" / "engram_peft")]
        pkg.__package__ = "engram_peft"
        pkg.__spec__ = importlib.machinery.ModuleSpec("engram_peft", loader=None)
        sys.modules["engram_peft"] = pkg
        # Ensure peft is importable (needed by engram_peft.utils.peft_patches).
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="data/models/Qwen3.5-0.8B",
        help="local Qwen3.5 checkpoint directory",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--output", default="outputs/qwen35-e2e.json")
    args = parser.parse_args()

    _install_torch_compat()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ecfg, emodel = _load_engram_peft_submodules()

    model_path = Path(args.model)
    if not model_path.is_dir():
        raise SystemExit(f"model directory not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.float32
    )
    model.eval()

    hidden_size = model.config.hidden_size
    config = ecfg.EngramConfig(
        hidden_size=hidden_size,
        embedding_dim=160,
        ngram_sizes=[2, 3],
        n_head_per_ngram=8,
        target_layers=[1],
        engram_vocab_size_per_ngram=[64, 64],
        compressed_vocab_size=model.config.vocab_size,
        pad_id=tokenizer.eos_token_id,
        tokenizer_name_or_path=str(model_path),
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

    engram_model = emodel.get_engram_model(
        model, config, tokenizer, train_mode="engram_only"
    )
    engram_model.eval()

    ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    t0 = time.time()
    with torch.no_grad():
        out = engram_model(ids)
    forward_seconds = time.time() - t0
    logits = out.logits if hasattr(out, "logits") else out
    finite = bool(torch.isfinite(logits).all().item())

    t0 = time.time()
    with torch.no_grad():
        generated = engram_model.generate(
            input_ids=ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    generate_seconds = time.time() - t0
    text = tokenizer.decode(generated[0], skip_special_tokens=True)

    result = {
        "model": str(model_path),
        "hidden_size": hidden_size,
        "engine": "qwen_ple",
        "table_spec": "PLE_QWEN_V1",
        "table_source": "memory",
        "prime_sizes": PRIME_SIZES,
        "forward_shape": list(logits.shape),
        "forward_finite": finite,
        "forward_seconds": forward_seconds,
        "generate_seconds": generate_seconds,
        "max_new_tokens": args.max_new_tokens,
        "prompt": args.prompt,
        "generated": text,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[run_qwen35_e2e] result written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
