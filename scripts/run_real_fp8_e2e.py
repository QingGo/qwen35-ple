#!/usr/bin/env python3
"""Real FP8 PLE e2e smoke using config-driven EngramDB injection.

This script uses the same lightweight engram-peft submodule loader as
``run_qwen35_e2e.py`` so it can run in environments where the full engram-peft
package dependencies (TRL/datasets) are not installed.

Usage:

    PYTHONPATH=src:../EngramDB/python \\
    python scripts/run_real_fp8_e2e.py \\
        --model /path/to/Qwen3.5-0.8B \\
        --store-dir /path/to/qwen38-rows \\
        --ple-model-dir /path/to/qwen38-ple
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

PRIME_SIZES = None  # real PLE table uses the production 20M prime layout


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
    import importlib.util

    if "engram_peft" in sys.modules:
        return

    pkg = types.ModuleType("engram_peft")
    root = Path(__file__).resolve().parents[1].parent / "engram-peft"
    pkg.__path__ = [str(root / "src" / "engram_peft")]
    pkg.__package__ = "engram_peft"
    pkg.__spec__ = importlib.machinery.ModuleSpec("engram_peft", loader=None)
    sys.modules["engram_peft"] = pkg

    try:
        import peft  # noqa: F401
    except ImportError as exc:
        raise SystemExit("engram-peft requires peft; install it first") from exc

    import engram_peft.config  # noqa: F401
    import engram_peft.model  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--store-dir", default="/Volumes/My Passport/qwen38-rows")
    parser.add_argument("--ple-model-dir", default="/Volumes/My Passport/qwen38-ple")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--cache-size", type=int, default=4096)
    parser.add_argument("--output", default="outputs/real-fp8-e2e.json")
    args = parser.parse_args()

    _install_torch_compat()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _load_engram_peft_submodules()

    import engram_peft.config as ecfg
    import engram_peft.model as emodel

    model_path = Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.float32
    )
    model.eval()

    config = ecfg.EngramConfig(
        hidden_size=model.config.hidden_size,
        embedding_dim=2560,
        ngram_sizes=[2, 3],
        n_head_per_ngram=8,
        target_layers=[1],
        engram_vocab_size_per_ngram=[160_000_000, 160_000_000],
        compressed_vocab_size=model.config.vocab_size,
        pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 2,
        tokenizer_name_or_path=str(model_path),
        engine="qwen_ple",
        table_spec="PLE_QWEN_V1",
        table_source="engramdb:store",
        table_store_path=args.store_dir,
        table_model_dir=args.ple_model_dir,
        table_dtype="float8_e4m3fn",
        table_cache_size=args.cache_size,
        enable_tokenizer_compression=False,
        hc_mult=1,
        conv_kernel_size=4,
        conv_dilation=3,
        gating_zero_init=True,
        conv_zero_init=True,
    )

    engram_model = emodel.get_engram_model(
        model, config, tokenizer, train_mode="engram_only"
    )
    engram_model.eval()

    ids = tokenizer("Real FP8 PLE e2e", return_tensors="pt").input_ids
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.steps):
            out = engram_model(ids)
            logits = out.logits if hasattr(out, "logits") else out
            assert torch.isfinite(logits).all(), "non-finite logits"
            generated = engram_model.generate(
                input_ids=ids, max_new_tokens=2, do_sample=False
            )
            assert generated.shape[-1] > ids.shape[-1]
            ids = generated[:, -3:]
    elapsed = time.time() - t0

    result = {
        "model": str(model_path),
        "rows_dir": args.store_dir,
        "ple_model_dir": args.ple_model_dir,
        "steps": args.steps,
        "cache_size": args.cache_size,
        "elapsed_s": elapsed,
        "logits_finite": True,
        "generated_shape": list(generated.shape),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("REAL_FP8_E2E_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
