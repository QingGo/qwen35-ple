#!/usr/bin/env python3
"""Phase 0 experiment harness: formal PPL three-arm comparison.

This script establishes the reproducible Phase 0 protocol:

* fixed train/val split (no validation leakage)
* three arms: no-reader baseline, real PLE reader, shuffled control reader
* 3+ seeds with aggregate mean/std
* optional QA probes: log-likelihood (`--qa`) and greedy exact-match generation
  (`--qa-exact-match`, with live PLE injection for real/control)
* one command can run the whole matrix

Usage:

    PYTHONPATH=src:../EngramDB/python \
    python scripts/run_phase0.py \
        --features data/ple-adapter-features-20k \
        --steps 20 --seq-len 128 --seeds 0 1 2 \
        --modes no-reader real control \
        --qa --qa-exact-match --output outputs/phase0.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from qwen35_ple.eval.prompting import format_qa_prompt
from qwen35_ple.eval.resume import (
    load_partial_results,
    merge_results,
    write_partial_result,
)
from qwen35_ple.live_store import LiveETStore, LiveETViewStore
from qwen35_ple.reader import (
    EngramReader,
    MLPValueReader,
    OfficialSourceQwenReader,
    QwenEngramReader,
    ShortConv,
    install_reader_hook,
)
from qwen35_ple.reader_registry import (
    ENGRAM_V1,
    MLP_VALUE_V1,
    OFFICIAL_SOURCE_QWEN_V1,
    SIMPLE_V1,
    load_reader_with_extra,
    reader_config_from_args,
    save_reader,
)
from qwen35_ple.real_ple import resolve_ple_weight_scale
from qwen35_ple.serving.bundle import make_bundle, save_bundle

DEFAULT_QA = [
    {
        "task": "triviaqa",
        "question": "What is the capital of France?",
        "answer": "Paris",
    },
    {
        "task": "triviaqa",
        "question": "What is the largest planet in the Solar System?",
        "answer": "Jupiter",
    },
    {
        "task": "triviaqa",
        "question": "What is the chemical symbol for gold?",
        "answer": "Au",
    },
    {
        "task": "nq",
        "question": "Who wrote Romeo and Juliet?",
        "answer": "William Shakespeare",
    },
    {
        "task": "nq",
        "question": "In which country is the city of Kyoto?",
        "answer": "Japan",
    },
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
        torch.get_default_device = lambda: torch.device("cpu")
    if not hasattr(torch, "set_default_device"):
        torch.set_default_device = lambda device: None
    _orig = torch.is_autocast_enabled

    def _autocast(device_type=None):
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


def _resolve_backbone_dtype(name: str):
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[str(name or "float32").lower()]


def _load_model(model_path: str, device: str = "cpu", dtype: str = "float32"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_dtype = _resolve_backbone_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    # Newer Transformers releases may still load the checkpoint in its original
    # bf16 dtype even when dtype=float32 is requested.  The default research path
    # assumes float32 on CPU/GPU; --backbone-dtype bfloat16 is the escape hatch
    # for 2B/4B backbones that do not fit in float32 on one GPU.
    if next(model.parameters()).dtype != target_dtype:
        model = model.to(target_dtype)
    model.eval()
    if device != "cpu":
        model = model.to(device)
    return tokenizer, model


def save_lora_adapter(model, path: Path) -> Path:
    """Persist only the LoRA adapter weights (never a full backbone)."""
    from peft import get_peft_model_state_dict

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(model)
    torch.save({k: v.detach().cpu() for k, v in state.items()}, path / "adapter.pt")
    meta = {
        "format": "peft-state-dict-v1",
        "num_tensors": len(state),
        "num_params": int(sum(v.numel() for v in state.values())),
    }
    (path / "adapter.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_lora_adapter(model, path) -> dict:
    """Load adapter weights saved by :func:`save_lora_adapter` in place."""
    from peft import set_peft_model_state_dict

    path = Path(path)
    state = torch.load(path / "adapter.pt", map_location="cpu", weights_only=True)
    result = set_peft_model_state_dict(model, state)
    meta_path = path / "adapter.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    loaded = len(state)
    return {"path": str(path), "tensors": loaded, "meta": meta, "result": str(result)}


def _apply_lora(model, args) -> dict:
    """Wrap the frozen backbone in LoRA adapters; returns metadata for logging."""
    from peft import LoraConfig, get_peft_model

    raw_targets = str(getattr(args, "lora_target_modules", "") or "").strip()
    if raw_targets in {"", "all-linear"}:
        target_modules = "all-linear"
        target_list: list[str] = ["all-linear"]
    else:
        target_list = [t.strip() for t in raw_targets.split(",") if t.strip()]
        target_modules = target_list
    config = LoraConfig(
        r=int(getattr(args, "lora_r", 16)),
        lora_alpha=int(getattr(args, "lora_alpha", 32)),
        lora_dropout=float(getattr(args, "lora_dropout", 0.0)),
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config)
    trainable = 0
    for name, param in peft_model.named_parameters():
        param.requires_grad_(("lora_" in name) or name.startswith("modules_to_save"))
        if param.requires_grad:
            trainable += int(param.numel())
    return {
        "r": int(config.r),
        "alpha": int(config.lora_alpha),
        "dropout": float(config.lora_dropout),
        "target_modules": target_list,
        "trainable_params": trainable,
    }


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
    if hasattr(e_t, "view") and not isinstance(e_t, np.ndarray):
        return (tokens[:cut], e_t.view(0, cut)), (
            tokens[cut:],
            e_t.view(cut, len(tokens) - cut),
        )
    train = (tokens[:cut], e_t[:cut])
    val = (tokens[cut:], e_t[cut:])
    return train, val


def _e_t_slice(e_t: Any, start: int, length: int) -> np.ndarray:
    """Return e_t[start:start+length], fetching lazily when in live-store mode."""
    if hasattr(e_t, "get"):
        return e_t.get(start, length)
    return e_t[start : start + length]


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
            ids = (
                torch.from_numpy(tokens[start : start + seq_len][None, :])
                .long()
                .to(device)
            )
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
    val_tokens: np.ndarray | None = None,
    val_e_t: np.ndarray | None = None,
    val_every: int = 0,
    max_val_windows: int = 4,
    qa_sft_items: list[dict] | None = None,
    qa_sft_weight: float = 0.0,
    qa_sft_answer_only: bool = True,
    qa_sft_control: bool = False,
    qa_sft_log_every: int = 0,
    gate_reg_weight: float = 0.0,
    qa_sft_warmup_steps: int = 0,
    train_backbone: bool = False,
) -> tuple[list[float], list[dict]]:
    """Train the reader on corpus next-token loss and/or QA answer-only loss.

    ``qa_sft_weight`` is the per-step probability of drawing a QA SFT item
    instead of a corpus window.  ``1.0`` means QA-only; ``0.5`` means an even
    mix; ``0.0`` reproduces the original corpus-only training.
    """
    assert len(tokens) > seq_len + 1
    params = list(reader.parameters())
    if short_conv is not None:
        params += list(short_conv.parameters())
    if train_backbone:
        params += [p for p in model.parameters() if p.requires_grad]
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(params, lr=lr)
    rng = random.Random(seed)
    qa_rng = random.Random(seed * 10007 + 17)
    losses = []
    val_curve: list[dict] = []
    for step in range(steps):
        use_qa = (
            step >= int(qa_sft_warmup_steps)
            and bool(qa_sft_items)
            and qa_sft_weight > 0.0
            and (
                qa_sft_weight >= 1.0 or qa_rng.random() < qa_sft_weight
            )
        )
        if use_qa:
            item = qa_rng.choice(qa_sft_items or [])
            ids = (
                torch.from_numpy(np.asarray(item["ids"], dtype=np.int64)[None, :])
                .long()
                .to(device)
            )
            ets_np = np.asarray(item["e_t"], dtype=np.float32)
            if qa_sft_control:
                perm = np.random.default_rng(seed * 100000 + step).permutation(
                    len(ets_np)
                )
                ets_np = ets_np[perm]
            ets = torch.from_numpy(ets_np[None, :]).float().to(device)
            model._current_ple_e_t = ets
            optimizer.zero_grad()
            out = model(input_ids=ids)
            logits = out.logits
            labels = ids.clone()
            if qa_sft_answer_only:
                labels[:, : int(item["prompt_len"])] = -100
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            loss_kind = "qa"
        else:
            start = rng.randint(0, len(tokens) - seq_len - 1)
            ids = (
                torch.from_numpy(tokens[start : start + seq_len][None, :])
                .long()
                .to(device)
            )
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
            loss_kind = "corpus"
        if gate_reg_weight > 0.0:
            gate_raw = getattr(reader, "last_gate_raw", None)
            if gate_raw is not None:
                loss = loss + gate_reg_weight * gate_raw.mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        log_now = (step + 1) % 5 == 0 or step == 0
        if qa_sft_log_every > 0 and (step + 1) % qa_sft_log_every == 0:
            log_now = True
        if log_now:
            print(
                f"    step {step + 1}/{steps} [{loss_kind}]: "
                f"loss={loss.item():.4f}"
            )
        if (
            val_every > 0
            and (step + 1) % val_every == 0
            and val_tokens is not None
            and val_e_t is not None
        ):
            vloss = _window_loss(
                model,
                val_tokens,
                val_e_t,
                seq_len,
                max_windows=max_val_windows,
            )
            val_curve.append({"step": step + 1, "val_loss": float(vloss)})
            print(f"    step {step + 1}/{steps}: val_loss={vloss:.4f}")
    return losses, val_curve


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
        ans_tokens = tokenizer.encode(item["answer"], add_special_tokens=False)
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


def _parse_head_mask(spec: str | None) -> list[int] | None:
    """Parse a PLE head-mask spec into a list of head indices to zero.

    ``2gram`` keeps the 2-gram heads (zeroes heads 8..15), ``3gram`` keeps the
    3-gram heads (zeroes heads 0..7), and a comma list selects individual heads.
    """
    if spec is None:
        return None
    text = str(spec).strip().lower()
    if text in ("", "none", "full"):
        return None
    if text == "2gram":
        return list(range(8, 16))
    if text == "3gram":
        return list(range(8))
    if text == "all":
        return list(range(16))
    heads: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        head = int(part)
        if head < 0 or head >= 16:
            raise ValueError(f"PLE head index out of range: {head}")
        heads.append(head)
    return sorted(set(heads))


def _apply_head_mask(et: np.ndarray, head_mask: list[int] | None) -> np.ndarray:
    """Zero selected PLE heads in an ``[T, 2560]`` e_t array."""
    if not head_mask:
        return et
    if et.shape[-1] != 2560:
        raise ValueError(f"expected e_t width 2560, got {et.shape[-1]}")
    out = et.reshape(et.shape[0], 16, 160).copy()
    out[:, head_mask, :] = 0.0
    return out.reshape(et.shape[0], 2560)


def _qa_gold_nll(
    model,
    tokenizer,
    items: list[dict],
    qa_store,
    control: bool,
    seed: int,
    prompt_template: str | None = None,
    boolq_prompt_template: str | None = None,
    chat_template: bool = False,
    chat_enable_thinking: bool = False,
    head_mask: list[int] | None = None,
) -> dict:
    """Teacher-forced gold-answer NLL for the format-matched QA prompt.

    This is the format-insensitive complement to exact-match generation: it
    scores only the gold continuation tokens after the same prompt that
    generation uses.  Lower is better.  The continuation is always the raw
    answer tokenization, matching the QA-SFT convention and avoiding a
    leading-space tokenization mismatch between arms.
    """
    device = next(model.parameters()).device
    answers: list[dict[str, Any]] = []
    per_task_nll: dict[str, list[float]] = {}
    total_tokens = 0
    total_loss = 0.0
    for idx, item in enumerate(items):
        prompt_ids = _qa_prompt_ids(
            tokenizer,
            item,
            prompt_template,
            boolq_prompt_template,
            chat_template=chat_template,
            chat_enable_thinking=chat_enable_thinking,
        )
        answer = str(item.get("answer", ""))
        continuation = answer
        cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
        if not prompt_ids or not cont_ids:
            continue
        full_ids = list(prompt_ids) + list(cont_ids)
        start = len(prompt_ids)
        end = len(full_ids)

        if qa_store is not None:
            et = qa_store.fetch(full_ids)
            if control:
                rng = np.random.default_rng(seed * 1000 + idx)
                et = et[rng.permutation(len(et))]
            et = _apply_head_mask(et, head_mask)
            model._current_ple_e_t = (
                torch.from_numpy(et).float().unsqueeze(0).to(device)
            )
        else:
            model._current_ple_e_t = None

        ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(ids)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attention_mask)
        logits = out.logits[0]
        shift_logits = logits[start - 1 : end - 1, :]
        labels = ids[0, start:end]
        loss = F.cross_entropy(shift_logits, labels)
        value = float(loss.item())
        n_tokens = int(end - start)
        answers.append(
            {
                "task": str(item.get("task", "unknown")),
                "question": str(item.get("question", "")),
                "answer": answer,
                "gold_nll": value,
                "n_tokens": n_tokens,
                "continuation": continuation,
            }
        )
        per_task_nll.setdefault(str(item.get("task", "unknown")), []).append(value)
        total_loss += value * n_tokens
        total_tokens += n_tokens
        if (idx + 1) % 200 == 0:
            print(f"    gold-NLL {idx + 1}/{len(items)}", flush=True)

    metrics = {
        f"qa_{task}_nll": float(np.mean(vals))
        for task, vals in sorted(per_task_nll.items())
    }
    metrics["qa_mean_nll"] = (
        float(np.mean([a["gold_nll"] for a in answers])) if answers else float("nan")
    )
    metrics["qa_token_mean_nll"] = (
        float(total_loss / total_tokens) if total_tokens else float("nan")
    )
    metrics["qa_n"] = float(len(answers))
    return {"metrics": metrics, "answers": answers}


def _load_qa_file(path: str | Path | None) -> list[dict]:
    """Load a QA file that matches the Phase 0 default schema.

    Accepts either a JSON list or JSONL (one JSON object per line).  The
    Phase 1 KB split uses JSONL for ``qa.train.jsonl`` / ``qa.eval.jsonl``.
    """
    if path is None:
        return list(DEFAULT_QA)
    text = Path(path).read_text(encoding="utf-8")
    data: list[Any]
    if text.lstrip().startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise TypeError("--qa-file must be a JSON list or JSONL records")
        data = parsed
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    out = []
    for item in data:
        if not isinstance(item, dict):
            raise TypeError("--qa-file items must be JSON objects")
        if "question" not in item or "answer" not in item:
            raise ValueError("each --qa-file item must contain 'question' and 'answer'")
        out.append(
            {
                "task": str(item.get("task", "qa")),
                "question": str(item["question"]),
                "answer": str(item["answer"]),
            }
        )
    return out


def _load_qa_sft_file(path: str | Path) -> list[dict]:
    """Load QA SFT data from a JSON list or JSONL file.

    Each record must contain ``question`` and ``answer``; ``task`` is optional
    and is used for the BoolQ prompt override.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    records: list[dict] = []
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise TypeError(f"{path}: expected a JSON list")
        records = data
    else:
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError(f"{path}:{line_no}: expected a JSON object")
            records.append(item)
    out: list[dict] = []
    for item in records:
        if "question" not in item or "answer" not in item:
            raise ValueError(
                f"{path}: each QA SFT item must contain 'question' and 'answer'"
            )
        out.append(
            {
                "task": str(item.get("task", "qa")),
                "question": str(item["question"]),
                "answer": str(item["answer"]),
            }
        )
    return out



class _QAEtStore:
    """Persistent EngramDB reader for arbitrary QA-token sequences.

    This is used by the exact-match generation path.  It opens the Store once
    and fetches the PLE rows for the currently generated token sequence on every
    decoding step, so we can inject e_t for both the prompt and generated
    tokens without materializing a full e_t array.
    """

    def __init__(self, rows_dir: str, scale: float) -> None:
        import engramdb

        from qwen35_ple.real_ple import real_spec

        spec = real_spec()
        self.store = engramdb.Store(
            rows_dir,
            shards=spec.shards,
            rows_per_shard=spec.rows_per_shard,
            width=160,
        )
        self.scale = float(scale)

    def fetch(self, ids: list[int] | np.ndarray) -> np.ndarray:
        import engramdb

        from qwen35_ple.real_ple import rowids_from_tokens

        rowids = rowids_from_tokens(np.asarray(ids, dtype=np.int64))
        arr = engramdb.fetch_e_t_tensor(
            self.store,
            rowids.reshape(-1).tolist(),
            scale=self.scale,
            num_heads=16,
            head_dim=160,
            dtype=None,
            out_dtype=None,
        )
        return arr.reshape(len(ids), 2560).numpy()

    def fetch_many(self, sequences: list[list[int]]) -> list[np.ndarray]:
        """Fetch e_t for several token sequences with one Store call.

        The returned list has one ``[len(sequence), 2560]`` array per input
        sequence.  Batching the rowids avoids one Store round-trip per sequence
        during batched generation.
        """
        import engramdb

        from qwen35_ple.real_ple import rowids_from_tokens

        lengths = [len(seq) for seq in sequences]
        if not lengths:
            return []
        flat_parts = [
            rowids_from_tokens(np.asarray(seq, dtype=np.int64)).reshape(-1)
            for seq in sequences
        ]
        flat = np.concatenate(flat_parts) if len(flat_parts) > 1 else flat_parts[0]
        arr = (
            engramdb.fetch_e_t_tensor(
                self.store,
                flat.tolist(),
                scale=self.scale,
                num_heads=16,
                head_dim=160,
                dtype=None,
                out_dtype=None,
            )
            .reshape(sum(lengths), 2560)
            .numpy()
        )
        out: list[np.ndarray] = []
        offset = 0
        for length in lengths:
            out.append(arr[offset : offset + length])
            offset += length
        return out

    def close(self) -> None:
        self.store.close()


def _build_qa_sft_cache(
    items: list[dict],
    tokenizer,
    qa_store: _QAEtStore,
    prompt_template: str | None,
    boolq_prompt_template: str | None,
    max_len: int,
    eos_id: int | None,
    lazy: bool = False,
    chat_template: bool = False,
    chat_enable_thinking: bool = False,
    chat_eos_id: int | None = None,
) -> list[dict]:
    """Tokenize QA SFT items and (optionally) fetch their PLE rows.

    The returned items contain ``ids`` (prompt + answer + EOS), ``prompt_len``
    so the training loop can mask prompt tokens, and ``e_t`` unless
    ``lazy=True``.  In lazy mode the caller wraps the result in
    :class:`_LazyQASFTCache`, which fetches PLE rows on demand and keeps only a
    small LRU cache in memory.  This is required for the 6000-item standard
    SFT split, whose full e_t materialization would be tens of GB.
    """
    cache: list[dict] = []
    skipped = 0
    for item in items:
        if chat_template:
            prompt_ids = _qa_prompt_ids(
                tokenizer,
                item,
                None,
                None,
                chat_template=True,
                chat_enable_thinking=chat_enable_thinking,
            )
        else:
            prompt_text = format_qa_prompt(
                item["question"],
                task=item.get("task"),
                template=prompt_template,
                boolq_template=boolq_prompt_template,
            )
            prompt_ids = list(
                tokenizer.encode(prompt_text, add_special_tokens=False)
            )
        answer_ids = list(
            tokenizer.encode(str(item["answer"]), add_special_tokens=False)
        )
        if not answer_ids:
            skipped += 1
            continue
        answer_eos = chat_eos_id if chat_template else eos_id
        if answer_eos is not None:
            answer_ids = answer_ids + [int(answer_eos)]
        # Keep the tail of the prompt (question + instructions) when the
        # passage is too long.  Dropping the beginning of a long BoolQ passage
        # is preferable to dropping the question/answer pair entirely.
        budget = int(max_len) - len(answer_ids)
        if budget < 1:
            skipped += 1
            continue
        if len(prompt_ids) > budget:
            prompt_ids = prompt_ids[-budget:]
        ids = prompt_ids + answer_ids
        entry: dict = {
            "task": item.get("task", "qa"),
            "question": item["question"],
            "answer": item["answer"],
            "ids": np.asarray(ids, dtype=np.int64),
            "prompt_len": len(prompt_ids),
            "answer_len": len(answer_ids),
        }
        if not lazy:
            e_t = qa_store.fetch(ids)
            if e_t.shape[0] != len(ids):
                skipped += 1
                continue
            entry["e_t"] = np.asarray(e_t, dtype=np.float32)
        cache.append(entry)
    if skipped:
        print(f"[phase0] QA SFT cache skipped {skipped}/{len(items)} items")
    return cache


class _LazyQASFTCache:
    """Sequence of QA SFT items whose PLE rows are fetched on demand.

    ``random.Random.choice`` only needs ``__len__`` / ``__getitem__``.  The
    small LRU cache bounds memory while still avoiding repeated Store calls for
    the most recently sampled items.
    """

    def __init__(self, items: list[dict], qa_store: _QAEtStore, max_cached: int = 128):
        self.items = items
        self.qa_store = qa_store
        self.max_cached = max(1, int(max_cached))
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        cached = self._cache.get(idx)
        if cached is not None:
            return cached
        item = dict(self.items[idx])
        item["e_t"] = self.qa_store.fetch(item["ids"])
        if len(self._cache) >= self.max_cached:
            self._cache.pop(next(iter(self._cache)))
        self._cache[idx] = item
        return item



_NUMBER_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_NUMBER_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def _expand_number_words(text: str) -> str:
    """Convert English number words in a normalized phrase to digits."""
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        if (
            words[i] in _NUMBER_UNITS
            or words[i] in _NUMBER_TENS
            or words[i] in {"hundred", "thousand"}
        ):
            total = 0
            current = 0
            while i < len(words):
                w = words[i]
                if w in _NUMBER_UNITS:
                    current += _NUMBER_UNITS[w]
                elif w == "hundred":
                    current *= 100
                elif w in _NUMBER_TENS:
                    current += _NUMBER_TENS[w]
                elif w == "thousand":
                    total += current * 1000
                    current = 0
                else:
                    break
                i += 1
            total += current
            out.append(str(total))
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def _normalize_answer(text: str) -> str:
    """SQuAD-style normalization with number-word canonicalization."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    words = [w for w in text.split() if w not in {"a", "an", "the"}]
    return _expand_number_words(" ".join(words))


def _qa_prompt_ids(
    tokenizer,
    item: dict,
    prompt_template: str | None,
    boolq_prompt_template: str | None,
    chat_template: bool = False,
    chat_enable_thinking: bool = False,
) -> list[int]:
    """Return prompt token IDs for one QA item.

    ``chat_template=False`` keeps the Phase 0/2 completion-style prompt.  When
    true, use the tokenizer's native chat template (optionally with thinking),
    which is the model's in-distribution instruction format.
    """
    if chat_template:
        messages = [{"role": "user", "content": str(item["question"])}]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=bool(chat_enable_thinking),
        )
        # Transformers versions differ: some return a list, others a
        # BatchEncoding/dict with an ``input_ids`` field.
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        elif hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        return [int(x) for x in encoded]
    return list(
        tokenizer.encode(
            format_qa_prompt(
                item["question"],
                task=item.get("task"),
                template=prompt_template,
                boolq_template=boolq_prompt_template,
            ),
            add_special_tokens=False,
        )
    )


def _qa_exact_match(
    model,
    tokenizer,
    items: list[dict],
    qa_store: _QAEtStore | None,
    control: bool,
    seed: int,
    max_new_tokens: int,
    batch_size: int = 1,
    max_batch_tokens: int = 0,
    prompt_template: str | None = None,
    boolq_prompt_template: str | None = None,
    chat_template: bool = False,
    chat_enable_thinking: bool = False,
    head_mask: list[int] | None = None,
) -> dict:
    """Greedy exact-match QA generation with live PLE injection.

    For real/control, every decoding step fetches the e_t rows for the current
    token sequence and injects them through the installed reader hook.  For
    no-reader, ``qa_store`` is None and the same greedy loop runs without PLE.

    Questions are grouped into batches.  ``batch_size`` is the maximum number
    of questions per batch and ``max_batch_tokens`` is an optional token budget
    (roughly ``batch_size * padded_length``).  Grouping by prompt length and
    token budget keeps padding bounded and avoids OOM on long BoolQ passages.
    """
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0
    stop_ids: set[int] = set()
    if eos_id is not None:
        stop_ids.add(int(eos_id))
    if chat_template:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0:
            stop_ids.add(int(im_end_id))
    batch_size = max(1, int(batch_size))
    max_batch_tokens = max(0, int(max_batch_tokens))

    prompt_ids = {
        idx: _qa_prompt_ids(
            tokenizer,
            item,
            prompt_template,
            boolq_prompt_template,
            chat_template=chat_template,
            chat_enable_thinking=chat_enable_thinking,
        )
        for idx, item in enumerate(items)
    }
    ordered = sorted(enumerate(items), key=lambda pair: len(prompt_ids[pair[0]]))

    batches: list[list[tuple[int, dict]]] = []
    current: list[tuple[int, dict]] = []
    current_max = 0
    for idx, item in ordered:
        prompt_len = len(prompt_ids[idx])
        next_max = max(current_max, prompt_len)
        over_items = len(current) + 1 > batch_size
        over_tokens = (
            max_batch_tokens > 0
            and (next_max + max_new_tokens) * (len(current) + 1) > max_batch_tokens
        )
        if current and (over_items or over_tokens):
            batches.append(current)
            current = []
            current_max = 0
        current.append((idx, item))
        current_max = max(current_max, prompt_len)
    if current:
        batches.append(current)

    records: list[tuple[int, dict, bool]] = []
    for batch in batches:
        states = []
        for idx, item in batch:
            print(
                f"    QA {idx + 1}/{len(items)} [{item['task']}] {item['question'][:90]}",
                flush=True,
            )
            states.append(
                {
                    "idx": idx,
                    "item": item,
                    "ids": list(prompt_ids[idx]),
                    "generated": [],
                    "finished": False,
                }
            )

        with torch.no_grad():
            for _ in range(max_new_tokens):
                active = [state for state in states if not state["finished"]]
                if not active:
                    break
                max_len = max(len(state["ids"]) for state in active)
                input_ids = torch.full(
                    (len(active), max_len),
                    int(pad_id),
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.zeros(
                    (len(active), max_len), dtype=torch.long, device=device
                )
                for row, state in enumerate(active):
                    length = len(state["ids"])
                    input_ids[row, :length] = torch.tensor(
                        state["ids"], dtype=torch.long, device=device
                    )
                    attention_mask[row, :length] = 1

                if qa_store is not None:
                    et_list = qa_store.fetch_many([state["ids"] for state in active])
                    e_t_padded = np.zeros(
                        (len(active), max_len, 2560), dtype=np.float32
                    )
                    for row, (state, et_np) in enumerate(
                        zip(active, et_list, strict=True)
                    ):
                        if control:
                            rng = np.random.default_rng(seed * 1000 + state["idx"])
                            et_np = et_np[rng.permutation(len(et_np))]
                        et_np = _apply_head_mask(et_np, head_mask)
                        e_t_padded[row, : len(state["ids"])] = et_np
                    model._current_ple_e_t = (
                        torch.from_numpy(e_t_padded).float().to(device)
                    )
                else:
                    model._current_ple_e_t = None

                out = model(input_ids=input_ids, attention_mask=attention_mask)
                for row, state in enumerate(active):
                    logits = out.logits[row, len(state["ids"]) - 1]
                    next_id = int(torch.argmax(logits).item())
                    if next_id in stop_ids:
                        state["finished"] = True
                    else:
                        state["generated"].append(next_id)
                        state["ids"].append(next_id)

        for state in states:
            item = state["item"]
            generated_text = tokenizer.decode(
                state["generated"], skip_special_tokens=True
            )
            hit = _normalize_answer(item["answer"]) in _normalize_answer(generated_text)
            records.append(
                (
                    state["idx"],
                    {
                        "task": item["task"],
                        "question": item["question"],
                        "answer": item["answer"],
                        "generated": generated_text,
                        "correct": hit,
                    },
                    hit,
                )
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    records.sort(key=lambda record: record[0])
    answers = [record[1] for record in records]
    per_task_correct: dict[str, list[bool]] = {}
    for _, answer, hit in records:
        per_task_correct.setdefault(answer["task"], []).append(hit)

    metrics: dict[str, float] = {}
    for task, hits in sorted(per_task_correct.items()):
        metrics[f"qa_{task}_em"] = float(np.mean(hits))
    all_hits = [answer["correct"] for answer in answers]
    metrics["qa_em_mean"] = float(np.mean(all_hits)) if all_hits else float("nan")
    metrics["qa_n"] = float(len(all_hits))
    return {"metrics": metrics, "answers": answers}


def _format_reader_save_path(template: str, mode: str, seed: int) -> Path:
    """Expand ``{mode}`` / ``{seed}`` placeholders in a checkpoint path."""
    return Path(str(template).replace("{mode}", mode).replace("{seed}", str(seed)))


def _bundle_memory_from_args(args: Any) -> dict[str, Any] | None:
    """Build an EngramDB bundle memory section from Phase 0 CLI args."""
    if getattr(args, "store_p_view", None):
        memory: dict[str, Any] = {
            "type": "view",
            "view": {"path": str(Path(args.store_p_view).resolve())},
        }
        if getattr(args, "store_p_slot_index", None):
            memory["slot_index"] = {
                "type": "disk",
                "path": str(Path(args.store_p_slot_index).resolve()),
            }
        else:
            memory["sequential_view"] = True
        return memory
    if getattr(args, "live_store", False) and getattr(args, "rows_dir", None):
        return {
            "type": "store",
            "store": {
                "path": str(Path(args.rows_dir).resolve()),
                "shards": 128,
                "rows_per_shard": 2_500_012,
                "width": 160,
            },
        }
    return None


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
    qa_exact_items,
    qa_store,
    lora_meta: dict | None = None,
):
    head_mask = _parse_head_mask(getattr(args, "qa_head_mask", None))
    if mode == "no-reader":
        val_loss = _window_loss(model, val_tokens, val_e_t, args.seq_len)
        qa = (
            _qa_loglik(model, tokenizer, qa_items, control=False, seed=seed)
            if args.qa
            else None
        )
        qa_exact = None
        if args.qa_exact_match:
            qa_exact = _qa_exact_match(
                model,
                tokenizer,
                qa_exact_items,
                None,
                control=False,
                seed=seed,
                max_new_tokens=args.qa_max_new_tokens,
                batch_size=getattr(args, "qa_batch_size", 1),
                max_batch_tokens=getattr(args, "qa_batch_max_tokens", 0),
                prompt_template=getattr(args, "qa_prompt_template", None),
                boolq_prompt_template=getattr(
                    args, "qa_boolq_prompt_template", None
                ),
                chat_template=bool(getattr(args, "qa_chat_template", False)),
                chat_enable_thinking=bool(
                    getattr(args, "qa_chat_enable_thinking", False)
                ),
                head_mask=head_mask,
            )
        qa_gold = None
        if getattr(args, "qa_gold_nll", False) and qa_exact_items:
            qa_gold = _qa_gold_nll(
                model,
                tokenizer,
                qa_exact_items,
                None,
                control=False,
                seed=seed,
                prompt_template=getattr(args, "qa_prompt_template", None),
                boolq_prompt_template=getattr(
                    args, "qa_boolq_prompt_template", None
                ),
                chat_template=bool(getattr(args, "qa_chat_template", False)),
                chat_enable_thinking=bool(
                    getattr(args, "qa_chat_enable_thinking", False)
                ),
                head_mask=head_mask,
            )
        return {
            "mode": mode,
            "seed": seed,
            "val_loss": val_loss,
            "val_ppl": math.exp(val_loss) if math.isfinite(val_loss) else None,
            "qa": qa,
            "qa_exact": qa_exact,
            "qa_gold": qa_gold,
        }

    torch.manual_seed(seed)
    random.seed(seed)

    if args.reader == "official":
        reader_name = OFFICIAL_SOURCE_QWEN_V1
    elif args.reader == "engram":
        reader_name = ENGRAM_V1
    elif args.reader == "mlp":
        reader_name = MLP_VALUE_V1
    else:
        reader_name = SIMPLE_V1

    short_conv = None
    if getattr(args, "load_reader", None):
        load_path = _format_reader_save_path(args.load_reader, mode, seed)
        reader, extra_state = load_reader_with_extra(
            load_path,
            device=args.device,
        )
        if args.reader == "simple" and args.short_conv:
            if "short_conv" not in extra_state:
                raise SystemExit(
                    "--load-reader checkpoint does not contain --short-conv state; "
                    "train again with --save-reader to include it"
                )
            short_conv = ShortConv(model.config.hidden_size)
            short_conv.load_state_dict(extra_state["short_conv"])
            if args.device != "cpu":
                short_conv = short_conv.to(args.device)
        train_losses: list[float] = []
        print(f"  [{mode}] seed={seed} loaded reader from {load_path}")
    else:
        if args.reader == "official":
            reader = OfficialSourceQwenReader.from_official_checkpoint(
                args.official_reader_path,
                d_target=model.config.hidden_size,
                freeze_source=not bool(
                    getattr(args, "unfreeze_official_source", False)
                ),
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
        elif args.reader == "mlp":
            reader = MLPValueReader(
                model.config.hidden_size,
                d_mem=2560,
                hidden=int(getattr(args, "mlp_hidden", 256) or 256),
                zero_init_v=getattr(args, "zero_init_v", True),
            )
            short_conv = None
        else:
            reader = EngramReader(
                model.config.hidden_size,
                num_branches=args.branches,
                zero_init_v=args.zero_init_v,
            )
            short_conv = (
                ShortConv(model.config.hidden_size) if args.short_conv else None
            )

        if args.device != "cpu":
            reader = reader.to(args.device)
            if short_conv is not None:
                short_conv = short_conv.to(args.device)

    gate_override = getattr(args, "gate_override", None)
    if gate_override is not None and hasattr(reader, "gate_override"):
        reader.gate_override = float(gate_override)

    handle = install_reader_hook(model, args.layer, reader, short_conv)

    e_t = train_e_t
    if mode == "control":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(e_t))
        e_t = e_t.permuted(perm) if hasattr(e_t, "permuted") else e_t[perm]

    val_eval_e_t = val_e_t
    if mode == "control":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(val_e_t))
        val_eval_e_t = (
            val_e_t.permuted(perm) if hasattr(val_e_t, "permuted") else val_e_t[perm]
        )

    if getattr(args, "load_reader", None):
        # Loaded-checkpoint evaluation mode: no training, only eval/QA below.
        val_curve: list[dict] = []
    else:
        print(f"  [{mode}] seed={seed} training ...")
        train_losses, val_curve = _train_reader(
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
            val_tokens=val_tokens if getattr(args, "val_every", 0) else None,
            val_e_t=val_eval_e_t if getattr(args, "val_every", 0) else None,
            val_every=getattr(args, "val_every", 0),
            qa_sft_items=getattr(args, "qa_sft_items", None),
            qa_sft_weight=float(getattr(args, "qa_sft_weight", 0.0) or 0.0),
            qa_sft_answer_only=not bool(
                getattr(args, "qa_sft_full_loss", False)
            ),
            qa_sft_control=(mode == "control"),
            qa_sft_log_every=int(getattr(args, "qa_sft_log_every", 0) or 0),
            gate_reg_weight=float(getattr(args, "gate_reg_weight", 0.0) or 0.0),
            qa_sft_warmup_steps=int(
                getattr(args, "qa_sft_warmup_steps", 0) or 0
            ),
            train_backbone=bool(getattr(args, "finetune_backbone", False)),
        )

    if getattr(args, "finetune_backbone", False):
        for p in model.parameters():
            p.grad = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    qa_exact = None
    if args.qa_exact_match:
        qa_exact = _qa_exact_match(
            model,
            tokenizer,
            qa_exact_items,
            qa_store,
            control=(mode == "control"),
            seed=seed,
            max_new_tokens=args.qa_max_new_tokens,
            batch_size=getattr(args, "qa_batch_size", 1),
            max_batch_tokens=getattr(args, "qa_batch_max_tokens", 0),
            prompt_template=getattr(args, "qa_prompt_template", None),
            boolq_prompt_template=getattr(args, "qa_boolq_prompt_template", None),
            chat_template=bool(getattr(args, "qa_chat_template", False)),
            chat_enable_thinking=bool(
                getattr(args, "qa_chat_enable_thinking", False)
            ),
            head_mask=head_mask,
        )

    qa_gold = None
    if getattr(args, "qa_gold_nll", False) and qa_exact_items:
        qa_gold = _qa_gold_nll(
            model,
            tokenizer,
            qa_exact_items,
            qa_store,
            control=(mode == "control"),
            seed=seed,
            prompt_template=getattr(args, "qa_prompt_template", None),
            boolq_prompt_template=getattr(args, "qa_boolq_prompt_template", None),
            chat_template=bool(getattr(args, "qa_chat_template", False)),
            chat_enable_thinking=bool(
                getattr(args, "qa_chat_enable_thinking", False)
            ),
            head_mask=head_mask,
        )

    if getattr(args, "save_reader", None) and not getattr(args, "load_reader", None):
        save_path = _format_reader_save_path(args.save_reader, mode, seed)
        extra_state = None
        if args.reader == "simple" and short_conv is not None:
            extra_state = {"short_conv": short_conv.state_dict()}
        save_reader(
            reader,
            save_path,
            name=reader_name,
            version="1",
            config=reader_config_from_args(args, model.config.hidden_size, reader_name),
            extra_state=extra_state,
        )
        print(f"  [{mode}] saved reader -> {save_path}")

        if getattr(args, "save_bundle", None):
            bundle_path = _format_reader_save_path(args.save_bundle, mode, seed)
            memory = _bundle_memory_from_args(args)
            if memory is None:
                print(
                    f"  [{mode}] warning: --save-bundle requires --live-store "
                    "or --store-p-view; skipping bundle"
                )
            else:
                reader_config = reader_config_from_args(
                    args,
                    model.config.hidden_size,
                    reader_name,
                )
                bundle = make_bundle(
                    backbone_path=args.model,
                    memory=memory,
                    ple={
                        "ple_embed_dim": 2560,
                        "num_heads": 16,
                        "head_dim": 160,
                        "scale": float(getattr(args, "applied_scale", 1.0)),
                    },
                    readers=[
                        {
                            "name": reader_name,
                            "version": "1",
                            "path": str(save_path.resolve()),
                            "options": reader_config,
                        }
                    ],
                )
                save_bundle(bundle, bundle_path)
                print(f"  [{mode}] saved bundle -> {bundle_path}")

    adapter_path = None
    if getattr(args, "lora", False) and getattr(args, "save_reader", None):
        # Adapter-only checkpoint: a LoRA row must never write a full backbone
        # (disk retention policy; see docs/round-151 section 9).
        adapter_path = Path(
            _format_reader_save_path(str(args.save_reader), mode, seed)
        ).with_suffix(".lora")
        adapter_path.mkdir(parents=True, exist_ok=True)
        save_lora_adapter(model, adapter_path)
        print(f"  [{mode}] saved LoRA adapter -> {adapter_path}")

    handle.remove()
    return {
        "mode": mode,
        "seed": seed,
        "ple_off": bool(getattr(args, "ple_off", False)),
        "gate_override": getattr(args, "gate_override", None),
        "val_loss": val_loss,
        "val_ppl": math.exp(val_loss) if math.isfinite(val_loss) else None,
        "train_losses": train_losses,
        "train_final_loss": train_losses[-1] if train_losses else None,
        "val_curve": val_curve,
        "qa": qa,
        "qa_exact": qa_exact,
        "qa_gold": qa_gold,
        "lora": lora_meta,
        "adapter_path": str(adapter_path) if adapter_path else None,
    }


def _summarize(results: list[dict], modes: list[str]) -> dict:
    summary = {}
    for mode in modes:
        vals = [
            r["val_loss"]
            for r in results
            if r["mode"] == mode and np.isfinite(r["val_loss"])
        ]
        qa_vals = [
            r["qa_exact"]["metrics"]["qa_em_mean"]
            for r in results
            if r["mode"] == mode
            and r.get("qa_exact") is not None
            and np.isfinite(r["qa_exact"]["metrics"]["qa_em_mean"])
        ]
        entry: dict[str, Any] = {
            "n_seeds": len(vals),
            "val_loss_mean": float(np.mean(vals)) if vals else None,
            "val_loss_std": float(np.std(vals)) if vals else None,
            "val_ppl_mean": float(np.exp(np.mean(vals))) if vals else None,
            "details": [r for r in results if r["mode"] == mode],
        }
        if qa_vals:
            entry["qa_em_mean"] = float(np.mean(qa_vals))
            entry["qa_em_std"] = float(np.std(qa_vals))
        summary[mode] = entry
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--features", default="data/ple-adapter-features-20k")
    parser.add_argument("--rows-dir", default="/Volumes/My Passport/qwen38-rows")
    parser.add_argument("--model-dir", default="/Volumes/My Passport/qwen38-ple")
    parser.add_argument("--live-store", action="store_true")
    parser.add_argument("--store-p-view", default=None)
    parser.add_argument("--store-p-slot-indices", default=None)
    parser.add_argument("--store-p-slot-index", default=None)
    parser.add_argument(
        "--access-order",
        action="store_true",
        help="read Store-P slots in sorted physical order",
    )
    parser.add_argument("--tokens-npy", default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--branches", type=int, default=1)
    parser.add_argument(
        "--reader", choices=["simple", "engram", "official", "mlp"], default="simple"
    )
    parser.add_argument("--official-reader-path", default="data/official_ple_reader.pt")
    parser.add_argument(
        "--mlp-hidden", type=int, default=256, help="MLP value reader hidden width"
    )
    parser.add_argument("--bridge-mlp", action="store_true")
    parser.add_argument("--bridge-hidden", type=int, default=None)
    parser.add_argument("--out-mlp", action="store_true")
    parser.add_argument("--out-hidden", type=int, default=None)
    parser.add_argument("--short-conv", action="store_true")
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=4)
    parser.add_argument("--dilation", type=int, default=3)
    parser.add_argument("--zero-init-v", action="store_true")
    parser.add_argument(
        "--save-reader",
        default=None,
        help=(
            "save the trained reader checkpoint; supports {mode} and {seed} "
            "placeholders, e.g. outputs/reader-{mode}-seed{seed}.pt"
        ),
    )
    parser.add_argument(
        "--load-reader",
        default=None,
        help=(
            "load a reader checkpoint saved by --save-reader and skip training; "
            "intended for QA/eval-only reruns"
        ),
    )
    parser.add_argument(
        "--save-bundle",
        default=None,
        help=(
            "also write an EngramDB-compatible bundle manifest; supports "
            "{mode} and {seed} placeholders"
        ),
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--val-every",
        type=int,
        default=0,
        help="compute validation loss every N training steps (0=only final)",
    )
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
    parser.add_argument(
        "--qa", action="store_true", help="run minimal QA log-likelihood probes"
    )
    parser.add_argument(
        "--qa-exact-match",
        action="store_true",
        help="run greedy exact-match QA generation with live PLE injection",
    )
    parser.add_argument(
        "--qa-gold-nll",
        action="store_true",
        help=(
            "teacher-forced gold-answer NLL for the same prompt protocol; "
            "a format-insensitive complement to --qa-exact-match"
        ),
    )
    parser.add_argument(
        "--unfreeze-official-source",
        action="store_true",
        help=(
            "when training an official-source reader, unfreeze the source "
            "key/value/norm/conv projections (table stays frozen)"
        ),
    )
    parser.add_argument(
        "--qa-head-mask",
        default=None,
        help=(
            "zero selected PLE heads during QA: '2gram' keeps 2-gram heads, "
            "'3gram' keeps 3-gram heads, 'all', or a comma list of head indices"
        ),
    )
    parser.add_argument("--qa-max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--qa-max-items",
        type=int,
        default=0,
        help=(
            "use only the first N QA items for generation, gold NLL and exact "
            "match (0 = all); intended for smoke tests, never for reported "
            "numbers"
        ),
    )
    parser.add_argument(
        "--qa-batch-size",
        type=int,
        default=1,
        help="number of questions to decode together (1 = original sequential path)",
    )
    parser.add_argument(
        "--qa-batch-max-tokens",
        type=int,
        default=0,
        help="optional token budget per QA batch (0 = only --qa-batch-size)",
    )
    parser.add_argument(
        "--qa-prompt-template",
        default=None,
        help=(
            "optional instruction-style QA prompt with a {question} placeholder; "
            "default keeps the raw question"
        ),
    )
    parser.add_argument(
        "--qa-boolq-prompt-template",
        default=None,
        help="optional BoolQ-specific override for --qa-prompt-template",
    )
    parser.add_argument(
        "--qa-chat-template",
        action="store_true",
        help=(
            "use the tokenizer's native chat template instead of the "
            "completion-style Question/Answer prompt"
        ),
    )
    parser.add_argument(
        "--qa-chat-enable-thinking",
        action="store_true",
        help=(
            "when --qa-chat-template is set, enable the model's thinking mode "
            "(requires a much larger --qa-max-new-tokens)"
        ),
    )
    parser.add_argument(
        "--qa-file",
        default=None,
        help="optional JSON list of {question, answer, task} for exact-match QA",
    )
    parser.add_argument(
        "--qa-sft-file",
        default=None,
        help=(
            "optional JSON/JSONL QA file for answer-only reader SFT; "
            "records must contain question and answer"
        ),
    )
    parser.add_argument(
        "--qa-sft-weight",
        type=float,
        default=0.0,
        help=(
            "per-step probability of drawing a QA SFT item instead of a "
            "corpus window (0=off, 1=QA-only, 0.5=even mix)"
        ),
    )
    parser.add_argument(
        "--qa-sft-max-len",
        type=int,
        default=512,
        help="maximum token length for a QA SFT item (prompt is left-truncated)",
    )
    parser.add_argument(
        "--qa-sft-prompt-template",
        default=None,
        help="prompt template for QA SFT; defaults to --qa-prompt-template",
    )
    parser.add_argument(
        "--qa-sft-boolq-prompt-template",
        default=None,
        help="BoolQ prompt template for QA SFT; defaults to the QA BoolQ template",
    )
    parser.add_argument(
        "--qa-sft-full-loss",
        action="store_true",
        help="train on all QA tokens instead of answer-only (default: answer-only)",
    )
    parser.add_argument(
        "--qa-sft-log-every",
        type=int,
        default=0,
        help="print an extra QA/corpus loss line every N steps",
    )
    parser.add_argument(
        "--gate-reg-weight",
        type=float,
        default=0.0,
        help=(
            "auxiliary loss weight on the mean reader gate value; pushes the "
            "gate closed when PLE is not useful (0=off)"
        ),
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help=(
            "adapt the backbone with LoRA instead of full fine-tuning: keeps "
            "the base weights frozen and trains only low-rank adapters plus "
            "the reader (gentler than --finetune-backbone; use with "
            "--gate-override 0.0 for a no-PLE arm)"
        ),
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default 2x rank)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout (default 0.0, deterministic)",
    )
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help=(
            "comma-separated LoRA target module names, or 'all-linear' to let "
            "peft select every Linear/Conv1D projection"
        ),
    )
    parser.add_argument(
        "--backbone-dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help=(
            "dtype for the loaded backbone; keep float32 for the 0.8B "
            "research path, use bfloat16 to fit 2B/4B backbones on one GPU"
        ),
    )
    parser.add_argument(
        "--finetune-backbone",
        action="store_true",
        help=(
            "unfreeze all backbone parameters and train them jointly with the "
            "reader (use with --gate-override 0.0 for a no-PLE control)"
        ),
    )
    parser.add_argument(
        "--ple-off",
        action="store_true",
        help=(
            "keep the PLE reader in the model but suppress its contribution "
            "entirely; unlike --modes no-reader this still trains the backbone "
            "and the reader, so it is the correct no-PLE cell of an adaptation "
            "2x2 (no reader store is loaded)"
        ),
    )
    parser.add_argument(
        "--gate-override",
        type=float,
        default=None,
        help=(
            "force every reader gate to this constant value for causal "
            "ablation (e.g. 0.0 = closed, 1.0 = fully open)"
        ),
    )
    parser.add_argument(
        "--qa-sft-warmup-steps",
        type=int,
        default=0,
        help=(
            "run corpus-only training for the first N steps before enabling "
            "the QA SFT mix (two-stage curriculum)"
        ),
    )
    parser.add_argument(
        "--qa-sft-lazy",
        action="store_true",
        help=(
            "fetch PLE rows for QA SFT items on demand instead of "
            "materializing the full e_t cache"
        ),
    )
    parser.add_argument(
        "--qa-sft-lazy-cache",
        type=int,
        default=128,
        help="number of QA SFT items to keep in the lazy e_t cache",
    )
    parser.add_argument(
        "--qa-sft-chat-template",
        action="store_true",
        help=(
            "format QA SFT prompts with the tokenizer's native chat template "
            "and use <|im_end|> as the answer EOS"
        ),
    )
    parser.add_argument(
        "--qa-sft-chat-enable-thinking",
        action="store_true",
        help="enable thinking when formatting QA SFT chat prompts",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed (mode, seed) partial results and skip them",
    )
    parser.add_argument(
        "--partial-dir",
        default=None,
        help="directory for per-(mode, seed) partial JSON files",
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="optional directory for an extra copy of the final and partial JSONs",
    )
    parser.add_argument("--output", default="outputs/phase0.json")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    _install_torch_compat()
    out_path = Path(args.output)
    partial_dir = (
        Path(args.partial_dir)
        if getattr(args, "partial_dir", None)
        else out_path.parent / f"{out_path.stem}-partial"
    )
    backup_dir = (
        Path(args.backup_dir) if getattr(args, "backup_dir", None) else None
    )
    existing_results: dict[tuple[str, int], dict[str, Any]] = {}
    if getattr(args, "resume", False):
        existing_results = load_partial_results(partial_dir, out_path)
        if out_path.exists():
            try:
                previous = json.loads(out_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
            for item in previous.get("results", []):
                if isinstance(item, dict) and "mode" in item and "seed" in item:
                    existing_results[
                        (str(item["mode"]), int(item["seed"]))
                    ] = item
        if existing_results:
            print(
                f"[phase0] resume: loaded {len(existing_results)} completed "
                f"(mode, seed) results"
            )
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

        from qwen35_ple.real_ple import resolve_ple_weight_scale

        applied_scale = resolve_ple_weight_scale(
            model_dir=args.model_dir, scale=args.scale
        )
        import engramdb

        if args.store_p_view:
            print(
                f"[phase0] live-store Store-P: {len(tokens)} tokens, "
                f"view={args.store_p_view} (scale={applied_scale:.6g}) ..."
            )
            if args.store_p_slot_index:
                from qwen35_ple.real_ple import rowids_from_tokens
                from qwen35_ple.slot_index import SlotIndex

                slot_index = SlotIndex.load(args.store_p_slot_index)
                rowids = rowids_from_tokens(tokens)
                slot_indices = slot_index.to_slots(rowids)
                print(
                    f"[phase0] Store-P generic slot index: {len(slot_indices)} tokens "
                    f"mapped to {len(slot_index)} view slots"
                )
            elif args.store_p_slot_indices:
                slot_indices = np.load(args.store_p_slot_indices).astype(np.int64)
                if len(slot_indices) < len(tokens):
                    raise SystemExit(
                        f"store-p slot_indices length {len(slot_indices)} < tokens {len(tokens)}"
                    )
                slot_indices = slot_indices[: len(tokens)]
            else:
                slot_indices = np.arange(len(tokens), dtype=np.int64)
            view = engramdb.View(args.store_p_view)
            e_t = LiveETViewStore(
                view,
                slot_indices,
                applied_scale,
                num_heads=16,
                head_dim=160,
                embedding_dim=2560,
                view_path=args.store_p_view,
                access_order=args.access_order,
            )
            live_store_handle = e_t
            print(
                f"[phase0] Store-P ready: {len(tokens)} tokens, "
                f"slot_indices={len(slot_indices)}, no full e_t allocated"
            )
        else:
            from qwen35_ple.real_ple import rowids_from_tokens

            print(
                f"[phase0] live-store: {len(tokens)} tokens, rowids from "
                f"{args.rows_dir} (scale={applied_scale:.6g}) ..."
            )
            t0 = time.time()
            rowids = rowids_from_tokens(tokens)
            rowid_s = time.time() - t0
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
        tokens, e_t, applied_scale = _load_features(
            feature_dir, args.model_dir, args.scale
        )
        print(
            f"[phase0] tokens={len(tokens)} e_t={e_t.shape} scale={applied_scale:.6g}"
        )

    (train_tokens, train_e_t), (val_tokens, val_e_t) = _split(
        tokens, e_t, args.val_frac
    )
    print(
        f"[phase0] split train={len(train_tokens)} val={len(val_tokens)} "
        f"val_frac={args.val_frac}"
    )

    tokenizer, model = _load_model(
        args.model, args.device, getattr(args, "backbone_dtype", "float32")
    )
    for p in model.parameters():
        p.requires_grad_(False)
    lora_meta: dict | None = None
    if getattr(args, "lora", False):
        lora_meta = _apply_lora(model, args)
        print(
            f"[phase0] LoRA enabled: r={lora_meta['r']} alpha={lora_meta['alpha']} "
            f"targets={len(lora_meta['target_modules'])} "
            f"trainable={lora_meta['trainable_params']:,}"
        )
    if getattr(args, "finetune_backbone", False):
        for p in model.parameters():
            p.requires_grad_(True)
        print("[phase0] backbone fine-tuning enabled: all parameters trainable")

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

    qa_exact_items = None
    qa_sft_items: list[dict] | None = None
    qa_store = None
    needs_reader_store = (
        any(mode != "no-reader" for mode in args.modes)
        and not getattr(args, "ple_off", False)
    )
    needs_qa_store = needs_reader_store and (
        bool(args.qa_exact_match) or bool(args.qa_sft_file) or bool(args.qa_gold_nll)
    )
    if args.qa_exact_match or args.qa_gold_nll:
        qa_exact_items = _load_qa_file(args.qa_file)
        qa_max_items = int(getattr(args, "qa_max_items", 0) or 0)
        if qa_max_items > 0:
            qa_exact_items = qa_exact_items[:qa_max_items]
            print(f"[phase0] --qa-max-items: evaluating only {len(qa_exact_items)} items")
        print(
            f"[phase0] preparing QA eval: {len(qa_exact_items)} items, "
            f"max_new_tokens={args.qa_max_new_tokens}, "
            f"gold_nll={bool(args.qa_gold_nll)}"
        )
    if args.qa_sft_file and needs_reader_store:
        qa_sft_raw = _load_qa_sft_file(args.qa_sft_file)
        if qa_store is None:
            qa_store = _QAEtStore(args.rows_dir, applied_scale)
        chat_eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if not isinstance(chat_eos_id, int) or chat_eos_id < 0:
            chat_eos_id = tokenizer.eos_token_id
        qa_sft_entries = _build_qa_sft_cache(
            qa_sft_raw,
            tokenizer,
            qa_store,
            prompt_template=(
                args.qa_sft_prompt_template or args.qa_prompt_template
            ),
            boolq_prompt_template=(
                args.qa_sft_boolq_prompt_template
                or args.qa_boolq_prompt_template
            ),
            max_len=int(args.qa_sft_max_len),
            eos_id=tokenizer.eos_token_id,
            lazy=bool(getattr(args, "qa_sft_lazy", False)),
            chat_template=bool(getattr(args, "qa_sft_chat_template", False)),
            chat_enable_thinking=bool(
                getattr(args, "qa_sft_chat_enable_thinking", False)
            ),
            chat_eos_id=chat_eos_id,
        )
        task_counts: dict[str, int] = {}
        for item in qa_sft_entries:
            task = str(item.get("task", "qa"))
            task_counts[task] = task_counts.get(task, 0) + 1
        mean_len = (
            sum(len(item["ids"]) for item in qa_sft_entries)
            / len(qa_sft_entries)
            if qa_sft_entries
            else 0.0
        )
        if getattr(args, "qa_sft_lazy", False):
            qa_sft_items = _LazyQASFTCache(
                qa_sft_entries,
                qa_store,
                max_cached=int(getattr(args, "qa_sft_lazy_cache", 128) or 128),
            )
        else:
            qa_sft_items = qa_sft_entries
        print(
            f"[phase0] QA SFT cache: {len(qa_sft_entries)} items, "
            f"mean_len={mean_len:.1f}, tasks={task_counts}, "
            f"weight={args.qa_sft_weight}, lazy={bool(getattr(args, 'qa_sft_lazy', False))}"
        )
    if needs_qa_store and qa_store is None:
        qa_store = _QAEtStore(args.rows_dir, applied_scale)

    args.qa_sft_items = qa_sft_items
    new_results: list[dict[str, Any]] = []
    args.applied_scale = applied_scale
    for seed in args.seeds:
        print(f"=== seed {seed} ===")
        for mode in args.modes:
            key = (mode, int(seed))
            if key in existing_results:
                print(f"  mode={mode} seed={seed} already complete; skipping")
                continue
            print(f"  mode={mode}")
            if live_store_handle is not None and hasattr(
                live_store_handle, "reset_stats"
            ):
                live_store_handle.reset_stats()
            model._ple_disabled = bool(getattr(args, "ple_off", False))
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
                qa_exact_items,
                qa_store,
                lora_meta=lora_meta,
            )
            if live_store_handle is not None:
                stats = getattr(live_store_handle, "stats", None)
                if hasattr(stats, "as_dict"):
                    fetch = stats.as_dict()
                elif isinstance(stats, dict):
                    fetch = stats
                else:
                    fetch = None
                if fetch is not None:
                    res["fetch_stats"] = fetch
                    windows = int(fetch.get("windows", 0))
                    seconds = float(fetch.get("fetch_seconds", 0.0))
                    res["fetch_ms_per_window"] = (
                        seconds * 1000.0 / windows if windows else None
                    )
                    tokens = int(fetch.get("tokens", 0))
                    res["fetch_ms_per_token"] = (
                        seconds * 1000.0 / tokens if tokens else None
                    )
            partial_path = write_partial_result(partial_dir, out_path, res)
            print(f"  [{mode}] partial -> {partial_path}")
            new_results.append(res)

    all_results = merge_results(
        existing_results,
        new_results,
        modes=args.modes,
        seeds=args.seeds,
    )
    summary = _summarize(all_results, args.modes)
    result = {
        "config": {
            "model": args.model,
            "features": args.features,
            "rows_dir": args.rows_dir,
            "live_store": bool(args.live_store),
            "store_p_view": args.store_p_view,
            "store_p_slot_indices": args.store_p_slot_indices,
            "store_p_slot_index": args.store_p_slot_index,
            "access_order": bool(args.access_order),
            "layer": args.layer,
            "branches": args.branches,
            "reader": args.reader,
            "save_reader": args.save_reader,
            "load_reader": args.load_reader,
            "save_bundle": args.save_bundle,
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
            "qa": bool(args.qa),
            "qa_exact_match": bool(args.qa_exact_match),
            "qa_max_new_tokens": args.qa_max_new_tokens,
            "qa_batch_size": args.qa_batch_size,
            "qa_batch_max_tokens": args.qa_batch_max_tokens,
            "qa_prompt_template": args.qa_prompt_template,
            "qa_boolq_prompt_template": args.qa_boolq_prompt_template,
            "qa_file": args.qa_file,
            "resume": bool(args.resume),
            "partial_dir": str(partial_dir),
            "backup_dir": str(backup_dir) if backup_dir is not None else None,
        },
        "summary": summary,
        "results": all_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, backup_dir / out_path.name)
        for partial in sorted(partial_dir.glob(f"{out_path.stem}-*.json")):
            shutil.copy2(partial, backup_dir / partial.name)
        print(f"[phase0] backup -> {backup_dir}")
    if qa_store is not None:
        qa_store.close()
        print("[phase0] QA store closed")
    if live_store_handle is not None:
        live_store_handle.close()
        print("[phase0] live-store closed")
    print(f"[phase0] saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
