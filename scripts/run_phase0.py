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
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

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

def _load_model(model_path: str, device: str = "cpu"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    # Newer Transformers releases may still load the checkpoint in its original
    # bf16 dtype even when dtype=float32 is requested.  Our reader and eval
    # path currently assume float32 on CPU/GPU, so force a consistent dtype.
    if next(model.parameters()).dtype != torch.float32:
        model = model.to(torch.float32)
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
    if hasattr(e_t, "view") and not isinstance(e_t, np.ndarray):
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
    val_tokens: np.ndarray | None = None,
    val_e_t: np.ndarray | None = None,
    val_every: int = 0,
    max_val_windows: int = 4,
) -> tuple[list[float], list[dict]]:
    assert len(tokens) > seq_len + 1
    params = [reader] + ([short_conv] if short_conv is not None else [])
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(torch.nn.ModuleList(params).parameters(), lr=lr)
    rng = random.Random(seed)
    losses = []
    val_curve: list[dict] = []
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
        if val_every > 0 and (step + 1) % val_every == 0 and val_tokens is not None and val_e_t is not None:
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


def _load_qa_file(path: str | Path | None) -> list[dict]:
    """Load a QA file that matches the Phase 0 default schema."""
    if path is None:
        return list(DEFAULT_QA)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("--qa-file must be a JSON list of {question, answer} items")
    out = []
    for item in data:
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

    def close(self) -> None:
        self.store.close()


_NUMBER_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_NUMBER_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _expand_number_words(text: str) -> str:
    """Convert English number words in a normalized phrase to digits."""
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        if words[i] in _NUMBER_UNITS or words[i] in _NUMBER_TENS or words[i] in {"hundred", "thousand"}:
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


def _qa_exact_match(
    model,
    tokenizer,
    items: list[dict],
    qa_store: _QAEtStore | None,
    control: bool,
    seed: int,
    max_new_tokens: int,
) -> dict:
    """Greedy exact-match QA generation with live PLE injection.

    For real/control, every decoding step fetches the e_t rows for the current
    token sequence and injects them through the installed reader hook.  For
    no-reader, ``qa_store`` is None and the same greedy loop runs without PLE.
    """
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    answers: list[dict] = []
    per_task_correct: dict[str, list[bool]] = {}

    for idx, item in enumerate(items):
        print(
            f"    QA {idx + 1}/{len(items)} [{item['task']}] {item['question'][:90]}",
            flush=True,
        )
        qids = tokenizer.encode(item["question"], add_special_tokens=False)
        ids = list(qids)
        generated_ids: list[int] = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if qa_store is not None:
                    et_np = qa_store.fetch(ids)
                    if control:
                        rng = np.random.default_rng(seed * 1000 + idx)
                        et_np = et_np[rng.permutation(len(et_np))]
                    model._current_ple_e_t = (
                        torch.from_numpy(et_np[None, :]).float().to(device)
                    )
                else:
                    model._current_ple_e_t = None
                out = model(input_ids=torch.tensor([ids], dtype=torch.long, device=device))
                logits = out.logits
                next_id = int(torch.argmax(logits[0, -1]).item())
                if eos_id is not None and next_id == eos_id:
                    break
                generated_ids.append(next_id)
                ids.append(next_id)

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        hit = _normalize_answer(item["answer"]) in _normalize_answer(generated_text)
        answers.append(
            {
                "task": item["task"],
                "question": item["question"],
                "answer": item["answer"],
                "generated": generated_text,
                "correct": hit,
            }
        )
        per_task_correct.setdefault(item["task"], []).append(hit)

    metrics: dict[str, float] = {}
    for task, hits in sorted(per_task_correct.items()):
        metrics[f"qa_{task}_em"] = float(np.mean(hits))
    all_hits = [a["correct"] for a in answers]
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
):
    if mode == "no-reader":
        val_loss = _window_loss(model, val_tokens, val_e_t, args.seq_len)
        qa = _qa_loglik(model, tokenizer, qa_items, control=False, seed=seed) if args.qa else None
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
            )
        return {
            "mode": mode,
            "seed": seed,
            "val_loss": val_loss,
            "val_ppl": math.exp(val_loss) if math.isfinite(val_loss) else None,
            "qa": qa,
            "qa_exact": qa_exact,
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
        reader, extra_state = load_reader_with_extra(
            args.load_reader,
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
        print(f"  [{mode}] seed={seed} loaded reader from {args.load_reader}")
    else:
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

    val_eval_e_t = val_e_t
    if mode == "control":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(val_e_t))
        val_eval_e_t = val_e_t.permuted(perm) if hasattr(val_e_t, "permuted") else val_e_t[perm]

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
        )

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

    handle.remove()
    return {
        "mode": mode,
        "seed": seed,
        "val_loss": val_loss,
        "val_ppl": math.exp(val_loss) if math.isfinite(val_loss) else None,
        "train_losses": train_losses,
        "train_final_loss": train_losses[-1] if train_losses else None,
        "val_curve": val_curve,
        "qa": qa,
        "qa_exact": qa_exact,
    }


def _summarize(results: list[dict], modes: list[str]) -> dict:
    summary = {}
    for mode in modes:
        vals = [r["val_loss"] for r in results if r["mode"] == mode and np.isfinite(r["val_loss"])]
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
    parser.add_argument("--access-order", action="store_true", help="read Store-P slots in sorted physical order")
    parser.add_argument("--tokens-npy", default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--branches", type=int, default=1)
    parser.add_argument("--reader", choices=["simple", "engram", "official", "mlp"], default="simple")
    parser.add_argument("--official-reader-path", default="data/official_ple_reader.pt")
    parser.add_argument("--mlp-hidden", type=int, default=256, help="MLP value reader hidden width")
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
    parser.add_argument("--val-every", type=int, default=0, help="compute validation loss every N training steps (0=only final)")
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
    parser.add_argument(
        "--qa-exact-match",
        action="store_true",
        help="run greedy exact-match QA generation with live PLE injection",
    )
    parser.add_argument("--qa-max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--qa-file",
        default=None,
        help="optional JSON list of {question, answer, task} for exact-match QA",
    )
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

    qa_exact_items = None
    qa_store = None
    if args.qa_exact_match:
        qa_exact_items = _load_qa_file(args.qa_file)
        print(
            f"[phase0] preparing exact-match QA: {len(qa_exact_items)} items, "
            f"max_new_tokens={args.qa_max_new_tokens}"
        )
        if any(mode != "no-reader" for mode in args.modes):
            qa_store = _QAEtStore(args.rows_dir, applied_scale)

    all_results = []
    args.applied_scale = applied_scale
    for seed in args.seeds:
        print(f"=== seed {seed} ===")
        for mode in args.modes:
            print(f"  mode={mode}")
            if live_store_handle is not None and hasattr(
                live_store_handle, "reset_stats"
            ):
                live_store_handle.reset_stats()
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
            all_results.append(res)

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
            "qa_file": args.qa_file,
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
