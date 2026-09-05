#!/usr/bin/env python3
"""P0 PLE evidence repair: true local tasks, same-domain banks, per-task Deltas.

This script builds:

1. Real local-task eval positions:
   - code continuation from Python source files;
   - name and number positions from Wiki text;
2. Same-domain PLE memory banks:
   - code bank from held-out split of Python files;
   - wiki bank from held-out split of Wiki docs;
   - matching shuffled control banks;
3. Per-task metrics:
   - base log-prob;
   - real/control n-gram log-prob;
   - fused log-prob using global vs per-task calibrated parameters;
   - top-1 hits;
   - Delta = E[log p_m - log p_b], and real-control difference;
4. Optional MoRA adapter comparison for multi-source PLE+MoRA evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from qwen35_ple.addressable_memory import AddressableNgramMemory
from qwen35_ple.fusion import calibrate_ngram_fusion, fuse_ngram_logits, softmax
from qwen35_ple.router import load_fusion_router_config


def _load_model(model_path: str, adapter: str | None, device: str):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model


def _python_files(root: Path, max_files: int | None) -> list[Path]:
    files = []
    for p in root.rglob("*.py"):
        if ".venv" in p.parts or ".git" in p.parts:
            continue
        files.append(p)
    files.sort()
    return files[:max_files] if max_files is not None else files


def _load_wiki_docs(path: str, max_docs: int | None) -> list[str]:
    docs = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(obj.get("text") or "")
            if text.strip():
                docs.append(text.strip())
            if max_docs is not None and len(docs) >= max_docs:
                break
    return docs


def _split(seqs: list[list[int]], train_frac: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(seqs)))
    rng.shuffle(idx)
    n_train = max(1, int(round(len(seqs) * train_frac)))
    return [seqs[i] for i in idx[:n_train]], [seqs[i] for i in idx[n_train:]]


def _build_memory(seqs, *, shuffle: bool, seed: int, max_order: int = 4) -> AddressableNgramMemory:
    mem = AddressableNgramMemory(min_order=2, max_order=max_order)
    rng = random.Random(seed)
    for value_id, seq in enumerate(seqs):
        if shuffle:
            seq = list(seq)
            rng.shuffle(seq)
        mem.add_document(seq, value_id=value_id)
    return mem


def _token_category(tokenizer, tok_id: int) -> str:
    try:
        raw = tokenizer.convert_ids_to_tokens([tok_id])[0]
        text = raw.lstrip("Ġ▁")
    except Exception:
        text = ""
    if text and text[0].isdigit():
        return "number"
    if text and text[0].isupper():
        return "name"
    return "general"


def _sample_positions(
    seqs: list[list[int]],
    *,
    context_len: int,
    max_per_doc: int,
    seed: int,
    category: str | None = None,
):
    rng = random.Random(seed)
    out = []
    for seq in seqs:
        candidates = []
        for i in range(context_len, len(seq) - 1):
            if category is None or _token_category_local(seq[i], category):
                candidates.append(i)
        if not candidates:
            continue
        rng.shuffle(candidates)
        out.extend((seq, i) for i in candidates[:max_per_doc])
    # Cap total (caller can further limit)
    return out


def _token_category_local(tok_id: int, category: str) -> bool:
    # We use category string only for caller-level filter; actual filtering is
    # done in _sample_positions_by_category using tokenizer.
    raise NotImplementedError


def _sample_positions_by_category(
    tokenizer,
    seqs: list[list[int]],
    *,
    context_len: int,
    max_per_doc: int,
    seed: int,
    category: str | None = None,
):
    rng = random.Random(seed)
    out = []
    for seq in seqs:
        candidates = []
        for i in range(context_len, len(seq) - 1):
            cat = _token_category(tokenizer, seq[i])
            if category is None or cat == category:
                candidates.append(i)
        if not candidates:
            continue
        rng.shuffle(candidates)
        for i in candidates[:max_per_doc]:
            out.append((seq, i))
    return out


def _precompute_position(
    model,
    context: list[int],
    target: int,
    device: str,
    dist=None,
):
    ids = torch.tensor([context], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0, -1].float().cpu().numpy()
    return {
        "logits": logits,
        "target": target,
        "dist": dist,
    }


def _score_precomputed(rec, *, scale: float = 0.0, bias: float = 0.0, temperature: float = 1.0):
    logits = rec["logits"]
    target = rec["target"]
    dist = rec["dist"]
    base_logprob = float(math.log(max(softmax(logits)[target], 1e-12)))
    ngram_logprob = None
    fused_logprob = None
    ngram_hit = False
    fused_hit = False
    if dist is not None:
        p = float(dist.get(target, 0.0))
        ngram_logprob = float(math.log(max(p, 1e-12)))
        fused = fuse_ngram_logits(logits, dist, scale=scale, bias=bias, temperature=temperature)
        fused_logprob = float(math.log(max(softmax(fused)[target], 1e-12)))
        fused_hit = bool(int(np.argmax(fused)) == target)
        ngram_hit = bool(max(dist, key=dist.get) == target)
    return {
        "base_logprob": base_logprob,
        "ngram_logprob": ngram_logprob,
        "fused_logprob": fused_logprob,
        "base_hit": bool(int(np.argmax(logits)) == target),
        "ngram_hit": ngram_hit,
        "fused_hit": fused_hit,
    }


def _aggregate(entries: list[dict]) -> dict:
    if not entries:
        return {}
    base_lp = np.mean([e["base_logprob"] for e in entries])
    real_lp = np.mean([e["ngram_logprob"] for e in entries if e["ngram_logprob"] is not None])
    fused_lp = np.mean([e["fused_logprob"] for e in entries if e["fused_logprob"] is not None])
    n = len(entries)
    return {
        "n": n,
        "base_logprob": float(base_lp),
        "ngram_logprob": float(real_lp) if entries else None,
        "fused_logprob": float(fused_lp) if entries else None,
        "delta_ngram_vs_base": float(real_lp - base_lp) if entries else None,
        "delta_fused_vs_base": float(fused_lp - base_lp) if entries else None,
        "base_top1": float(np.mean([1.0 if e["base_hit"] else 0.0 for e in entries])) if entries else None,
        "ngram_top1": float(np.mean([1.0 if e["ngram_hit"] else 0.0 for e in entries if e["ngram_logprob"] is not None])) if entries else None,
        "fused_top1": float(np.mean([1.0 if e["fused_hit"] else 0.0 for e in entries if e["fused_logprob"] is not None])) if entries else None,
    }


def _calibrate(
    base_logits_list: list[np.ndarray],
    targets: list[int],
    dist_list: list[dict | None],
    *,
    scale_grid: np.ndarray | None = None,
    bias_grid: np.ndarray | None = None,
) -> dict:
    if scale_grid is None:
        scale_grid = np.linspace(-2.0, 4.0, 9)
    if bias_grid is None:
        bias_grid = np.linspace(-4.0, 3.0, 9)
    return calibrate_ngram_fusion(
        base_logits_list,
        targets,
        dist_list,
        scale_grid=scale_grid,
        bias_grid=bias_grid,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--code-root", default=".")
    parser.add_argument("--wiki-path", default="data/sources/wikitext.jsonl")
    parser.add_argument("--max-code-files", type=int, default=200)
    parser.add_argument("--max-wiki-docs", type=int, default=400)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--max-per-doc-code", type=int, default=8)
    parser.add_argument("--max-per-doc-wiki", type=int, default=4)
    parser.add_argument("--max-eval-positions", type=int, default=240)
    parser.add_argument("--calib-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/ple-evidence-p0.json")
    args = parser.parse_args()

    tokenizer, model = _load_model(args.model, args.adapter, args.device)

    # Build code and wiki sequence splits.
    py_files = _python_files(Path(args.code_root), args.max_code_files)
    code_seqs = []
    for f in py_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > 8:
            code_seqs.append(ids)
    print(f"[ple-p0] code files={len(py_files)} seqs={len(code_seqs)}", flush=True)

    wiki_docs = _load_wiki_docs(args.wiki_path, args.max_wiki_docs)
    wiki_seqs = [
        tokenizer.encode(d, add_special_tokens=False)
        for d in wiki_docs
        if len(tokenizer.encode(d, add_special_tokens=False)) > 16
    ]
    print(f"[ple-p0] wiki docs={len(wiki_docs)} seqs={len(wiki_seqs)}", flush=True)

    code_train, code_eval = _split(code_seqs, args.train_frac, args.seed)
    wiki_train, wiki_eval = _split(wiki_seqs, args.train_frac, args.seed)

    code_mem_real = _build_memory(code_train, shuffle=False, seed=args.seed, max_order=args.max_order)
    code_mem_ctrl = _build_memory(code_train, shuffle=True, seed=args.seed, max_order=args.max_order)
    wiki_mem_real = _build_memory(wiki_train, shuffle=False, seed=args.seed, max_order=args.max_order)
    wiki_mem_ctrl = _build_memory(wiki_train, shuffle=True, seed=args.seed, max_order=args.max_order)

    config = load_fusion_router_config("configs/ngram-fusion-router.json")
    fusion = config.get("fusion", {})
    global_scale = float(fusion.get("scale", 1.0))
    global_bias = float(fusion.get("bias", 0.0))
    global_temp = float(fusion.get("temperature", 1.0))

    tasks = {
        "code": (code_eval, code_mem_real, code_mem_ctrl, None),
        "name": (wiki_eval, wiki_mem_real, wiki_mem_ctrl, "name"),
        "number": (wiki_eval, wiki_mem_real, wiki_mem_ctrl, "number"),
    }

    results = {}
    for task_name, (eval_seqs, mem_real, mem_ctrl, category) in tasks.items():
        positions = _sample_positions_by_category(
            tokenizer,
            eval_seqs,
            context_len=args.context_len,
            max_per_doc=args.max_per_doc_code if task_name == "code" else args.max_per_doc_wiki,
            seed=args.seed,
            category=category,
        )
        rng = random.Random(args.seed)
        rng.shuffle(positions)
        positions = positions[: args.max_eval_positions]
        print(f"[ple-p0] task={task_name} positions={len(positions)}", flush=True)

        # Calibration split.
        calib_positions = positions[: int(len(positions) * args.calib_frac)]
        eval_positions = positions[int(len(positions) * args.calib_frac):]

        calib_logits, calib_targets, calib_real_dists, calib_ctrl_dists = [], [], [], []
        for seq, i in calib_positions:
            context = seq[max(0, i - args.context_len): i]
            target = seq[i]
            ids = torch.tensor([context], dtype=torch.long, device=args.device)
            with torch.no_grad():
                logits = model(input_ids=ids, use_cache=False).logits[0, -1].float().cpu().numpy()
            calib_logits.append(logits)
            calib_targets.append(target)
            calib_real_dists.append(mem_real.continuation_distribution(context)[0] if mem_real.continuation_distribution(context) else None)
            calib_ctrl_dists.append(mem_ctrl.continuation_distribution(context)[0] if mem_ctrl.continuation_distribution(context) else None)

        real_cal = _calibrate(calib_logits, calib_targets, calib_real_dists)
        ctrl_cal = _calibrate(calib_logits, calib_targets, calib_ctrl_dists)

        precomputed_real = []
        precomputed_ctrl = []
        for seq, i in eval_positions:
            context = seq[max(0, i - args.context_len): i]
            target = seq[i]
            real_dist = mem_real.continuation_distribution(context)[0] if mem_real.continuation_distribution(context) else None
            ctrl_dist = mem_ctrl.continuation_distribution(context)[0] if mem_ctrl.continuation_distribution(context) else None
            rec = _precompute_position(model, context, target, args.device)
            precomputed_real.append({**rec, "dist": real_dist})
            precomputed_ctrl.append({**rec, "dist": ctrl_dist})

        def _eval_set(precomputed, scale, bias):
            entries = [
                _score_precomputed(r, scale=scale, bias=bias, temperature=global_temp)
                for r in precomputed
            ]
            return _aggregate(entries)

        real_global = _eval_set(precomputed_real, global_scale, global_bias)
        real_calibrated = _eval_set(precomputed_real, real_cal["best_scale"], real_cal["best_bias"])
        ctrl_global = _eval_set(precomputed_ctrl, global_scale, global_bias)
        ctrl_calibrated = _eval_set(precomputed_ctrl, ctrl_cal["best_scale"], ctrl_cal["best_bias"])

        results[task_name] = {
            "n_raw_positions": len(positions),
            "n_calibration": len(calib_positions),
            "n_eval": len(eval_positions),
            "calibration_real": real_cal,
            "calibration_control": ctrl_cal,
            "real_global": real_global,
            "real_calibrated": real_calibrated,
            "control_global": ctrl_global,
            "control_calibrated": ctrl_calibrated,
            "real_vs_control_delta": (
                (real_calibrated.get("delta_ngram_vs_base") or 0.0)
                - (ctrl_calibrated.get("delta_ngram_vs_base") or 0.0)
            ) if real_calibrated and ctrl_calibrated else None,
            "real_vs_control_fused_delta": (
                (real_calibrated.get("delta_fused_vs_base") or 0.0)
                - (ctrl_calibrated.get("delta_fused_vs_base") or 0.0)
            ) if real_calibrated and ctrl_calibrated else None,
        }
        print(json.dumps({k: results[task_name][k] for k in ["real_calibrated", "control_calibrated", "real_vs_control_delta"]}, ensure_ascii=False), flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "ple-evidence-p0-v1", "config": vars(args), "results": results, "model": args.model, "adapter": args.adapter}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[ple-p0] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
