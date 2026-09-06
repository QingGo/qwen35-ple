#!/usr/bin/env python3
"""PLE unsubstitutability ablation: base vs BM25 vs n-gram retrieval vs PLE fusion.

This script evaluates whether calibrated PLE logit fusion is meaningfully better
than cheaper external-memory baselines on the same real local tasks used by the
P0 evidence protocol.

Conditions per eval point:

* ``base``          : frozen model next-token log-prob;
* ``bm25``          : prepend BM25-retrieved same-domain docs to the context;
* ``ngram_retrieval``: prepend exact n-gram-retrieved docs to the context;
* ``ple_fusion``    : calibrated log-linear fusion of the PLE continuation
                      distribution with base logits;
* ``ngram_raw``     : raw PLE continuation distribution probability (no base).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from qwen35_ple.addressable_memory import AddressableNgramMemory
from qwen35_ple.fusion import calibrate_ngram_fusion, fuse_ngram_logits, softmax
from qwen35_ple.rag import BM25Index, NgramKeyRetriever


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


def _split_texts(texts: list[str], train_frac: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    n_train = max(1, round(len(texts) * train_frac))
    train = [texts[i] for i in idx[:n_train]]
    eval_ = [texts[i] for i in idx[n_train:]]
    return train, eval_


def _token_category(tokenizer, tok_id: int) -> str:
    try:
        raw = tokenizer.convert_ids_to_tokens([tok_id])[0]
        text = raw.lstrip("Ġ▁")
    except (IndexError, KeyError, TypeError, ValueError):
        text = ""
    if text and text[0].isdigit():
        return "number"
    if text and text[0].isupper():
        return "name"
    return "general"


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


def _build_memory(train_texts: list[str], tokenizer, *, max_order: int, seed: int):
    mem = AddressableNgramMemory(min_order=2, max_order=max_order)
    for value_id, text in enumerate(train_texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            mem.add_document(ids, value_id=value_id)
    return mem


def _logprob(logits: np.ndarray, target: int) -> float:
    return float(math.log(max(softmax(logits)[target], 1e-12)))


def _predict_logprob(
    model,
    tokenizer,
    input_ids: list[int],
    target: int,
    device: str,
) -> tuple[float, bool]:
    if not input_ids:
        return -math.inf, False
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0, -1].float().cpu().numpy()
    return _logprob(logits, target), bool(int(np.argmax(logits)) == target)


def _retrieved_docs_text(
    retriever,
    query_text: str,
    docs: list[str],
    top_k: int,
) -> str:
    if retriever is None:
        return ""
    if isinstance(retriever, BM25Index):
        hits = retriever.search(query_text, top_k=top_k)
        return "\n\n".join(docs[i] for i in hits)
    # NgramKeyRetriever
    hits = retriever.search(query_text, top_k=top_k)
    return "\n\n".join(docs[i] for i in hits if 0 <= i < len(docs))


def _aggregate(entries: list[dict]) -> dict:
    if not entries:
        return {}
    out: dict[str, float] = {}
    for key in ["base_logprob", "bm25_logprob", "ngram_retrieval_logprob", "ple_logprob", "bm25_ple_logprob", "ngram_raw_logprob"]:
        vals = [e[key] for e in entries if e.get(key) is not None and not math.isinf(e[key])]
        out[key] = float(np.mean(vals)) if vals else None
    for key in ["base_hit", "bm25_hit", "ngram_retrieval_hit", "ple_hit", "bm25_ple_hit", "ngram_raw_hit"]:
        vals = [1.0 if e.get(key) else 0.0 for e in entries]
        out[key] = float(np.mean(vals)) if vals else None
    out["n"] = len(entries)
    out["delta_bm25_vs_base"] = (
        out["bm25_logprob"] - out["base_logprob"]
        if out.get("bm25_logprob") is not None and out.get("base_logprob") is not None
        else None
    )
    out["delta_ngram_retrieval_vs_base"] = (
        out["ngram_retrieval_logprob"] - out["base_logprob"]
        if out.get("ngram_retrieval_logprob") is not None and out.get("base_logprob") is not None
        else None
    )
    out["delta_ple_vs_base"] = (
        out["ple_logprob"] - out["base_logprob"]
        if out.get("ple_logprob") is not None and out.get("base_logprob") is not None
        else None
    )
    out["delta_ple_vs_bm25"] = (
        out["ple_logprob"] - out["bm25_logprob"]
        if out.get("ple_logprob") is not None and out.get("bm25_logprob") is not None
        else None
    )
    out["delta_ple_vs_ngram_retrieval"] = (
        out["ple_logprob"] - out["ngram_retrieval_logprob"]
        if out.get("ple_logprob") is not None and out.get("ngram_retrieval_logprob") is not None
        else None
    )
    out["delta_bm25_ple_vs_base"] = (
        out["bm25_ple_logprob"] - out["base_logprob"]
        if out.get("bm25_ple_logprob") is not None and out.get("base_logprob") is not None
        else None
    )
    out["delta_bm25_ple_vs_bm25"] = (
        out["bm25_ple_logprob"] - out["bm25_logprob"]
        if out.get("bm25_ple_logprob") is not None and out.get("bm25_logprob") is not None
        else None
    )
    out["delta_bm25_ple_vs_ple"] = (
        out["bm25_ple_logprob"] - out["ple_logprob"]
        if out.get("bm25_ple_logprob") is not None and out.get("ple_logprob") is not None
        else None
    )
    return out


def _calibrate_ple(
    model,
    tokenizer,
    device,
    positions,
    memory,
    *,
    context_len: int,
    scale_grid: np.ndarray | None = None,
    bias_grid: np.ndarray | None = None,
):
    base_logits_list = []
    target_ids = []
    dist_list = []
    for seq, i in positions:
        context = seq[max(0, i - context_len) : i]
        target = seq[i]
        ids = torch.tensor([context], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False).logits[0, -1].float().cpu().numpy()
        base_logits_list.append(logits)
        target_ids.append(target)
        dist = memory.continuation_distribution(context)
        dist_list.append(dist[0] if dist else None)
    if scale_grid is None:
        scale_grid = np.linspace(-2.0, 4.0, 9)
    if bias_grid is None:
        bias_grid = np.linspace(-4.0, 3.0, 9)
    return calibrate_ngram_fusion(
        base_logits_list,
        target_ids,
        dist_list,
        scale_grid=scale_grid,
        bias_grid=bias_grid,
    )


def evaluate_task(
    model,
    tokenizer,
    *,
    task_name: str,
    eval_seqs: list[list[int]],
    train_texts: list[str],
    memory,
    bm25,
    ngram_retriever,
    context_len: int,
    max_per_doc: int,
    seed: int,
    category: str | None,
    calib_frac: float,
    max_eval_positions: int,
    top_k: int,
    max_rag_tokens: int,
    device: str,
    global_temp: float,
):
    positions = _sample_positions_by_category(
        tokenizer,
        eval_seqs,
        context_len=context_len,
        max_per_doc=max_per_doc,
        seed=seed,
        category=category,
    )
    rng = random.Random(seed)
    rng.shuffle(positions)
    positions = positions[:max_eval_positions]
    n_calib = int(len(positions) * calib_frac)
    calib_positions = positions[:n_calib]
    eval_positions = positions[n_calib:]
    print(f"[ple-baseline] {task_name}: positions={len(positions)} calib={len(calib_positions)} eval={len(eval_positions)}", flush=True)

    cal = _calibrate_ple(
        model,
        tokenizer,
        device,
        calib_positions,
        memory,
        context_len=context_len,
    )
    if not cal:
        print(f"[ple-baseline] {task_name}: calibration failed", flush=True)
        return {}

    scale = float(cal["best_scale"])
    bias = float(cal["best_bias"])
    rows = []
    t0 = time.time()
    for seq, i in eval_positions:
        context = seq[max(0, i - context_len) : i]
        target = seq[i]
        query_text = tokenizer.decode(context, skip_special_tokens=True)
        base_lp, base_hit = _predict_logprob(model, tokenizer, context, target, device)

        # BM25 retrieval as context prepending.
        bm25_text = _retrieved_docs_text(bm25, query_text, train_texts, top_k)
        bm25_ids = tokenizer.encode(bm25_text, add_special_tokens=False) if bm25_text else []
        bm25_ids = bm25_ids[:max_rag_tokens]
        bm25_lp, bm25_hit = _predict_logprob(
            model, tokenizer, bm25_ids + context, target, device
        ) if bm25_ids else (base_lp, base_hit)

        # Exact n-gram retrieval as context prepending.
        ngram_ctx_text = _retrieved_docs_text(ngram_retriever, query_text, train_texts, top_k)
        ngram_ctx_ids = tokenizer.encode(ngram_ctx_text, add_special_tokens=False) if ngram_ctx_text else []
        ngram_ctx_ids = ngram_ctx_ids[:max_rag_tokens]
        ngram_ctx_lp, ngram_ctx_hit = _predict_logprob(
            model, tokenizer, ngram_ctx_ids + context, target, device
        ) if ngram_ctx_ids else (base_lp, base_hit)

        # PLE calibrated log-linear fusion.
        dist = memory.continuation_distribution(context)
        dist = dist[0] if dist else None
        if dist:
            ids = torch.tensor([context], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids=ids, use_cache=False).logits[0, -1].float().cpu().numpy()
            fused = fuse_ngram_logits(
                logits, dist, scale=scale, bias=bias, temperature=global_temp
            )
            ple_lp = _logprob(fused, target)
            ple_hit = bool(int(np.argmax(fused)) == target)
        else:
            ple_lp = base_lp
            ple_hit = base_hit

        # BM25 + PLE: use BM25 context to condition the base model, then apply
        # the calibrated PLE n-gram prior on top of the resulting logits.
        bm25_ple_lp = bm25_lp
        bm25_ple_hit = bm25_hit
        if dist and bm25_ids:
            ids = torch.tensor([bm25_ids + context], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids=ids, use_cache=False).logits[0, -1].float().cpu().numpy()
            fused = fuse_ngram_logits(
                logits, dist, scale=scale, bias=bias, temperature=global_temp
            )
            bm25_ple_lp = _logprob(fused, target)
            bm25_ple_hit = bool(int(np.argmax(fused)) == target)

        # Raw n-gram memory distribution (no base model).
        if dist:
            ngram_raw_lp = float(math.log(max(dist.get(target, 0.0), 1e-12)))
            ngram_raw_hit = bool(max(dist, key=dist.get) == target)
        else:
            ngram_raw_lp = None
            ngram_raw_hit = False

        rows.append(
            {
                "base_logprob": base_lp,
                "base_hit": base_hit,
                "bm25_logprob": bm25_lp,
                "bm25_hit": bm25_hit,
                "ngram_retrieval_logprob": ngram_ctx_lp,
                "ngram_retrieval_hit": ngram_ctx_hit,
                "ple_logprob": ple_lp,
                "ple_hit": ple_hit,
                "bm25_ple_logprob": bm25_ple_lp,
                "bm25_ple_hit": bm25_ple_hit,
                "ngram_raw_logprob": ngram_raw_lp,
                "ngram_raw_hit": ngram_raw_hit,
            }
        )
        if (len(rows) % 25) == 0:
            print(f"[ple-baseline] {task_name}: {len(rows)}/{len(eval_positions)}", flush=True)

    summary = _aggregate(rows)
    summary["calibration"] = cal
    summary["runtime_seconds"] = time.time() - t0
    return summary


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
    parser.add_argument("--max-eval-positions", type=int, default=160)
    parser.add_argument("--calib-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-rag-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/ple-baseline-ablation.json")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = _load_model(args.model, args.adapter, args.device)

    code_files = _python_files(Path(args.code_root), args.max_code_files)
    code_texts = []
    for f in code_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        if text.strip():
            code_texts.append(text)
    print(f"[ple-baseline] code files={len(code_files)} texts={len(code_texts)}", flush=True)

    wiki_texts = _load_wiki_docs(args.wiki_path, args.max_wiki_docs)
    print(f"[ple-baseline] wiki docs={len(wiki_texts)}", flush=True)

    code_train_texts, code_eval_texts = _split_texts(code_texts, args.train_frac, args.seed)
    wiki_train_texts, wiki_eval_texts = _split_texts(wiki_texts, args.train_frac, args.seed)

    code_mem = _build_memory(code_train_texts, tokenizer, max_order=args.max_order, seed=args.seed)
    wiki_mem = _build_memory(wiki_train_texts, tokenizer, max_order=args.max_order, seed=args.seed)

    code_bm25 = BM25Index(code_train_texts)
    wiki_bm25 = BM25Index(wiki_train_texts)
    code_ngram_retriever = NgramKeyRetriever(
        code_mem,
        tokenizer=lambda text: tokenizer.encode(text, add_special_tokens=False),
        top_k=args.top_k,
    )
    wiki_ngram_retriever = NgramKeyRetriever(
        wiki_mem,
        tokenizer=lambda text: tokenizer.encode(text, add_special_tokens=False),
        top_k=args.top_k,
    )

    code_eval_seqs = [
        tokenizer.encode(t, add_special_tokens=False) for t in code_eval_texts
        if len(tokenizer.encode(t, add_special_tokens=False)) > 8
    ]
    wiki_eval_seqs = [
        tokenizer.encode(t, add_special_tokens=False) for t in wiki_eval_texts
        if len(tokenizer.encode(t, add_special_tokens=False)) > 16
    ]

    tasks = {
        "code": {
            "eval_seqs": code_eval_seqs,
            "train_texts": code_train_texts,
            "memory": code_mem,
            "bm25": code_bm25,
            "ngram_retriever": code_ngram_retriever,
            "max_per_doc": args.max_per_doc_code,
            "category": None,
        },
        "name": {
            "eval_seqs": wiki_eval_seqs,
            "train_texts": wiki_train_texts,
            "memory": wiki_mem,
            "bm25": wiki_bm25,
            "ngram_retriever": wiki_ngram_retriever,
            "max_per_doc": args.max_per_doc_wiki,
            "category": "name",
        },
        "number": {
            "eval_seqs": wiki_eval_seqs,
            "train_texts": wiki_train_texts,
            "memory": wiki_mem,
            "bm25": wiki_bm25,
            "ngram_retriever": wiki_ngram_retriever,
            "max_per_doc": args.max_per_doc_wiki,
            "category": "number",
        },
    }

    results = {}
    for task_name, cfg in tasks.items():
        results[task_name] = evaluate_task(
            model,
            tokenizer,
            task_name=task_name,
            eval_seqs=cfg["eval_seqs"],
            train_texts=cfg["train_texts"],
            memory=cfg["memory"],
            bm25=cfg["bm25"],
            ngram_retriever=cfg["ngram_retriever"],
            context_len=args.context_len,
            max_per_doc=cfg["max_per_doc"],
            seed=args.seed,
            category=cfg["category"],
            calib_frac=args.calib_frac,
            max_eval_positions=args.max_eval_positions,
            top_k=args.top_k,
            max_rag_tokens=args.max_rag_tokens,
            device=args.device,
            global_temp=1.0,
        )
        print(json.dumps({k: results[task_name].get(k) for k in [
            "n", "base_logprob", "bm25_logprob", "ngram_retrieval_logprob",
            "ple_logprob", "bm25_ple_logprob", "delta_bm25_vs_base",
            "delta_ngram_retrieval_vs_base", "delta_ple_vs_base",
            "delta_ple_vs_bm25", "delta_ple_vs_ngram_retrieval",
            "delta_bm25_ple_vs_base", "delta_bm25_ple_vs_bm25", "delta_bm25_ple_vs_ple",
        ]}, ensure_ascii=False), flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "ple-baseline-ablation-v1",
        "config": vars(args),
        "model": args.model,
        "adapter": args.adapter,
        "results": results,
        "runtime_seconds": time.time() - t0,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[ple-baseline] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
