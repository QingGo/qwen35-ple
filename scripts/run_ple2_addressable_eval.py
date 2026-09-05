#!/usr/bin/env python3
"""PLE-2: evaluate addressable n-gram external memory real vs control.

This is the first empirical check for the PLE-2 architecture:

* discrete n-gram key;
* external value (document / chunk / entity id);
* continuation recall (top-k);
* retrieval evidence: does a retrieved value actually contain the n-gram
  transition that leads to the observed next token?

Real memory is built from original document token order.  Control memory is
built from the same documents with token order shuffled inside each document,
so both share the same unigram reservoirs but not the original key->value
lexical associations.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from qwen35_ple.addressable_memory import AddressableNgramMemory

CODE_ROOTS = [
    Path("src/qwen35_ple"),
    Path("scripts"),
    Path("tests"),
    Path("/Users/zeng/code/engram-peft/src"),
]


def _load_tokenizer(model_dir: str):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def load_wiki_docs(limit: int) -> list[str]:
    docs: list[str] = []
    with Path("data/sources/wikitext.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(obj.get("text") or "")
            if text.strip():
                docs.append(text.strip())
            if len(docs) >= limit:
                break
    return docs


def load_code_docs(limit: int) -> list[str]:
    files: list[Path] = []
    for root in CODE_ROOTS:
        if root.exists():
            files.extend(sorted(root.rglob("*.py")))
    files = sorted(set(files))
    out = []
    for p in files[:limit]:
        try:
            out.append(p.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return out


def tokenize_docs(docs, tokenizer) -> list[list[int]]:
    return [tokenizer.encode(d, add_special_tokens=False) for d in docs]


def split_sequences(seqs, train_frac, seed):
    rng = random.Random(seed)
    indices = list(range(len(seqs)))
    rng.shuffle(indices)
    n_train = max(1, round(len(seqs) * train_frac))
    train = [seqs[i] for i in indices[:n_train]]
    eval_ = [seqs[i] for i in indices[n_train:]]
    return train, eval_


def build_memory(seqs, *, max_order, shuffle, seed):
    mem = AddressableNgramMemory(min_order=2, max_order=max_order)
    rng = random.Random(seed) if shuffle else None
    value_seqs = []
    for value_id, seq in enumerate(seqs):
        if shuffle:
            seq = list(seq)
            rng.shuffle(seq)
        value_seqs.append(seq)
        mem.add_document(seq, value_id=value_id)
    return mem, value_seqs


def contains_transition(seq, ctx, target, max_order) -> bool:
    """Return True if ``target`` immediately follows any suffix of ``ctx`` in seq."""
    n = min(max_order, len(ctx))
    while n >= 2:
        suffix = tuple(ctx[-n:])
        for i in range(n, len(seq)):
            if tuple(seq[i - n : i]) == suffix and seq[i] == target:
                return True
        n -= 1
    return False


def evaluate(mem, value_seqs, seqs, *, max_positions, max_order, seed):
    positions = []
    for si, seq in enumerate(seqs):
        for i in range(1, len(seq)):
            positions.append((si, i))
    rng = random.Random(seed)
    if max_positions and len(positions) > max_positions:
        positions = rng.sample(positions, max_positions)

    rows = []
    for si, i in positions:
        seq = seqs[si]
        ctx = seq[:i]
        y = seq[i]
        real_top = mem.topk(ctx, k=5)
        real_tokens = [t for t, _, _ in real_top]
        match_order = real_top[0][2] if real_top else 0
        cont_hit = {
            "1": any(t == y for t in real_tokens[:1]),
            "3": any(t == y for t in real_tokens[:3]),
            "5": any(t == y for t in real_tokens[:5]),
        }
        matches = mem.retrieve(ctx, top_k=3)
        retrieval_hit = False
        retrieval_hit_any = False
        for m in matches:
            if m.value_id >= len(value_seqs):
                continue
            vs = value_seqs[m.value_id]
            if contains_transition(vs, ctx, y, max_order):
                retrieval_hit = True
            if y in vs:
                retrieval_hit_any = True
        rows.append({
            "seq": si,
            "pos": i,
            "target": int(y),
            "cont_hit_1": cont_hit["1"],
            "cont_hit_3": cont_hit["3"],
            "cont_hit_5": cont_hit["5"],
            "retrieval_hit": retrieval_hit,
            "retrieval_hit_any": retrieval_hit_any,
            "match_order": match_order,
        })

    summary = {
        "n": len(rows),
        "cont_top1": float(np.mean([r["cont_hit_1"] for r in rows])),
        "cont_top3": float(np.mean([r["cont_hit_3"] for r in rows])),
        "cont_top5": float(np.mean([r["cont_hit_5"] for r in rows])),
        "retrieval_exact_hit": float(np.mean([r["retrieval_hit"] for r in rows])),
        "retrieval_any_hit": float(np.mean([r["retrieval_hit_any"] for r in rows])),
        "mean_match_order": float(np.mean([r["match_order"] for r in rows])),
    }
    return summary, rows


def format_summary(s: dict) -> str:
    return (
        f"n={s['n']} top1={s['cont_top1']:.4f} top3={s['cont_top3']:.4f} "
        f"top5={s['cont_top5']:.4f} ret_exact={s['retrieval_exact_hit']:.4f} "
        f"ret_any={s['retrieval_any_hit']:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/Users/zeng/code/LLM-CompileForge/models/Qwen/Qwen3.5-0.8B.local-backup")
    parser.add_argument("--wiki-limit", type=int, default=300)
    parser.add_argument("--code-limit", type=int, default=120)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--max-position", type=int, default=2000)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/ple2-addressable-eval.json")
    parser.add_argument("--report", default="outputs/ple2-addressable-eval-report.md")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer = _load_tokenizer(args.model_dir)
    wiki_seqs = tokenize_docs(load_wiki_docs(args.wiki_limit), tokenizer)
    wiki_train, wiki_eval = split_sequences(wiki_seqs, args.train_frac, args.seed)
    code_seqs = tokenize_docs(load_code_docs(args.code_limit), tokenizer)
    code_train, code_eval = split_sequences(code_seqs, args.train_frac, args.seed)

    results = {
        "schema": "ple2-addressable-eval-v1",
        "seed": args.seed,
        "max_order": args.max_order,
        "domains": {},
    }

    for name, train, eval_ in [("wiki", wiki_train, wiki_eval), ("code", code_train, code_eval)]:
        real_mem, real_values = build_memory(train, max_order=args.max_order, shuffle=False, seed=args.seed)
        ctrl_mem, ctrl_values = build_memory(train, max_order=args.max_order, shuffle=True, seed=args.seed)
        real_summary, _ = evaluate(
            real_mem, real_values, eval_,
            max_positions=args.max_position, max_order=args.max_order, seed=args.seed,
        )
        ctrl_summary, _ = evaluate(
            ctrl_mem, ctrl_values, eval_,
            max_positions=args.max_position, max_order=args.max_order, seed=args.seed,
        )
        results["domains"][name] = {
            "real": real_summary,
            "control": ctrl_summary,
            "deltas": {
                "cont_top1": real_summary["cont_top1"] - ctrl_summary["cont_top1"],
                "cont_top3": real_summary["cont_top3"] - ctrl_summary["cont_top3"],
                "cont_top5": real_summary["cont_top5"] - ctrl_summary["cont_top5"],
                "retrieval_exact_hit": real_summary["retrieval_exact_hit"] - ctrl_summary["retrieval_exact_hit"],
                "retrieval_any_hit": real_summary["retrieval_any_hit"] - ctrl_summary["retrieval_any_hit"],
            },
        }
        print(f"[ple2] {name} real: {format_summary(real_summary)}", flush=True)
        print(f"[ple2] {name} ctrl: {format_summary(ctrl_summary)}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PLE-2：可寻址 n-gram 外部记忆 real vs control",
        "",
        f"- Seed: {args.seed}",
        f"- Max order: {args.max_order}",
        f"- Max positions: {args.max_position}",
        "",
        "| Domain | Model | top1 | top3 | top5 | retrieval exact | retrieval any |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, d in results["domains"].items():
        for model in ["real", "control"]:
            s = d[model]
            lines.append(
                f"| {name} | {model} | {s['cont_top1']:.4f} | {s['cont_top3']:.4f} "
                f"| {s['cont_top5']:.4f} | {s['retrieval_exact_hit']:.4f} | {s['retrieval_any_hit']:.4f} |"
            )
    lines += ["", "## Δ (real - control)", "", "| Domain | top1 | top3 | top5 | retrieval exact | retrieval any |", "|---|---:|---:|---:|---:|---:|"]
    for name, d in results["domains"].items():
        dt = d["deltas"]
        lines.append(
            f"| {name} | {dt['cont_top1']:.4f} | {dt['cont_top3']:.4f} "
            f"| {dt['cont_top5']:.4f} | {dt['retrieval_exact_hit']:.4f} | {dt['retrieval_any_hit']:.4f} |"
        )
    rep.write_text("\n".join(lines), encoding="utf-8")

    print(f"[ple2] wrote {out} and {rep} in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
