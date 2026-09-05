#!/usr/bin/env python3
"""PLE-2 semantic-value addressable memory + 3-seed evaluation.

Values are upgraded from whole documents to semantic units:

* code: AST function/class blocks;
* wiki: paragraph chunks.

The script runs three seeds and aggregates retrieval/continuation recall,
allowing a first 3-seed evidence check for PLE-2 addressable memory.
"""

from __future__ import annotations

import argparse
import ast
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


def load_code_blocks(limit_files: int) -> list[str]:
    files: list[Path] = []
    for root in CODE_ROOTS:
        if root.exists():
            files.extend(sorted(root.rglob("*.py")))
    files = sorted(set(files))[:limit_files]
    blocks: list[str] = []
    for path in files:
        try:
            src = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            blocks.append(src)
            continue
        found = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                seg = ast.get_source_segment(src, node)
                if seg and seg.strip():
                    blocks.append(seg)
                    found = True
        if not found:
            blocks.append(src)
    return blocks


def split_sequences(seqs, train_frac, seed):
    rng = random.Random(seed)
    indices = list(range(len(seqs)))
    rng.shuffle(indices)
    n_train = max(1, int(round(len(seqs) * train_frac)))
    return [seqs[i] for i in indices[:n_train]], [seqs[i] for i in indices[n_train:]]


def contains_transition(seq, ctx, target, max_order) -> bool:
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
    hits = {"cont1": [], "cont3": [], "ret_exact": [], "ret_any": []}
    for si, i in positions:
        seq = seqs[si]
        ctx = seq[:i]
        y = seq[i]
        top = mem.topk(ctx, k=5)
        toks = [t for t, _, _ in top]
        hits["cont1"].append(any(t == y for t in toks[:1]))
        hits["cont3"].append(any(t == y for t in toks[:3]))
        ret_exact = False
        ret_any = False
        for m in mem.retrieve(ctx, top_k=3):
            if m.value_id >= len(value_seqs):
                continue
            vs = value_seqs[m.value_id]
            if contains_transition(vs, ctx, y, max_order):
                ret_exact = True
            if y in vs:
                ret_any = True
        hits["ret_exact"].append(ret_exact)
        hits["ret_any"].append(ret_any)
    return {
        "n": len(positions),
        "cont_top1": float(np.mean(hits["cont1"])),
        "cont_top3": float(np.mean(hits["cont3"])),
        "retrieval_exact": float(np.mean(hits["ret_exact"])),
        "retrieval_any": float(np.mean(hits["ret_any"])),
    }


def run_domain(name, seqs, tokenizer, *, max_order, max_positions, seeds):
    per_seed = []
    for seed in seeds:
        train, eval_ = split_sequences(seqs, 0.8, seed)
        mem = AddressableNgramMemory(min_order=2, max_order=max_order)
        value_seqs = []
        for i, seq in enumerate(train):
            if not seq:
                continue
            value_seqs.append(seq)
            mem.add_document(seq, value_id=i)
        res = evaluate(mem, value_seqs, eval_, max_positions=max_positions, max_order=max_order, seed=seed)
        per_seed.append(res)
        print(f"[sem] {name} seed={seed} {json.dumps(res)}", flush=True)
    keys = ["cont_top1", "cont_top3", "retrieval_exact", "retrieval_any"]
    agg = {"seeds": per_seed}
    for k in keys:
        vals = [r[k] for r in per_seed]
        agg[k] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return agg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/Users/zeng/code/LLM-CompileForge/models/Qwen/Qwen3.5-0.8B.local-backup")
    parser.add_argument("--wiki-limit", type=int, default=120)
    parser.add_argument("--code-files", type=int, default=80)
    parser.add_argument("--max-position", type=int, default=500)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output", default="outputs/ple2-semantic-values-3seed.json")
    parser.add_argument("--report", default="outputs/ple2-semantic-values-3seed.md")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer = _load_tokenizer(args.model_dir)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    wiki_blocks = load_wiki_docs(args.wiki_limit)
    wiki_seqs = [tokenizer.encode(d, add_special_tokens=False) for d in wiki_blocks]
    code_blocks = load_code_blocks(args.code_files)
    code_seqs = [tokenizer.encode(b, add_special_tokens=False) for b in code_blocks]

    results = {
        "schema": "ple2-semantic-values-3seed-v1",
        "seeds": seeds,
        "domains": {},
    }
    results["domains"]["wiki"] = run_domain("wiki", wiki_seqs, tokenizer, max_order=args.max_order, max_positions=args.max_position, seeds=seeds)
    results["domains"]["code"] = run_domain("code", code_seqs, tokenizer, max_order=args.max_order, max_positions=args.max_position, seeds=seeds)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PLE-2 语义 value（函数块/段落）3-seed 结果",
        "",
        f"- Seeds: {seeds}",
        f"- Max positions per seed: {args.max_position}",
        "",
        "| Domain | Cont@1 | Cont@3 | Ret exact | Ret any |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, d in results["domains"].items():
        lines.append(
            f"| {name} | {d['cont_top1']:.4f} | {d['cont_top3']:.4f} "
            f"| {d['retrieval_exact']:.4f} | {d['retrieval_any']:.4f} |"
        )
    rep.write_text("\n".join(lines), encoding="utf-8")
    print(f"[sem] wrote {out} and {rep} in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
