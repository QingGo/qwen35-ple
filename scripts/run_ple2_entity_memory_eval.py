#!/usr/bin/env python3
"""PLE-2 entity-value memory smoke evaluation.

Entity values are simple QA snippets ``question + answer``.  The test checks
whether exact n-gram addresses built from these entity snippets can recall the
answer's first token for held-out questions, compared to a shuffled control.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from qwen35_ple.addressable_memory import AddressableNgramMemory


def _load_tokenizer(model_dir: str):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def load_qa(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate(mem, seqs, targets, max_order=4, top_k=5):
    hits = {"1": [], "3": [], "5": []}
    for seq, target in zip(seqs, targets):
        ctx = seq[:-len(target)] if len(target) < len(seq) else seq[:-1] or []
        top = mem.topk(ctx, k=top_k)
        toks = [t for t, _, _ in top]
        hits["1"].append(int(target[0] in toks[:1]) if target else 0)
        hits["3"].append(int(target[0] in toks[:3]) if target else 0)
        hits["5"].append(int(target[0] in toks[:5]) if target else 0)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in hits.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/Users/zeng/code/LLM-CompileForge/models/Qwen/Qwen3.5-0.8B.local-backup")
    parser.add_argument("--qa-file", default="data/qa-expanded-150.json")
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output", default="outputs/ple2-entity-memory-eval.json")
    parser.add_argument("--report", default="outputs/ple2-entity-memory-eval.md")
    args = parser.parse_args()

    tokenizer = _load_tokenizer(args.model_dir)
    qa = load_qa(args.qa_file)
    items = []
    for x in qa:
        q = str(x.get("question", ""))
        a = str(x.get("answer", ""))
        if q and a:
            items.append((q, a))

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results = {"seeds": [], "mean": {}}
    for seed in seeds:
        rng = random.Random(seed)
        indices = list(range(len(items)))
        rng.shuffle(indices)
        n_train = max(1, int(len(items) * 0.8))
        train_ids = indices[:n_train]
        eval_ids = indices[n_train:]

        real = AddressableNgramMemory(min_order=2, max_order=args.max_order)
        ctrl = AddressableNgramMemory(min_order=2, max_order=args.max_order)
        for i in train_ids:
            q, a = items[i]
            seq = tokenizer.encode(f"{q} {a}", add_special_tokens=False)
            if seq:
                real.add_document(seq, value_id=i)
                shuf = list(seq)
                rng.shuffle(shuf)
                ctrl.add_document(shuf, value_id=i)

        eval_seqs = []
        targets = []
        for i in eval_ids:
            q, a = items[i]
            seq = tokenizer.encode(f"{q} {a}", add_special_tokens=False)
            ans = tokenizer.encode(a, add_special_tokens=False)
            if seq and ans:
                eval_seqs.append(seq)
                targets.append(ans)

        real_res = evaluate(real, eval_seqs, targets, max_order=args.max_order)
        ctrl_res = evaluate(ctrl, eval_seqs, targets, max_order=args.max_order)
        entry = {"seed": seed, "n": len(eval_seqs), "real": real_res, "control": ctrl_res}
        results["seeds"].append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    for k in ["1", "3", "5"]:
        vals = [s["real"][k] for s in results["seeds"]]
        cvals = [s["control"][k] for s in results["seeds"]]
        results["mean"][f"real@{k}"] = float(np.mean(vals)) if vals else 0.0
        results["mean"][f"control@{k}"] = float(np.mean(cvals)) if cvals else 0.0
        results["mean"][f"delta@{k}"] = results["mean"][f"real@{k}"] - results["mean"][f"control@{k}"]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PLE-2 实体 value（QA snippet）记忆评测",
        "",
        "| Seed | N | real@1 | control@1 | real@3 | control@3 | real@5 | control@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in results["seeds"]:
        lines.append(
            f"| {s['seed']} | {s['n']} | {s['real']['1']:.3f} | {s['control']['1']:.3f} "
            f"| {s['real']['3']:.3f} | {s['control']['3']:.3f} | {s['real']['5']:.3f} | {s['control']['5']:.3f} |"
        )
    m = results["mean"]
    lines.append(
        f"| mean | - | {m['real@1']:.3f} | {m['control@1']:.3f} | {m['real@3']:.3f} "
        f"| {m['control@3']:.3f} | {m['real@5']:.3f} | {m['control@5']:.3f} |"
    )
    rep.write_text("\n".join(lines), encoding="utf-8")
    print(f"[entity] wrote {out} and {rep}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
