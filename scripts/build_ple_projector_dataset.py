#!/usr/bin/env python3
"""Build a reusable PLE projector local-continuation dataset.

This script does not run the backbone.  It samples the same kind of P0 local
continuation points used by ``scripts/train_ple_projector.py`` and writes them
as JSONL, so larger 1k/10k datasets can be built and audited cheaply.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from qwen35_ple.addressable_memory import AddressableNgramMemory


def _load_code_corpus(path: str | None) -> list[str]:
    if not path:
        return []
    texts = []
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
                texts.append(text.strip())
    return texts


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


def _python_files(root: Path, max_files: int | None) -> list[Path]:
    files = []
    for p in root.rglob("*.py"):
        if ".venv" in p.parts or ".git" in p.parts:
            continue
        files.append(p)
    files.sort()
    return files[:max_files] if max_files is not None else files


def _split_texts(texts: list[str], train_frac: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    n_train = max(1, round(len(texts) * train_frac))
    return [texts[i] for i in idx[:n_train]], [texts[i] for i in idx[n_train:]]


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


def _sample_positions(
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


def _build_memory(train_texts: list[str], tokenizer, *, max_order: int):
    mem = AddressableNgramMemory(min_order=2, max_order=max_order)
    for value_id, text in enumerate(train_texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            mem.add_document(ids, value_id=value_id)
    return mem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--code-root", default=".")
    parser.add_argument("--code-corpus", default=None)
    parser.add_argument("--wiki-path", default="data/sources/wikitext.jsonl")
    parser.add_argument("--max-code-files", type=int, default=200)
    parser.add_argument("--max-wiki-docs", type=int, default=400)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--max-per-doc-code", type=int, default=20)
    parser.add_argument("--max-per-doc-wiki", type=int, default=10)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="data/ple-projector-dataset.jsonl")
    args = parser.parse_args()

    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    code_texts = []
    for f in _python_files(Path(args.code_root), args.max_code_files):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if text.strip():
            code_texts.append(text.strip())
    code_texts.extend(_load_code_corpus(args.code_corpus))
    wiki_texts = _load_wiki_docs(args.wiki_path, args.max_wiki_docs)

    code_train, code_eval = _split_texts(code_texts, args.train_frac, args.seed)
    wiki_train, wiki_eval = _split_texts(wiki_texts, args.train_frac, args.seed)
    code_mem = _build_memory(code_train, tokenizer, max_order=args.max_order)
    wiki_mem = _build_memory(wiki_train, tokenizer, max_order=args.max_order)

    code_eval_seqs = [
        tokenizer.encode(t, add_special_tokens=False)
        for t in code_eval
        if len(tokenizer.encode(t, add_special_tokens=False)) > 8
    ]
    wiki_eval_seqs = [
        tokenizer.encode(t, add_special_tokens=False)
        for t in wiki_eval
        if len(tokenizer.encode(t, add_special_tokens=False)) > 16
    ]

    positions = []
    for task, seqs, mem, max_per_doc, category in [
        ("code", code_eval_seqs, code_mem, args.max_per_doc_code, None),
        ("name", wiki_eval_seqs, wiki_mem, args.max_per_doc_wiki, "name"),
        ("number", wiki_eval_seqs, wiki_mem, args.max_per_doc_wiki, "number"),
    ]:
        sampled = _sample_positions(
            tokenizer,
            seqs,
            context_len=args.context_len,
            max_per_doc=max_per_doc,
            seed=args.seed,
            category=category,
        )
        for seq, i in sampled:
            context = seq[max(0, i - args.context_len) : i]
            dist_res = mem.continuation_distribution(context)
            if dist_res is None:
                continue
            positions.append(
                {
                    "task": task,
                    "context": context,
                    "target": seq[i],
                    "dist": dist_res[0],
                    "order": dist_res[1],
                }
            )

    rng = random.Random(args.seed)
    rng.shuffle(positions)
    positions = positions[: args.max_samples]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in positions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[dataset] wrote {len(positions)} samples to {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
