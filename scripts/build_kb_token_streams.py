#!/usr/bin/env python3
"""Convert KB/QA splits into token streams for unseen-KB memory experiments.

Expected input layout (produced by ``scripts/build_phase1_kb_split.py``)::

    <kb-dir>/kb.train.jsonl
    <kb-dir>/kb.eval.jsonl

Output::

    <output-dir>/train.tokens.npy
    <output-dir>/eval.tokens.npy
    <output-dir>/manifest.json

The reader can be trained on ``train.tokens.npy`` and then loaded/evalled on
``eval.tokens.npy``.  Because ``kb.eval`` is disjoint from ``kb.train``, this
provides the unseen-KB condition required by the memory-reading protocol.

Usage::

    python scripts/build_kb_token_streams.py \
      --kb-dir data/phase1/kb-wiki \
      --tokenizer data/models/Qwen3.5-0.8B \
      --output-dir data/phase1/kb-wiki-tokens
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_docs(path: Path) -> list[str]:
    docs: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(obj.get("text") or obj.get("content") or "")
            if text.strip():
                docs.append(text.strip())
    return docs


def _tokenize_docs(tokenizer, docs: list[str]) -> np.ndarray:
    ids: list[int] = []
    for idx, doc in enumerate(docs):
        ids.extend(tokenizer.encode(doc, add_special_tokens=False))
        if idx + 1 < len(docs):
            eos = tokenizer.eos_token_id
            if eos is not None:
                ids.append(eos)
    return np.asarray(ids, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-dir", required=True)
    parser.add_argument("--tokenizer", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    kb_dir = Path(args.kb_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True
    )

    outputs: dict[str, Path] = {}
    manifest: dict = {"kb_dir": str(kb_dir.resolve()), "tokenizer": args.tokenizer}
    for name in ("train", "eval"):
        src = kb_dir / f"kb.{name}.jsonl"
        if not src.exists():
            raise SystemExit(f"missing {src}")
        dest = out_dir / f"{name}.tokens.npy"
        if dest.exists() and not args.force:
            print(f"[kb-tokens] skip existing {dest}", flush=True)
            continue
        docs = _load_docs(src)
        arr = _tokenize_docs(tokenizer, docs)
        np.save(dest, arr)
        outputs[name] = dest
        manifest[f"{name}_docs"] = len(docs)
        manifest[f"{name}_tokens"] = len(arr)
        print(
            f"[kb-tokens] {name}: docs={len(docs)} tokens={len(arr)} -> {dest}",
            flush=True,
        )

    manifest["outputs"] = {k: str(v.resolve()) for k, v in outputs.items()}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
