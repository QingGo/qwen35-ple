#!/usr/bin/env python3
"""Activation-patching style mechanism probe for the PLE reader.

For a small QA sample this script generates greedy outputs under several
injection conditions:

* ``no-reader``  : no PLE e_t (reader installed but disabled)
* ``real``       : real PLE e_t fetched from the EngramDB store
* ``control``    : real PLE e_t but token-order shuffled (same as Phase 0 control)
* ``random``     : random vectors with the same per-token L2 norms as real e_t
* ``zero``       : all-zero PLE e_t

The main purpose is to see whether the observed QA differences are causally
driven by the *content* of the injected e_t, or merely by the presence of any
injection/reader perturbation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

CONDITIONS = ["no-reader", "real", "control", "random", "zero"]


def _load_model(model_path: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    if next(model.parameters()).dtype != torch.float32:
        model = model.to(torch.float32)
    model.eval()
    return tokenizer, model


def _load_reader(reader_path: str):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from qwen35_ple.reader_registry import load_reader_with_extra
    return load_reader_with_extra(reader_path, device="cpu")


def _install_reader(model, reader, layer: int):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from qwen35_ple.reader import install_reader_hook
    return install_reader_hook(model, layer, reader)


def _load_qa(path: str, limit: int | None, tasks: list[str] | None):
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if tasks:
        items = [x for x in items if x.get("task") in tasks]
    if limit is not None:
        items = items[:limit]
    return items


def _normalize_answer(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(w for w in text.split() if w not in {"a", "an", "the"})


class QAEtStore:
    """Fetch real PLE e_t for arbitrary token sequences from EngramDB."""

    def __init__(self, rows_dir: str, scale: float):
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


def _random_like(real_et: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=real_et.shape).astype(np.float32)
    real_norms = np.linalg.norm(real_et, axis=1, keepdims=True)
    noise_norms = np.linalg.norm(noise, axis=1, keepdims=True)
    noise_norms[noise_norms == 0] = 1.0
    return noise * (real_norms / noise_norms)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--reader", default="outputs/reader-real-seed0.pt")
    parser.add_argument("--rows-dir", default="/home/zeng/qwen38-rows")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.0002)
    parser.add_argument("--qa-file", default="data/qa-expanded-150.json")
    parser.add_argument("--tasks", nargs="+", default=None, choices=["triviaqa", "nq", "boolq"])
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--output", default="outputs/mechanism-patching.json")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = _load_model(args.model)
    print(f"loaded model in {time.time()-t0:.1f}s", flush=True)

    reader, _ = _load_reader(args.reader)
    handle = _install_reader(model, reader, args.layer)
    qa_store = QAEtStore(args.rows_dir, args.scale)
    items = _load_qa(args.qa_file, args.limit, args.tasks)
    print(f"running {len(items)} QA under {args.conditions}", flush=True)

    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    results = []

    try:
        for idx, item in enumerate(items):
            qids = tokenizer.encode(item["question"], add_special_tokens=False)
            cond_outputs = {}
            for cond in args.conditions:
                ids = list(qids)
                generated_ids: list[int] = []
                with torch.no_grad():
                    for step in range(args.max_new_tokens):
                        if cond == "no-reader":
                            model._current_ple_e_t = None
                        else:
                            real_et = qa_store.fetch(ids)
                            if cond == "real":
                                et_np = real_et
                            elif cond == "control":
                                rng = np.random.default_rng(args.seed * 1000 + idx)
                                et_np = real_et[rng.permutation(len(real_et))]
                            elif cond == "random":
                                et_np = _random_like(real_et, args.seed * 1000 + idx + step)
                            elif cond == "zero":
                                et_np = np.zeros_like(real_et)
                            else:
                                raise ValueError(cond)
                            model._current_ple_e_t = torch.from_numpy(et_np[None, :]).float().to(device)
                        out = model(input_ids=torch.tensor([ids], dtype=torch.long, device=device))
                        next_id = int(torch.argmax(out.logits[0, -1]).item())
                        if eos_id is not None and next_id == eos_id:
                            break
                        generated_ids.append(next_id)
                        ids.append(next_id)
                gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                cond_outputs[cond] = {
                    "generated": gen_text,
                    "correct": _normalize_answer(item["answer"]) in _normalize_answer(gen_text),
                    "n_tokens": len(generated_ids),
                }
                print(f"  [{idx}] {cond:10s} {item['task']:8s} correct={cond_outputs[cond]['correct']} :: {gen_text[:120]!r}", flush=True)

            results.append(
                {
                    "task": item["task"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "conditions": cond_outputs,
                }
            )
    finally:
        qa_store.close()
        handle.remove()

    summary = {}
    for cond in args.conditions:
        hits = [r["conditions"][cond]["correct"] for r in results]
        summary[cond] = {
            "n": len(hits),
            "em": float(np.mean(hits)) if hits else None,
            "correct": int(sum(hits)),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": vars(args),
                "summary": summary,
                "results": results,
                "runtime_seconds": time.time() - t0,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("summary:", summary, flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
