#!/usr/bin/env python3
"""Evaluate the P1 memory interface on the rare-token QA benchmark.

The script compares three tied conditions:

* ``no-memory``: frozen backbone only;
* ``real``: frozen backbone + P1 memory module using the real exact bank;
* ``control``: same P1 memory module using the control bank (same exact keys,
  shuffled PLE values).

The primary metric is answer-token log-probability (and top-1 exact-match) on
rare vs common QA items.  A positive real > control gap on rare items is the P1
gate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qwen35_ple.memory.bank import ExactNgramBank
from qwen35_ple.memory.token_mem import P1MemoryModule


def _load_model(model_path: str, device: str):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model


def _load_qa(path: str, limit: int | None) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("items") if isinstance(data, dict) else data
    if limit is not None:
        items = items[:limit]
    return items


class QAEtStore:
    """Minimal live PLE e_t fetcher for QA tokens (optional)."""

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


def _load_module(checkpoint: str, d_model: int, d_mem: int, vocab_size: int, device: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    module = P1MemoryModule(
        d_model=d_model,
        d_mem=d_mem,
        vocab_size=vocab_size,
        n_heads=int(cfg.get("n_heads", 4)),
        head_hidden=int(cfg.get("head_hidden", 256)),
        router_hidden=int(cfg.get("router_hidden", 64)),
    ).to(device)
    module.load_state_dict(payload["state_dict"])
    module.eval()
    return module


def _answer_logprob(logits: torch.Tensor, ids: np.ndarray, answer_start: int) -> float:
    """Sum log-prob of answer tokens given the full sequence."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    total = 0.0
    n = 0
    start = max(0, answer_start - 1)
    for t in range(start, len(ids) - 1):
        total += float(log_probs[t, int(ids[t + 1])])
        n += 1
    return total / max(1, n)


def _top1_exact(
    logits: torch.Tensor, ids: np.ndarray, answer_start: int, answer_text: str
) -> bool:
    """Check whether the most likely token at the first answer position is the
    first answer token (a weak exact-match proxy)."""
    t = max(0, answer_start - 1)
    if t >= len(ids) - 1:
        return False
    pred = int(torch.argmax(logits[t]))
    # For a single-token answer this is exact; for multi-token answers this is
    # only a first-token hit and is reported separately.
    return pred == int(ids[t + 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--checkpoint", default="outputs/p1-memory-real.pt")
    parser.add_argument("--bank-real", default="data/exact-ple-bank.npz")
    parser.add_argument("--bank-control", default="data/exact-ple-bank-control.npz")
    parser.add_argument("--qa-file", default="data/rare-kb-v1.json")
    parser.add_argument("--rows-dir", default=None)
    parser.add_argument("--scale", type=float, default=0.00019931793212890625)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/p1-memory-eval.json")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = _load_model(args.model, args.device)
    bank_real = ExactNgramBank.load(args.bank_real)
    bank_control = ExactNgramBank.load(args.bank_control)
    module = _load_module(
        args.checkpoint,
        int(model.config.hidden_size),
        int(bank_real.d_mem),
        int(model.config.vocab_size),
        args.device,
    )
    items = _load_qa(args.qa_file, args.limit)
    print(
        f"[eval-p1] items={len(items)} banks={bank_real.num_entries}/"
        f"{bank_control.num_entries}",
        flush=True,
    )

    qa_store = None
    if args.rows_dir:
        qa_store = QAEtStore(args.rows_dir, args.scale)

    results: list[dict] = []
    try:
        for idx, item in enumerate(items):
            question = str(item.get("question", ""))
            answer = str(item.get("answer", ""))
            qids = tokenizer.encode(question, add_special_tokens=False)
            full_ids = tokenizer.encode(f"{question} {answer}", add_special_tokens=False)
            answer_start = len(qids)
            if len(full_ids) < 2:
                continue
            fallback = None
            if qa_store is not None:
                fallback = qa_store.fetch(full_ids)

            mem_real, _ = bank_real.lookup_multi(full_ids, fallback=fallback)
            mem_control, _ = bank_control.lookup_multi(full_ids, fallback=fallback)

            ids = torch.from_numpy(np.asarray(full_ids, dtype=np.int64)).unsqueeze(0).to(args.device)
            with torch.no_grad():
                out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
                h = out.hidden_states[args.layer][0].float()
                base_logits = out.logits.float()

                m_real = torch.from_numpy(mem_real).unsqueeze(0).to(args.device)
                m_control = torch.from_numpy(mem_control).unsqueeze(0).to(args.device)
                fused_real, _, alpha_real = module(h.unsqueeze(0), m_real, base_logits)
                fused_control, _, alpha_control = module(h.unsqueeze(0), m_control, base_logits)

            conditions = {
                "no-memory": base_logits[0],
                "real": fused_real[0],
                "control": fused_control[0],
            }
            entry: dict = {
                "id": str(item.get("id", idx)),
                "task": str(item.get("task", "qa")),
                "is_rare": bool(item.get("is_rare", False)),
                "answer": answer,
                "answer_start": answer_start,
                "first_token": int(full_ids[answer_start]) if answer_start < len(full_ids) else None,
                "conditions": {},
            }
            for cond, logits in conditions.items():
                entry["conditions"][cond] = {
                    "answer_logprob": _answer_logprob(logits, np.asarray(full_ids), answer_start),
                    "first_token_hit": _top1_exact(
                        logits, np.asarray(full_ids), answer_start, answer
                    ),
                    "mean_alpha": (
                        float(alpha_real.mean()) if cond == "real"
                        else float(alpha_control.mean()) if cond == "control"
                        else None
                    ),
                }
            results.append(entry)
            if (idx + 1) % 25 == 0:
                print(f"  [{idx + 1}/{len(items)}]", flush=True)
    finally:
        if qa_store is not None:
            qa_store.close()

    # Summaries.
    def avg(values: list[float | None]) -> float | None:
        vals = [v for v in values if v is not None]
        return float(np.mean(vals)) if vals else None

    summary: dict = {}
    rare = [r for r in results if r["is_rare"]]
    common = [r for r in results if not r["is_rare"]]
    for group_name, group in [("all", results), ("rare", rare), ("common", common)]:
        row: dict = {"n": len(group)}
        for cond in ["no-memory", "real", "control"]:
            row[f"{cond}_answer_logprob"] = avg(
                [r["conditions"][cond]["answer_logprob"] for r in group]
            )
            row[f"{cond}_first_token_hit"] = avg(
                [1.0 if r["conditions"][cond]["first_token_hit"] else 0.0 for r in group]
            )
        row["real_minus_control_logprob"] = (
            row["real_answer_logprob"] - row["control_answer_logprob"]
            if row["real_answer_logprob"] is not None
            and row["control_answer_logprob"] is not None
            else None
        )
        row["real_minus_control_first_hit"] = (
            row["real_first_token_hit"] - row["control_first_token_hit"]
            if row["real_first_token_hit"] is not None
            and row["control_first_token_hit"] is not None
            else None
        )
        summary[group_name] = row

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
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[eval-p1] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
