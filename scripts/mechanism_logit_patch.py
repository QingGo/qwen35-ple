#!/usr/bin/env python3
"""Fast logit-level activation patching for the PLE reader.

This is the cheap version of ``mechanism_patching.py``.  Instead of
generating token-by-token, it performs one forward pass per condition on the
full question (+ optionally answer) sequence and records:

* answer-token cross-entropy / log-probability
* next-token entropy after the question
* top-1 predicted token after the question
* logit deltas relative to the no-reader condition

Conditions:
    no-reader / real / control / random / zero
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
import torch.nn.functional as F

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


def _install_reader(model, reader, layer: int, inject_scale: float = 1.0):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    if inject_scale == 1.0:
        from qwen35_ple.reader import install_reader_hook
        return install_reader_hook(model, layer, reader)

    layer_module = model.model.layers[layer]

    def post_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        current = getattr(model, "_current_ple_e_t", None)
        if current is not None and current.shape[1] == hidden.shape[1]:
            contribution = reader(hidden, current)
            new_hidden = hidden + contribution * inject_scale
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden
        return output

    return layer_module.register_forward_hook(post_hook)


def _load_qa(path: str, limit: int | None, tasks: list[str] | None):
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if tasks:
        items = [x for x in items if x.get("task") in tasks]
    if limit is not None:
        items = items[:limit]
    return items


class QAEtStore:
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


def _topk(logits: torch.Tensor, k: int = 5):
    vals, idxs = torch.topk(logits, k)
    return [int(i) for i in idxs.tolist()], [float(v) for v in vals.tolist()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--reader", default="outputs/reader-real-seed0.pt")
    parser.add_argument("--rows-dir", default="/home/zeng/qwen38-rows")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--inject-scale", type=float, default=1.0, help="multiplier on reader output contribution")
    parser.add_argument("--scale", type=float, default=0.0002)
    parser.add_argument("--qa-file", default="data/qa-expanded-150.json")
    parser.add_argument("--tasks", nargs="+", default=None, choices=["triviaqa", "nq", "boolq"])
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--output", default="outputs/mechanism-logit-patch.json")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = _load_model(args.model)
    print(f"loaded model in {time.time()-t0:.1f}s", flush=True)

    reader, _ = _load_reader(args.reader)
    handle = _install_reader(model, reader, args.layer, args.inject_scale)
    qa_store = QAEtStore(args.rows_dir, args.scale)
    items = _load_qa(args.qa_file, args.limit, args.tasks)
    device = next(model.parameters()).device
    results = []

    try:
        for idx, item in enumerate(items):
            qids = tokenizer.encode(item["question"], add_special_tokens=False)
            full_text = item["question"] + " " + item["answer"]
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            answer_start = len(qids)
            # answer token span (without the separator space, good enough for CE)
            ans_ids = full_ids[answer_start:]
            cond_out = {}
            baseline_logits = None

            for cond in args.conditions:
                if cond == "no-reader":
                    model._current_ple_e_t = None
                    et_np = None
                else:
                    real_et = qa_store.fetch(full_ids)
                    if cond == "real":
                        et_np = real_et
                    elif cond == "control":
                        rng = np.random.default_rng(args.seed * 1000 + idx)
                        et_np = real_et[rng.permutation(len(real_et))]
                    elif cond == "random":
                        et_np = _random_like(real_et, args.seed * 1000 + idx)
                    elif cond == "zero":
                        et_np = np.zeros_like(real_et)
                    else:
                        raise ValueError(cond)
                    model._current_ple_e_t = torch.from_numpy(et_np[None, :]).float().to(device)

                ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model(input_ids=ids_t)
                logits = out.logits[0]  # [T, V]

                # Answer-token loss over the generated/answer span.
                if ans_ids:
                    loss = F.cross_entropy(
                        logits[answer_start - 1 : -1].reshape(-1, logits.size(-1)),
                        torch.tensor(ans_ids, device=device),
                    )
                    answer_logprob = -float(loss.item())
                else:
                    loss = None
                    answer_logprob = None

                next_logits = logits[-1]
                probs = F.softmax(next_logits, dim=-1)
                eps = 1e-12
                entropy = float(-(probs * torch.log(probs + eps)).sum().item())
                top_ids, _ = _topk(next_logits, k=5)
                top_text = [tokenizer.decode([i], skip_special_tokens=True) for i in top_ids]

                if baseline_logits is None:
                    baseline_logits = next_logits
                    delta_text = None
                    gold_delta = None
                else:
                    delta = next_logits - baseline_logits
                    # logit delta on the gold first answer token if available
                    if ans_ids:
                        first_ans = ans_ids[0]
                        gold_delta = float(delta[first_ans].item())
                    else:
                        gold_delta = None
                    top_delta_ids, _ = _topk(delta, k=5)
                    delta_text = [tokenizer.decode([i], skip_special_tokens=True) for i in top_delta_ids]

                cond_out[cond] = {
                    "answer_logprob": answer_logprob,
                    "next_entropy": entropy,
                    "next_top1": tokenizer.decode([int(top_ids[0])], skip_special_tokens=True),
                    "next_top5": top_text,
                    "gold_first_answer": tokenizer.decode([int(ans_ids[0])], skip_special_tokens=True) if ans_ids else None,
                    "gold_first_delta_vs_no_reader": gold_delta,
                    "top5_delta_vs_no_reader": delta_text,
                    "n_tokens": len(full_ids),
                }
                print(f"  [{idx}] {cond:10s} {item['task']:8s} loss={loss.item() if loss is not None else float('nan'):.4f} top1={cond_out[cond]['next_top1']!r}", flush=True)

            results.append(
                {
                    "task": item["task"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "conditions": cond_out,
                }
            )
    finally:
        qa_store.close()
        handle.remove()

    summary = {}
    for cond in args.conditions:
        vals = [r["conditions"][cond]["answer_logprob"] for r in results if r["conditions"][cond]["answer_logprob"] is not None]
        ent = [r["conditions"][cond]["next_entropy"] for r in results]
        summary[cond] = {
            "n": len(vals),
            "mean_answer_logprob": float(np.mean(vals)) if vals else None,
            "mean_next_entropy": float(np.mean(ent)) if ent else None,
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
