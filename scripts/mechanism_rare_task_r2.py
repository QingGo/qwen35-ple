#!/usr/bin/env python3
"""Task-level conditional-information probe for rare-token knowledge QA.

Phase A metric.  For every answer-token position in a knowledge question, we
collect:

    H  = backbone hidden state at the current position,
    E  = real PLE feature at that position,
    Y  = input embedding of the next token.

Then we fit ridge regressions on rare and common subsets and report:

    ΔR²_real      = R²(Y; [H,E]) - R²(Y; H)
    ΔR²_control   = R²(Y; [H,E_shuffled]) - R²(Y; H)
    ΔR²_perp      = R²(Y; [H,E⊥]) - R²(Y; H)

A positive real-vs-control gap on rare items is the Phase-A causal signal we
need before investing in reader training.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def _load_model(model_path: str, device: str = "cpu"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    if next(model.parameters()).dtype != torch.float32:
        model = model.to(torch.float32)
    if device != "cpu":
        model = model.to(device)
    model.eval()
    return tokenizer, model


def _load_qa(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        items = data.get("items")
    else:
        items = data
    if not isinstance(items, list):
        raise TypeError("QA file must be a JSON list or a dict with 'items'")
    return items


def _load_token_freq(tokens_path: str | None) -> dict[int, int]:
    if not tokens_path:
        return {}
    tokens = np.load(tokens_path, mmap_mode="r").astype(np.int64)
    vals, counts = np.unique(tokens, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts, strict=True)}


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


def _hidden(model, ids: list[int], layer: int) -> np.ndarray:
    device = next(model.parameters()).device
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=ids_t, output_hidden_states=True)
    h = out.hidden_states[layer][0]
    return h.detach().cpu().float().numpy()


def _center(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    return train - mean, test - mean


def _ridge_fit_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, lam: float
) -> np.ndarray:
    d = train_x.shape[1]
    xtx = train_x.T @ train_x + lam * np.eye(d)
    xty = train_x.T @ train_y
    beta = np.linalg.solve(xtx, xty)
    return test_x @ beta


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum(y_true**2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _residualize(
    train_h: np.ndarray,
    test_h: np.ndarray,
    train_e: np.ndarray,
    test_e: np.ndarray,
    lam: float,
):
    d = train_h.shape[1]
    beta = np.linalg.solve(
        train_h.T @ train_h + lam * np.eye(d),
        train_h.T @ train_e,
    )
    return train_e - train_h @ beta, test_e - test_h @ beta


def _subset_metrics(
    H: np.ndarray,
    E: np.ndarray,
    Y: np.ndarray,
    control_E: np.ndarray,
    test_frac: float,
    seed: int,
    lam: float,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(H)
    if n < 8:
        return {
            "n": n,
            "r2_H": float("nan"),
            "r2_HE_real": float("nan"),
            "r2_HE_control": float("nan"),
            "delta_real": float("nan"),
            "delta_control": float("nan"),
        }
    perm = rng.permutation(n)
    cut = max(1, int(n * (1.0 - test_frac)))
    if cut >= n:
        cut = n - 1
    tr = perm[:cut]
    te = perm[cut:]

    H_tr, H_te = _center(H[tr], H[te])
    E_tr, E_te = _center(E[tr], E[te])
    C_tr, C_te = _center(control_E[tr], control_E[te])
    Y_tr, Y_te = _center(Y[tr], Y[te])

    pred_h = _ridge_fit_predict(H_tr, Y_tr, H_te, lam)
    r2_h = _r2(Y_te, pred_h)

    HE_tr = np.concatenate([H_tr, E_tr], axis=1)
    HE_te = np.concatenate([H_te, E_te], axis=1)
    pred_he = _ridge_fit_predict(HE_tr, Y_tr, HE_te, lam)
    r2_he = _r2(Y_te, pred_he)

    HC_tr = np.concatenate([H_tr, C_tr], axis=1)
    HC_te = np.concatenate([H_te, C_te], axis=1)
    pred_hc = _ridge_fit_predict(HC_tr, Y_tr, HC_te, lam)
    r2_hc = _r2(Y_te, pred_hc)

    Ep_tr, Ep_te = _residualize(H_tr, H_te, E_tr, E_te, lam)
    HEp_tr = np.concatenate([H_tr, Ep_tr], axis=1)
    HEp_te = np.concatenate([H_te, Ep_te], axis=1)
    pred_hep = _ridge_fit_predict(HEp_tr, Y_tr, HEp_te, lam)
    r2_hep = _r2(Y_te, pred_hep)

    return {
        "n": n,
        "n_train": len(tr),
        "n_test": len(te),
        "r2_H": r2_h,
        "r2_HE_real": r2_he,
        "r2_HE_control": r2_hc,
        "r2_HE_perp": r2_hep,
        "delta_real": r2_he - r2_h,
        "delta_control": r2_hc - r2_h,
        "delta_perp": r2_hep - r2_h,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--qa-file", default="data/rare-kb-v1.json")
    parser.add_argument("--rows-dir", default="/home/zeng/qwen38-rows")
    parser.add_argument("--token-freq", default="data/wet-1m-tokens.npy")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.00019931793212890625)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="multiple seeds; overrides --seed")
    parser.add_argument("--lam", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="outputs/mechanism-rare-task-r2.json")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = _load_model(args.model, device=args.device)
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

    items = _load_qa(args.qa_file)
    if args.limit:
        items = items[: args.limit]
    token_freq = _load_token_freq(args.token_freq)
    qa_store = QAEtStore(args.rows_dir, args.scale)

    # Precompute token embedding table for Y.
    embed = model.get_input_embeddings().weight.detach().cpu().float().numpy()

    samples: list[dict] = []
    try:
        for idx, item in enumerate(items):
            question = str(item.get("question", ""))
            answer = str(item.get("answer", ""))
            qids = tokenizer.encode(question, add_special_tokens=False)
            full_text = f"{question} {answer}"
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            answer_start = len(qids)
            if len(full_ids) < 2:
                continue
            e_t = qa_store.fetch(full_ids)
            h = _hidden(model, full_ids, args.layer)

            # Only positions that predict an answer token (or following text).
            start = max(0, answer_start - 1)
            is_rare = bool(item.get("is_rare", False))
            for t in range(start, len(full_ids) - 1):
                nxt = int(full_ids[t + 1])
                samples.append(
                    {
                        "h": h[t],
                        "e": e_t[t],
                        "y": embed[nxt],
                        "next_token": nxt,
                        "is_rare": is_rare,
                        "answer_min_freq": int(item.get("answer_word_min_freq", 0)),
                        "next_freq": token_freq.get(nxt, 0),
                        "item_id": str(item.get("id", idx)),
                        "task": str(item.get("task", "qa")),
                    }
                )
            if (idx + 1) % 25 == 0:
                print(
                    f"  [{idx+1}/{len(items)}] collected {len(samples)} samples",
                    flush=True,
                )
    finally:
        qa_store.close()

    print(f"collected {len(samples)} samples in {time.time()-t0:.1f}s", flush=True)
    if not samples:
        print("no samples collected", flush=True)
        return 1

    H = np.stack([s["h"] for s in samples]).astype(np.float32)
    E = np.stack([s["e"] for s in samples]).astype(np.float32)
    Y = np.stack([s["y"] for s in samples]).astype(np.float32)
    rare_flags = np.asarray([s["is_rare"] for s in samples], dtype=bool)

    # Control: shuffle E within each split later; here we just create a
    # deterministic per-sample permutation for the whole set.  This destroys
    # the E->Y mapping while preserving marginal E distribution.
    rng = np.random.default_rng(args.seed + 999)
    perm = rng.permutation(len(E))
    C = E[perm].copy()

    # Subsets.  Also create two numeric-split views by next-token frequency.
    rare_idx = np.flatnonzero(rare_flags)
    common_idx = np.flatnonzero(~rare_flags)
    freq = np.asarray([s["next_freq"] for s in samples], dtype=np.int64)
    rare_by_token_idx = np.flatnonzero((freq > 0) & (freq <= 3))
    common_by_token_idx = np.flatnonzero(freq > 30)

    seeds = args.seeds if args.seeds is not None else [args.seed]
    subsets = [
        ("rare_item", rare_idx),
        ("common_item", common_idx),
        ("rare_token", rare_by_token_idx),
        ("common_token", common_by_token_idx),
    ]
    result: dict = {
        "config": vars(args),
        "n_samples": len(samples),
        "n_rare_items": int(rare_flags.sum()),
        "n_common_items": int((~rare_flags).sum()),
        "metrics": {},
        "metrics_by_seed": {},
    }
    for name, idx in subsets:
        if len(idx) < 8:
            result["metrics"][name] = {"n": len(idx), "skipped": True}
            continue
        per_seed = []
        for seed in seeds:
            m = _subset_metrics(
                H[idx], E[idx], Y[idx], C[idx],
                test_frac=args.test_frac,
                seed=seed,
                lam=args.lam,
            )
            per_seed.append(m)
        result["metrics_by_seed"][name] = per_seed
        keys = ["r2_H", "r2_HE_real", "r2_HE_control", "r2_HE_perp",
                "delta_real", "delta_control", "delta_perp"]
        agg: dict = {"n": len(idx), "n_seeds": len(per_seed)}
        for key in keys:
            vals = [m[key] for m in per_seed if np.isfinite(m[key])]
            if vals:
                agg[key + "_mean"] = float(np.mean(vals))
                agg[key + "_std"] = float(np.std(vals))
        result["metrics"][name] = agg

    result["runtime_seconds"] = time.time() - t0
    result["sample_counts"] = {
        "rare_item": len(rare_idx),
        "common_item": len(common_idx),
        "rare_token": len(rare_by_token_idx),
        "common_token": len(common_by_token_idx),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
