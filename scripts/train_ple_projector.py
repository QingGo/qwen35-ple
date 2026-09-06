#!/usr/bin/env python3
"""Train the Phase A PLE Projector.

The projector maps a frozen backbone hidden state plus lexical memory features
to a (scale, bias) correction applied to the PLE n-gram log-prior:

    fused = base_logits + scale * log p_memory + bias

Only the small MLP is trained, with next-token cross-entropy.
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
import torch.nn.functional as F

from qwen35_ple.addressable_memory import AddressableNgramMemory
from qwen35_ple.fusion import calibrate_ngram_fusion, fuse_ngram_logits, softmax
from qwen35_ple.projector import (
    PleProjector,
    add_task_features,
    compute_memory_features,
)


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


def _load_dataset(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(obj)
    return rows


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


def _build_memory(train_texts: list[str], tokenizer, *, max_order: int):
    mem = AddressableNgramMemory(min_order=2, max_order=max_order)
    for value_id, text in enumerate(train_texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            mem.add_document(ids, value_id=value_id)
    return mem


def _forward(model, context: list[int], device: str):
    ids = torch.tensor([context], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
        logits = out.logits[0, -1].float().cpu().numpy()
        hidden = out.hidden_states[-1][0, -1].float().cpu().numpy()
    return logits, hidden


def _make_samples(
    model,
    tokenizer,
    *,
    code_texts: list[str],
    wiki_texts: list[str],
    seed: int,
    max_order: int,
    context_len: int,
    max_per_doc_code: int,
    max_per_doc_wiki: int,
    max_samples: int,
    train_frac: float,
) -> tuple[list[dict], list[dict]]:
    code_train, code_eval = _split_texts(code_texts, train_frac, seed)
    wiki_train, wiki_eval = _split_texts(wiki_texts, train_frac, seed)
    code_mem = _build_memory(code_train, tokenizer, max_order=max_order)
    wiki_mem = _build_memory(wiki_train, tokenizer, max_order=max_order)

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
        ("code", code_eval_seqs, code_mem, max_per_doc_code, None),
        ("name", wiki_eval_seqs, wiki_mem, max_per_doc_wiki, "name"),
        ("number", wiki_eval_seqs, wiki_mem, max_per_doc_wiki, "number"),
    ]:
        sampled = _sample_positions_by_category(
            tokenizer,
            seqs,
            context_len=context_len,
            max_per_doc=max_per_doc,
            seed=seed,
            category=category,
        )
        for seq, i in sampled:
            context = seq[max(0, i - context_len) : i]
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

    rng = random.Random(seed)
    rng.shuffle(positions)
    n_train = int(len(positions) * train_frac)
    train = positions[: min(max_samples, n_train)]
    eval_ = positions[n_train : n_train + max_samples]
    return train, eval_


def _features_from_sample(logits: np.ndarray, sample: dict) -> dict[str, float]:
    features = compute_memory_features(
        logits, sample["dist"], matched_order=sample.get("order")
    )
    return add_task_features(features, sample.get("task"))


def _eval_condition(
    model,
    samples: list[dict],
    device: str,
    *,
    projector: PleProjector | None = None,
    fixed_scale: float | None = None,
    fixed_bias: float | None = None,
    temperature: float = 1.0,
) -> dict:
    nll = 0.0
    hit = 0.0
    per_task: dict[str, dict[str, float]] = {}
    for sample in samples:
        logits, hidden = _forward(model, sample["context"], device)
        dist = sample["dist"]
        target = sample["target"]
        if projector is not None:
            features = _features_from_sample(logits, sample)
            scale, bias = projector.predict_np(hidden, features)
        else:
            scale = fixed_scale if fixed_scale is not None else 0.0
            bias = fixed_bias if fixed_bias is not None else 0.0
        fused = fuse_ngram_logits(
            logits, dist, scale=scale, bias=bias, temperature=temperature
        )
        p = softmax(fused)
        nll += -math.log(max(float(p[target]), 1e-12))
        hit += 1.0 if int(np.argmax(fused)) == target else 0.0
        key = sample["task"]
        b = per_task.setdefault(key, {"n": 0.0, "nll": 0.0, "hit": 0.0})
        b["n"] += 1
        b["nll"] += -math.log(max(float(p[target]), 1e-12))
        b["hit"] += 1.0 if int(np.argmax(fused)) == target else 0.0
    n = max(1, len(samples))
    return {
        "n": len(samples),
        "nll": nll / n,
        "hit": hit / n,
        "per_task": {
            k: {
                "n": int(v["n"]),
                "nll": v["nll"] / v["n"],
                "hit": v["hit"] / v["n"],
            }
            for k, v in per_task.items()
        },
    }


def _eval_paired(
    model,
    samples: list[dict],
    device: str,
    *,
    projector: PleProjector,
    fixed_scale: float,
    fixed_bias: float,
    temperature: float = 1.0,
) -> dict:
    """Evaluate base/fixed/projector on the same samples and return paired rows."""
    rows = []
    for sample in samples:
        logits, hidden = _forward(model, sample["context"], device)
        dist = sample["dist"]
        target = sample["target"]
        base_p = softmax(logits)
        base_nll = -math.log(max(float(base_p[target]), 1e-12))
        base_hit = 1.0 if int(np.argmax(logits)) == target else 0.0

        fixed = fuse_ngram_logits(
            logits, dist, scale=fixed_scale, bias=fixed_bias, temperature=temperature
        )
        fixed_p = softmax(fixed)
        fixed_nll = -math.log(max(float(fixed_p[target]), 1e-12))
        fixed_hit = 1.0 if int(np.argmax(fixed)) == target else 0.0

        features = _features_from_sample(logits, sample)
        proj_scale, proj_bias = projector.predict_np(hidden, features)
        proj = fuse_ngram_logits(
            logits, dist, scale=proj_scale, bias=proj_bias, temperature=temperature
        )
        proj_p = softmax(proj)
        proj_nll = -math.log(max(float(proj_p[target]), 1e-12))
        proj_hit = 1.0 if int(np.argmax(proj)) == target else 0.0

        rows.append({
            "task": sample["task"],
            "target": target,
            "base_nll": base_nll,
            "base_hit": base_hit,
            "fixed_nll": fixed_nll,
            "fixed_hit": fixed_hit,
            "fixed_scale": fixed_scale,
            "fixed_bias": fixed_bias,
            "proj_nll": proj_nll,
            "proj_hit": proj_hit,
            "proj_scale": proj_scale,
            "proj_bias": proj_bias,
        })

    def _mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    def _mean_bool(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    per_task: dict[str, dict[str, float]] = {}
    for r in rows:
        t = r["task"]
        b = per_task.setdefault(t, {"n": 0.0, "base_nll": 0.0, "fixed_nll": 0.0, "proj_nll": 0.0, "base_hit": 0.0, "fixed_hit": 0.0, "proj_hit": 0.0})
        b["n"] += 1
        b["base_nll"] += r["base_nll"]
        b["fixed_nll"] += r["fixed_nll"]
        b["proj_nll"] += r["proj_nll"]
        b["base_hit"] += r["base_hit"]
        b["fixed_hit"] += r["fixed_hit"]
        b["proj_hit"] += r["proj_hit"]

    return {
        "n": len(rows),
        "base_nll": _mean("base_nll"),
        "fixed_nll": _mean("fixed_nll"),
        "proj_nll": _mean("proj_nll"),
        "base_hit": _mean_bool("base_hit"),
        "fixed_hit": _mean_bool("fixed_hit"),
        "proj_hit": _mean_bool("proj_hit"),
        "delta_fixed_vs_base_nll": _mean("base_nll") - _mean("fixed_nll"),
        "delta_proj_vs_fixed_nll": _mean("fixed_nll") - _mean("proj_nll"),
        "delta_proj_vs_base_nll": _mean("base_nll") - _mean("proj_nll"),
        "per_task": {
            k: {
                "n": int(v["n"]),
                "base_nll": v["base_nll"] / v["n"],
                "fixed_nll": v["fixed_nll"] / v["n"],
                "proj_nll": v["proj_nll"] / v["n"],
                "base_hit": v["base_hit"] / v["n"],
                "fixed_hit": v["fixed_hit"] / v["n"],
                "proj_hit": v["proj_hit"] / v["n"],
            }
            for k, v in per_task.items()
        },
        "rows": rows,
    }


def _calibrate_fixed(
    model, train_samples: list[dict], device: str, *, temperature: float = 1.0
) -> dict:
    base_logits_list = []
    target_ids = []
    dist_list = []
    for sample in train_samples[:40]:
        logits, _ = _forward(model, sample["context"], device)
        base_logits_list.append(logits)
        target_ids.append(sample["target"])
        dist_list.append(sample["dist"])
    if not base_logits_list:
        return {"best_scale": 0.0, "best_bias": 0.0}
    return calibrate_ngram_fusion(
        base_logits_list,
        target_ids,
        dist_list,
        scale_grid=np.linspace(-2.0, 4.0, 9),
        bias_grid=np.linspace(-4.0, 3.0, 9),
    )


def _train_projector(
    model,
    train_samples: list[dict],
    eval_samples: list[dict],
    device: str,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    num_layers: int,
    seed: int,
    temperature: float,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    projector = PleProjector(
        model.config.hidden_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        zero_init=True,
    ).to(device)

    feature_rows = []
    for sample in train_samples[:100]:
        logits, _ = _forward(model, sample["context"], device)
        feature_rows.append(_features_from_sample(logits, sample))
    if feature_rows:
        arr = np.asarray(
            [[float(f.get(n, 0.0)) for n in projector.feature_names] for f in feature_rows],
            dtype=np.float32,
        )
        projector.set_feature_stats(arr.mean(axis=0), arr.std(axis=0))
    else:
        projector.set_feature_stats(
            np.zeros(projector.num_features), np.ones(projector.num_features)
        )

    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)
    history = []
    t0 = time.time()
    for step in range(steps):
        batch = rng.sample(train_samples, min(batch_size, len(train_samples)))
        optimizer.zero_grad()
        total_loss = 0.0
        for sample in batch:
            logits, hidden = _forward(model, sample["context"], device)
            features = _features_from_sample(logits, sample)
            hidden_t = torch.as_tensor(hidden, dtype=torch.float32, device=device).unsqueeze(0)
            feat_t = torch.as_tensor(
                [[float(features.get(n, 0.0)) for n in projector.feature_names]],
                dtype=torch.float32,
                device=device,
            )
            scale, bias = projector(hidden_t, feat_t)
            base_t = torch.as_tensor(logits, dtype=torch.float32, device=device)
            logp_t = torch.zeros_like(base_t)
            support_t = torch.zeros_like(base_t)
            for tok, prob in sample["dist"].items():
                if 0 <= tok < base_t.numel() and prob > 0:
                    logp_t[tok] = math.log(float(prob)) / temperature
                    support_t[tok] = 1.0
            fused = base_t + scale[0] * logp_t + bias[0] * support_t
            target = torch.tensor([sample["target"]], dtype=torch.long, device=device)
            loss = F.cross_entropy(fused.unsqueeze(0), target) / len(batch)
            loss.backward()
            total_loss += float(loss) * len(batch)
        optimizer.step()
        history.append(total_loss / len(batch))
        if (step + 1) % max(1, steps // 10) == 0:
            print(
                f"[projector] step={step + 1}/{steps} loss={history[-1]:.4f} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    return {
        "projector": projector,
        "projector_dict": projector.to_dict(),
        "train": _eval_condition(
            model, train_samples[:40], device, projector=projector, temperature=temperature
        ),
        "loss_history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--code-root", default=".")
    parser.add_argument("--code-corpus", default=None, help="optional JSONL code corpus")
    parser.add_argument("--dataset", default=None, help="optional prebuilt projector dataset JSONL")
    parser.add_argument("--wiki-path", default="data/sources/wikitext.jsonl")
    parser.add_argument("--max-code-files", type=int, default=200)
    parser.add_argument("--max-wiki-docs", type=int, default=400)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--max-per-doc-code", type=int, default=6)
    parser.add_argument("--max-per-doc-wiki", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=120)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/ple-projector-v0.json")
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
    if args.code_corpus:
        code_texts.extend(_load_code_corpus(args.code_corpus))
    wiki_texts = _load_wiki_docs(args.wiki_path, args.max_wiki_docs)
    print(f"[ple-projector] code={len(code_texts)} wiki={len(wiki_texts)}", flush=True)

    if args.dataset:
        rows = _load_dataset(args.dataset)
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        n_train = int(len(rows) * args.train_frac)
        train_samples = rows[:n_train]
        eval_samples = rows[n_train:]
        print(
            f"[ple-projector] dataset={args.dataset} rows={len(rows)} "
            f"train={len(train_samples)} eval={len(eval_samples)}",
            flush=True,
        )
    else:
        train_samples, eval_samples = _make_samples(
        model,
        tokenizer,
        code_texts=code_texts,
        wiki_texts=wiki_texts,
        seed=args.seed,
        max_order=args.max_order,
        context_len=args.context_len,
        max_per_doc_code=args.max_per_doc_code,
        max_per_doc_wiki=args.max_per_doc_wiki,
        max_samples=args.max_samples,
        train_frac=args.train_frac,
    )
    print(
        f"[ple-projector] samples train={len(train_samples)} eval={len(eval_samples)}",
        flush=True,
    )

    calib = _calibrate_fixed(
        model, train_samples, args.device, temperature=args.temperature
    )
    print(
        f"[ple-projector] fixed calib scale={calib.get('best_scale')} "
        f"bias={calib.get('best_bias')}",
        flush=True,
    )

    trained = _train_projector(
        model,
        train_samples,
        eval_samples,
        args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seed=args.seed,
        temperature=args.temperature,
    )
    paired_eval = _eval_paired(
        model,
        eval_samples,
        args.device,
        projector=trained["projector"],
        fixed_scale=float(calib.get("best_scale", 0.0)),
        fixed_bias=float(calib.get("best_bias", 0.0)),
        temperature=args.temperature,
    )
    base_eval = {
        "n": paired_eval["n"],
        "nll": paired_eval["base_nll"],
        "hit": paired_eval["base_hit"],
    }
    fixed_eval = {
        "n": paired_eval["n"],
        "nll": paired_eval["fixed_nll"],
        "hit": paired_eval["fixed_hit"],
    }
    projector_eval = {
        "n": paired_eval["n"],
        "nll": paired_eval["proj_nll"],
        "hit": paired_eval["proj_hit"],
    }
    result = {
        "schema": "ple-projector-v1-experiment",
        "config": vars(args),
        "fixed_calibration": calib,
        "base_eval": base_eval,
        "fixed_eval": fixed_eval,
        "projector_eval": projector_eval,
        "paired_eval": paired_eval,
        "projector_train": trained["train"],
        "loss_history": trained["loss_history"],
        "runtime_seconds": time.time() - t0,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    from qwen35_ple.projector import save_projector

    projector_path = out.parent / f"{out.stem}.projector.json"
    save_projector(projector_path, trained["projector_dict"])
    print(f"[ple-projector] wrote {out} and {projector_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
