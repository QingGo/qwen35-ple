#!/usr/bin/env python3
"""PLE-1: n-gram lexical memory real vs control.

Builds two training-free n-gram LMs from the same training sequences:
  * real    : original token order
  * control : token order shuffled inside each sequence

Both share the same unigram distribution, so a real-vs-control gap isolates
ordered n-gram structure rather than marginal token frequency.

Optional --base-sample loads the Qwen3.5-0.8B base model and performs
log-linear n-gram fusion on a small CPU sample, estimating the best lambda and
the empirical NLL reduction in bits.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qwen35_ple.ngram_lm import NgramLM

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


def _load_model(model_dir: str):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


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
    texts = []
    for p in files[:limit]:
        try:
            texts.append(p.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return texts


def tokenize_docs(docs, tokenizer) -> list[list[int]]:
    return [tokenizer.encode(d, add_special_tokens=False) for d in docs]


def split_sequences(seqs, train_frac: float, seed: int):
    rng = random.Random(seed)
    indices = list(range(len(seqs)))
    rng.shuffle(indices)
    n_train = max(1, round(len(seqs) * train_frac))
    train = [seqs[i] for i in indices[:n_train]]
    eval_ = [seqs[i] for i in indices[n_train:]]
    return train, eval_


def build_ngram(seqs, *, max_order: int, shuffle: bool, seed: int) -> NgramLM:
    lm = NgramLM(max_order=max_order)
    rng = random.Random(seed) if shuffle else None
    for seq in seqs:
        if shuffle:
            seq = list(seq)
            rng.shuffle(seq)
        lm.add_sequence(seq)
    return lm


def classify_token(token_id: int, tokenizer) -> set[str]:
    text = tokenizer.decode([token_id], skip_special_tokens=True)
    cats: set[str] = set()
    if any(ch.isdigit() for ch in text):
        cats.add("number")
    stripped = text.strip()
    if stripped and stripped[0].isupper() and stripped.isalpha():
        cats.add("name")
    return cats


@dataclass
class EvalStats:
    n: int = 0
    real_lp: list[float] | None = None
    control_lp: list[float] | None = None
    real_top1: list[int] | None = None
    control_top1: list[int] | None = None
    real_hit: list[int] | None = None
    control_hit: list[int] | None = None

    def __post_init__(self) -> None:
        if self.real_lp is None:
            self.real_lp = []
            self.control_lp = []
            self.real_top1 = []
            self.control_top1 = []
            self.real_hit = []
            self.control_hit = []

    def add(self, real_lp, control_lp, real_top1, control_top1, real_hit, control_hit) -> None:
        self.real_lp.append(float(real_lp))
        self.control_lp.append(float(control_lp))
        self.real_top1.append(int(real_top1))
        self.control_top1.append(int(control_top1))
        self.real_hit.append(int(real_hit))
        self.control_hit.append(int(control_hit))
        self.n += 1

    def summary(self) -> dict:
        if not self.n:
            return {"n": 0}
        real = np.asarray(self.real_lp, dtype=np.float64)
        control = np.asarray(self.control_lp, dtype=np.float64)
        diff = real - control
        real_top = np.asarray(self.real_top1, dtype=np.float64)
        control_top = np.asarray(self.control_top1, dtype=np.float64)
        tstat = 0.0
        pvalue = 1.0
        if self.n > 1:
            sd = float(np.std(diff, ddof=1))
            if sd > 0:
                tstat = float(np.mean(diff) / (sd / math.sqrt(self.n)))
                pvalue = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(tstat) / math.sqrt(2.0))))
        return {
            "n": self.n,
            "real_mean_logprob": float(np.mean(real)),
            "control_mean_logprob": float(np.mean(control)),
            "delta_logprob": float(np.mean(diff)),
            "delta_logprob_median": float(np.median(diff)),
            "delta_logprob_std": float(np.std(diff, ddof=1)) if self.n > 1 else 0.0,
            "paired_t": tstat,
            "paired_p": pvalue,
            "real_top1": float(np.mean(real_top)),
            "control_top1": float(np.mean(control_top)),
            "delta_top1": float(np.mean(real_top) - np.mean(control_top)),
            "real_dist_hit": float(np.mean(self.real_hit)),
            "control_dist_hit": float(np.mean(self.control_hit)),
            "real_perplexity": float(np.exp(-np.mean(real))),
            "control_perplexity": float(np.exp(-np.mean(control))),
        }


def evaluate_positions(real_lm, control_lm, seqs, tokenizer, *, max_positions, max_order, seed):
    positions = []
    for si, seq in enumerate(seqs):
        for i in range(1, len(seq)):
            positions.append((si, i))
    rng = random.Random(seed)
    if max_positions and len(positions) > max_positions:
        positions = rng.sample(positions, max_positions)

    all_stats = EvalStats()
    by_cat: dict[str, EvalStats] = {}
    rows = []

    for si, i in positions:
        seq = seqs[si]
        ctx = seq[:i]
        y = seq[i]
        dist_r = real_lm.distribution(ctx)
        dist_c = control_lm.distribution(ctx)
        lp_r = real_lm.logprob(y, ctx)
        lp_c = control_lm.logprob(y, ctx)
        top_r = max(dist_r.items(), key=lambda x: x[1])[0] if dist_r else None
        top_c = max(dist_c.items(), key=lambda x: x[1])[0] if dist_c else None
        hit_r = top_r == y
        hit_c = top_c == y
        all_stats.add(lp_r, lp_c, hit_r, hit_c, dist_r is not None, dist_c is not None)
        cats = classify_token(y, tokenizer) or {"other"}
        for cat in cats:
            st = by_cat.setdefault(cat, EvalStats())
            st.add(lp_r, lp_c, hit_r, hit_c, dist_r is not None, dist_c is not None)
        rows.append({
            "seq": si,
            "pos": i,
            "target": int(y),
            "categories": sorted(cats),
            "real_lp": lp_r,
            "control_lp": lp_c,
            "real_top1": hit_r,
            "control_top1": hit_c,
        })

    summary = {"all": all_stats.summary()}
    for cat in sorted(by_cat):
        summary[cat] = by_cat[cat].summary()
    return summary, rows


def softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max()
    e = np.exp(logits - m)
    return e / e.sum()


def optimize_fusion(base_logits_list, target_ids, dist_list, *, lam_grid=None):
    if not base_logits_list:
        return {}
    if lam_grid is None:
        lam_grid = np.linspace(-2.0, 6.0, 161)
    best_lam = float(lam_grid[0])
    best_loss = float("inf")
    losses = []
    for lam in lam_grid:
        total = 0.0
        for logits, target, dist in zip(base_logits_list, target_ids, dist_list):
            fused = logits.copy()
            if dist:
                for tok, p in dist.items():
                    if 0 <= tok < len(fused) and p > 0:
                        fused[tok] += float(lam) * math.log(p)
            probs = softmax(fused)
            total -= math.log(max(float(probs[target]), 1e-12))
        avg = float(total / len(base_logits_list))
        if avg < best_loss:
            best_loss = avg
            best_lam = float(lam)
        losses.append(avg)
    return {
        "best_lambda": best_lam,
        "best_mean_nll": best_loss,
        "loss_curve": losses,
        "lambda_grid": lam_grid.tolist(),
    }


def run_base_fusion(model, tokenizer, seqs, real_lm, control_lm, *, sample_size, seed, context_len=128):
    import torch

    positions = []
    for si, seq in enumerate(seqs):
        for i in range(1, len(seq)):
            positions.append((si, i))
    rng = random.Random(seed)
    if len(positions) > sample_size:
        positions = rng.sample(positions, sample_size)

    base_logits = []
    target_ids = []
    real_dists = []
    control_dists = []
    base_lps = []
    real_lps = []
    control_lps = []
    real_top1s = []
    control_top1s = []
    base_top1s = []
    ctx_lens = []

    for si, i in positions:
        seq = seqs[si]
        # N-gram only uses the last few tokens; for the base model we feed a
        # bounded local context to keep CPU inference tractable.
        ngram_ctx = seq[:i]
        ctx = ngram_ctx[-context_len:]
        y = seq[i]
        ctx_lens.append(len(ctx))
        ids = torch.tensor([ctx], dtype=torch.long)
        with torch.no_grad():
            out = model(ids)
            logits = out.logits[0, -1].float().cpu().numpy()
        probs = softmax(logits)
        base_lps.append(float(math.log(max(probs[y], 1e-12))))
        base_top1s.append(bool(int(np.argmax(logits)) == y))
        dr = real_lm.distribution(ngram_ctx)
        dc = control_lm.distribution(ngram_ctx)
        real_dists.append(dr)
        control_dists.append(dc)
        real_lps.append(real_lm.logprob(y, ngram_ctx))
        control_lps.append(control_lm.logprob(y, ngram_ctx))
        real_top1s.append(bool(max(dr.items(), key=lambda x: x[1])[0] == y) if dr else False)
        control_top1s.append(bool(max(dc.items(), key=lambda x: x[1])[0] == y) if dc else False)
        base_logits.append(logits)
        target_ids.append(y)

    real_fusion = optimize_fusion(base_logits, target_ids, real_dists)
    control_fusion = optimize_fusion(base_logits, target_ids, control_dists)
    base_nll = float(-np.mean(
        [math.log(max(softmax(logits)[t], 1e-12)) for logits, t in zip(base_logits, target_ids)]
    ))

    return {
        "n": len(positions),
        "base_mean_logprob": float(np.mean(base_lps)),
        "base_top1": float(np.mean([int(x) for x in base_top1s])),
        "base_nll": base_nll,
        "real_ngram_mean_logprob": float(np.mean(real_lps)),
        "control_ngram_mean_logprob": float(np.mean(control_lps)),
        "real_ngram_top1": float(np.mean([int(x) for x in real_top1s])),
        "control_ngram_top1": float(np.mean([int(x) for x in control_top1s])),
        "base_plus_real": real_fusion,
        "base_plus_control": control_fusion,
        "base_plus_real_delta_nll_nats": base_nll - real_fusion["best_mean_nll"],
        "base_plus_control_delta_nll_nats": base_nll - control_fusion["best_mean_nll"],
        "base_plus_real_delta_nll_bits": (base_nll - real_fusion["best_mean_nll"]) / math.log(2.0),
        "base_plus_control_delta_nll_bits": (base_nll - control_fusion["best_mean_nll"]) / math.log(2.0),
        "real_over_control_delta_bits": (control_fusion["best_mean_nll"] - real_fusion["best_mean_nll"]) / math.log(2.0),
    }


def write_report(results: dict, report_path: Path) -> None:
    lines = [
        "# PLE-1：N-gram 词法记忆 real vs control 结果",
        "",
        f"- Seed: {results.get('seed')}",
        f"- Max n-gram order: {results.get('max_order')}",
        f"- Max eval positions per domain: {results.get('max_position_per_domain')}",
        "",
        "## 总览",
        "",
        "| Domain | N | real logprob | control logprob | Δ logprob | real top1 | control top1 | Δ top1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, d in results.get("domains", {}).items():
        s = d["summary"]["all"]
        lines.append(
            f"| {name} | {s['n']} | {s['real_mean_logprob']:.3f} | {s['control_mean_logprob']:.3f} "
            f"| {s['delta_logprob']:.3f} | {s['real_top1']:.3f} | {s['control_top1']:.3f} | {s['delta_top1']:.3f} |"
        )
    lines += ["", "## 分类（target token 类别）", ""]
    for name, d in results.get("domains", {}).items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Category | N | Δ logprob | real top1 | control top1 | paired p |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for cat, s in d["summary"].items():
            if cat == "all":
                continue
            lines.append(
                f"| {cat} | {s['n']} | {s['delta_logprob']:.3f} | {s['real_top1']:.3f} "
                f"| {s['control_top1']:.3f} | {s['paired_p']:.4f} |"
            )
        lines.append("")
    if any("base_fusion" in d for d in results.get("domains", {}).values()):
        lines += ["", "## Base model fusion", ""]
        for name, d in results.get("domains", {}).items():
            bf = d.get("base_fusion")
            if not bf:
                continue
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"- n={bf['n']}")
            lines.append(f"- base mean logprob: {bf['base_mean_logprob']:.4f}")
            lines.append(f"- real ngram mean logprob: {bf['real_ngram_mean_logprob']:.4f}")
            lines.append(f"- control ngram mean logprob: {bf['control_ngram_mean_logprob']:.4f}")
            lines.append(f"- base NLL: {bf['base_nll']:.4f}")
            lines.append(f"- base+real best λ: {bf['base_plus_real'].get('best_lambda','N/A')}")
            lines.append(f"- base+real NLL: {bf['base_plus_real'].get('best_mean_nll','N/A')}")
            lines.append(f"- base+control best λ: {bf['base_plus_control'].get('best_lambda','N/A')}")
            lines.append(f"- base+control NLL: {bf['base_plus_control'].get('best_mean_nll','N/A')}")
            lines.append(f"- Δ NLL bits (base→real): {bf['base_plus_real_delta_nll_bits']:.4f}")
            lines.append(f"- Δ NLL bits (base→control): {bf['base_plus_control_delta_nll_bits']:.4f}")
            lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/Users/zeng/code/LLM-CompileForge/models/Qwen/Qwen3.5-0.8B.local-backup")
    parser.add_argument("--wiki-limit", type=int, default=300)
    parser.add_argument("--code-limit", type=int, default=120)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--max-position", type=int, default=4000)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-sample", type=int, default=0)
    parser.add_argument("--base-context-len", type=int, default=128)
    parser.add_argument("--base-domains", default="wiki,code", help="comma-separated domains to run base fusion on")
    parser.add_argument("--output", default="outputs/ple1-ngram-eval.json")
    parser.add_argument("--report", default="outputs/ple1-ngram-eval-report.md")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer = _load_tokenizer(args.model_dir)

    wiki_seqs = tokenize_docs(load_wiki_docs(args.wiki_limit), tokenizer)
    wiki_train, wiki_eval = split_sequences(wiki_seqs, args.train_frac, args.seed)
    print(f"[ple1] wiki docs={len(wiki_seqs)} train_tokens={sum(len(s) for s in wiki_train)} eval_tokens={sum(len(s) for s in wiki_eval)}", flush=True)

    code_seqs = tokenize_docs(load_code_docs(args.code_limit), tokenizer)
    code_train, code_eval = split_sequences(code_seqs, args.train_frac, args.seed)
    print(f"[ple1] code files={len(code_seqs)} train_tokens={sum(len(s) for s in code_train)} eval_tokens={sum(len(s) for s in code_eval)}", flush=True)

    results = {
        "schema": "ple1-ngram-eval-v1",
        "seed": args.seed,
        "max_order": args.max_order,
        "max_position_per_domain": args.max_position,
        "wiki_train_tokens": sum(len(s) for s in wiki_train),
        "wiki_eval_tokens": sum(len(s) for s in wiki_eval),
        "code_train_tokens": sum(len(s) for s in code_train),
        "code_eval_tokens": sum(len(s) for s in code_eval),
        "domains": {},
    }

    base_model = None
    base_domains = set()
    if args.base_sample > 0:
        print("[ple1] loading base model for fusion sample", flush=True)
        base_model = _load_model(args.model_dir)
        base_domains = {x.strip() for x in args.base_domains.split(",") if x.strip()}

    for name, train, eval_ in [("wiki", wiki_train, wiki_eval), ("code", code_train, code_eval)]:
        t1 = time.time()
        real_lm = build_ngram(train, max_order=args.max_order, shuffle=False, seed=args.seed)
        control_lm = build_ngram(train, max_order=args.max_order, shuffle=True, seed=args.seed)
        print(f"[ple1] {name}: built real/control in {time.time()-t1:.1f}s", flush=True)
        summary, _ = evaluate_positions(
            real_lm, control_lm, eval_, tokenizer,
            max_positions=args.max_position, max_order=args.max_order, seed=args.seed,
        )
        domain = {
            "train_sequences": len(train),
            "eval_sequences": len(eval_),
            "summary": summary,
        }
        if base_model is not None and name in base_domains:
            domain["base_fusion"] = run_base_fusion(
                base_model, tokenizer, eval_, real_lm, control_lm,
                sample_size=args.base_sample, seed=args.seed, context_len=args.base_context_len,
            )
        results["domains"][name] = domain
        print(f"[ple1] {name}: {json.dumps(summary.get('all', {}), ensure_ascii=False)}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    write_report(results, rep)
    print(f"[ple1] wrote {out} and {rep} in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
