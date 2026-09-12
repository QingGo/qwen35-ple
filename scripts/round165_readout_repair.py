#!/usr/bin/env python3
"""Round 165: if we replace the read-out with a non-collapsing one, does the
output start responding to table content?

Pre-registration: ``docs/round-165-readout-repair-preregistration.md``.
The variants, the norm-matching rule, the endpoints and the verdict labels were
frozen there before this file was run; do not tune them to a result.

Round 164 established that the injected vector is nearly a constant function of
its input: ``PR(c) = 1.0014`` and ``cos(c_real, c_shuf) = 0.9966``.  The paper
says in §4.7 that this leaves two explanations unseparated -- the window has
nothing to give beyond ``h_t``, or the read-out fails to transmit what the
window has.  Only the first is unfixable, so separating them changes what the
null result is evidence for.

This script keeps the backbone, layer, prompts, table, checkpoint and PLE scale
fixed and changes *only* the map ``e_t -> c_t``.  Every variant is norm-matched
per item to production's ``||c_t||``, i.e. same injection budget, different
read-out.

Because attention is causal, injecting at the answer position affects only that
position and later ones, so the per-position accounting is exact.

Usage::

    PYTHONPATH=src python scripts/round165_readout_repair.py \
      --model /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B \
      --rows-dir /root/autodl-tmp/qwen35-ple/qwen38-rows \
      --reader /root/autodl-tmp/qwen35-ple/outputs/round162-0.8B-nosft600/reader-wiki-seed0.pt \
      --qa-file data/qa-standard/eval-600b.jsonl \
      --output outputs/round165/readout-repair.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

PROMPT = "Question: {question}\nAnswer:"
BOOLQ_PROMPT = "Question: {question}\nAnswer with one word, Yes or No:"

# Frozen in the pre-registration.  ``oracle_ungated`` was called
# ``oracle_linear`` there; the loaded reader's ``out_proj`` turned out to be a
# three-layer MLP rather than a single Linear, so the variant applies that
# module as-is.  Renamed before any result existed, to keep the label honest.
VARIANTS = (
    "production",
    "centered",
    "random_proj",
    "oracle_ungated",
    "positive_control",
)
NON_COLLAPSING = ("centered", "random_proj", "oracle_ungated")
CONDITIONS = ("real", "shuf")
N_BOOTSTRAP = 2000


# --------------------------------------------------------------------------
# statistics (pure numpy, unit-tested without a GPU)
# --------------------------------------------------------------------------


def participation_ratio(mat: np.ndarray) -> float:
    """``(sum lambda)^2 / sum lambda^2`` over covariance eigenvalues.

    ``mat`` is [n, d]; centered internally.
    """
    x = np.asarray(mat, dtype=np.float64)
    if x.shape[0] < 2:
        raise ValueError("participation ratio needs at least two rows")
    x = x - x.mean(axis=0, keepdims=True)
    # Eigenvalues of the [n, n] Gram matrix are the nonzero ones of the
    # covariance up to a constant factor, which cancels in the ratio.
    gram = x @ x.T
    lam = np.linalg.eigvalsh(gram)
    lam = np.clip(lam, 0.0, None)
    denom = float(np.sum(lam**2))
    if denom <= 0:
        raise ValueError("zero variance; the read-out is a constant")
    return float(np.sum(lam) ** 2 / denom)


def bootstrap_ci(values: np.ndarray, n: int = N_BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean."""
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.shape[0], size=(n, x.shape[0]))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation between two categorical distributions."""
    return float(0.5 * np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def mean_pairwise_cosine(mat: np.ndarray) -> float:
    sub = np.asarray(mat, dtype=np.float64)
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-norm vector; the hook is not recording")
    unit = sub / norms
    n = unit.shape[0]
    if n < 2:
        raise ValueError("need at least two rows")
    gram = unit @ unit.T
    iu = np.triu_indices(n, k=1)
    return float(gram[iu].mean())


def demean_ci(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    return bootstrap_ci(values, seed=seed)


# --------------------------------------------------------------------------
# read-out variants
# --------------------------------------------------------------------------


def build_variant_vectors(
    variant: str,
    *,
    c_prod: np.ndarray,
    c_bar: np.ndarray,
    value_proj_e: np.ndarray | None,
    oracle_out: np.ndarray | None,
    rand_w: np.ndarray | None,
    gold_emb: np.ndarray | None,
    target_norm: float,
    pos: int = -1,
    match: bool = True,
) -> np.ndarray:
    """Return the [T, d] vector this variant injects, norm-matched at ``pos``.

    ``c_prod`` is production's contribution for the whole sequence, already
    computed with the reader's own causal convolution.  ``oracle_out`` is the
    ungated read-out ``out_proj(value_proj(e_t))``, precomputed by the caller
    because ``out_proj`` may be an MLP rather than a single matrix.

    Norm matching is done on the answer position ``pos`` and the resulting
    scalar is applied to every position, so each variant injects the same
    magnitude as production at the position whose logits we read.  Production
    itself is passed ``match=False``: it is the reference, and rescaling it
    would stop it being the production read-out.
    """
    if variant == "production":
        raw = c_prod
    elif variant == "centered":
        raw = c_prod - c_bar
    elif variant == "random_proj":
        if value_proj_e is None or rand_w is None:
            raise ValueError("random_proj needs value_proj(e_t) and a random matrix")
        raw = value_proj_e @ rand_w.T
    elif variant == "oracle_ungated":
        if oracle_out is None:
            raise ValueError("oracle_ungated needs a precomputed out_proj(value_proj(e_t))")
        raw = oracle_out
    elif variant == "positive_control":
        if gold_emb is None:
            raise ValueError("positive_control needs the gold token embedding")
        raw = np.broadcast_to(gold_emb, c_prod.shape).copy()
    else:
        raise ValueError(f"unknown variant {variant!r}")

    if not match:
        return raw

    anchor = float(np.linalg.norm(raw[pos]))
    if anchor <= 0:
        raise ValueError(f"variant {variant!r} produced a zero vector at position {pos}")
    return raw * (target_norm / anchor)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows-dir", required=True)
    parser.add_argument("--reader", required=True)
    parser.add_argument("--model-dir", default="/root/autodl-tmp/qwen35-ple/models/qwen38_ple")
    parser.add_argument("--qa-file", default="data/qa-standard/eval-600b.jsonl")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--backbone-dtype", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_phase0 as p0

    from qwen35_ple.reader import _base_backbone
    from qwen35_ple.real_ple import resolve_ple_weight_scale

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")

    items = [json.loads(line) for line in Path(args.qa_file).read_text().splitlines() if line.strip()]
    if args.max_items:
        items = items[: args.max_items]
    print(f"[r165] {len(items)} items, variants={variants}", flush=True)

    tokenizer, model = p0._load_model(args.model, args.device, args.backbone_dtype)
    for param in model.parameters():
        param.requires_grad_(False)

    scale = resolve_ple_weight_scale(model_dir=args.model_dir, scale=None)
    store = p0._QAEtStore(args.rows_dir, scale)
    reader, _extra = p0.load_reader_with_extra(Path(args.reader), device=args.device)
    reader = reader.to(args.device)
    reader.eval()
    print(f"[r165] PLE scale={scale:.6g}", flush=True)

    backbone = _base_backbone(model)
    layer_module = backbone.model.layers[args.layer]

    # ---- pass 1: production contributions at the answer position ----------
    c_last_prod: list[np.ndarray] = []
    cnorm_prod: list[float] = []
    captures: list[dict] = []

    def capture_hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captures.append({"h": hidden.detach()})
        return output

    # Kept installed across both passes: pass 2 needs the same pre-injection
    # layer output to rebuild each variant.  ``captures`` is cleared before
    # every forward, so index 0 is always the current call.
    handle = layer_module.register_forward_hook(capture_hook)
    with torch.no_grad():
        for idx, item in enumerate(items):
            ids = p0._qa_prompt_ids(tokenizer, item, PROMPT, BOOLQ_PROMPT)
            et = store.fetch(np.asarray(ids, dtype=np.int64))
            captures.clear()
            inp = torch.tensor([ids], dtype=torch.long, device=args.device)
            model(input_ids=inp)
            h_all = captures[0]["h"][0]                                  # [T, d]
            et_t = torch.from_numpy(et[None, :, :]).float().to(args.device)
            c_all = reader(h_all[None], et_t)[0]                         # [T, d]
            last = c_all[-1].float().cpu().numpy()
            c_last_prod.append(last)
            cnorm_prod.append(float(np.linalg.norm(last)))

    C_prod = np.stack(c_last_prod).astype(np.float64)
    c_bar = C_prod.mean(axis=0)
    target_norms = np.asarray(cnorm_prod, dtype=np.float64)
    print(
        f"[r165] production: PR={participation_ratio(C_prod):.4f} "
        f"||c||={target_norms.mean():.4f}+-{target_norms.std():.4f}",
        flush=True,
    )

    # ---- frozen auxiliary projections for the oracle variants -------------
    # ``out_proj`` may be an MLP (the released checkpoint's is a 3-layer
    # Sequential), so it is applied as a module rather than as a matrix.
    if "oracle_ungated" in variants and not hasattr(reader, "out_proj"):
        raise SystemExit(
            f"oracle_ungated needs a reader with out_proj; {args.reader} has none"
        )

    rng = np.random.default_rng(args.seed)
    d_target = int(model.config.hidden_size)
    rand_w = rng.standard_normal((d_target, 2560)) / np.sqrt(2560)

    input_emb = model.get_input_embeddings().weight

    # ---- pass 2: per-variant, per-condition gold NLL ----------------------
    inject: dict[str, torch.Tensor | None] = {"vec": None}

    def inject_hook(module, inputs, output):
        vec = inject["vec"]
        if vec is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        new = hidden + vec
        if isinstance(output, tuple):
            return (new,) + output[1:]
        return new

    ihandle = layer_module.register_forward_hook(inject_hook)

    per_item: list[dict] = []
    tv_last: dict[tuple[str, str], list[float]] = {}
    tv_vs_off: dict[str, list[float]] = {}
    variant_last_vectors: dict[tuple[str, str], list[np.ndarray]] = {}

    with torch.no_grad():
        for idx, item in enumerate(items):
            # Reset before the clean forward: otherwise the previous item's
            # vector would contaminate the hidden states we read here.
            inject["vec"] = None
            prompt_ids = p0._qa_prompt_ids(tokenizer, item, PROMPT, BOOLQ_PROMPT)
            gold_ids = tokenizer.encode(str(item["answer"]))
            if not gold_ids:
                continue
            full = list(prompt_ids) + list(gold_ids)
            pos = len(prompt_ids) - 1
            inp = torch.tensor([full], dtype=torch.long, device=args.device)

            et_full = store.fetch(np.asarray(full, dtype=np.int64))

            captures.clear()
            model(input_ids=inp)
            h_all = captures[0]["h"][0]                                  # [T, d]

            # PLE off reference.
            off_logits = model(input_ids=inp).logits[0, pos].float()
            off_logp = torch.log_softmax(off_logits, dim=-1)

            et_t = torch.from_numpy(et_full[None, :, :]).float().to(args.device)
            c_prod_full = reader(h_all[None], et_t)[0].float().cpu().numpy()
            gold_id = int(gold_ids[0])
            gold_emb = input_emb[gold_id].detach().float().cpu().numpy()

            # Shuffled-rows control: same permutation convention as run_phase0.
            perm_rng = np.random.default_rng(args.seed * 1000 + idx)
            perm = perm_rng.permutation(len(full))
            et_shuf = et_full[perm]
            et_shuf_t = torch.from_numpy(et_shuf[None, :, :]).float().to(args.device)
            c_prod_shuf = reader(h_all[None], et_shuf_t)[0].float().cpu().numpy()

            row: dict = {"task": item.get("task"), "n_gold_tokens": len(gold_ids)}
            logits_by: dict[tuple[str, str], torch.Tensor] = {}

            for variant in variants:
                for cond in CONDITIONS:
                    if cond == "real":
                        c_prod_full_cond = c_prod_full
                        et_cond = et_full
                    else:
                        c_prod_full_cond = c_prod_shuf
                        et_cond = et_shuf

                    vp = None
                    oracle_out = None
                    if variant in ("random_proj", "oracle_ungated"):
                        et_cond_t = torch.from_numpy(et_cond).float().to(args.device)
                        vp_t = reader.value_proj(et_cond_t)
                        vp = vp_t.float().cpu().numpy()
                        if variant == "oracle_ungated":
                            # Gate and short convolution skipped; the trained
                            # out_proj is applied to the raw value projection.
                            oracle_out = reader.out_proj(vp_t).float().cpu().numpy()

                    vec = build_variant_vectors(
                        variant,
                        c_prod=c_prod_full_cond,
                        c_bar=c_bar,
                        value_proj_e=vp,
                        oracle_out=oracle_out,
                        rand_w=rand_w,
                        gold_emb=gold_emb,
                        target_norm=float(target_norms[idx]),
                        pos=pos,
                        match=variant != "production",
                    )
                    variant_last_vectors.setdefault((variant, cond), []).append(vec[pos])

                    inject["vec"] = torch.from_numpy(vec[None, :, :]).to(
                        args.device, dtype=h_all.dtype
                    )
                    logits = model(input_ids=inp).logits[0, pos].float()
                    logits_by[(variant, cond)] = logits

                    logp = torch.log_softmax(logits, dim=-1)
                    nll = float(-logp[gold_id])
                    row[f"nll_{variant}_{cond}"] = nll

            inject["vec"] = None

            off_nll = float(-off_logp[gold_id])
            row["nll_off"] = off_nll

            # TV at the answer position, real vs shuf, per variant.
            for variant in variants:
                p_real = torch.softmax(logits_by[(variant, "real")], dim=-1).cpu().numpy()
                p_shuf = torch.softmax(logits_by[(variant, "shuf")], dim=-1).cpu().numpy()
                tv_last.setdefault((variant, "real_vs_shuf"), []).append(
                    total_variation(p_real, p_shuf)
                )
                p_off = torch.softmax(off_logits, dim=-1).cpu().numpy()
                tv_vs_off.setdefault(variant, []).append(total_variation(p_real, p_off))

            per_item.append(row)
            if (idx + 1) % 50 == 0:
                print(f"[r165] {idx + 1}/{len(items)}", flush=True)

    ihandle.remove()
    handle.remove()

    # ---- aggregate --------------------------------------------------------
    def _col(name: str) -> np.ndarray:
        return np.asarray([r[name] for r in per_item], dtype=np.float64)

    report: dict = {
        "round": 165,
        "preregistration": "docs/round-165-readout-repair-preregistration.md",
        "model": args.model,
        "reader": args.reader,
        "qa_file": args.qa_file,
        "layer": args.layer,
        "n_items": len(per_item),
        "variants": variants,
        "readout_diagnostics": {},
        "end_to_end": {},
        "nll_off_mean": float(_col("nll_off").mean()),
    }

    for variant in variants:
        vecs = np.stack(variant_last_vectors[(variant, "real")]).astype(np.float64)
        shuf = np.stack(variant_last_vectors[(variant, "shuf")]).astype(np.float64)
        report["readout_diagnostics"][variant] = {
            "PR_c": participation_ratio(vecs),
            "cnorm_mean": float(np.linalg.norm(vecs, axis=1).mean()),
            "cos_real_vs_shuf": mean_pairwise_cosine(
                np.stack([vecs.mean(axis=0), shuf.mean(axis=0)])
            ),
            "cos_real_vs_shuf_peritem_mean": float(
                np.mean(
                    np.sum(vecs * shuf, axis=1)
                    / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(shuf, axis=1))
                )
            ),
        }

    for variant in variants:
        d = _col(f"nll_{variant}_shuf") - _col(f"nll_{variant}_real")
        lo, hi = demean_ci(d, seed=args.seed)
        tv = np.asarray(tv_last[(variant, "real_vs_shuf")], dtype=np.float64)
        tv_off = np.asarray(tv_vs_off[variant], dtype=np.float64)
        tv_lo, tv_hi = bootstrap_ci(tv, seed=args.seed)
        off_lo, off_hi = bootstrap_ci(tv_off, seed=args.seed)
        report["end_to_end"][variant] = {
            "dNLL_gold_mean": float(d.mean()),
            "dNLL_gold_ci95": [lo, hi],
            "tv_real_vs_shuf_mean": float(tv.mean()),
            "tv_real_vs_shuf_ci95": [tv_lo, tv_hi],
            "tv_vs_ple_off_mean": float(tv_off.mean()),
            "tv_vs_ple_off_ci95": [off_lo, off_hi],
        }

    # ---- frozen verdict ---------------------------------------------------
    pc = report["end_to_end"]["positive_control"]["tv_vs_ple_off_ci95"]
    report["verdict"] = _verdict(report, pc)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[r165] verdict={report['verdict']}", flush=True)
    print(f"[r165] wrote {out}", flush=True)
    return 0


def _verdict(report: dict, positive_control_ci: list[float]) -> str:
    """Apply the pre-registered rule, in the frozen order."""
    if positive_control_ci[0] <= 0:
        return "UNDERPOWERED"

    for variant in NON_COLLAPSING:
        if variant not in report["end_to_end"]:
            continue
        lo, _hi = report["end_to_end"][variant]["dNLL_gold_ci95"]
        if lo > 0:
            prod_tv = report["end_to_end"]["production"]["tv_real_vs_shuf_ci95"]
            var_tv = report["end_to_end"][variant]["tv_real_vs_shuf_ci95"]
            if var_tv[0] > prod_tv[1]:
                return "READOUT_IS_THE_CONSTRAINT"
    return "BOUND_SURVIVES_REPAIRED_READOUT"


if __name__ == "__main__":
    raise SystemExit(main())
