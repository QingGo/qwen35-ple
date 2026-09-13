#!/usr/bin/env python3
"""Round 167 Stage 1a: does the PLE graft change the backbone's *effective depth*?

Why this is the experiment that was missing
------------------------------------------
Every negative result we own was measured on the **knowledge** axis: does content
from the table reach the output at the answer position?  That axis is where the
round-157 bound says the answer is *forced* to be zero, and it is also the axis
on which the official Engram paper reports its **smallest** gains
(MMLU +3.4 vs BBH +5.0, ARC-C +3.7, HumanEval +3.0).

Engram's own mechanism claim is different: the memory "relieves the backbone's
early layers from static reconstruction, effectively deepening the network".
That is a claim about **effective depth**, and we have never measured it --
`logit lens` appears in this repo only as a two-word TODO (round-26 V185).

So this script measures the depth axis under PLE on/off, using the instruments
of Csordás et al. (arXiv 2505.13898) and the effective-depth operationalisation
of arXiv 2512.14064:

* **logit-lens KL** -- KL(final || intermediate) per layer; the model's
  prediction stabilises where this falls.
* **top-5 overlap** -- agreement between the intermediate and final top-5.
* **residual cosine** -- ``cos(sublayer contribution, residual)``; the standard
  reading is composition (near zero), refinement (positive), erasure (negative).
* **effective depth** -- first layer where KL drops below half its max, and
  first layer where top-5 overlap exceeds 0.3.

Conditions: ``off`` (no reader), ``real`` (frozen wiki rows), ``shuf`` (the same
rows permuted, the round-162 control).

The three possible outcomes and what they mean are pre-registered in
``docs/round-167-stage1-effective-depth-preregistration.md``.

Usage::

    PYTHONPATH=src python scripts/round167_effective_depth.py \
      --model /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B \
      --rows-dir /root/autodl-tmp/qwen35-ple/qwen38-rows \
      --reader outputs/round162-0.8B-nosft600/reader-wiki-seed0.pt \
      --qa-file data/qa-standard/eval-600b.jsonl \
      --output outputs/round167/effective-depth-0.8B.json
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

TOP_K = 5
OVERLAP_THRESHOLD = 0.3
KL_HALF_MAX = 0.5

CONDITIONS = ("off", "real", "shuf")


# ---------------------------------------------------------------------------
# Instrument definitions (kept as pure functions so tests can pin them)
# ---------------------------------------------------------------------------


def logit_lens_kl(p_final: np.ndarray, p_layer: np.ndarray) -> float:
    """KL(p_final || p_layer) in nats, with an epsilon floor for zeros."""
    eps = 1e-12
    p = np.clip(np.asarray(p_final, dtype=np.float64), eps, None)
    q = np.clip(np.asarray(p_layer, dtype=np.float64), eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def top_k_overlap(logits_layer: np.ndarray, logits_final: np.ndarray, k: int = TOP_K) -> float:
    """Fraction of the final top-k that the layer's top-k already contains."""
    a = set(np.argsort(-np.asarray(logits_layer))[:k].tolist())
    b = set(np.argsort(-np.asarray(logits_final))[:k].tolist())
    return float(len(a & b) / k)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def effective_depth_from_kl(kl_by_layer: list[float]) -> int | None:
    """First layer whose KL is at or below half the maximum observed KL."""
    arr = np.asarray(kl_by_layer, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    threshold = KL_HALF_MAX * float(finite.max())
    below = np.where(np.isfinite(arr) & (arr <= threshold))[0]
    return int(below[0]) if below.size else None


def effective_depth_from_overlap(overlap_by_layer: list[float]) -> int | None:
    """First layer whose top-k overlap with the final prediction exceeds 0.3."""
    arr = np.asarray(overlap_by_layer, dtype=np.float64)
    above = np.where(np.isfinite(arr) & (arr > OVERLAP_THRESHOLD))[0]
    return int(above[0]) if above.size else None


def effective_depth_from_cosine(cos_by_layer: list[float]) -> int | None:
    """First layer where the contribution cosine turns and *stays* positive.

    Mirrors arXiv 2512.14064: the transition from feature composition
    (negative or near-zero similarity) to refinement (positive similarity).  The
    rule is "positive here and positive at the next layer", which resists a lone
    noisy spike without the multi-layer lag a trailing moving average would
    introduce -- a lag of 2+ layers could mask exactly the small depth shift this
    experiment is looking for.
    """
    arr = np.asarray(cos_by_layer, dtype=np.float64)
    n = arr.size
    if n == 0:
        return None
    positive = np.isfinite(arr) & (arr > 0.0)
    for i in range(n - 1):
        if positive[i] and positive[i + 1]:
            return int(i)
    # A curve that ends positive but never has two consecutive positives still
    # has a transition; report its first positive layer.
    hits = np.where(positive)[0]
    return int(hits[0]) if hits.size else None


# ---------------------------------------------------------------------------


def _resolve_attn_module(layer):
    """Find the attention submodule of a decoder layer.

    Qwen3.5 is a hybrid stack: the layer exposes ``linear_attn`` (a
    ``Qwen3_5GatedDeltaNet``) rather than the ``self_attn`` a vanilla Qwen would
    have.  Hard-coding ``self_attn`` therefore fails on this backbone, so probe
    the known names and fall back to any child whose name contains ``attn``.
    """
    for name in ("self_attn", "linear_attn", "attn", "attention"):
        mod = getattr(layer, name, None)
        if mod is not None:
            return name, mod
    for name, mod in layer.named_children():
        if "attn" in name:
            return name, mod
    raise RuntimeError(
        f"no attention submodule on {type(layer).__name__}; children: "
        f"{[n for n, _ in layer.named_children()]}"
    )


def _resolve_head(model):
    """Return (final_norm, lm_head) with a few tolerant attribute paths."""
    inner = getattr(model, "model", None)
    norm = getattr(inner, "norm", None) if inner is not None else None
    if norm is None:
        norm = getattr(model, "norm", None)
    head = getattr(model, "lm_head", None)
    if head is None and inner is not None:
        head = getattr(inner, "lm_head", None)
    if norm is None or head is None:
        raise RuntimeError(
            "could not resolve final norm / lm_head; "
            f"have model attrs {[a for a in dir(model) if not a.startswith('_')][:20]}"
        )
    return norm, head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--model-dir", default="/root/autodl-tmp/qwen35-ple/models/qwen38_ple")
    ap.add_argument("--qa-file", default="data/qa-standard/eval-600b.jsonl")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--backbone-dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import torch

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_phase0 as p0

    from qwen35_ple.reader import install_reader_hook
    from qwen35_ple.real_ple import resolve_ple_weight_scale

    items = [
        json.loads(line)
        for line in Path(args.qa_file).read_text().splitlines()
        if line.strip()
    ]
    if args.max_items:
        items = items[: args.max_items]
    if not items:
        raise SystemExit(f"no items in {args.qa_file}")

    tokenizer, model = p0._load_model(args.model, args.device, args.backbone_dtype)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    scale = resolve_ple_weight_scale(model_dir=args.model_dir, scale=None)
    store = p0._QAEtStore(args.rows_dir, scale)
    reader, _extra = p0.load_reader_with_extra(Path(args.reader), device=args.device)
    reader = reader.to(args.device)
    reader.eval()

    n_layers = len(model.model.layers)
    norm, head = _resolve_head(model)
    print(
        f"[r167-1a] {len(items)} items, {n_layers} layers, layer={args.layer}, "
        f"PLE scale={scale:.6g}",
        flush=True,
    )

    # Per-layer sublayer contributions for the residual-cosine instrument.
    contrib: dict[str, object] = {}

    def make_attn_hook(i: int):
        def hook(module, inputs, output):
            contrib[f"attn_{i}"] = output[0] if isinstance(output, tuple) else output
            return output

        return hook

    def make_mlp_hook(i: int):
        def hook(module, inputs, output):
            contrib[f"mlp_{i}"] = output
            return output

        return hook

    handles = []
    attn_name = None
    for i, layer in enumerate(model.model.layers):
        attn_name, attn_mod = _resolve_attn_module(layer)
        handles.append(attn_mod.register_forward_hook(make_attn_hook(i)))
        handles.append(layer.mlp.register_forward_hook(make_mlp_hook(i)))
    print(f"[r167-1a] attention submodule resolved as {attn_name!r}", flush=True)
    # A reader handle is always installed; the "off" condition disables injection
    # by clearing ``model._current_ple_e_t`` before the forward pass, which is the
    # same switch ``run_phase0`` uses for its ple-off arm.
    handles.append(install_reader_hook(model, args.layer, reader, None))

    # kl[cond][l], overlap[cond][l], and the three cosine families.
    kl = {c: [[] for _ in range(n_layers + 1)] for c in CONDITIONS}
    ov = {c: [[] for _ in range(n_layers + 1)] for c in CONDITIONS}
    cos_layer = {c: [[] for _ in range(n_layers)] for c in CONDITIONS}
    cos_attn = {c: [[] for _ in range(n_layers)] for c in CONDITIONS}
    cos_mlp = {c: [[] for _ in range(n_layers)] for c in CONDITIONS}
    tasks: list[str] = []
    injection_deltas: list[float] = []

    with torch.no_grad():
        for idx, item in enumerate(items):
            ids = p0._qa_prompt_ids(tokenizer, item, PROMPT, BOOLQ_PROMPT)
            pos = len(ids) - 1
            et = store.fetch(np.asarray(ids, dtype=np.int64))
            perm = np.random.default_rng(args.seed * 1000 + idx).permutation(len(ids))
            tasks.append(str(item.get("task", "unknown")))

            prev_final: np.ndarray | None = None
            for cond in CONDITIONS:
                contrib.clear()
                if cond == "off":
                    model._current_ple_e_t = None
                else:
                    et_cur = et if cond == "real" else et[perm]
                    model._current_ple_e_t = (
                        torch.from_numpy(et_cur[None]).float().to(args.device)
                    )

                out = model(
                    input_ids=torch.tensor([ids], dtype=torch.long, device=args.device),
                    output_hidden_states=True,
                )
                hs = out.hidden_states  # tuple of [1, T, D], length n_layers+1

                h_final = hs[-1][0, pos].float()
                logits_final = head(norm(h_final[None]))[0].float().cpu().numpy()
                p_final = np.exp(logits_final - logits_final.max())
                p_final /= p_final.sum()

                if cond == "off":
                    prev_final = p_final
                elif prev_final is not None:
                    injection_deltas.append(
                        float(np.abs(p_final - prev_final).sum())
                    )

                for l in range(n_layers + 1):
                    h_l = hs[l][0, pos].float()
                    logits_l = head(norm(h_l[None]))[0].float().cpu().numpy()
                    p_l = np.exp(logits_l - logits_l.max())
                    p_l /= p_l.sum()
                    kl[cond][l].append(logit_lens_kl(p_final, p_l))
                    ov[cond][l].append(top_k_overlap(logits_l, logits_final))

                for l in range(n_layers):
                    h_l = hs[l][0, pos].float().cpu().numpy().astype(np.float64)
                    h_next = hs[l + 1][0, pos].float().cpu().numpy().astype(np.float64)
                    a_l = contrib.get(f"attn_{l}")
                    m_l = contrib.get(f"mlp_{l}")
                    if a_l is not None:
                        a_np = a_l[0, pos].float().cpu().numpy().astype(np.float64)
                        cos_attn[cond][l].append(cosine(a_np, h_l))
                    if m_l is not None:
                        m_np = m_l[0, pos].float().cpu().numpy().astype(np.float64)
                        pre = h_l + (a_np if a_l is not None else 0.0)
                        cos_mlp[cond][l].append(cosine(m_np, pre))
                    cos_layer[cond][l].append(cosine(h_next - h_l, h_l))

            if (idx + 1) % 25 == 0:
                print(f"[r167-1a] {idx + 1}/{len(items)}", flush=True)

    for h in handles:
        h.remove()

    def _mean(rows: list[list[float]]) -> list[float]:
        return [float(np.nanmean(v)) if len(v) else float("nan") for v in rows]

    report: dict = {
        "config": {
            "model": args.model,
            "reader": args.reader,
            "qa_file": args.qa_file,
            "layer": args.layer,
            "max_items": len(items),
            "n_layers": n_layers,
            "backbone_dtype": args.backbone_dtype,
            "ple_weight_scale": scale,
        },
        "tasks": {t: tasks.count(t) for t in sorted(set(tasks))},
        "sanity": {
            "mean_total_variation_off_vs_real": (
                float(np.mean(injection_deltas)) if injection_deltas else None
            ),
            "n_injection_comparisons": len(injection_deltas),
            "injection_is_live": bool(injection_deltas and np.mean(injection_deltas) > 1e-6),
        },
        "conditions": {},
    }

    for cond in CONDITIONS:
        kl_mean = _mean(kl[cond])
        ov_mean = _mean(ov[cond])
        cl_mean = _mean(cos_layer[cond])
        ca_mean = _mean(cos_attn[cond])
        cm_mean = _mean(cos_mlp[cond])
        report["conditions"][cond] = {
            "kl_by_layer": kl_mean,
            "top5_overlap_by_layer": ov_mean,
            "cos_layer_by_layer": cl_mean,
            "cos_attn_by_layer": ca_mean,
            "cos_mlp_by_layer": cm_mean,
            "effective_depth": {
                "from_kl_half_max": effective_depth_from_kl(kl_mean),
                "from_top5_overlap_0.3": effective_depth_from_overlap(ov_mean),
                "from_residual_cosine": effective_depth_from_cosine(cl_mean),
            },
            "final_kl": kl_mean[-1],
            "final_overlap": ov_mean[-1],
        }

    # The headline contrast: does injection move the transition earlier?
    ed = {c: report["conditions"][c]["effective_depth"] for c in CONDITIONS}
    report["verdict_inputs"] = {
        "effective_depth_by_condition": ed,
        "depth_gain_real_vs_off": {
            k: (
                None
                if ed["off"][k] is None or ed["real"][k] is None
                else int(ed["off"][k] - ed["real"][k])
            )
            for k in ed["off"]
        },
        "depth_gain_shuf_vs_off": {
            k: (
                None
                if ed["off"][k] is None or ed["shuf"][k] is None
                else int(ed["off"][k] - ed["shuf"][k])
            )
            for k in ed["off"]
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== effective depth (layer index of the transition) ===")
    print(f"{'instrument':<28} {'off':>6} {'real':>6} {'shuf':>6}")
    for k in ("from_kl_half_max", "from_top5_overlap_0.3", "from_residual_cosine"):
        print(
            f"{k:<28} {ed['off'][k]!s:>6} {ed['real'][k]!s:>6} {ed['shuf'][k]!s:>6}"
        )
    print(f"\ninjection live: {report['sanity']['injection_is_live']} "
          f"(mean TV off-vs-real {report['sanity']['mean_total_variation_off_vs_real']})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
