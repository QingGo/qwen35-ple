#!/usr/bin/env python3
"""Round 168 Stage 1.5c: measure the marginal-advantage (``margin``) distribution.

The question
------------
Every measurement this project has of "how much can the memory add" is limited
by the decoder used to read the table (round-168 TD-9: the CMI probe's *linear*
decoder was blind to complementary information, excess -0.44 vs +2.2 for an
MLP).  The one quantity that is **not** decoder-limited is

    margin(t) = L_bb(t) - L_cnt(t)

where ``L_bb`` is the backbone's per-position NLL and ``L_cnt`` is a count-based
**trigram** model's NLL on the same stream.  The trigram is not an arbitrary
baseline: round 167 proved that every PLE row id is a function of at most three
tokens (``k * n = 12`` characters, ``r = 9``), so

    I(Y ; e_{t-r:t} | h_t)  <=  I(Y ; w_t | h_t) ,      |w_t| <= 3 tokens.

Hence *any* memory of this design is bounded by the best trigram predictor, and
the count model estimates exactly that.  Where ``margin`` is positive the
trigram alone beats the backbone alone, so the trigram is not redundant with the
backbone state; where the whole distribution is negative there is no position a
prediction-replacing memory could improve, and Stage 3 can be closed for free.

Two frames, both measured
-------------------------
* ``bb_final``: the pure backbone's next-token NLL -- what the graft would have
  to beat end to end.
* ``bb_lens``: the logit lens at the **injection point** (the residual stream
  the reader writes into, ``hidden_states[layer + 1]``) -- the correct frame for
  the bound, because information the later layers can compute from ``h_t`` is
  already available without memory.  ``bb_lens >= bb_final`` always, so this
  frame is the more generous one for the memory and the one the pre-registered
  verdict uses.

Stages
------
``counts``   build MKN trigram tables over ``--train-npy`` and score every
             position of ``--eval-npy``.  CPU only.
``backbone`` per-position NLL of the pure backbone (no reader, no injection) at
             the final layer and at the injection point.  GPU minutes, no
             training.
``analyze``  combine, apply the pre-registered rule, write JSON + Markdown.
``all``      the three in order.

Usage (per domain)::

    python scripts/round168_margin_distribution.py --stage counts \
      --train-npy data/phase1/PURE_WIKI/tokens.npy \
      --eval-npy data/phase1/wikitext-heldout-decon/tokens.npy \
      --tag wiki --workdir outputs/round168/margin

    python scripts/round168_margin_distribution.py --stage backbone \
      --eval-npy data/phase1/wikitext-heldout-decon/tokens.npy \
      --tag wiki --workdir outputs/round168/margin \
      --model /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B

    python scripts/round168_margin_distribution.py --stage analyze \
      --tag wiki --workdir outputs/round168/margin
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen35_ple.margin import (
    MARGIN_QUANTILES,
    MIN_CONTEXT_COUNT,
    bucket_summary,
    cross_tab,
    gain_curve,
    margin_from_nll,
    positive_share,
    verdict_from_curve,
)

#: The pre-registered primaries: the injection-point lens is the frame the
#: decoder-independent bound is stated in; the final layer is the end-to-end
#: view.  Both are always reported, the verdict is taken on the former.
PRIMARY_MARGIN = "lens"


def log(msg: str) -> None:
    print(f"[r168-1.5c] {msg}", flush=True)


def peak_rss_gib() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0 / 1024.0
    except OSError:
        pass
    return float("nan")


def _paths(workdir: Path, tag: str) -> dict[str, Path]:
    return {
        "counts": workdir / f"{tag}-counts.npz",
        "backbone": workdir / f"{tag}-backbone.npz",
        "json": workdir / f"{tag}-margin.json",
        "md": workdir / f"{tag}-margin.md",
    }


# --------------------------------------------------------------------------
# stage: counts (CPU)
# --------------------------------------------------------------------------
def stage_counts(args: argparse.Namespace) -> int:
    """Score every eval position with an MKN trigram model.  No torch."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import bench_ngram_reference as bnr

    out_path = _paths(Path(args.workdir), args.tag)["counts"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train = bnr.load_tokens(Path(args.train_npy))
    stream = bnr.load_tokens(Path(args.eval_npy))
    max_tok = int(max(train.max(), stream.max()))
    # ``--uniform-vocab`` is the *floor* vocabulary, kept at round-158's 248047
    # for the wiki arm so its numbers stay comparable.  Other domains (stem's
    # tokenizer output reaches 248069) need the floor spread over strictly more
    # tokens, so raise it to cover the observed ids rather than crashing:
    # a floor that leaves tokens unreachable is a silent -inf, not a small bias.
    vocab = int(args.uniform_vocab)
    if max_tok >= vocab:
        log(f"raising uniform-vocab {vocab} -> {max_tok + 1} "
            f"(max observed token id {max_tok})")
        vocab = max_tok + 1
    M = int(args.order)

    t0 = time.time()
    log(f"counting {M}-grams over {train.shape[0]:,} training tokens (V={vocab})")
    raw, cont = bnr.build_counts(train, list(range(1, M + 1)), vocab)

    # Context frequency of the row-determining trigram: the PLE row used to
    # predict x[t] is selected by (x[t-M], ..., x[t-1]), so the raw count of
    # that n-gram in the count model's training stream is the proxy for "how
    # well estimated is this row" (design doc condition 3).
    keys_m = bnr.pack_contexts(raw[M][0], vocab)
    counts_m = raw[M][1].astype(np.int64)
    keys_1 = raw[1][0][:, 0].astype(np.int64)
    counts_1 = raw[1][1].astype(np.int64)

    notes: list[str] = []
    levels: dict[int, Any] = {}
    for k in range(1, M + 1):
        rows, ct = (cont[k] if k < M else raw[k])
        lvl = bnr.build_level(rows, ct, vocab)
        bnr.finalize_level(lvl, args.smoothing, args.addk, vocab, notes)
        levels[k] = lvl
    del raw, cont
    gc.collect()
    model = bnr.NgramModel(M, levels, vocab, args.smoothing, args.addk, notes)

    N = stream.shape[0]
    pos = np.arange(M - 1, N, dtype=np.int64)
    t1 = time.time()
    lp, top_level, pos_out, n_zero = model.stream_logprob(stream, pos)
    log(f"scored {pos.size:,} positions in {time.time() - t1:.1f}s "
        f"({n_zero} positions hit the uniform floor)")

    cnt_nll = np.full(N, np.nan, dtype=np.float64)
    cnt_nll[pos_out] = -np.asarray(lp, dtype=np.float64)
    top_lvl = np.zeros(N, dtype=np.int8)
    top_lvl[pos_out] = np.asarray(top_level, dtype=np.int8)

    qk = bnr.context_keys_at(stream, pos, M, vocab)
    idx = np.searchsorted(keys_m, qk)
    np.clip(idx, 0, max(keys_m.shape[0] - 1, 0), out=idx)
    hit = keys_m[idx] == qk if keys_m.shape[0] else np.zeros(pos.shape, dtype=bool)
    ctx_counts = np.zeros(N, dtype=np.int64)
    ctx_counts[pos] = np.where(hit, counts_m[idx], 0)

    tq = stream[pos].astype(np.int64)
    tidx = np.searchsorted(keys_1, tq)
    np.clip(tidx, 0, max(keys_1.shape[0] - 1, 0), out=tidx)
    thit = keys_1[tidx] == tq
    tgt_counts = np.zeros(N, dtype=np.int64)
    tgt_counts[pos] = np.where(thit, counts_1[tidx], 0)

    np.savez_compressed(
        out_path,
        cnt_nll=cnt_nll.astype(np.float32),
        top_level=top_lvl,
        context_counts=ctx_counts,
        target_counts=tgt_counts,
        positions=pos,
        train_npy=str(args.train_npy),
        eval_npy=str(args.eval_npy),
        order=np.int64(M),
        uniform_vocab=np.int64(vocab),
        smoothing=args.smoothing,
        n_train_tokens=np.int64(train.shape[0]),
        n_eval_tokens=np.int64(N),
    )
    log(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, "
        f"peak RSS {peak_rss_gib():.2f} GiB, {time.time() - t0:.1f}s total)")
    return 0


# --------------------------------------------------------------------------
# stage: backbone (GPU minutes, no training)
# --------------------------------------------------------------------------
def stage_backbone(args: argparse.Namespace) -> int:
    import torch

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_phase0 as p0
    from round167_effective_depth import _resolve_head

    out_path = _paths(Path(args.workdir), args.tag)["backbone"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ids = np.load(args.eval_npy).astype(np.int64)
    N = ids.shape[0]
    log(f"loading {args.model} on {args.device} ({args.backbone_dtype})")
    _tok, model = p0._load_model(args.model, args.device, args.backbone_dtype)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    # No reader is installed here, so nothing can inject; clearing the handle is
    # belt-and-braces for releases that default it to a tensor.
    model._current_ple_e_t = None

    text_model = getattr(model, "model", model)
    n_layers = len(text_model.layers)
    norm, head = _resolve_head(model)
    param_dtype = next(head.parameters()).dtype
    # The reader hook is a post-forward hook on ``layers[layer]``, so the
    # residual stream it writes into is ``hidden_states[layer + 1]``
    # (``hidden_states[0]`` is the embedding output).
    lens_idx = int(args.layer) + 1
    if not 0 <= lens_idx <= n_layers:
        raise SystemExit(f"lens index {lens_idx} outside 0..{n_layers}")
    log(f"{n_layers} layers, injection layer {args.layer} -> "
        f"lens on hidden_states[{lens_idx}], head dtype {param_dtype}")

    bb_final = np.full(N, np.nan, dtype=np.float64)
    bb_lens = np.full(N, np.nan, dtype=np.float64)
    B = int(args.chunk_tokens)
    sub = int(args.head_chunk)
    t0 = time.time()
    scored = 0
    for a in range(0, N, B):
        b = min(a + B, N)
        if b - a < 2:
            continue
        inp = torch.from_numpy(ids[a:b]).unsqueeze(0).to(args.device)
        with torch.no_grad():
            out = text_model(input_ids=inp, output_hidden_states=True)
        hs = out.hidden_states
        targets = torch.from_numpy(ids[a + 1 : b]).to(args.device)
        for idx, arr in ((n_layers, bb_final), (lens_idx, bb_lens)):
            h = hs[idx][0]  # [T, D]; logits at i predict token i+1
            T = b - a - 1
            for s in range(0, T, sub):
                e = min(s + sub, T)
                logits = head(norm(h[s:e].to(param_dtype).unsqueeze(0)))[0].float()
                lse = torch.logsumexp(logits, dim=-1)
                pick = logits[torch.arange(e - s, device=args.device), targets[s:e]]
                arr[a + 1 + s : a + 1 + e] = (
                    (lse - pick).double().cpu().numpy()
                )
                del logits, lse, pick
        scored += b - a - 1
        del out, hs, inp, targets
        if (a // B) % 20 == 0:
            done = b
            rate = done / max(time.time() - t0, 1e-9)
            log(f"  {done:,}/{N:,} tokens ({rate:,.0f} tok/s)")
    log(f"scored {scored:,} positions in {time.time() - t0:.1f}s "
        f"(peak RSS {peak_rss_gib():.2f} GiB)")

    np.savez_compressed(
        out_path,
        bb_final=bb_final.astype(np.float32),
        bb_lens=bb_lens.astype(np.float32),
        model=str(args.model),
        layer=np.int64(args.layer),
        lens_hidden_index=np.int64(lens_idx),
        backbone_dtype=args.backbone_dtype,
        chunk_tokens=np.int64(B),
        n_eval_tokens=np.int64(N),
        eval_npy=str(args.eval_npy),
    )
    log(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


# --------------------------------------------------------------------------
# stage: analyze
# --------------------------------------------------------------------------
def _analysis_for(
    name: str, bb: np.ndarray, cnt: np.ndarray, ctx: np.ndarray, tgt: np.ndarray,
    valid: np.ndarray, min_ctx: int,
) -> dict[str, object]:
    margin = margin_from_nll(bb[valid], cnt[valid])
    curve = gain_curve(
        margin, context_counts=ctx[valid], min_context_count=min_ctx,
    )
    verdict = verdict_from_curve(
        curve, positive_share_all=positive_share(margin), n_positions=int(margin.size),
    )
    return {
        "name": name,
        "n_positions": int(margin.size),
        "mean_bb_nll": float(bb[valid].mean()),
        "mean_cnt_nll": float(cnt[valid].mean()),
        "mean_margin": float(margin.mean()),
        "median_margin": float(np.median(margin)),
        "positive_share": positive_share(margin),
        "gain_curve": [e.to_dict() for e in curve],
        "verdict": verdict.to_dict(),
        "by_target_frequency": bucket_summary(
            margin, tgt[valid], name="target_count", bb_nll=bb[valid], cnt_nll=cnt[valid],
        ),
        "by_context_frequency": bucket_summary(
            margin, ctx[valid], name="context_count", bb_nll=bb[valid], cnt_nll=cnt[valid],
        ),
        "cross_tab": cross_tab(margin, tgt[valid], ctx[valid]),
    }


def stage_analyze(args: argparse.Namespace) -> int:
    paths = _paths(Path(args.workdir), args.tag)
    c = np.load(paths["counts"])
    b = np.load(paths["backbone"])
    cnt = np.asarray(c["cnt_nll"], dtype=np.float64)
    ctx = np.asarray(c["context_counts"], dtype=np.int64)
    tgt = np.asarray(c["target_counts"], dtype=np.int64)
    final = np.asarray(b["bb_final"], dtype=np.float64)
    lens = np.asarray(b["bb_lens"], dtype=np.float64)
    if not (cnt.shape == final.shape == lens.shape):
        raise SystemExit(
            f"array length mismatch: cnt {cnt.shape} final {final.shape} lens {lens.shape}"
        )

    # ``positions`` marks exactly the positions the count model scored; the
    # backbone leaves NaN wherever a chunk boundary (or the stream tail) made a
    # position unscorable, so intersecting the two is the whole alignment.
    valid = np.isfinite(cnt) & np.isfinite(final) & np.isfinite(lens)
    valid &= _position_mask(c, cnt.shape)
    n_total = int(cnt.shape[0])
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        raise SystemExit("no overlapping scored positions between the two stages")
    log(f"{n_valid:,}/{n_total:,} positions scored by both stages "
        f"({100.0 * n_valid / n_total:.2f}%)")

    per_frame = {
        name: _analysis_for(name, bb, cnt, ctx, tgt, valid, args.min_context_count)
        for name, bb in (("lens", lens), ("final", final))
    }
    primary = per_frame[PRIMARY_MARGIN]
    out = {
        "tag": args.tag,
        "order": int(c["order"]),
        "uniform_vocab": int(c["uniform_vocab"]),
        "smoothing": str(c["smoothing"]),
        "train_npy": str(c["train_npy"]),
        "eval_npy": str(c["eval_npy"]),
        "n_train_tokens": int(c["n_train_tokens"]),
        "n_eval_tokens": n_total,
        "n_scored_positions": n_valid,
        "coverage": n_valid / n_total,
        "model": str(b["model"]),
        "injection_layer": int(b["layer"]),
        "lens_hidden_index": int(b["lens_hidden_index"]),
        "backbone_dtype": str(b["backbone_dtype"]),
        "min_context_count": int(args.min_context_count),
        "quantiles": list(MARGIN_QUANTILES),
        "primary_frame": PRIMARY_MARGIN,
        "frames": per_frame,
        "verdict": primary["verdict"],
        "assumptions": [
            (
                "L_cnt is an MKN trigram model over the count model's training "
                "stream; it is a proxy for the best achievable trigram predictor, "
                "so a weak proxy makes margin pessimistic (negative results "
                "strong, positive weak)."
            ),
            (
                "The pre-registered verdict uses the injection-point logit-lens "
                "frame (the residual stream the reader writes into), which is the "
                "frame the decoder-independent bound is stated in."
            ),
            (
                "oracle_gain is the upper bound for a prediction-REPLACING memory "
                "only; a memory combined with the backbone could in principle beat "
                "both."
            ),
            (
                "WikiText is almost certainly in the Qwen pretraining data, so the "
                "backbone's NLL there is optimistic (memory looks worse than it "
                "is); the stem/code domains are the counterweight."
            ),
        ],
    }
    paths["json"].write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    paths["md"].write_text(render_markdown(out), encoding="utf-8")
    log(f"verdict[{PRIMARY_MARGIN}] = {primary['verdict']['label']}: "
        f"{primary['verdict']['reason']}")
    log(f"verdict[final] = {per_frame['final']['verdict']['label']}")
    log(f"wrote {paths['json']} and {paths['md']}")
    return 0


def _position_mask(c: Any, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(c["positions"], dtype=np.int64)] = True
    return mask


def render_markdown(out: dict[str, object]) -> str:
    p = out["frames"][out["primary_frame"]]  # type: ignore[index]
    f = out["frames"]["final"]  # type: ignore[index]
    lines: list[str] = []
    lines.append(f"# Stage 1.5c margin distribution — {out['tag']}")
    lines.append("")
    lines.append(f"* count training stream: `{out['train_npy']}` "
                 f"({out['n_train_tokens']:,} tokens)")
    lines.append(f"* eval stream: `{out['eval_npy']}` ({out['n_eval_tokens']:,} tokens)")
    lines.append(f"* scored positions: {out['n_scored_positions']:,} "
                 f"({100.0 * float(out['coverage']):.2f}%)")
    lines.append(f"* backbone: `{out['model']}`, injection layer {out['injection_layer']} "
                 f"-> lens on `hidden_states[{out['lens_hidden_index']}]`")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{p['verdict']['label']}** (primary frame = {out['primary_frame']})")
    lines.append("")
    lines.append(f"> {p['verdict']['reason']}")
    lines.append("")
    lines.append(f"Final-layer frame: **{f['verdict']['label']}** — {f['verdict']['reason']}")
    lines.append("")
    lines.append("## Means (nats)")
    lines.append("")
    lines.append("| frame | mean L_bb | mean L_cnt | mean margin | median margin | positive share |")
    lines.append("|---|---|---|---|---|---|")
    for name in ("lens", "final"):
        d = out["frames"][name]  # type: ignore[index]
        lines.append(
            f"| {name} | {d['mean_bb_nll']:.4f} | {d['mean_cnt_nll']:.4f} | "
            f"{d['mean_margin']:+.4f} | {d['median_margin']:+.4f} | "
            f"{d['positive_share']:.4f} |"
        )
    lines.append("")
    lines.append("## Oracle gain curve — `(1/N)·Σ max(0, margin)` on the top-q% tail")
    lines.append("")
    lines.append("| frame | pool | q% | n selected | mean margin | positive share | corpus NLL reduction |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in ("lens", "final"):
        for e in out["frames"][name]["gain_curve"]:  # type: ignore[index]
            lines.append(
                f"| {name} | {e['label']} | {e['q_percent']:g} | {e['n_selected']:,} | "
                f"{e['mean_margin_selected']:+.4f} | {e['share_positive_selected']:.3f} | "
                f"**{e['corpus_nll_reduction']:.5f}** |"
            )
    lines.append("")
    lines.append("## Margin by context frequency (row-quality axis)")
    lines.append("")
    lines.append("| ctx count | n | share | mean L_bb | mean L_cnt | mean margin | positive share |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in p["by_context_frequency"]:  # type: ignore[index]
        if r["n"] == 0:
            continue
        lines.append(
            f"| {r['bucket']} | {r['n']:,} | {r['share']:.4f} | "
            f"{_fmt(r['mean_bb_nll'])} | {_fmt(r['mean_cnt_nll'])} | "
            f"{_fmt(r['mean_margin'])} | {_fmt(r['share_positive'])} |"
        )
    lines.append("")
    lines.append("## Margin by target frequency (Nishida axis, primary frame)")
    lines.append("")
    lines.append("| target count | n | share | mean margin | positive share |")
    lines.append("|---|---|---|---|---|")
    for r in p["by_target_frequency"]:  # type: ignore[index]
        if r["n"] == 0:
            continue
        lines.append(
            f"| {r['bucket']} | {r['n']:,} | {r['share']:.4f} | "
            f"{_fmt(r['mean_margin'])} | {_fmt(r['share_positive'])} |"
        )
    lines.append("")
    lines.append("## Cross-tab: mean margin, target x context frequency")
    lines.append("")
    tab = p["cross_tab"]  # type: ignore[index]
    cells = tab["cells"]  # type: ignore[index]
    ctx_order = _ordered(c["context_bucket"] for c in cells)
    tgt_order = _ordered(c["target_bucket"] for c in cells)
    by_key = {(c["target_bucket"], c["context_bucket"]): c for c in cells}
    lines.append("| target \\ ctx | " + " | ".join(ctx_order) + " |")
    lines.append("|---" * (len(ctx_order) + 1) + "|")
    for tb in tgt_order:
        row = [tb]
        for cb in ctx_order:
            cell = by_key.get((tb, cb))
            if cell is None or cell["n"] == 0:
                row.append("·")
            else:
                row.append(f"{cell['mean_margin']:+.2f} ({cell['n']:,})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Assumptions / honesty")
    lines.append("")
    for a in out["assumptions"]:  # type: ignore[index]
        lines.append(f"* {a}")
    lines.append("")
    return "\n".join(lines)


def _fmt(v: object) -> str:
    return "—" if v is None else f"{float(v):+.4f}"  # type: ignore[arg-type]


def _ordered(values: Any) -> list[str]:
    """De-duplicate while preserving the bucket order the cross-tab emits."""
    out: list[str] = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True,
                    choices=["counts", "backbone", "analyze", "all"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workdir", default="outputs/round168/margin")
    ap.add_argument("--train-npy", default="data/phase1/PURE_WIKI/tokens.npy")
    ap.add_argument("--eval-npy", default="data/phase1/wikitext-heldout-decon/tokens.npy")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--uniform-vocab", type=int, default=248047)
    ap.add_argument("--smoothing", default="mkn")
    ap.add_argument("--addk", type=float, default=0.1)
    ap.add_argument("--min-context-count", type=int, default=MIN_CONTEXT_COUNT)
    # backbone
    ap.add_argument("--model", default="/root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone-dtype", default="float32")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--chunk-tokens", type=int, default=4096)
    ap.add_argument("--head-chunk", type=int, default=256)
    args = ap.parse_args()

    if args.stage in ("counts", "all"):
        stage_counts(args)
    if args.stage in ("backbone", "all"):
        stage_backbone(args)
    if args.stage in ("analyze", "all"):
        stage_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
