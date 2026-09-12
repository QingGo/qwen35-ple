#!/usr/bin/env python3
"""Round 165 (part A): where does the injected vector's collapse come from?

Pre-registration: ``docs/round-165-readout-repair-preregistration.md``.

The paper reports that the injected vector has an effective dimensionality of
1.001 over 600 prompts and concludes from a random-linear-map control that "the
collapse to 1.001 is the read-out's".  That control maps the *hidden state*,
not the addressed row, so it does not in fact localise the collapse: the
read-out's input is ``(h_t, e_t)``, and if ``e_t`` is itself the same row for
every item then a perfectly faithful read-out would still emit the same vector.

This script walks the reader's own arithmetic at the answer position and
measures the rank and the content-sensitivity of every intermediate stage::

    e_t  ->  value_proj(e_t)  ->  branch_sum  ->  c_t = out_proj(branch_sum)

with ``h_t`` beside it as the reference the paper already reports (1.480).
Content-sensitivity is ``cos(real, shuffled rows)`` at each stage, using the
same row permutation as ``run_phase0``'s control mode.

Usage::

    PYTHONPATH=src python scripts/round165_collapse_provenance.py \
      --model /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B \
      --rows-dir /root/autodl-tmp/qwen35-ple/qwen38-rows \
      --reader /root/autodl-tmp/qwen35-ple/outputs/round162-0.8B-nosft600/reader-wiki-seed0.pt \
      --output outputs/round165/collapse-provenance.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

PROMPT = "Question: {question}\nAnswer:"
BOOLQ_PROMPT = "Question: {question}\nAnswer with one word, Yes or No:"

# Ordered from the addressing (the read-out's input) to the injected vector.
STAGES = ("e_t", "value_proj", "branch_sum", "c_t", "h_t")
N_BOOTSTRAP = 2000


def participation_ratio(mat: np.ndarray) -> float:
    """``(sum lambda)^2 / sum lambda^2`` over the centered covariance.

    Returns ``nan`` when every row is identical, because the collapse is then
    total and the ratio is undefined rather than 1 -- reporting 1.0 there would
    hide the strongest possible form of the effect.
    """
    x = np.asarray(mat, dtype=np.float64)
    if x.shape[0] < 2:
        raise ValueError("participation ratio needs at least two rows")
    x = x - x.mean(axis=0, keepdims=True)
    gram = x @ x.T
    lam = np.clip(np.linalg.eigvalsh(gram), 0.0, None)
    denom = float(np.sum(lam**2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(lam) ** 2 / denom)


def bootstrap_ci(values: np.ndarray, n: int = N_BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.shape[0], size=(n, x.shape[0]))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def per_item_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine between two [n, d] arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    if np.any(na == 0) or np.any(nb == 0):
        raise ValueError("zero-norm row; the hook is not recording")
    return np.sum(a * b, axis=1) / (na * nb)


def n_distinct_rows(mat: np.ndarray) -> int:
    """How many byte-distinct rows the addressing actually returns."""
    return len({row.tobytes() for row in np.ascontiguousarray(mat)})


def stage_summary(
    real: np.ndarray, shuf: np.ndarray, seed: int = 0
) -> dict:
    cos = per_item_cosine(real, shuf)
    lo, hi = bootstrap_ci(cos, seed=seed)
    return {
        "PR": participation_ratio(real),
        "PR_is_undefined_all_rows_identical": bool(np.isnan(participation_ratio(real))),
        "cnorm_mean": float(np.linalg.norm(real, axis=1).mean()),
        "cos_real_vs_shuf_mean": float(cos.mean()),
        "cos_real_vs_shuf_ci95": [lo, hi],
        "n_distinct_rows": n_distinct_rows(real),
    }


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
    parser.add_argument(
        "--offsets",
        default="0,12,40",
        help=(
            "positions to measure, as token distances back from the answer "
            "position.  0 is the position whose logits the arms actually read; "
            "larger offsets sit inside the passage, where the addressed trigram "
            "is item-specific rather than the fixed template."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    offsets = sorted({int(x) for x in args.offsets.split(",") if x.strip() != ""})
    if not offsets or offsets[0] < 0:
        raise SystemExit(f"bad --offsets {args.offsets!r}")

    import torch

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_phase0 as p0

    from qwen35_ple.real_ple import resolve_ple_weight_scale

    items = [json.loads(line) for line in Path(args.qa_file).read_text().splitlines() if line.strip()]
    if args.max_items:
        items = items[: args.max_items]

    tokenizer, model = p0._load_model(args.model, args.device, args.backbone_dtype)
    for param in model.parameters():
        param.requires_grad_(False)

    scale = resolve_ple_weight_scale(model_dir=args.model_dir, scale=None)
    store = p0._QAEtStore(args.rows_dir, scale)
    reader, _extra = p0.load_reader_with_extra(Path(args.reader), device=args.device)
    reader = reader.to(args.device)
    reader.eval()
    print(f"[r165A] {len(items)} items, PLE scale={scale:.6g}", flush=True)

    layer_module = model.model.layers[args.layer]

    captured: dict[str, object] = {}

    def h_hook(module, inputs, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output
        return output

    def value_hook(module, inputs, output):
        captured["value_proj"] = output
        return output

    def out_hook(module, inputs, output):
        # The Sequential's input is branch_sum and its output is c_t.
        captured["branch_sum"] = inputs[0]
        captured["c_t"] = output
        return output

    handles = [
        layer_module.register_forward_hook(h_hook),
        reader.value_proj.register_forward_hook(value_hook),
        reader.out_proj.register_forward_hook(out_hook),
    ]

    # One entry per offset; every offset is read off the same reader call.
    rows: dict[int, dict[str, list[np.ndarray]]] = {
        off: {s: [] for s in STAGES} for off in offsets
    }
    shuf_rows: dict[int, dict[str, list[np.ndarray]]] = {
        off: {s: [] for s in STAGES} for off in offsets
    }
    tasks: list[str] = []

    with torch.no_grad():
        for idx, item in enumerate(items):
            ids = p0._qa_prompt_ids(tokenizer, item, PROMPT, BOOLQ_PROMPT)
            pos = len(ids) - 1
            et = store.fetch(np.asarray(ids, dtype=np.int64))

            captured.clear()
            model(input_ids=torch.tensor([ids], dtype=torch.long, device=args.device))
            h_all = captured["h"][0]
            h_np = h_all.float().cpu().numpy().astype(np.float64)

            perm_rng = np.random.default_rng(args.seed * 1000 + idx)
            perm = perm_rng.permutation(len(ids))

            for _cond, et_cur, sink in (
                ("real", et, rows),
                ("shuf", et[perm], shuf_rows),
            ):
                et_t = torch.from_numpy(et_cur[None]).float().to(args.device)
                for key in ("value_proj", "branch_sum", "c_t"):
                    captured.pop(key, None)
                reader(h_all[None], et_t)
                vp_full = captured["value_proj"][0].float().cpu().numpy().astype(np.float64)
                bs_full = captured["branch_sum"][0].float().cpu().numpy().astype(np.float64)
                c_full = captured["c_t"][0].float().cpu().numpy().astype(np.float64)

                for off in offsets:
                    p = max(0, pos - off)
                    sink[off]["e_t"].append(et_cur[p].astype(np.float64))
                    sink[off]["value_proj"].append(vp_full[p])
                    sink[off]["branch_sum"].append(bs_full[p])
                    sink[off]["c_t"].append(c_full[p])
                    sink[off]["h_t"].append(h_np[p])

            tasks.append(str(item.get("task")))
            if (idx + 1) % 100 == 0:
                print(f"[r165A] {idx + 1}/{len(items)}", flush=True)

    for handle in handles:
        handle.remove()

    tasks_arr = np.asarray(tasks)
    report: dict = {
        "round": "165A",
        "preregistration": "docs/round-165-readout-repair-preregistration.md",
        "model": args.model,
        "reader": args.reader,
        "layer": args.layer,
        "n_items": len(tasks),
        "tasks": dict(Counter(tasks)),
        "stage_order": list(STAGES),
        "offsets": offsets,
        "by_offset": {},
    }

    for off in offsets:
        block: dict = {"overall": {}, "within_task": {}}
        for stage in STAGES:
            real = np.stack(rows[off][stage])
            shuf = np.stack(shuf_rows[off][stage])
            block["overall"][stage] = {
                **stage_summary(real, shuf, seed=args.seed),
                "dim": int(real.shape[1]),
                "duplicate_row_fraction": float(1.0 - n_distinct_rows(real) / real.shape[0]),
            }
            block["within_task"][stage] = {
                t: stage_summary(real[tasks_arr == t], shuf[tasks_arr == t], seed=args.seed)
                for t in sorted(set(tasks))
            }
        report["by_offset"][str(off)] = block

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    for off in offsets:
        block = report["by_offset"][str(off)]
        print(f"--- offset {off} tokens back from the answer position ---")
        print(f"{'stage':13s} {'dim':>5s} {'PR':>8s} {'dup%':>7s} {'cos(r,s)':>9s} {'||x||':>8s}")
        for stage in STAGES:
            s = block["overall"][stage]
            print(
                f"{stage:13s} {s['dim']:5d} {s['PR']:8.3f} "
                f"{100 * s['duplicate_row_fraction']:7.1f} "
                f"{s['cos_real_vs_shuf_mean']:9.4f} {s['cnorm_mean']:8.3f}",
                flush=True,
            )
    print(f"[r165A] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
