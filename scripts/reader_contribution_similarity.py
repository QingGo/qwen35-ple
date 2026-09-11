#!/usr/bin/env python3
"""Round 162: does the training corpus change the vector actually injected?

The decisive arm comparison of round 162 is 1 (PURE_WIKI) vs 2 (PURE_CODE) vs
3 (PURE_STEM).  If those arms produce indistinguishable generation formats, two
very different worlds are consistent with the data:

* **content is irrelevant** -- the readers compute genuinely different vectors
  and the model simply does not care about their content; or
* **the manipulation never reached the injection point** -- the three readers
  happen to compute (nearly) the same vector *on these prompts*, so no
  comparison of their effects could ever have shown a difference.

This script separates them.  It installs each reader in turn, runs the *same*
prompt set through the *same* frozen backbone, and records
``model._last_reader_contribution`` -- the exact tensor the reader hook adds to
the residual stream -- at the pre-answer position.  It then reports, pairwise:

* mean per-item **cosine similarity** between the two arms' injected vectors,
* mean per-item **relative L2 distance** ``‖c_A - c_B‖ / ‖c_A‖``,
* the two arms' mean norms (the magnitude control, computed on identical
  inputs instead of during decoding).

A near-1 cosine with a near-0 relative distance means the corpus manipulation
produced the same injection and the format question is unanswerable *by this
manipulation*; a low cosine means the readers are functionally different and a
null format result is a real statement about the model.

Usage::

    python scripts/reader_contribution_similarity.py \
      --model /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B \
      --rows-dir /root/autodl-tmp/qwen35-ple/qwen38-rows \
      --qa-file data/qa-standard/eval.jsonl --max-items 192 \
      --reader wiki=$OUT/reader-wiki-seed0.pt \
      --reader code=$OUT/reader-code-seed0.pt \
      --reader stem=$OUT/reader-stem-seed0.pt \
      --output $OUT/contribution-similarity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_phase0 as p0  # noqa: E402  (needs scripts/ on sys.path first)
from qwen35_ple.reader import install_reader_hook  # noqa: E402


def _load_items(path: Path, max_items: int) -> list[dict]:
    items = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if max_items <= 0:
        return items
    # Balanced head: take the same number from each task so a 192-item sample is
    # 64/64/64 rather than 192 BoolQ passages.
    by_task: dict[str, list[dict]] = {}
    for item in items:
        by_task.setdefault(item.get("task", "?"), []).append(item)
    per_task = max(1, max_items // max(1, len(by_task)))
    out: list[dict] = []
    for task in sorted(by_task):
        out.extend(by_task[task][:per_task])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", default="/root/autodl-tmp/qwen35-ple/models/qwen38_ple")
    parser.add_argument("--rows-dir", required=True)
    parser.add_argument("--official-reader-path", default="data/official_ple_reader.pt")
    parser.add_argument("--qa-file", default="data/qa-standard/eval.jsonl")
    parser.add_argument("--max-items", type=int, default=192)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--backbone-dtype", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--reader", action="append", required=True, metavar="NAME=PATH",
        help="reader checkpoint to measure (repeatable)",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    prompt_template = "Question: {question}\nAnswer:"
    boolq_template = "Question: {question}\nAnswer with one word, Yes or No:"

    items = _load_items(Path(args.qa_file), args.max_items)
    print(f"[contrib-sim] {len(items)} prompts", flush=True)

    tokenizer, model = p0._load_model(args.model, args.device, args.backbone_dtype)
    for param in model.parameters():
        param.requires_grad_(False)

    import torch

    from qwen35_ple.real_ple import resolve_ple_weight_scale

    scale = resolve_ple_weight_scale(model_dir=args.model_dir, scale=None)
    store = p0._QAEtStore(args.rows_dir, scale)
    print(f"[contrib-sim] PLE scale={scale:.6g}", flush=True)

    results: dict[str, dict] = {}
    vectors: dict[str, np.ndarray] = {}
    norms: dict[str, np.ndarray] = {}
    hidden_norms: dict[str, np.ndarray] = {}

    for spec in args.reader:
        name, _, path = spec.partition("=")
        # The checkpoint carries the reader name/version/constructor config, so
        # the arm is rebuilt exactly as it was trained -- no architecture is
        # re-specified here that could silently differ from the arm's own.
        reader, _extra = p0.load_reader_with_extra(Path(path), device=args.device)
        reader = reader.to(args.device)
        reader.eval()

        handle = install_reader_hook(model, args.layer, reader)
        for stale in ("_last_reader_contribution", "_last_reader_hidden"):
            if hasattr(model, stale):
                delattr(model, stale)

        contribs: list[np.ndarray] = []
        cnorms: list[float] = []
        hnorms: list[float] = []
        with torch.no_grad():
            for item in items:
                ids = p0._qa_prompt_ids(
                    tokenizer, item, prompt_template, boolq_template
                )
                et = store.fetch(np.asarray(ids, dtype=np.int64))
                model._current_ple_e_t = (
                    torch.from_numpy(et[None, :, :]).float().to(args.device)
                )
                inp = torch.tensor([ids], dtype=torch.long, device=args.device)
                model(input_ids=inp)
                contrib = model._last_reader_contribution
                hidden = model._last_reader_hidden
                pos = len(ids) - 1
                c = contrib[0, pos].float().cpu().numpy()
                h = hidden[0, pos].float().cpu().numpy()
                contribs.append(c)
                cnorms.append(float(np.linalg.norm(c)))
                hnorms.append(float(np.linalg.norm(h)))
        handle.remove()
        arr = np.stack(contribs)
        vectors[name] = arr
        norms[name] = np.asarray(cnorms)
        hidden_norms[name] = np.asarray(hnorms)
        results[name] = {
            "checkpoint": path,
            "n": len(contribs),
            "cnorm_mean": float(np.mean(cnorms)),
            "cnorm_median": float(np.median(cnorms)),
            "hnorm_mean": float(np.mean(hnorms)),
            "ratio_mean": float(np.mean(norms[name] / hidden_norms[name])),
            "ratio_median": float(np.median(norms[name] / hidden_norms[name])),
        }
        print(
            f"[contrib-sim] {name}: ||c||={results[name]['cnorm_mean']:.4f} "
            f"||h||={results[name]['hnorm_mean']:.4f} ratio={results[name]['ratio_mean']:.4f}",
            flush=True,
        )
        if args.device != "cpu":
            torch.cuda.empty_cache()

    names = list(vectors)
    pairwise: dict[str, dict] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            va, vb = vectors[a], vectors[b]
            cos = np.sum(va * vb, axis=1) / (
                np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-12
            )
            rel = np.linalg.norm(va - vb, axis=1) / (np.linalg.norm(va, axis=1) + 1e-12)
            pairwise[f"{a}|{b}"] = {
                "cosine_mean": float(np.mean(cos)),
                "cosine_sem": float(np.std(cos, ddof=1) / np.sqrt(len(cos))),
                "cosine_median": float(np.median(cos)),
                "cosine_min": float(np.min(cos)),
                "rel_l2_mean": float(np.mean(rel)),
                "rel_l2_sem": float(np.std(rel, ddof=1) / np.sqrt(len(rel))),
                "norm_ratio_b_over_a": float(np.mean(norms[b] / norms[a])),
            }
            print(
                f"[contrib-sim] {a} vs {b}: cos={pairwise[f'{a}|{b}']['cosine_mean']:.4f} "
                f"relL2={pairwise[f'{a}|{b}']['rel_l2_mean']:.4f}",
                flush=True,
            )

    payload = {
        "model": args.model,
        "backbone_dtype": args.backbone_dtype,
        "layer": args.layer,
        "qa_file": args.qa_file,
        "prompt_template": prompt_template,
        "boolq_prompt_template": boolq_template,
        "n_items": len(items),
        "per_arm": results,
        "pairwise": pairwise,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[contrib-sim] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
