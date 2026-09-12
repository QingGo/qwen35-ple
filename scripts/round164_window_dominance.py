#!/usr/bin/env python3
"""Round 164 Part B: is the injected vector a property of the template or the item?

Pre-registration: ``docs/round-164-window-composition-preregistration.md``.
The rule below was frozen before this file existed; do not edit the thresholds
or the verdict labels to fit a result.

Part A measured, with no model, that the 12-token addressing window at the
answer position is 10/12 fixed template for BoolQ and 3/12 for nq/triviaqa.
That predicts: for a *fixed* reader, the injected vector ``c`` should be more
similar across BoolQ items than across the short-answer items -- while the
hidden states ``h``, which see the whole passage, are not.

Statistics (mean pairwise cosine within a group)::

    S_c(g), S_h(g)          g in {boolq, short}, short = nq u triviaqa
    dS_c = S_c(boolq) - S_c(short)
    dS_h = S_h(boolq) - S_h(short)

Verdict, by permutation test on task labels (1000 permutations, group sizes
fixed at 200/400)::

    WINDOW_DOMINATED   dS_c > 0 and p_c < 0.01 and dS_h <= 0
    ITEM_DOMINATED     dS_c <= 0
    CONFOUNDED         dS_c > 0 and p_c < 0.01 but dS_h > 0

No generation and no training: one forward pass per item per reader, reading
``model._last_reader_contribution`` and ``model._last_reader_hidden`` at the
last prompt position.

Usage::

    python scripts/round164_window_dominance.py \
      --model /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B \
      --rows-dir /root/autodl-tmp/qwen35-ple/qwen38-rows \
      --qa-file data/qa-standard/eval-600b.jsonl \
      --reader wiki=$OUT/reader-wiki-seed0.pt \
      --reader wiki-shuf=$OUT/reader-wiki-shuf-seed0.pt \
      --output $OUT/window-dominance.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# ``run_phase0`` and the reader hook pull in torch, so they are imported inside
# ``main`` -- the statistics helpers below are pure numpy and are unit-tested
# without a GPU (tests/test_round164_statistics.py).

PROMPT = "Question: {question}\nAnswer:"
BOOLQ_PROMPT = "Question: {question}\nAnswer with one word, Yes or No:"

N_PERMUTATIONS = 1000
SHORT_TASKS = ("nq", "triviaqa")


def _load_all_items(path: Path) -> list[dict]:
    """Every item, in file order -- no sampling, so group sizes match the doc."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _mean_pairwise_cosine(mat: np.ndarray, idx: np.ndarray) -> float:
    """Mean cosine over unordered pairs within ``idx``.

    ``idx`` is a boolean mask, so the group size is ``sub.shape[0]`` -- *not*
    ``len(idx)``, which would be the length of the full item list.
    """
    sub = mat[idx]
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-norm vector in group; hook is not recording")
    n = sub.shape[0]
    if n < 2:
        return float("nan")
    unit = sub / norms
    gram = unit @ unit.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(gram[iu]))


def _delta(mat: np.ndarray, is_boolq: np.ndarray) -> float:
    return _mean_pairwise_cosine(mat, is_boolq) - _mean_pairwise_cosine(mat, ~is_boolq)


def _permutation_p(mat: np.ndarray, is_boolq: np.ndarray, observed: float, seed: int) -> float:
    """One-sided p for dS > 0 under exchangeable task labels, sizes held fixed."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(N_PERMUTATIONS):
        if _delta(mat, rng.permutation(is_boolq)) >= observed:
            hits += 1
    return (hits + 1) / (N_PERMUTATIONS + 1)


def _verdict(ds_c: float, p_c: float, ds_h: float) -> str:
    if ds_c <= 0:
        return "ITEM_DOMINATED"
    if p_c < 0.01 and ds_h <= 0:
        return "WINDOW_DOMINATED"
    if p_c < 0.01 and ds_h > 0:
        return "CONFOUNDED"
    return "INCONCLUSIVE"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", default="/root/autodl-tmp/qwen35-ple/models/qwen38_ple")
    parser.add_argument("--rows-dir", required=True)
    parser.add_argument("--official-reader-path", default="data/official_ple_reader.pt")
    parser.add_argument("--qa-file", default="data/qa-standard/eval-600b.jsonl")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--backbone-dtype", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reader", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "reader NAME gets control (row-permuted) rows, reproducing the "
            "round-162 wiki-shuf condition: same checkpoint, same addressing, "
            "shuffled e_t.  Permutation seed is seed*1000 + item index, matching "
            "run_phase0's control path exactly."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--vectors-dir", default=None, help="optional .npy dump per reader")
    args = parser.parse_args(argv)

    items = _load_all_items(Path(args.qa_file))
    tasks = np.array([it.get("task", "?") for it in items])
    is_boolq = tasks == "boolq"
    n_boolq = int(is_boolq.sum())
    n_short = int((~is_boolq).sum())
    print(f"[round164B] {len(items)} items: boolq={n_boolq} short={n_short}", flush=True)
    unknown = set(tasks) - {"boolq", *SHORT_TASKS}
    if unknown:
        raise SystemExit(f"unexpected tasks in --qa-file: {sorted(unknown)}")
    if n_boolq == 0 or n_short == 0:
        raise SystemExit("both groups must be non-empty for the permutation test")

    import torch

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_phase0 as p0

    from qwen35_ple.reader import install_reader_hook
    from qwen35_ple.real_ple import resolve_ple_weight_scale

    tokenizer, model = p0._load_model(args.model, args.device, args.backbone_dtype)
    for param in model.parameters():
        param.requires_grad_(False)

    import torch

    scale = resolve_ple_weight_scale(model_dir=args.model_dir, scale=None)
    store = p0._QAEtStore(args.rows_dir, scale)
    print(f"[round164B] PLE scale={scale:.6g}", flush=True)

    per_reader: dict[str, dict] = {}
    verdicts: dict[str, str] = {}

    for spec in args.reader:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--reader must be NAME=PATH, got {spec!r}")
        reader, _extra = p0.load_reader_with_extra(Path(path), device=args.device)
        reader = reader.to(args.device)
        reader.eval()

        handle = install_reader_hook(model, args.layer, reader)
        for stale in ("_last_reader_contribution", "_last_reader_hidden"):
            if hasattr(model, stale):
                delattr(model, stale)

        contribs: list[np.ndarray] = []
        hiddens: list[np.ndarray] = []
        use_control = name in set(args.control)
        with torch.no_grad():
            for idx, item in enumerate(items):
                ids = p0._qa_prompt_ids(tokenizer, item, PROMPT, BOOLQ_PROMPT)
                et = store.fetch(np.asarray(ids, dtype=np.int64))
                if use_control:
                    # Same permutation run_phase0 applies in its control mode.
                    perm_rng = np.random.default_rng(args.seed * 1000 + idx)
                    et = et[perm_rng.permutation(len(et))]
                model._current_ple_e_t = (
                    torch.from_numpy(et[None, :, :]).float().to(args.device)
                )
                inp = torch.tensor([ids], dtype=torch.long, device=args.device)
                model(input_ids=inp)
                pos = len(ids) - 1
                contribs.append(model._last_reader_contribution[0, pos].float().cpu().numpy())
                hiddens.append(model._last_reader_hidden[0, pos].float().cpu().numpy())
        handle.remove()

        C = np.stack(contribs).astype(np.float64)
        H = np.stack(hiddens).astype(np.float64)

        s_c_boolq = _mean_pairwise_cosine(C, is_boolq)
        s_c_short = _mean_pairwise_cosine(C, ~is_boolq)
        s_h_boolq = _mean_pairwise_cosine(H, is_boolq)
        s_h_short = _mean_pairwise_cosine(H, ~is_boolq)
        ds_c = s_c_boolq - s_c_short
        ds_h = s_h_boolq - s_h_short
        p_c = _permutation_p(C, is_boolq, ds_c, args.seed)
        p_h = _permutation_p(H, is_boolq, ds_h, args.seed)

        # Anti-vacuity: a constant c would give S_c == 1.0 in both groups.
        degenerate = bool(np.isclose(s_c_boolq, 1.0) and np.isclose(s_c_short, 1.0))

        verdict = _verdict(ds_c, p_c, ds_h)
        verdicts[name] = verdict
        per_reader[name] = {
            "checkpoint": path,
            "control_rows": use_control,
            "n": len(items),
            "S_c_boolq": s_c_boolq,
            "S_c_short": s_c_short,
            "S_h_boolq": s_h_boolq,
            "S_h_short": s_h_short,
            "dS_c": ds_c,
            "dS_h": ds_h,
            "p_c": p_c,
            "p_h": p_h,
            "cnorm_mean": float(np.mean(np.linalg.norm(C, axis=1))),
            "cnorm_std": float(np.std(np.linalg.norm(C, axis=1))),
            "hnorm_mean": float(np.mean(np.linalg.norm(H, axis=1))),
            "degenerate_constant_c": degenerate,
            "verdict": verdict,
        }
        print(
            f"[round164B] {name}: S_c boolq={s_c_boolq:.4f} short={s_c_short:.4f} "
            f"(dS_c={ds_c:+.4f} p={p_c:.4f}) | "
            f"S_h boolq={s_h_boolq:.4f} short={s_h_short:.4f} (dS_h={ds_h:+.4f}) "
            f"-> {verdict}",
            flush=True,
        )

        if args.vectors_dir:
            vd = Path(args.vectors_dir)
            vd.mkdir(parents=True, exist_ok=True)
            np.save(vd / f"{name}-c.npy", C)
            np.save(vd / f"{name}-h.npy", H)

        if args.device != "cpu":
            torch.cuda.empty_cache()

    payload = {
        "round": "164B",
        "preregistration": "docs/round-164-window-composition-preregistration.md",
        "model": args.model,
        "layer": args.layer,
        "backbone_dtype": args.backbone_dtype,
        "qa_file": args.qa_file,
        "prompt_template": PROMPT,
        "boolq_prompt_template": BOOLQ_PROMPT,
        "n_permutations": N_PERMUTATIONS,
        "control_readers": list(args.control),
        "n_boolq": n_boolq,
        "n_short": n_short,
        "tasks": {t: int((tasks == t).sum()) for t in sorted(set(tasks))},
        "per_reader": per_reader,
        "verdicts": verdicts,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[round164B] wrote {args.output}")

    if all(v == "ITEM_DOMINATED" for v in verdicts.values()):
        print("[round164B] P1 refuted: the injected vector is not more template-like "
              "for BoolQ than for the short-answer tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
