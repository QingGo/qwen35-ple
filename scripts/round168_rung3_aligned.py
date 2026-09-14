#!/usr/bin/env python
"""Round 168 Stage 1.5g -- the aligned rung-3 (row content) layer.

Two jobs, both frozen in ``docs/round-168-stage1.5g-preregistration.md``:

1. ``--export-positions``: rebuild, from the Stage 1.5c artifacts alone, the
   EXACT position set those measurements scored.  The alignment is the three
   lines the margin analysis itself uses::

       valid  = isfinite(cnt_nll) & isfinite(bb_final) & isfinite(bb_lens)
       mask   = zeros(n, bool);  mask[positions] = True
       valid &= mask
       aligned = flatnonzero(valid)          # indices into the eval stream

   This is deterministic and needs no GPU and no row table.  The count is
   checked against the ``n_scored_positions`` Stage 1.5c recorded, so a silent
   drift in the definition cannot pass unnoticed.

2. ``--aggregate``: read the probe JSONs (one per K), compute
   ``R = probe_raw_rows top-1 / count_trigram top-1`` and apply the frozen
   section 3 rule through :mod:`qwen35_ple.rung3_verdict`.

Nothing here decides anything on its own; the verdict module owns the rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen35_ple.rung3_verdict import (
    PRIMARY_K,
    SENSITIVITY_K,
    apply_rule,
    collect_inputs,
    render_markdown,
)


def log(msg: str) -> None:
    print(f"[rung3] {msg}", flush=True)


def aligned_positions(counts_path: Path, backbone_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """The Stage 1.5c position set, rebuilt from its own two artifacts."""
    c = np.load(counts_path)
    b = np.load(backbone_path)
    cnt = np.asarray(c["cnt_nll"], dtype=np.float64)
    final = np.asarray(b["bb_final"], dtype=np.float64)
    lens = np.asarray(b["bb_lens"], dtype=np.float64)
    if not (cnt.shape == final.shape == lens.shape):
        raise SystemExit(
            f"length mismatch: cnt {cnt.shape} final {final.shape} lens {lens.shape}"
        )
    valid = np.isfinite(cnt) & np.isfinite(final) & np.isfinite(lens)
    mask = np.zeros(cnt.shape[0], dtype=bool)
    mask[np.asarray(c["positions"], dtype=np.int64)] = True
    valid &= mask
    pos = np.flatnonzero(valid).astype(np.int64)
    meta = {
        "counts_npz": str(counts_path),
        "backbone_npz": str(backbone_path),
        "n_eval_tokens": int(cnt.shape[0]),
        "n_aligned": int(pos.size),
        "eval_npy": str(c["eval_npy"]),
        "train_npy": str(c["train_npy"]),
        "min_context_count": 5,
        "first_position": int(pos[0]) if pos.size else None,
        "last_position": int(pos[-1]) if pos.size else None,
    }
    return pos, meta


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def cmd_export(args: argparse.Namespace) -> int:
    pos, meta = aligned_positions(Path(args.counts), Path(args.backbone))
    recorded = None
    margin_json = Path(args.margin_json) if args.margin_json else None
    if margin_json is not None and margin_json.exists():
        recorded = int(_load_json(margin_json)["n_scored_positions"])
        meta["recorded_n_scored_positions"] = recorded
        meta["matches_recorded"] = bool(recorded == pos.size)
        if recorded != pos.size:
            raise SystemExit(
                f"ALIGNMENT DRIFT: rebuilt {pos.size:,} positions but Stage 1.5c "
                f"recorded {recorded:,}.  Refusing to write a set that is not the one "
                f"the margin measurements used."
            )
        log(f"alignment verified against 1.5c: {pos.size:,} positions (exact match)")
    else:
        log(f"WARNING: no margin json given, cannot verify against 1.5c ({pos.size:,} positions)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # A position whose +1 target falls off the end of the stream cannot be
    # scored.  Stage 1.5c's mask is over scored positions and does not know about
    # this, so it is exactly one position per stream; it is removed here (the
    # probe requires t + 1 < T) and the count is recorded so the arithmetic closes.
    t_eval = int(meta["n_eval_tokens"])
    tail_mask = pos + 1 >= t_eval
    tail = int(tail_mask.sum())
    usable = pos[~tail_mask]
    np.save(out, usable)
    meta["dropped_no_next_token"] = tail
    meta["n_usable_for_next_token"] = int(usable.size)
    meta["saved_positions"] = int(usable.size)
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    log(f"wrote {out} and {meta_path}")
    log(
        f"  aligned={pos.size:,}  no-next-token dropped={tail}  saved={usable.size:,}"
    )
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    per_k: dict[int, dict[str, Any]] = {}
    for spec in args.probe:
        if "=" not in spec:
            raise SystemExit(f"--probe expects K=path, got {spec!r}")
        k_str, path_str = spec.split("=", 1)
        path = Path(path_str)
        if not path.exists():
            raise SystemExit(f"probe JSON not found: {path}")
        per_k[int(k_str)] = _load_json(path)
        log(f"loaded K={k_str} from {path}")

    provenance = {
        "positions_npy": str(args.positions),
        "eval_npy": str(args.eval_tokens),
        "primary_k": PRIMARY_K,
        "sensitivity_k": SENSITIVITY_K,
    }
    result = apply_rule(collect_inputs(per_k, provenance=provenance))
    result["tag"] = args.tag

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    md = render_markdown(result)
    out_md = Path(args.out_md) if args.out_md else out_json.with_suffix(".md")
    out_md.write_text(md)
    log(f"wrote {out_json} and {out_md}")
    print()
    print(md)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    e = sub.add_parser("export-positions", help="rebuild the 1.5c aligned position set")
    e.add_argument("--counts", required=True)
    e.add_argument("--backbone", required=True)
    e.add_argument("--margin-json", default="", help="1.5c margin json, for the drift check")
    e.add_argument("--out", required=True, help="output .npy of int64 positions")
    e.set_defaults(func=cmd_export)

    a = sub.add_parser("aggregate", help="apply the frozen section 3 rule")
    a.add_argument("--tag", default="wiki")
    a.add_argument("--probe", action="append", required=True, metavar="K=PATH",
                   help="probe JSON for one K; pass twice for K=5000 and K=1000")
    a.add_argument("--positions", default="")
    a.add_argument("--eval-tokens", default="")
    a.add_argument("--out-json", required=True)
    a.add_argument("--out-md", default="")
    a.set_defaults(func=cmd_aggregate)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
