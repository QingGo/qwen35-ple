#!/usr/bin/env python3
"""Round 167 Stage 1a: apply the pre-registered decision rule.

The rule is frozen in
``docs/round-167-stage1-effective-depth-preregistration.md`` section 4 and was
written before any number existed.  It is reproduced here verbatim so that the
verdict cannot drift with the data:

    IF mean TV(off, real) <= 1e-6            => UNDERPOWERED
    ELSE IF >=2 instruments have gain_real >= 2 and gain_shuf < gain_real
                                             => DEPTH_FREED
    ELSE IF gain_real >= 2 and gain_shuf >= gain_real
                                             => PERTURBATION_ONLY
    ELSE IF all instruments |gain_real| < 2  => DEPTH_NULL
    ELSE                                     => MIXED

where ``gain = ED(off) - ED(real)`` and ED is the effective-depth layer index
(first layer at which the instrument says the prediction has stabilised).  A
positive gain means injection moved the transition EARLIER, i.e. the graft
freed depth.

Usage::

    python scripts/round167_stage1a_verdict.py \
      --input outputs/round167/effective-depth-0.8B.json \
      --input outputs/round167/effective-depth-2B.json \
      --input outputs/round167/effective-depth-4B.json \
      --output outputs/round167/stage1a-verdict.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Frozen in the pre-registration; do not tune after seeing results.
DEPTH_SHIFT_LAYERS = 2
UNDERPOWERED_TV = 1e-6
INSTRUMENTS = ("from_kl_half_max", "from_top5_overlap_0.3", "from_residual_cosine")

VERDICTS = ("UNDERPOWERED", "DEPTH_FREED", "PERTURBATION_ONLY", "DEPTH_NULL", "MIXED")


def decide(report: dict) -> dict:
    """Apply the frozen rule to one backbone's report."""
    sanity = report["sanity"]
    tv = sanity.get("mean_total_variation_off_vs_real")
    ed = report["verdict_inputs"]["effective_depth_by_condition"]

    gains: dict[str, dict[str, int | None]] = {}
    for inst in INSTRUMENTS:
        off, real, shuf = ed["off"][inst], ed["real"][inst], ed["shuf"][inst]
        gains[inst] = {
            "off": off,
            "real": real,
            "shuf": shuf,
            "gain_real": None if off is None or real is None else int(off - real),
            "gain_shuf": None if off is None or shuf is None else int(off - shuf),
        }

    if tv is None or tv <= UNDERPOWERED_TV:
        verdict = "UNDERPOWERED"
    else:
        freed = [
            i
            for i, g in gains.items()
            if g["gain_real"] is not None
            and g["gain_real"] >= DEPTH_SHIFT_LAYERS
            and g["gain_shuf"] is not None
            and g["gain_shuf"] < g["gain_real"]
        ]
        any_real_shift = [
            i
            for i, g in gains.items()
            if g["gain_real"] is not None and g["gain_real"] >= DEPTH_SHIFT_LAYERS
        ]
        all_null = all(
            g["gain_real"] is not None and abs(g["gain_real"]) < DEPTH_SHIFT_LAYERS
            for g in gains.values()
        )
        if len(freed) >= 2:
            verdict = "DEPTH_FREED"
        elif any_real_shift and all(
            g["gain_shuf"] is not None and g["gain_real"] is not None
            and g["gain_shuf"] >= g["gain_real"]
            for g in gains.values()
            if g["gain_real"] is not None and g["gain_real"] >= DEPTH_SHIFT_LAYERS
        ):
            verdict = "PERTURBATION_ONLY"
        elif all_null:
            verdict = "DEPTH_NULL"
        else:
            verdict = "MIXED"

    return {
        "verdict": verdict,
        "mean_tv_off_vs_real": tv,
        "injection_is_live": sanity.get("injection_is_live"),
        "n_items": report["config"]["max_items"],
        "n_layers": report["config"]["n_layers"],
        "injection_layer": report["config"]["layer"],
        "gains": gains,
        "freed_instruments": [
            i
            for i in INSTRUMENTS
            if gains[i]["gain_real"] is not None
            and gains[i]["gain_real"] >= DEPTH_SHIFT_LAYERS
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out: dict = {
        "rule": {
            "depth_shift_layers": DEPTH_SHIFT_LAYERS,
            "underpowered_tv": UNDERPOWERED_TV,
            "instruments": list(INSTRUMENTS),
            "source": "docs/round-167-stage1-effective-depth-preregistration.md section 4",
        },
        "backbones": {},
    }

    for spec in args.input:
        p = Path(spec)
        if not p.exists():
            out["backbones"][p.stem] = {"error": "missing"}
            continue
        report = json.loads(p.read_text())
        name = report["config"]["model"].rstrip("/").split("/")[-1]
        out["backbones"][name] = decide(report)

    out["verdict_counts"] = {
        v: sum(1 for b in out["backbones"].values() if b.get("verdict") == v)
        for v in VERDICTS
    }

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'backbone':<24} {'verdict':<18} {'tv':>8}  gains (off-real / off-shuf)")
    for name, b in out["backbones"].items():
        if "error" in b:
            print(f"{name:<24} {b['error']}")
            continue
        g = b["gains"]
        pretty = "  ".join(
            f"{i.split('_')[1][:6]}:{g[i]['gain_real']}/{g[i]['gain_shuf']}"
            for i in INSTRUMENTS
        )
        print(f"{name:<24} {b['verdict']:<18} {b['mean_tv_off_vs_real']:>8.4f}  {pretty}")
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
