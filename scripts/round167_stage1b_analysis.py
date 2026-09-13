#!/usr/bin/env python3
"""Round 167 Stage 1b: apply the pre-registered read-out-collapse decision rule.

The rule is frozen in ``docs/round-167-stage1b-preregistration.md`` section 4
before any provenance number was read, and is reproduced here verbatim:

    baseline = PR(c_t) @ offset 12 for the `prod` variant   (historical 1.020)
    IF a variant has PR(e_t)@12 < 50                 => that variant INVALID
    IF |PR(c_t)[prod] - 1.020| > 0.05                => BASELINE_NOT_REPRODUCED
    ELSE IF some v has PR(c_t)[v] >= 5*baseline and
            PR(c_t)[v] >= 5.0                        => COLLAPSE_IS_RECIPE
    ELSE IF max_v PR(c_t)[v] < 2.0                   => COLLAPSE_IS_ARCHITECTURAL
    ELSE                                             => PARTIAL

The question it answers: round-165A measured the read-out collapsing a
137-dimensional input to 1.020, and three independent results (the official
`conv_zero_init`, arXiv 2510.06954's condensation->rank-collapse, and the
ICML 2026 implicit-low-rank-bias-from-depth result) all predict exactly that for
a zero-initialised 2-layer MLP -- which is what production trains. So is the
collapse a *recipe* artifact or an *architectural* limit? Nothing in this repo
had ever perturbed either knob (round-167 TD-3a: `zero_init_out` was a literal).

Usage::

    python scripts/round167_stage1b_analysis.py \
      --dir outputs/round167/stage1b \
      --variants prod nozero linear linear-nozero \
      --output outputs/round167/stage1b-analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HISTORICAL_BASELINE_PR_C_T = 1.020
BASELINE_TOLERANCE = 0.05
MIN_E_T_PR = 50.0
RECIPE_MULTIPLE = 5.0
RECIPE_ABSOLUTE = 5.0
ARCHITECTURAL_CEILING = 2.0
OFFSET = "12"
STAGES = ("e_t", "value_proj", "branch_sum", "c_t")

VERDICTS = (
    "INVALID",
    "BASELINE_NOT_REPRODUCED",
    "COLLAPSE_IS_RECIPE",
    "COLLAPSE_IS_ARCHITECTURAL",
    "PARTIAL",
)


def extract(provenance: dict, offset: str = OFFSET) -> dict[str, dict[str, float]]:
    """Pull PR (and the cosine) per stage out of a round-165 provenance report."""
    block = provenance["by_offset"][offset]["overall"]
    return {
        stage: {
            "PR": float(block[stage]["PR"]),
            "cos_real_vs_shuf": float(block[stage]["cos_real_vs_shuf_mean"]),
            "dim": int(block[stage]["dim"]),
            "duplicate_row_fraction": float(block[stage]["duplicate_row_fraction"]),
        }
        for stage in STAGES
    }


def decide(per_variant: dict[str, dict[str, dict[str, float]]]) -> dict:
    """Apply the frozen rule to per-variant stage statistics."""
    if "prod" not in per_variant:
        return {"verdict": "INVALID", "reason": "no `prod` variant to use as baseline"}

    invalid = [
        v
        for v, stages in per_variant.items()
        if stages["e_t"]["PR"] < MIN_E_T_PR
    ]
    prod_c_t = per_variant["prod"]["c_t"]["PR"]

    if invalid:
        return {
            "verdict": "INVALID",
            "reason": f"addressing sanity failed (PR(e_t)<{MIN_E_T_PR}) for {invalid}",
            "invalid_variants": invalid,
        }

    # The tolerance is inclusive: 1.020 + 0.05 must be accepted, and in binary
    # floating point it evaluates to 0.050000000000000044, so a bare `>`
    # rejects a value that is exactly on the stated boundary.
    if abs(prod_c_t - HISTORICAL_BASELINE_PR_C_T) > BASELINE_TOLERANCE + 1e-9:
        return {
            "verdict": "BASELINE_NOT_REPRODUCED",
            "reason": (
                f"prod PR(c_t)@offset {OFFSET} = {prod_c_t:.3f}, expected "
                f"{HISTORICAL_BASELINE_PR_C_T} +- {BASELINE_TOLERANCE}"
            ),
        }

    recipe = [
        v
        for v, stages in per_variant.items()
        if stages["c_t"]["PR"] >= RECIPE_MULTIPLE * prod_c_t
        and stages["c_t"]["PR"] >= RECIPE_ABSOLUTE
    ]
    if recipe:
        return {
            "verdict": "COLLAPSE_IS_RECIPE",
            "reason": (
                f"PR(c_t) rises to {max(per_variant[v]['c_t']['PR'] for v in recipe):.3f} "
                f"from a baseline of {prod_c_t:.3f} (>= {RECIPE_MULTIPLE}x and "
                f">= {RECIPE_ABSOLUTE})"
            ),
            "relieved_variants": recipe,
        }

    ceiling = max(stages["c_t"]["PR"] for stages in per_variant.values())
    if ceiling < ARCHITECTURAL_CEILING:
        return {
            "verdict": "COLLAPSE_IS_ARCHITECTURAL",
            "reason": (
                f"every configuration stays below PR(c_t) {ARCHITECTURAL_CEILING}: "
                f"max is {ceiling:.3f}"
            ),
        }

    return {
        "verdict": "PARTIAL",
        "reason": f"best PR(c_t) is {ceiling:.3f}: neither relieved nor flat",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = Path(args.dir)
    per_variant: dict[str, dict[str, dict[str, float]]] = {}
    missing: list[str] = []
    for v in args.variants:
        p = d / f"provenance-{v}.json"
        if not p.exists():
            missing.append(v)
            continue
        per_variant[v] = extract(json.loads(p.read_text()))

    report: dict = {
        "rule": {
            "offset": OFFSET,
            "historical_baseline_pr_c_t": HISTORICAL_BASELINE_PR_C_T,
            "baseline_tolerance": BASELINE_TOLERANCE,
            "min_e_t_pr": MIN_E_T_PR,
            "recipe_multiple": RECIPE_MULTIPLE,
            "recipe_absolute": RECIPE_ABSOLUTE,
            "architectural_ceiling": ARCHITECTURAL_CEILING,
            "source": "docs/round-167-stage1b-preregistration.md section 4",
        },
        "missing": missing,
        "per_variant": per_variant,
    }

    if missing:
        report["verdict"] = {
            "verdict": "INCOMPLETE",
            "reason": f"missing provenance for {missing}",
        }
    else:
        report["verdict"] = decide(per_variant)

    # Descriptive only: how much each stage attenuates the effective rank, and
    # whether the read-out geometry tracks the input at all.
    report["attenuation"] = {
        v: {
            "e_t_to_value_proj": _ratio(stages, "e_t", "value_proj"),
            "value_proj_to_branch_sum": _ratio(stages, "value_proj", "branch_sum"),
            "branch_sum_to_c_t": _ratio(stages, "branch_sum", "c_t"),
            "overall": _ratio(stages, "e_t", "c_t"),
        }
        for v, stages in per_variant.items()
    }

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{'variant':<16} {'PR(e_t)':>9} {'value_proj':>11} {'branch_sum':>11} {'c_t':>8}")
    for v, stages in per_variant.items():
        print(
            f"{v:<16} {stages['e_t']['PR']:>9.3f} {stages['value_proj']['PR']:>11.3f} "
            f"{stages['branch_sum']['PR']:>11.3f} {stages['c_t']['PR']:>8.3f}"
        )
    print(f"\nverdict: {report['verdict']['verdict']} -- {report['verdict']['reason']}")
    print(f"wrote {dst}")
    return 0


def _ratio(stages: dict[str, dict[str, float]], a: str, b: str) -> float | None:
    """``PR(a) / PR(b)``; the attenuation applied by the stage(s) between them."""
    pb = stages[b]["PR"]
    return float(stages[a]["PR"] / pb) if pb else None


if __name__ == "__main__":
    raise SystemExit(main())
