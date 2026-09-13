#!/usr/bin/env python3
"""Round 167 Stage 0.1: run the nuisance audit over every regime we have.

This is the *evidence* half of TD-1.  ``src/qwen35_ple/metric_audit.py`` supplies
the statistic and ``tests/test_metric_audit.py`` proves it recovers known ground
truth on synthetic arms; this script applies it to the real round-162 / round-166
arms and prints, per regime and per arm pair, whether each headline metric is
reporting the arms or the generated length.

The strongest thing it shows is a *contrast within the same data*: on identical
arm pairs, ``prose_markers`` is confounded by length while ``chat_scaffold_rate``
is not.  A framework that flagged everything would be useless; this one
separates them.

Usage::

    python scripts/audit_metric_nuisance.py \
      --regime 0.8B=outputs/round162/0.8B \
      --regime 4B=outputs/round162/4B \
      --output outputs/round167/metric-nuisance-audit.json

Exit status is 0 unless ``--strict`` is passed, in which case any metric whose
measured verdict contradicts its declaration in ``METRIC_DECLARATIONS`` fails the
run.  ``--strict`` is what CI should eventually use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from classify_format import classify, domain_fingerprint

from qwen35_ple.metric_audit import (
    METRIC_DECLARATIONS,
    NUISANCE_SENSITIVE,
    paired_nuisance_contrast,
)

#: Arm pairs to contrast, and the marker each directional prediction reads.
#: Mirrors ``scripts/audit_directional_length_confound.py`` so the round-166
#: numbers and these are computed on the same pairs.
ARM_PAIRS = (
    ("wiki", "code"),
    ("stem", "wiki"),
    ("wiki-shuf", "wiki"),
    # The pair that produces the paper's headline format claim: zero injection
    # versus a real injection.  Sign convention is ``m_a - m_b``, so placing
    # ``ple-off`` first makes a positive scaffold difference mean "injection
    # suppressed the scaffold", matching how the claim is written.
    ("ple-off", "wiki"),
    ("ple-off", "code"),
    # Sanity: these two arms must be byte-identical, so every metric reads zero.
    ("ple-off", "no-reader"),
)


def _load_arm(d: Path, arm: str) -> list[dict] | None:
    f = d / f"arm-{arm}.json"
    if not f.exists():
        return None
    payload = json.loads(f.read_text())
    return payload["results"][0]["qa_exact"]["answers"]


def _marker_values(answers: list[dict], marker: str) -> np.ndarray:
    return np.array(
        [domain_fingerprint(x["generated"])[marker] for x in answers], dtype=float
    )


def _label_values(answers: list[dict], label: str) -> np.ndarray:
    return np.array([1.0 if classify(x["generated"]) == label else 0.0 for x in answers], dtype=float)


def _empty_values(answers: list[dict]) -> np.ndarray:
    return np.array([1.0 if not str(x["generated"] or "").strip() else 0.0 for x in answers], dtype=float)


#: How many leading characters the position-0 control reads.  Fixed, so length
#: beyond this prefix cannot influence the metric by construction.
PREFIX_CHARS = 32


def _scaffold_in_prefix(answers: list[dict]) -> np.ndarray:
    from classify_format import SCAFFOLD_MARKERS

    vals = []
    for x in answers:
        head = str(x["generated"] or "")[:PREFIX_CHARS]
        vals.append(1.0 if any(m in head for m in SCAFFOLD_MARKERS) else 0.0)
    return np.array(vals, dtype=float)


METRIC_FNS = {
    "prose_markers": lambda a: _marker_values(a, "prose"),
    "code_markers": lambda a: _marker_values(a, "code"),
    "math_markers": lambda a: _marker_values(a, "math"),
    "chat_scaffold_rate": lambda a: _label_values(a, "chat_scaffold"),
    "empty_rate": lambda a: _empty_values(a),
    "scaffold_in_first_32_chars": lambda a: _scaffold_in_prefix(a),
}


def analyse_regime(d: Path) -> dict:
    out: dict = {"regime_dir": str(d), "arm_pairs": {}, "arm_lengths": {}}
    loaded: dict[str, list[dict]] = {}
    for arm in {a for pair in ARM_PAIRS for a in pair}:
        ans = _load_arm(d, arm)
        if ans is not None:
            loaded[arm] = ans
            n = np.array([x["n_generated"] for x in ans], dtype=np.int64)
            out["arm_lengths"][arm] = {
                "n_items": int(n.size),
                "mean_n_generated": float(n.mean()),
                "pct_at_32_cap": float(100.0 * (n >= 32).mean()),
            }

    for arm_a, arm_b in ARM_PAIRS:
        if arm_a not in loaded or arm_b not in loaded:
            continue
        A, B = loaded[arm_a], loaded[arm_b]
        if len(A) != len(B):
            out["arm_pairs"][f"{arm_a}|{arm_b}"] = {
                "skipped": f"unpaired: {len(A)} vs {len(B)} items"
            }
            continue
        n_a = np.array([x["n_generated"] for x in A], dtype=float)
        n_b = np.array([x["n_generated"] for x in B], dtype=float)
        entry: dict = {"metrics": {}}
        for name, fn in METRIC_FNS.items():
            ct = paired_nuisance_contrast(fn(A), fn(B), n_a, n_b)
            declared = METRIC_DECLARATIONS[name].requires_adjustment
            measured_needs_adjustment = ct.verdict == NUISANCE_SENSITIVE
            # The gate is one-directional: a metric that *turns out* confounded
            # must have been declared as needing adjustment.  The reverse
            # (declared conservative, measured clean on this pair) is fine.
            agrees = (not measured_needs_adjustment) or declared
            entry["metrics"][name] = {
                **ct.as_dict(),
                "declared_requires_adjustment": declared,
                "declaration_agrees": agrees,
            }
        out["arm_pairs"][f"{arm_a}|{arm_b}"] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--output", required=True)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--strict", action="store_true", help="fail on declaration disagreement")
    args = ap.parse_args()

    report: dict = {"regimes": {}, "declarations": {k: v.__dict__ for k, v in METRIC_DECLARATIONS.items()}}
    for spec in args.regime:
        name, _, d = spec.partition("=")
        if not d:
            raise SystemExit(f"--regime must be NAME=DIR, got {spec!r}")
        report["regimes"][name] = analyse_regime(Path(d))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# Metric nuisance audit (round 167 Stage 0.1)", ""]
    lines.append("`nuisance` = `n_generated`. Verdicts: "
                 "`NUISANCE_SENSITIVE` = the raw arm difference is mostly generated length; "
                 "`SURVIVES_ADJUSTMENT` = it is still there at equal length; "
                 "`NO_EFFECT` = no raw difference to explain.")
    lines.append("")
    disagreements: list[str] = []

    for name, r in report["regimes"].items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| arm pair | metric | raw | adj | share | slope/tok | verdict | declared |")
        lines.append("|---|---|---:|---:|---:|---:|---|---|")
        for pair, e in r["arm_pairs"].items():
            if "skipped" in e:
                lines.append(f"| `{pair}` | – | – | – | – | – | {e['skipped']} | – |")
                continue
            for metric, m in e["metrics"].items():
                share = "–" if m["nuisance_share"] is None else f"{m['nuisance_share']:+.3f}"
                lines.append(
                    f"| `{pair}` | {metric} | {m['raw_diff']:+.4f} | "
                    f"{m['adjusted_diff']:+.4f} | {share} | "
                    f"{m['nuisance_slope']:+.4f} | {m['verdict']} | "
                    f"{'adjust' if m['declared_requires_adjustment'] else 'as-is'} |"
                )
                if not m["declaration_agrees"]:
                    disagreements.append(f"{name}/{pair}/{metric}: {m['verdict']}")
        lines.append("")

    if disagreements:
        lines.append("## Declaration disagreements")
        lines.append("")
        for d in disagreements:
            lines.append(f"* {d}")
        lines.append("")

    md = "\n".join(lines) + "\n"
    if args.markdown:
        mp = Path(args.markdown)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(md)
    print(md)
    print(f"wrote {out_path}")

    if args.strict and disagreements:
        print(f"STRICT: {len(disagreements)} declaration disagreement(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
