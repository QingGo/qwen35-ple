#!/usr/bin/env python
"""Round 169: turn the overnight artifacts into a summary, mechanically.

This is deliberately NOT prose.  The agent writes the interpretation; what has to
be automatic is the arithmetic -- every Delta, every verdict, every consistency
check, pulled from the JSON the runs actually produced, so that nothing in the
summary can drift from the artifacts while nobody is watching.

It reads every ``eval-*.json`` under the wiki root and the exploratory root, and
prints, per arm:

* the three mean NLLs (pure backbone / frozen rows / trained rows)
* Delta = L_frozen - L_trained, with its SE and t
* the frequency-stratified Delta
* the snapshot-vs-shard consistency check
* the injection effect vs the pure backbone

and then applies, VERBATIM, the frozen section 3 rule only where the
pre-registration licensed it (wiki).  Exploratory domains are reported with the
same arithmetic but explicitly NOT given a verdict, because the pre-registration
froze the wiki stream and position set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Pre-registration section 3, frozen.  Repeated here rather than imported so
#: that a later edit to the module cannot silently change what this summary says.
CARRY_FLOOR = 0.50
DEGRADED_FLOOR = 0.20
DEAD_MARGIN = 0.005
#: Section 3: ROWS_RESHAPEABLE also requires the null to stay near zero.
NULL_TOLERANCE = 0.01
#: Section 3: the effect floor, from margin.py's VERDICT_FLOOR_NATS.
EFFECT_FLOOR = 0.05
#: A positive Delta means the trained rows are BETTER (lower NLL) than frozen.
def rule(delta: float, se: float | None, delta_shuf: float | None) -> tuple[str, str]:
    """The frozen section 3 chain, applied to the paired Delta."""
    t = None if (se in (None, 0)) else abs(delta) / se
    if delta <= -EFFECT_FLOOR and (t is None or t > 3):
        why = f"Delta={delta:+.5f} <= -{EFFECT_FLOOR}"
        return "ROWS_HARMED", (f"{why} with |t|={t:.1f}" if t else why)
    if delta >= EFFECT_FLOOR and (t is None or t > 3):
        if delta_shuf is not None and delta_shuf > NULL_TOLERANCE:
            return (
                "ROWS_RESHAPEABLE_BUT_NULL_MOVED",
                (
                    f"Delta={delta:+.5f} >= {EFFECT_FLOOR}, but the shuffled-target arm also "
                    f"gained {delta_shuf:+.5f} > {NULL_TOLERANCE}: the gain is not content-specific"
                ),
            )
        return "ROWS_RESHAPEABLE", f"Delta={delta:+.5f} >= {EFFECT_FLOOR} with |t|={t:.1f}"
    return "ROWS_STUCK", f"|Delta|={abs(delta):.5f} is inside the +/-{EFFECT_FLOOR} effect floor"


def load_evals(root: Path) -> list[tuple[str, dict]]:
    out = []
    for p in sorted(root.glob("eval-*.json")):
        try:
            out.append((p.stem.replace("eval-", ""), json.loads(p.read_text())))
        except (OSError, json.JSONDecodeError) as exc:
            out.append((p.stem, {"_error": f"{type(exc).__name__}: {exc}"}))
    return out


def fmt(v, spec: str = ".5f") -> str:
    return "-" if v is None else format(v, spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="wiki round169 output dir")
    ap.add_argument("--exploratory", default="", help="exploratory subdir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wiki_root = Path(args.root)
    groups: list[tuple[str, Path, bool]] = [("wiki (pre-registered)", wiki_root, True)]
    if args.exploratory:
        ex = Path(args.exploratory)
        for tag in ("code", "stem"):
            if ex.exists():
                groups.append((f"{tag} (EXPLORATORY)", ex, False))

    lines: list[str] = ["# Round 169 overnight summary (mechanical)", ""]
    lines.append(
        "Every number below is read from the JSON the runs wrote. The interpretation "
        "is written separately; this file exists so the arithmetic cannot drift from "
        "the artifacts."
    )
    lines.append("")

    for title, root, is_prereg in groups:
        evals = load_evals(root)
        lines.append(f"## {title}")
        lines.append("")
        if not evals:
            lines.append("*no `eval-*.json` found*")
            lines.append("")
            continue
        lines.append("| arm | pure backbone | frozen rows | trained rows | Delta | SE | t |")
        lines.append("|---|---|---|---|---|---|---|")
        table = {}
        for name, d in evals:
            if "_error" in d:
                lines.append(f"| {name} | *unreadable: {d['_error']}* | | | | | |")
                continue
            mn = d.get("mean_nll", {})
            dl = d.get("delta", {}) or {}
            table[name] = dl.get("mean")
            lines.append(
                f"| {name} | {fmt(mn.get('none_pure_backbone'))} | {fmt(mn.get('frozen_rows'))} | "
                f"{fmt(mn.get('trained_rows'))} | **{fmt(dl.get('mean'))}** | "
                f"{fmt(dl.get('se'), '.6f')} | {fmt(dl.get('t'), '.2f')} |"
            )
        lines.append("")

        # delta for the shuffled-target control, if present
        shuf = next((v for k, v in table.items() if "shuf" in k), None)
        for name, d in evals:
            if "_error" in d:
                continue
            lines.append(f"### {name}")
            lines.append("")
            cons = d.get("consistency") or {}
            if cons:
                lines.append(
                    f"* snapshot-vs-shard consistency: max|diff| = "
                    f"{fmt(cons.get('max_abs_diff'), '.3e')} over {cons.get('n')} positions"
                )
            inj = d.get("injection_effect_vs_pure_backbone") or {}
            for k, v in inj.items():
                if isinstance(v, dict):
                    lines.append(f"* {k}: {fmt(v.get('mean'))} nats (t={fmt(v.get('t'), '.1f')})")
            bands = d.get("by_context_frequency")
            if bands:
                lines.append("")
                lines.append("| context-count band | n | Delta | t |")
                lines.append("|---|---|---|---|")
                for b, v in bands.items():
                    lines.append(f"| {b} | {v.get('n'):,} | {fmt(v.get('mean'))} | {fmt(v.get('t'), '.1f')} |")
            if d.get("by_context_frequency_error"):
                lines.append(f"* frequency breakdown unavailable: {d['by_context_frequency_error']}")
            lines.append("")

        # the verdict, only where the pre-registration licensed one
        real = next((v for k, v in table.items() if k.startswith("real")), None)
        se = next(
            (d.get("delta", {}).get("se") for k, d in evals if k.startswith("real") and "_error" not in d),
            None,
        )
        if real is None:
            lines.append("*no `real` arm found: no verdict*")
            lines.append("")
            continue
        label, why = rule(real, se, shuf)
        if is_prereg:
            lines.append(f"**Frozen section 3 verdict: `{label}`** -- {why}")
            lines.append("")
        else:
            lines.append(
                f"*the same arithmetic would read `{label}` ({why}), but the pre-registration "
                f"froze the wiki stream and position set, so this domain is EXPLORATORY and "
                f"carries no verdict.*"
            )
            lines.append("")

    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
