#!/usr/bin/env python
"""Round 169, EXPLORATORY: report the shape of R(K) for ladder rung 3.

This script deliberately does NOT apply the frozen section 3 rule.  That rule,
frozen in ``docs/round-168-stage1.5g-preregistration.md``, decides on K=5000
(primary) with K=1000 as the sensitivity check, and its verdict already stands.
Extra K values are exploratory and cannot revise it -- so this reporter refuses
to emit a verdict at all, and says so in its own output.

The question the curve answers
------------------------------
Rung 3 returned R = 0.51-0.74: the rows recover roughly half to three quarters of
what an explicit count table recovers on the same positions.  Two explanations
survive, and they predict different curves:

* **measurement-limited** -- a smaller candidate set K weakens the explicit-count
  baseline (it has fewer tokens to estimate from and a smaller argmax space), so
  the probe should catch up and ``R -> 1`` as K falls;
* **content-limited** -- the rows genuinely hold a degraded prior, in which case R
  plateaus at some value below 1 and does not care much about K.

Neither outcome is a verdict; the curve is the diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen35_ple.rung3_verdict import recovery, top1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--probe", action="append", required=True, metavar="K=PATH")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    rows = []
    for spec in args.probe:
        if "=" not in spec:
            raise SystemExit(f"--probe expects K=path, got {spec!r}")
        k_str, path_str = spec.split("=", 1)
        path = Path(path_str)
        if not path.exists():
            print(f"[rk] SKIP K={k_str}: {path} not found", file=sys.stderr)
            continue
        rep = json.loads(path.read_text())
        probe = top1(rep, "probe_raw_rows")
        count = top1(rep, "count_trigram")
        mlp = top1(rep, "probe_mlp_raw_rows")
        shuffled = top1(rep, "control_shuffled_train_rows")
        majority = top1(rep, "majority_train_prior")
        rows.append(
            {
                "K": int(k_str),
                "path": str(path),
                "stage": rep.get("stage"),
                "probe_top1": probe,
                "count_trigram_top1": count,
                "mlp_top1": mlp,
                "shuffled_rows_top1": shuffled,
                "majority_top1": majority,
                "floor": None if (shuffled is None or majority is None) else max(shuffled, majority),
                "R": recovery(rep),
                "mlp_over_linear": (
                    None if (mlp is None or probe in (None, 0)) else mlp / probe
                ),
            }
        )
    if not rows:
        raise SystemExit("no probe JSONs could be read")
    rows.sort(key=lambda r: r["K"])

    out = {
        "tag": args.tag,
        "exploratory": True,
        "verdict_withheld": (
            "This is an exploratory curve. The Stage 1.5g verdict is fixed by "
            "K=5000 (primary) and K=1000 (sensitivity) in "
            "docs/round-168-stage1.5g-preregistration.md section 3, and no K added "
            "here can change it."
        ),
        "rows": rows,
    }

    lines = [f"# R(K) curve (EXPLORATORY) - {args.tag}", "", "> " + out["verdict_withheld"], ""]
    lines.append("| K | probe top-1 | count trigram | R | MLP | MLP/linear | floor | stage |")
    lines.append("|---|---|---|---|---|---|---|---|")

    def fmt(v: float | None) -> str:
        return "-" if v is None else f"{v:.4f}"

    for r in rows:
        f = fmt
        lines.append(
            f"| {r['K']} | {f(r['probe_top1'])} | {f(r['count_trigram_top1'])} | "
            f"{f(r['R'])} | {f(r['mlp_top1'])} | {f(r['mlp_over_linear'])} | "
            f"{f(r['floor'])} | {r['stage']} |"
        )
    lines.append("")
    rs = [(r["K"], r["R"]) for r in rows if r["R"] is not None]
    if len(rs) >= 2:
        lo_k, lo_r = rs[0]
        hi_k, hi_r = rs[-1]
        lines.append(
            f"* R at the smallest K ({lo_k}) = {lo_r:.4f}; at the largest ({hi_k}) = {hi_r:.4f}. "
            f"A rise toward 1 as K falls supports the measurement-limited reading; a "
            f"plateau supports the content-limited one."
        )
    md = "\n".join(lines) + "\n"
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    if args.out_md:
        Path(args.out_md).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
