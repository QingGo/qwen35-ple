#!/usr/bin/env python3
"""Round 167 Stage 2.2: apply the pre-registered gate-selectivity rule.

The rule is frozen in ``docs/round-167-stage2-gate-selectivity-preregistration.md``
section 4 and reproduced in :mod:`qwen35_ple.stage2_gate_verdict`.  This script
is only the loader: it reads the queue's ``eval-*`` and ``gate-*`` JSON files and
hands them to the rule.

Note which template the rule reads: the gate is saturated *always-on*, and the
always-on regime is the **chat** template (the pre-registration section 2 says
so explicitly).  The chat reports are the ones keyed ``-chat``; the raw reports
are ignored by the rule and reported only for context.

Usage::

    python scripts/round167_stage2_verdict.py \
        --dir outputs/round167/stage2-gate \
        --output outputs/round167/stage2-gate/verdict.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen35_ple.stage2_gate_verdict import (
    apply_rule,
    collect_inputs,
    render_markdown,
)

DEFAULT_MODES = ("scalar", "per_dim")
DEFAULT_SEEDS = (0, 1, 2)


def log(msg: str) -> None:
    print(f"[r167-2.2] {msg}", flush=True)


def _load(path: Path) -> dict | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="outputs/round167/stage2-gate")
    ap.add_argument("--output", default=None)
    ap.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = ap.parse_args()

    work = Path(args.dir)
    if not work.is_dir():
        raise SystemExit(f"no such directory: {work}")

    eval_reports: dict[tuple[str, str, int], dict] = {}
    for mode in args.modes:
        for arm in ("real", "control"):
            for seed in args.seeds:
                report = _load(work / f"eval-{mode}-seed{seed}-chat.json")
                if report is not None:
                    eval_reports[(mode, arm, seed)] = report
    gate_reports: dict[tuple[str, int], dict] = {}
    for mode in args.modes:
        for seed in args.seeds:
            report = _load(work / f"gate-{mode}-seed{seed}.json")
            if report is not None:
                gate_reports[(mode, seed)] = report

    log(f"loaded {len(eval_reports)} chat eval reports, "
        f"{len(gate_reports)} gate reports")
    inputs = collect_inputs(
        eval_reports, gate_reports, modes=list(args.modes), seeds=list(args.seeds),
    )
    result = apply_rule(inputs)
    result["inputs"] = inputs.to_dict()
    result["source_dir"] = str(work)

    out_path = Path(args.output) if args.output else work / "verdict.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    log(f"verdict = {result['label']}")
    log(f"  {result['reason']}")
    log(f"wrote {out_path} and {md_path}")
    # INCOMPLETE means the queue has not finished; that is not a verdict.
    return 2 if result["label"] == "INCOMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
