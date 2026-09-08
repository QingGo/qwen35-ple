#!/usr/bin/env python3
"""Summarize unseen-KB per-arm eval JSONs produced by run_unseen_kb_experiment.sh.

Expected files:

    eval-real-seed0.json
    eval-control-seed0.json
    eval-no-reader-seed0.json

or any files with a single-mode Phase 0 summary.

Usage::

    python scripts/summarize_unseen_kb.py \
      --dir outputs/unseen-kb-pilot/unseen-readers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _summary_metrics(path: Path) -> tuple[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    if not isinstance(summary, dict) or not summary:
        return path.stem, {}
    mode, entry = next(iter(summary.items()))
    return mode, {
        "val_loss": entry.get("val_loss_mean"),
        "val_ppl": entry.get("val_ppl_mean"),
        "qa_em": entry.get("qa_em_mean"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows: list[dict] = []
    for path in sorted(Path(args.dir).glob("eval-*.json")):
        mode, metrics = _summary_metrics(path)
        rows.append({"file": path.name, "mode": mode, **metrics})

    print("=== Unseen-KB evaluation ===")
    for row in rows:
        print(
            f"{row['mode']:<12}"
            f"ppl={row.get('val_ppl')!s:<12}"
            f"loss={row.get('val_loss')!s:<12}"
            f"qa_em={row.get('qa_em')!s}"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[unseen-kb] wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
