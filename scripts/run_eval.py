#!/usr/bin/env python3
"""A0/A1 evaluation report entrypoint.

Expected input: two JSON files produced by an evaluation harness, for example:

    {
      "model": "qwen35-0.8b-a0",
      "metrics": {
        "knowledge_recall": 0.52,
        "long_context_score": 0.81,
        "gsm8k": 0.30
      }
    }

This script only compares and renders the report; it does not run models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen35_ple.eval.protocol import load_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0", required=True, help="A0 baseline result JSON")
    parser.add_argument("--a1", required=True, help="A1 PLE treatment result JSON")
    parser.add_argument("--output", default="outputs/ablation-report.md")
    args = parser.parse_args()

    comp = load_comparison(args.a0, args.a1)
    report = comp.to_report()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\n[run_eval] report written to {out}")


if __name__ == "__main__":
    main()
