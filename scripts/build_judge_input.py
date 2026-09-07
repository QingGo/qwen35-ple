#!/usr/bin/env python3
"""Convert a HumanEval ablation result JSON into LLM-judge input rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if "results" in data:
        rows = []
        for task_id, r in data["results"].items():
            for cond in ("base", "bm25_ple"):
                entry = r.get(cond)
                if not entry:
                    continue
                rows.append({
                    "task_id": task_id,
                    "condition": cond,
                    "question": r.get("prompt", ""),
                    "answer": entry.get("completion", ""),
                    "reference": "",
                })
    else:
        rows = data.get("rows", data if isinstance(data, list) else [])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[judge-input] wrote {len(rows)} rows to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
