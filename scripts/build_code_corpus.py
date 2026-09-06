#!/usr/bin/env python3
"""Build a JSONL code corpus from local Python files for PLE/code demos.

Each line is ``{"text": "<file content>", "path": "<relative path>"}`` so the
existing RAG/PLE serving utilities can consume it as a same-domain code corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=None, help="root to scan (repeatable)")
    parser.add_argument("--output", default="data/code-corpus.jsonl")
    parser.add_argument("--max-files", type=int, default=300)
    parser.add_argument("--min-chars", type=int, default=40)
    args = parser.parse_args()

    roots = [Path(r) for r in (args.root or ["src", "scripts", "tests"])]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"[code-corpus] skip missing root: {root}", flush=True)
            continue
        files.extend(sorted(root.rglob("*.py")))
    files = sorted(set(files))[: args.max_files]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for path in files:
            if ".venv" in path.parts or ".git" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if len(text) < args.min_chars:
                continue
            f.write(
                json.dumps(
                    {"text": text, "path": str(path)},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    print(f"[code-corpus] wrote {out} files={n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
