#!/usr/bin/env python3
"""Purified OPSD data preparation: filter/verify self-distillation examples.

This implements the local, low-resource version of Purified OPSD:

1. Load candidate examples (e.g. CAP-1 RAG self-distill data);
2. Verify each example:
   - math/arithmetic: parse a final numeric answer and, when a gold answer is
     available, require agreement;
   - code: require parseable Python code (or at least a function definition);
   - other: require non-trivial length;
3. Keep only verified examples and write them as purified training data;
4. The filtered data can then be fed to MoRA/QLoRA/LoRA training.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_last_number(text: str) -> str | None:
    hits = _NUM_RE.findall(text)
    return hits[-1] if hits else None


def _extract_code(text: str) -> str | None:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    if "def " in text:
        return text
    return None


def _verify_math(obj: dict) -> tuple[bool, str]:
    solution = str(obj.get("solution") or obj.get("answer") or "")
    if not solution.strip():
        return False, "empty solution"
    last = _extract_last_number(solution)
    if last is None:
        return False, "no numeric answer"
    gold = obj.get("answer")
    if gold is not None and str(gold).strip():
        gold_str = str(gold).strip()
        if gold_str != last and _normalize_number(gold_str) != _normalize_number(last):
            return False, "answer mismatch"
    return True, "math verified"


def _normalize_number(text: str) -> str:
    try:
        return str(float(text))
    except ValueError:
        return text.strip()


def _verify_code(obj: dict) -> tuple[bool, str]:
    solution = str(obj.get("solution") or obj.get("answer") or "")
    code = _extract_code(solution)
    if not code:
        return False, "no parseable code"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"
    return True, "code verified"


def _verify_generic(obj: dict) -> tuple[bool, str]:
    solution = str(obj.get("solution") or obj.get("answer") or "")
    if len(solution.strip()) < 20:
        return False, "too short"
    return True, "generic verified"


def _verify(obj: dict) -> tuple[bool, str]:
    category = str(obj.get("category") or "").lower()
    if category in {"gsm8k", "math", "arithmetic", "math-like"}:
        return _verify_math(obj)
    if category in {"humaneval", "mbpp", "code", "code-like", "humaneval-like", "mbpp-like"}:
        return _verify_code(obj)
    return _verify_generic(obj)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/cap1-rag-distill-160.jsonl")
    parser.add_argument("--output", default="data/purified-opsd-train.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    items = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            items.append(obj)
            if args.limit is not None and len(items) >= args.limit:
                break

    kept = []
    rejected = 0
    reasons = {}
    for obj in items:
        ok, reason = _verify(obj)
        if ok:
            kept.append(obj)
        else:
            rejected += 1
            reasons[reason] = reasons.get(reason, 0) + 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for obj in kept:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    meta = {
        "schema": "purified-opsd-v1",
        "input": args.input,
        "output": str(out),
        "input_n": len(items),
        "kept_n": len(kept),
        "rejected_n": rejected,
        "reject_reasons": reasons,
    }
    (out.parent / (out.stem + "-meta.json")).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
