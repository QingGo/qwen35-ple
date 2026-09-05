#!/usr/bin/env python3
"""Build formal-style benchmark files for local evaluation.

This script deliberately does not download official datasets.  It generates
deterministic, structured benchmark items in four common families:

- GSM8K-like arithmetic word problems;
- MATH-like algebra/arithmetic expressions;
- HumanEval-like Python function completion;
- MBPP-like programming tasks.

The outputs are JSONL files suitable for later model evaluation with the same
teacher-forced logprob / exact-match harness used elsewhere.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _build_gsm8k_like(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a = rng.randint(3, 99)
        b = rng.randint(3, 99)
        c = rng.randint(2, 12)
        if rng.random() < 0.5:
            question = f"Alice has {a} apples. She buys {b} more. How many apples does she have in total?"
            answer = str(a + b)
        else:
            question = f"A box contains {b} pencils. {c} pencils are removed. How many pencils remain?"
            answer = str(b - c)
        out.append(
            {
                "id": f"gsm8k-like-{i}",
                "category": "gsm8k",
                "problem": question,
                "answer": answer,
                "solution": f"Answer: {answer}",
            }
        )
    return out


def _build_math_like(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a = rng.randint(2, 20)
        b = rng.randint(2, 20)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            ans = a + b
        elif op == "-":
            ans = a - b
        else:
            ans = a * b
        expr = f"{a} {op} {b}"
        out.append(
            {
                "id": f"math-like-{i}",
                "category": "math",
                "problem": f"Evaluate the expression: {expr}",
                "answer": str(ans),
                "solution": f"Answer: {ans}",
            }
        )
    return out


def _build_humaneval_like(n: int, seed: int) -> list[dict]:
    templates = [
        ("add", "Return the sum of a and b.", "def add(a, b):\n    return a + b"),
        ("mul", "Return the product of a and b.", "def mul(a, b):\n    return a * b"),
        ("max3", "Return the maximum of a, b, and c.", "def max3(a, b, c):\n    return max(a, b, c)"),
        ("absval", "Return the absolute value of x.", "def absval(x):\n    return abs(x)"),
        ("is_even", "Return True if n is even, False otherwise.", "def is_even(n):\n    return n % 2 == 0"),
    ]
    out = []
    for i in range(n):
        name, desc, code = templates[i % len(templates)]
        out.append(
            {
                "id": f"humaneval-like-{i}",
                "category": "humaneval",
                "problem": f"Write a Python function.\n\n{desc}\n\nFunction name: {name}",
                "answer": code,
                "solution": code,
            }
        )
    return out


def _build_mbpp_like(n: int, seed: int) -> list[dict]:
    templates = [
        ("Return the first n even numbers as a list.", "def first_evens(n):\n    return [2 * i for i in range(n)]"),
        ("Return the length of a string without using len().", "def str_len(s):\n    return sum(1 for _ in s)"),
        ("Return the reversed list.", "def reverse_list(xs):\n    return xs[::-1]"),
        ("Return the number of vowels in a string.", "def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')"),
        ("Return the factorial of n.", "def fact(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r"),
    ]
    out = []
    for i in range(n):
        desc, code = templates[i % len(templates)]
        out.append(
            {
                "id": f"mbpp-like-{i}",
                "category": "mbpp",
                "problem": f"Write a Python function.\n\n{desc}",
                "answer": code,
                "solution": code,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/formal-benchmarks")
    parser.add_argument("--n-per-family", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    families = {
        "gsm8k-like": _build_gsm8k_like(args.n_per_family, args.seed),
        "math-like": _build_math_like(args.n_per_family, args.seed),
        "humaneval-like": _build_humaneval_like(args.n_per_family, args.seed),
        "mbpp-like": _build_mbpp_like(args.n_per_family, args.seed),
    }

    manifest = {"schema": "formal-benchmarks-v1", "seed": args.seed, "families": {}}
    for name, items in families.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        manifest["families"][name] = {"path": str(path), "n": len(items)}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
