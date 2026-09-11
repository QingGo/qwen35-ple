#!/usr/bin/env python3
"""Recompute exact-match under several reading rules, to audit the published one.

Round 156 found that the published EM rule is substring containment, which
credits an answer fused into a larger token: the scaffolding arm emits ``Nouser``
and ``no`` matches inside it, worth up to 33 points of inflation.  Every
falsification this project relies on was judged with that rule, and the
token-level replacement was never applied backwards to them.

This script re-scores existing artifacts offline.  It does NOT need torch, a GPU
or the row table: it reads ``qa_exact.answers`` from a Phase 0 JSON and applies
four rules to the same items.

The guard that makes the output trustworthy: the ``substring`` rule must
reproduce the stored ``correct`` flag on every single item, or the script aborts.
A reimplementation that cannot reproduce the published number cannot be trusted
to revise it.

Rules
-----
substring   norm(answer) in norm(generated)        -- what was published
word        norm(answer) contiguous in norm(generated).split()
exact       norm(answer) == norm(generated)
leading     norm(generated) starts with norm(answer)   (terse-answer check)

Usage
-----
    python scripts/audit_em_rules.py \
        --run lora-real=outputs/round152/lora-real.json \
        --run lora-control=outputs/round152/lora-control.json \
        --pair lora-real=lora-control
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any

RUN_PHASE0 = Path(__file__).resolve().parent / "run_phase0.py"
TASKS = ("boolq", "triviaqa", "nq")


def _load_normalizer() -> dict[str, Any]:
    """Exec the real normalization helpers out of run_phase0.py, without torch.

    Importing the module would pull in torch and the whole row-store stack; the
    functions are pure string processing, so extracting their source keeps this
    audit runnable anywhere -- including on a machine with no GPU.
    """
    tree = ast.parse(RUN_PHASE0.read_text(encoding="utf-8"))
    wanted_funcs = {"_normalize_answer", "_expand_number_words"}
    wanted_assigns = {"_NUMBER_UNITS", "_NUMBER_TENS"}
    picked: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
            picked.append(node)
        elif isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & wanted_assigns:
                picked.append(node)
    missing = wanted_funcs - {
        n.name for n in picked if isinstance(n, ast.FunctionDef)
    }
    if missing:
        raise SystemExit(f"could not extract {sorted(missing)} from {RUN_PHASE0}")
    namespace: dict[str, Any] = {"re": __import__("re")}
    # Executing the extracted source is the point: importing run_phase0 would pull
    # in torch and the row-store stack, and re-implementing the normalisation by
    # hand would defeat the reproduction guard that makes this audit trustworthy.
    # The source is this repository's own file, not external input.
    exec(  # noqa: S102
        compile(ast.Module(body=picked, type_ignores=[]), "<extracted>", "exec"),
        namespace,
    )
    return namespace


def _answers(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise SystemExit(f"{path}: no results entries")
    rows: list[dict[str, Any]] = []
    for entry in results:
        block = entry.get("qa_exact") or {}
        rows.extend(block.get("answers") or [])
    if not rows:
        raise SystemExit(f"{path}: no qa_exact answers (this audit needs generations)")
    return rows


def _rules(norm, answer: str, generated: str) -> dict[str, bool]:
    """Four readings of the same item.

    ``substring`` reproduces the published rule EXACTLY, including its quirk that
    an answer normalising to the empty string is scored correct -- because
    ``"" in anything`` is True, and ``_normalize_answer`` drops articles, so an
    item whose answer is "a" or "the" is credited no matter what the model said.
    Deviating from that here would break the reproduction guard below, which is
    the only thing making the other three columns trustworthy.  The quirk is
    counted separately and reported.
    """
    a = norm(answer)
    g = norm(generated)
    a_words = a.split()
    g_words = g.split()
    word = bool(a_words) and any(
        g_words[i : i + len(a_words)] == a_words
        for i in range(max(1, len(g_words) - len(a_words) + 1))
    )
    return {
        "substring": a in g,
        "word": word,
        "exact": bool(a) and g == a,
        "leading": bool(a) and g.startswith(a),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--pair", action="append", default=[], metavar="FIRST=SECOND")
    parser.add_argument("--title", default="EM rule audit")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.run:
        raise SystemExit("at least one --run LABEL=PATH is required")

    norm = _load_normalizer()["_normalize_answer"]

    per_run: dict[str, dict[tuple[str, str], dict[str, bool]]] = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run must be LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        rows = _answers(Path(path))
        scored: dict[tuple[str, str], dict[str, bool]] = {}
        mismatches = 0
        degenerate = 0
        for row in rows:
            key = (str(row.get("task")), str(row.get("question")))
            verdict = _rules(norm, str(row.get("answer", "")), str(row.get("generated", "")))
            scored[key] = verdict
            if not norm(str(row.get("answer", ""))):
                degenerate += 1
            stored = row.get("correct")
            if stored is not None and bool(stored) != verdict["substring"]:
                mismatches += 1
        if mismatches:
            raise SystemExit(
                f"{path}: the substring rule disagreed with the stored `correct` flag on "
                f"{mismatches}/{len(rows)} items, so this audit does not reproduce the "
                "published metric and must not be used to revise it"
            )
        per_run[label] = scored
        print(
            f"{label}: {len(rows)} items, substring rule reproduces `correct` exactly; "
            f"{degenerate} item(s) have an answer that normalises to empty and are "
            "credited unconditionally by that rule"
        )

    rules = ("substring", "word", "exact", "leading")
    lines: list[str] = [f"# {args.title}", ""]
    lines.append("## Exact match under each rule")
    lines.append("")
    lines.append("| Run | n | " + " | ".join(rules) + " |")
    lines.append("|---" * (2 + len(rules)) + "|")
    for label, scored in per_run.items():
        n = len(scored)
        cells = [f"{sum(v[r] for v in scored.values()) / n:.4f}" for r in rules]
        lines.append(f"| {label} | {n} | " + " | ".join(cells) + " |")

    if args.pair:
        lines += ["", "## Paired item-level deltas (positive = first label higher)", ""]
        lines.append("| Pair | Rule | n | Mean Δ | SEM | flips vs substring |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for spec in args.pair:
            if "=" not in spec:
                raise SystemExit(f"--pair must be FIRST=SECOND, got {spec!r}")
            first, second = spec.split("=", 1)
            if first not in per_run or second not in per_run:
                raise SystemExit(f"--pair {spec}: unknown label")
            shared = sorted(set(per_run[first]) & set(per_run[second]))
            for rule in rules:
                deltas = [
                    int(per_run[first][k][rule]) - int(per_run[second][k][rule])
                    for k in shared
                ]
                mean = sum(deltas) / len(deltas)
                var = sum((d - mean) ** 2 for d in deltas) / max(1, len(deltas) - 1)
                sem = math.sqrt(var / len(deltas))
                sub_deltas = [
                    int(per_run[first][k]["substring"]) - int(per_run[second][k]["substring"])
                    for k in shared
                ]
                flips = sum(
                    1
                    for a, b in zip(deltas, sub_deltas)
                    if (a > 0) != (b > 0) or (a == 0) != (b == 0)
                )
                lines.append(
                    f"| {first} vs {second} | {rule} | {len(shared)} | {mean:+.4f} | "
                    f"{sem:.4f} | {flips} |"
                )

    markdown = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
