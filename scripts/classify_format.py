#!/usr/bin/env python3
"""Round-162 output-format taxonomy: an explicit, auditable parser.

Why this exists
---------------
Round 156 was burned twice by string heuristics (a substring EM that inflated a
scaffolding arm by 33 points; a gold NLL that scored a token the model never
emits).  Round 162 asks whether the format shift caused by PLE injection is
*content-dependent* (a prior) or merely *a large perturbation* (an artifact),
and that question is answered from the **distribution of output formats**, so
the classifier has to be written down before the numbers and applied uniformly.

This module is deliberately rule-based and deterministic.  No model is called;
no threshold is tuned on the results.

Taxonomy (priority cascade, first match wins)
---------------------------------------------
1. ``empty``          -- ``text.strip() == ""``
2. ``chat_scaffold``  -- chat/thinking markers: ``<think>``, ``</think>``,
   ``<|im_start|>``, ``<|im_end|>``, a line consisting only of
   ``system``/``user``/``assistant``, or the literal ``assistant\\n`` /
   ``user\\n``.  (Superset of ``run_phase0._FORMAT_MARKERS``; the two are
   reported side by side so agreement is checkable.)
3. ``code_fence``     -- a ``` fence, or a line starting with a code keyword
   (``def``/``class``/``import``/``from X import``/``#include``/``public``/
   ``function``/``fn``/``printf``), or >= 3 lines ending in ``{``/``}``/``;``.
4. ``json``           -- stripped text starts with ``{``/``[`` and either
   ``json.loads`` succeeds or it matches a ``"key": value`` shape.
5. ``refusal``        -- a refusal / inability phrase in the first 200 chars.
6. ``raw_continuation`` -- non-empty and none of the above.
7. ``other``          -- non-empty but undecodable (U+FFFD present).

``raw_continuation`` is further split by the length of its first non-empty line,
because the round-156 headline was exactly "terse raw continuation vs chat
scaffolding":

* ``terse`` -- first non-empty line has <= ``TERSE_MAX_TOKENS`` whitespace tokens
  and no sentence-final punctuation (``.``/``?``/``!``/``:``);
* ``long``  -- everything else.

Domain-lexical fingerprint
--------------------------
For each generation we also count markers that indicate *what* the continuation
is about, independently of its structural format.  If the reader carries a
domain prior, a CODE-trained reader should move these markers even when the
structural label does not move:

* ``code``  -- ``def ``/``return ``/``import ``/``print(``/``self.``/``->``/
  a 4-space indented line/``# ``/``()``/``{}``/``;``
* ``math``  -- ``\\``/``$``/``^``/``\\frac``/``\\theta``/``=`` between digits/
  ``therefore``/``<=``/``>=``
* ``prose`` -- ``the ``/`` of ``/`` is ``/`` was ``/`` in ``/`` that ``
  (case-insensitive)

Usage::

    python scripts/classify_format.py outputs/round162/arm-*.json
    python scripts/classify_format.py --json outputs/round162/arm-code.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

TERSE_MAX_TOKENS = 6

LABELS = (
    "empty",
    "chat_scaffold",
    "code_fence",
    "json",
    "refusal",
    "raw_continuation",
    "other",
)

SCAFFOLD_MARKERS = (
    "<think>",
    "</think>",
    "<|im_start|>",
    "<|im_end|>",
    "assistant\n",
    "user\n",
)
_ROLE_LINE = re.compile(r"(?m)^\s*(system|user|assistant)\s*$")
_CODE_LINE = re.compile(
    r"(?m)^\s*(def |class |import |from \s*\S+\s+import|#include|public |"
    r"function |fn |printf\s*\(|using namespace)"
)
_CODE_TAIL = re.compile(r"[{};]\s*$")
_JSON_SHAPE = re.compile(r'^\s*[\[{]\s*"?[\w\-]+"?\s*:')
_REFUSAL = re.compile(
    r"(i'?m sorry|i am sorry|i cannot|i can'?t|as an ai|i'?m not able|"
    r"i don'?t know|unable to answer|not able to answer)",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.?!:]\s*$")
_INDENT4 = re.compile(r"(?m)^ {4}\S")

_CODE_MARKERS = (
    "def ",
    "return ",
    "import ",
    "print(",
    "self.",
    "->",
    "()",
    "{}",
    ";",
    "# ",
)
_MATH_MARKERS = ("\\", "$", "^", "\\frac", "\\theta", "therefore", "<=", ">=")
_PROSE_MARKERS = ("the ", " of ", " is ", " was ", " in ", " that ")


def classify(text: Any) -> str:
    """Return exactly one label from :data:`LABELS` for ``text``."""
    if text is None:
        return "empty"
    s = str(text)
    if "\ufffd" in s:
        return "other"
    if not s.strip():
        return "empty"
    if any(m in s for m in SCAFFOLD_MARKERS) or _ROLE_LINE.search(s):
        return "chat_scaffold"
    if "```" in s:
        return "code_fence"
    if _CODE_LINE.search(s):
        return "code_fence"
    if sum(1 for line in s.splitlines() if _CODE_TAIL.search(line)) >= 3:
        return "code_fence"
    stripped = s.strip()
    if stripped[0] in "{[":
        try:
            json.loads(stripped)
            return "json"
        except Exception:  # noqa: BLE001 - any parse failure means "not JSON"
            pass
        if _JSON_SHAPE.match(stripped):
            return "json"
    if _REFUSAL.search(s[:200]):
        return "refusal"
    return "raw_continuation"


def sub_label(text: Any) -> str:
    """Split ``raw_continuation`` into ``terse`` / ``long``; else return label."""
    label = classify(text)
    if label != "raw_continuation":
        return label
    first = next((ln for ln in str(text).splitlines() if ln.strip()), "")
    tokens = first.split()
    if len(tokens) <= TERSE_MAX_TOKENS and not _SENTENCE_END.search(first.strip()):
        return "terse"
    return "long"


def domain_fingerprint(text: Any) -> dict[str, int]:
    """Count domain-lexical markers (structural format aside)."""
    s = " " + str(text or "") + " "
    low = s.lower()
    code = sum(1 for m in _CODE_MARKERS if m in s)
    if _INDENT4.search(s):
        code += 1
    math = sum(1 for m in _MATH_MARKERS if m in s)
    if re.search(r"\d\s*=\s*\d", s):
        math += 1
    prose = sum(1 for m in _PROSE_MARKERS if m in low)
    return {"code": code, "math": math, "prose": prose}


def label_distribution(texts: list[Any]) -> dict[str, Any]:
    """Full label distribution plus the raw-continuation sub-split."""
    n = len(texts)
    counts = Counter(classify(t) for t in texts)
    subs = Counter(sub_label(t) for t in texts)
    dist = {label: counts.get(label, 0) / n if n else 0.0 for label in LABELS}
    dist_sub = {
        k: subs.get(k, 0) / n if n else 0.0
        for k in ("terse", "long", "chat_scaffold", "code_fence", "json", "refusal", "empty", "other")
    }
    return {"n": n, "counts": dict(counts), "dist": dist, "dist_sub": dist_sub}


def prefix_class(text: Any) -> str:
    """Classify the *leading surface* of a generation, below the taxonomy.

    Round-156's headline (``\\n\\n<think>...`` vs `` yes``) and its metric bug #2
    (the model emits the spaced token `` yes`` = 9542 while the scorer grades the
    unspaced ``yes`` = 9405) both live at a granularity the coarse taxonomy
    cannot see: two generations can both be ``raw_continuation``/``terse`` and
    still differ in exactly the way that decides whether the model is in
    completion mode or in chat mode.  So we record the leading whitespace
    structure separately:

    * ``empty``       -- no output at all
    * ``newline_lead``-- the first non-space character is preceded by a newline
      (``"\\n\\nYes"``; the round-156 zero-injection shape)
    * ``space_lead``  -- preceded only by spaces (``" Yes"``; the round-156
      injected shape)
    * ``no_lead``     -- content starts immediately (``"Yes"``)
    """
    s = str(text if text is not None else "")
    if not s.strip():
        return "empty"
    lead = s[: len(s) - len(s.lstrip())]
    if "\n" in lead:
        return "newline_lead"
    if lead:
        return "space_lead"
    return "no_lead"


PREFIX_CLASSES = ("newline_lead", "space_lead", "no_lead", "empty")

# First-token ids verified against the real Qwen3.5-0.8B tokenizer (round 162):
# the difference between them is exactly round-156 metric bug #2 -- the model
# emits the spaced token while the raw gold-NLL reading grades the unspaced one.
KNOWN_FIRST_TOKENS = {
    "271": "\\n\\n",
    "2233": " No",
    "7179": " Yes",
    "9542": " yes",
    "874": " no",
    "9405": "yes",
    "2083": "no",
}


def describe_token(token_id: str) -> str:
    """Human-readable form of a first-token id, when it is one we have decoded."""
    return KNOWN_FIRST_TOKENS.get(str(token_id), "?")



def first_token_key(answer: dict[str, Any], tokenizer: Any = None) -> str:
    """Return the first generated token id as a string key (or ``"none"``)."""
    ids = answer.get("generated_ids") or []
    if not ids:
        return "none"
    return str(int(ids[0]))


def load_arm(path: Path) -> dict[str, Any]:
    """Load one ``run_phase0.py`` output JSON (single-mode arms)."""
    payload = json.loads(Path(path).read_text())
    results = payload.get("results") or []
    if len(results) != 1:
        pairs = [(r.get("mode"), r.get("seed")) for r in results]
        raise SystemExit(
            f"{path}: expected exactly one (mode, seed) result, got {len(results)}: "
            f"{pairs}.  Run each arm with '--modes <one>' and '--seeds 0'."
        )
    result = results[0]
    qa_exact = result.get("qa_exact")
    if not qa_exact:
        raise SystemExit(f"{path}: no qa_exact block")
    return {
        "path": str(path),
        "mode": result.get("mode"),
        "ple_off": result.get("ple_off"),
        "results": result,
        "answers": qa_exact["answers"],
        "metrics": qa_exact.get("metrics", {}),
        "gold_metrics": (result.get("qa_gold") or {}).get("metrics", {}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="+", help="run_phase0 output JSON paths")
    parser.add_argument("--json", action="store_true", help="emit machine JSON")
    parser.add_argument(
        "--examples", type=int, default=0, help="print N example generations per label"
    )
    args = parser.parse_args(argv)

    out: dict[str, Any] = {}
    for arm_path in args.arms:
        arm = load_arm(Path(arm_path))
        name = Path(arm_path).stem
        texts = [a.get("generated", "") for a in arm["answers"]]
        entry = label_distribution(texts)
        entry["tasks"] = {}
        for task in sorted({a.get("task") for a in arm["answers"]}):
            task_texts = [a.get("generated", "") for a in arm["answers"] if a.get("task") == task]
            entry["tasks"][task] = label_distribution(task_texts)
        fp = [domain_fingerprint(t) for t in texts]
        entry["domain"] = {
            k: sum(f[k] for f in fp) / len(fp) if fp else 0.0 for k in ("code", "math", "prose")
        }
        entry["run_phase0_metrics"] = {
            k: v
            for k, v in arm["metrics"].items()
            if k.startswith("qa_fmt_") or k.startswith("qa_norm_") or "em_token" in k or k == "qa_n"
        }
        if args.examples:
            by_label: dict[str, list[str]] = {}
            for t in texts:
                by_label.setdefault(classify(t), []).append(str(t))
            entry["examples"] = {
                label: [repr(x)[:160] for x in items[: args.examples]]
                for label, items in sorted(by_label.items())
            }
        out[name] = entry

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    for name, entry in out.items():
        print(f"=== {name}  (n={entry['n']}) ===")
        for label in LABELS:
            print(f"  {label:<18} {entry['dist'][label]:>7.3f}  ({entry['counts'].get(label, 0)})")
        print(
            "  sub: "
            + "  ".join(f"{k}={entry['dist_sub'][k]:.3f}" for k in entry["dist_sub"])
        )
        print(
            "  domain means: "
            + "  ".join(f"{k}={v:.2f}" for k, v in entry["domain"].items())
        )
        for task, t in entry["tasks"].items():
            print(
                f"  [{task:<8}] n={t['n']:<4} scaffold={t['dist']['chat_scaffold']:.3f} "
                f"raw={t['dist']['raw_continuation']:.3f} empty={t['dist']['empty']:.3f} "
                f"fence={t['dist']['code_fence']:.3f} json={t['dist']['json']:.3f} "
                f"refusal={t['dist']['refusal']:.3f} terse={t['dist_sub']['terse']:.3f}"
            )
        if args.examples:
            for label, items in entry.get("examples", {}).items():
                print(f"  -- {label}:")
                for x in items:
                    print(f"       {x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
