#!/usr/bin/env python3
"""Analyze three-arm 150-QA results and check corpus overlap.

Usage:
    python scripts/analyze_qa_lines.py \
      --real outputs/phase0-live1m-qa150-loaded.json \
      --noreader outputs/phase0-live1m-qa150-noreader.json \
      --control outputs/phase0-live1m-qa150-control.json \
      --corpus data/wet-1m-one.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_answers(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results = data.get("results", [])
    answers: list[dict] = []
    for res in results:
        qa = res.get("qa_exact")
        if qa and qa.get("answers"):
            answers = qa["answers"]
            break
    return answers


_NUMBER_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_NUMBER_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _expand_number_words(text: str) -> str:
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        if words[i] in _NUMBER_UNITS or words[i] in _NUMBER_TENS or words[i] in {"hundred", "thousand"}:
            total = 0
            current = 0
            while i < len(words):
                w = words[i]
                if w in _NUMBER_UNITS:
                    current += _NUMBER_UNITS[w]
                elif w == "hundred":
                    current *= 100
                elif w in _NUMBER_TENS:
                    current += _NUMBER_TENS[w]
                elif w == "thousand":
                    total += current * 1000
                    current = 0
                else:
                    break
                i += 1
            total += current
            out.append(str(total))
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    words = [w for w in text.split() if w not in {"a", "an", "the"}]
    return _expand_number_words(" ".join(words))


def answer_in_corpus(answer: str, corpus_text: str) -> bool:
    norm = normalize_answer(answer)
    if not norm:
        return False
    # Both exact normalized phrase and the core answer should be checked.
    return norm in corpus_text.lower()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--noreader", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--corpus", default="data/wet-1m-one.txt")
    args = parser.parse_args()

    real = load_answers(args.real)
    no = load_answers(args.noreader)
    ctrl = load_answers(args.control)
    corpus = Path(args.corpus).read_text(encoding="utf-8", errors="ignore")

    assert len(real) == len(no) == len(ctrl), (len(real), len(no), len(ctrl))

    n = len(real)
    def em_mean(ans):
        return sum(a["correct"] for a in ans) / n

    print("=== Overall ===")
    print(f"n={n}  real={em_mean(real):.3f}  no-reader={em_mean(no):.3f}  control={em_mean(ctrl):.3f}")

    # Per task.
    tasks = sorted({a["task"] for a in real})
    print("\n=== Per task ===")
    for t in tasks:
        r = [a for a in real if a["task"] == t]
        nr = [a for a in no if a["task"] == t]
        c = [a for a in ctrl if a["task"] == t]
        print(f"{t:10s} real={sum(a['correct'] for a in r)/len(r):.3f} "
              f"no-reader={sum(a['correct'] for a in nr)/len(nr):.3f} "
              f"control={sum(a['correct'] for a in c)/len(c):.3f}")

    new_correct = []
    new_wrong = []
    mem_correct = []
    mem_wrong = []
    for i in range(n):
        r, nr, c = real[i], no[i], ctrl[i]
        # Keep row indices for cross-checking.
        row = {
            "index": i,
            "task": r["task"],
            "question": r["question"],
            "answer": r["answer"],
            "real_generated": r["generated"],
            "no_generated": nr["generated"],
            "control_generated": c["generated"],
            "real_correct": r["correct"],
            "no_correct": nr["correct"],
            "control_correct": c["correct"],
            "answer_in_corpus": answer_in_corpus(r["answer"], corpus),
        }
        if r["correct"] and not nr["correct"]:
            new_correct.append(row)
        if (not r["correct"]) and nr["correct"]:
            new_wrong.append(row)
        if r["correct"] and not c["correct"]:
            mem_correct.append(row)
        if r["correct"] is False and c["correct"] is True:
            mem_wrong.append(row)

    def dump(title, rows):
        print(f"\n=== {title} ({len(rows)}) ===")
        for row in rows:
            mark = "CORPUS_HIT" if row["answer_in_corpus"] else "not-in-corpus"
            print(f"[{row['index']}] {row['task']} | {row['question']} | answer={row['answer']} | {mark}")
            print(f"    real_gen: {row['real_generated'][:120]}")
            print(f"    no_gen:   {row['no_generated'][:120]}")
            print(f"    ctrl_gen: {row['control_generated'][:120]}")

    dump("New correct vs no-reader", new_correct)
    dump("New wrong vs no-reader", new_wrong)
    dump("New correct vs control (memory-specific)", mem_correct)
    dump("New wrong vs control (memory-specific)", mem_wrong)

    corpus_hits = sum(row["answer_in_corpus"] for row in new_correct)
    print("\n=== Corpus overlap among new-correct vs no-reader ===")
    print(f"new_correct_total={len(new_correct)}, answer_in_corpus={corpus_hits}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
