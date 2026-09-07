"""Answer extraction and normalization utilities for PLE grafting evals.

The existing Phase 0 harness used a short 16-token greedy generation and very
strict exact-match.  That protocol penalizes real/control readers that produce
explanatory/verbose answers.  This module provides the pieces needed for a more
robust protocol:

* normalize answers with SQuAD-style cleanup plus number-word expansion;
* match gold answers as either exact-normalized equality or a normalized
  substring containment (allowing ``The capital is Paris.`` to count for
  ``Paris``);
* extract a short answer from a longer generated response, preferring explicit
  answer markers and the final sentence when no marker is present.
"""

from __future__ import annotations

import re

_NUMBER_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_NUMBER_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_ARTICLES = {"a", "an", "the"}
_ANSWER_MARKERS = (
    "answer:",
    "the answer is:",
    "the answer is",
    "答案是",
    "答案：",
    "答案:",
    "therefore,",
    "so the answer is",
    "final answer:",
    "final answer",
)


def expand_number_words(text: str) -> str:
    """Convert common English number words in a normalized phrase to digits."""
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        if words[i] in _NUMBER_UNITS or words[i] in _NUMBER_TENS or words[i] in {
            "hundred",
            "thousand",
        }:
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


def normalize_answer(text: str, *, remove_articles: bool = True) -> str:
    """Normalize an answer/generation for tolerance-based matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    words = text.split()
    if remove_articles:
        words = [w for w in words if w not in _ARTICLES]
    return expand_number_words(" ".join(words))


def exact_match(prediction: str, gold: str) -> bool:
    """True when normalized prediction equals normalized gold."""
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    return bool(pred_norm) and pred_norm == gold_norm


def contains_match(prediction: str, gold: str) -> bool:
    """True when normalized gold appears in normalized prediction.

    This is the core lenient metric that prevents an explanation with the
    correct answer at the end from being counted as wrong.
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    if not gold_norm:
        return False
    return gold_norm in pred_norm


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s*", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_answer(generation: str) -> str:
    """Extract a concise answer candidate from a longer generated response."""
    text = generation.strip()
    if not text:
        return ""

    # 1. Explicit answer markers.
    lower = text.lower()
    marker_pos = -1
    marker_len = 0
    for marker in _ANSWER_MARKERS:
        pos = lower.rfind(marker)
        if pos >= 0 and pos + len(marker) > marker_pos + marker_len:
            marker_pos = pos
            marker_len = len(marker)
    if marker_pos >= 0:
        tail = text[marker_pos + marker_len :].strip()
        # Remove trailing explanatory boilerplate after the first newline.
        first_line = tail.split("\n", 1)[0].strip()
        if first_line:
            return _clean_extracted(first_line)

    # 2. Last sentence is usually the answer in a short explanatory response.
    sentences = _split_sentences(text)
    if sentences:
        return _clean_extracted(sentences[-1])

    # 3. Fallback: last non-empty line or whole text.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return _clean_extracted(lines[-1])
    return _clean_extracted(text)


def _clean_extracted(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(
        r"^(therefore|so|thus|hence|the answer is|answer)[,:.\s]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def score_answer(prediction: str, gold: str) -> dict[str, bool | str | float]:
    """Return both strict and lenient match signals for a single instance."""
    extracted = extract_answer(prediction)
    return {
        "exact": exact_match(prediction, gold),
        "contains": contains_match(prediction, gold),
        "extracted_exact": exact_match(extracted, gold),
        "extracted_contains": contains_match(extracted, gold),
        "extracted": extracted,
    }
