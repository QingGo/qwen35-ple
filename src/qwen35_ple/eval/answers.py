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


def _clean_v2_extracted(text: str) -> str:
    """Clean a v2 candidate without changing its lexical content."""
    text = re.sub(r"[*_`]+", "", text)
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


# ---------------------------------------------------------------------------
# v2 protocol
#
# The v1 extractor above uses the last sentence when no answer marker is
# present.  That is a poor fit for the Phase 2 generations, which often start
# with a direct answer and then continue with reasoning.  v2 prefers explicit
# answer markers, then the first meaningful sentence, and treats option lists
# in BoolQ as ambiguous instead of counting a generation that contains both
# ``yes`` and ``no`` as correct.
# ---------------------------------------------------------------------------

_V2_ANSWER_MARKERS = (
    r"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:)\s*",
    r"(?:答案|回答)\s*[:：]\s*",
)

_V2_YES_NO_ASSERTIONS = (
    r"(?:^|\n)\s*(?:assistant\s*)?\**(yes|no)\b",
    (
        r"\b(?:the\s+answer\s+is|answer\s+is|correct\s+answer\s+is|"
        r"i\s+(?:would\s+)?(?:say|choose|select))\s*\**\s*(yes|no)\b"
    ),
    r"\b(?:so|therefore|thus|hence)\s*(?:,|:)?\s*\**(yes|no)\b",
)


def _strip_generation_roles(text: str) -> str:
    text = text.strip()
    text = re.sub(r"</?think>\s*", " ", text, flags=re.IGNORECASE)
    # Chat-template control tokens may leak into generated text when the
    # stop-token set is incomplete; strip them before answer extraction.
    text = re.sub(r"<\|(?:im_start|im_end|endoftext)\|>", " ", text)
    text = re.sub(
        r"^\s*(?:assistant|user|system)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip(" \t\r\n?:")


def _first_meaningful_sentence(text: str) -> str:
    """Return the first non-empty sentence, skipping leading questions."""
    sentences = _split_sentences(text)
    for index, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not re.search(r"[A-Za-z0-9]", sentence):
            continue
        if re.fullmatch(r"</?think>", sentence, flags=re.IGNORECASE):
            continue
        # A leading question is usually the prompt, not the answer.  Skip it
        # when a following sentence exists.
        if sentence.endswith("?") and index + 1 < len(sentences):
            continue
        return sentence
    return ""


def _answer_marker_tail(text: str) -> str:
    """Return text after the last explicit answer marker, or ``""``."""
    best_end = -1
    for pattern in _V2_ANSWER_MARKERS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            best_end = max(best_end, match.end())
    if best_end < 0:
        return ""
    return text[best_end:].strip()


def _yes_no_from_tail(tail: str, task: str | None) -> str:
    """Extract yes/no from an explicit-answer tail."""
    tail = tail.strip()
    if not tail:
        return ""
    option = re.match(
        r"(?:option\s*)?[\(\*]*([A-D])[\)\.\*]*\s*[,:;]?\s*(.*)",
        tail,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if option:
        rest = option.group(2).strip()
        if rest:
            match = re.search(r"\b(yes|no)\b", rest, flags=re.IGNORECASE)
            if match:
                return match.group(1).lower()
            if task == "boolq":
                return ""
            return _first_meaningful_sentence(rest)
        if task == "boolq":
            return {"A": "yes", "B": "no"}.get(option.group(1).upper(), "")
    match = re.search(r"\b(yes|no)\b", tail, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return ""


def extract_answer_v2(generation: str, task: str | None = None) -> str:
    """Extract an answer candidate with the v2 protocol.

    ``task`` enables BoolQ-specific handling.  Without it, the function still
    prefers explicit answer markers and first sentences, but cannot reject an
    ambiguous yes/no option list.
    """
    text = _strip_generation_roles(generation)
    if not text:
        return ""

    tail = _answer_marker_tail(text)
    if tail:
        if task == "boolq":
            answer = _yes_no_from_tail(tail, task)
        else:
            answer = _clean_v2_extracted(_first_meaningful_sentence(tail))
        if answer:
            return answer

    if task == "boolq":
        assertions: list[tuple[int, str]] = []
        for pattern in _V2_YES_NO_ASSERTIONS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                assertions.append((match.start(), match.group(1).lower()))
        if assertions:
            assertions.sort(key=lambda item: item[0])
            return assertions[0][1]
        # Option lists such as ``A: Yes / B: No`` are deliberately treated as
        # ambiguous rather than as an answer.
        return ""

    first = _clean_v2_extracted(_first_meaningful_sentence(text))
    if first:
        return first
    return _clean_v2_extracted(extract_answer(generation))


def score_answer_v2(
    prediction: str, gold: str, task: str | None = None
) -> dict[str, bool | str | float]:
    """Return strict/lenient signals using the v2 extractor."""
    extracted = extract_answer_v2(prediction, task=task)
    return {
        "exact": exact_match(prediction, gold),
        "contains": contains_match(prediction, gold),
        "extracted_exact": exact_match(extracted, gold),
        "extracted_contains": contains_match(extracted, gold),
        "extracted": extracted,
    }
