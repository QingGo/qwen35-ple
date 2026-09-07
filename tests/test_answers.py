"""Tests for the improved Phase 1 answer matching/extraction protocol."""

from __future__ import annotations

from qwen35_ple.eval.answers import (
    contains_match,
    exact_match,
    extract_answer,
    normalize_answer,
    score_answer,
)


def test_normalize_basic() -> None:
    assert normalize_answer("The Capital of France!") == "capital of france"
    assert normalize_answer("42") == "42"
    assert normalize_answer("forty-two") == "42"  # hyphen -> space -> number words


def test_normalize_number_words() -> None:
    assert normalize_answer("twenty one") == "21"
    assert normalize_answer("one hundred five") == "105"


def test_exact_and_contains() -> None:
    assert exact_match("Paris", "Paris")
    assert not exact_match("The capital is Paris.", "Paris")
    assert contains_match("The capital is Paris.", "Paris")
    assert contains_match("The answer is Jupiter.", "Jupiter")
    assert not contains_match("The answer is Saturn.", "Jupiter")


def test_extract_answer_marker() -> None:
    gen = "Let me think. The answer is Paris."
    assert extract_answer(gen) == "Paris."


def test_extract_answer_final_sentence() -> None:
    gen = "Based on the evidence, the largest planet is Jupiter."
    extracted = extract_answer(gen)
    assert "Jupiter" in extracted
    assert contains_match(extracted, "Jupiter")


def test_score_answer_reports_lenient() -> None:
    score = score_answer("The correct answer is 42.", "42")
    assert score["exact"] is False
    assert score["contains"] is True
    assert score["extracted_contains"] is True
