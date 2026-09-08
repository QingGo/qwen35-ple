"""Tests for the improved Phase 1 answer matching/extraction protocol."""

from __future__ import annotations

from qwen35_ple.eval.answers import (
    contains_match,
    exact_match,
    extract_answer,
    extract_answer_v2,
    normalize_answer,
    score_answer,
    score_answer_v2,
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


def test_extract_answer_v2_prefers_answer_marker() -> None:
    gen = "Let me think. The correct answer is **Paris**. It is the capital."
    assert extract_answer_v2(gen) == "Paris."
    assert score_answer_v2(gen, "Paris")["extracted_exact"] is True


def test_extract_answer_v2_prefers_first_sentence() -> None:
    gen = "The capital of France is **Paris**. It is the largest city."
    assert extract_answer_v2(gen) == "The capital of France is Paris."
    assert score_answer_v2(gen, "Paris")["extracted_contains"] is True


def test_extract_answer_v2_skips_leading_question() -> None:
    gen = "What is the capital of France? Paris."
    assert extract_answer_v2(gen) == "Paris."


def test_extract_answer_v2_boolq_ambiguous_option_list() -> None:
    gen = "?\nA: Yes\nB: No\nAnswer:\n\n<think>\nWe need to check the passage."
    assert extract_answer_v2(gen, task="boolq") == ""
    assert score_answer_v2(gen, "yes", task="boolq")["extracted_exact"] is False


def test_extract_answer_v2_boolq_explicit_yes_no() -> None:
    assert extract_answer_v2("assistant\nYes, they are the same.", task="boolq") == "yes"
    assert extract_answer_v2("The correct answer is **B. No**.", task="boolq") == "no"
    assert extract_answer_v2("Answer: A. Yes.", task="boolq") == "yes"
