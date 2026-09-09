"""Tests for QA prompt formatting."""

from __future__ import annotations

import pytest

from qwen35_ple.eval.prompting import (
    DEFAULT_BOOLQ_PROMPT,
    DEFAULT_QA_PROMPT,
    format_qa_prompt,
)


def test_legacy_prompt_is_raw_question() -> None:
    assert format_qa_prompt("What is 2+2?") == "What is 2+2?"
    assert format_qa_prompt("Is the sky blue?", task="boolq") == "Is the sky blue?"


def test_template_applies_to_all_tasks() -> None:
    assert (
        format_qa_prompt("What is 2+2?", template=DEFAULT_QA_PROMPT)
        == "Question: What is 2+2?\nAnswer:"
    )


def test_boolq_template_overrides() -> None:
    assert (
        format_qa_prompt(
            "Is the sky blue?",
            task="boolq",
            template=DEFAULT_QA_PROMPT,
            boolq_template=DEFAULT_BOOLQ_PROMPT,
        )
        == "Question: Is the sky blue?\nAnswer with one word, Yes or No:"
    )


def test_template_must_contain_question() -> None:
    with pytest.raises(ValueError, match="question"):
        format_qa_prompt("x", template="Answer:")
