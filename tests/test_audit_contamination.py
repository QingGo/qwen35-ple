"""Unit tests for the strict contamination audit script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_contamination.py"


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_contamination", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_normalize_text(audit):
    assert audit.normalize_text("The Capital of France!") == "capital france"


def test_ngrams_short_text(audit):
    assert audit.ngrams("yes", 8) == {"yes"}


def test_ngrams_long_text(audit):
    grams = audit.ngrams("one two three four", 2)
    assert "one two" in grams
    assert "three four" in grams
    assert "two three" in grams


def test_overlap_ratio_exact(audit):
    ratio = audit.overlap_ratio(
        "isaac newton was born in england",
        "isaac newton",
        2,
    )
    assert ratio == 1.0


def test_overlap_ratio_zero(audit):
    ratio = audit.overlap_ratio(
        "completely different text",
        "isaac newton",
        2,
    )
    assert ratio == 0.0
