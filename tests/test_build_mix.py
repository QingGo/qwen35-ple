"""Unit tests for the reproducible mix builder (scripts/build_mix.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_mix.py"


def _load_build_mix():
    spec = importlib.util.spec_from_file_location("build_mix", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bm():
    return _load_build_mix()


def test_parse_ratios_defaults(bm):
    ratios = bm._parse_ratios(None)
    assert ratios["general"] == 50
    assert ratios["chat"] == 20
    assert ratios["wiki"] == 20
    assert ratios["cot"] + ratios["tool"] == 10


def test_parse_ratios_custom(bm):
    ratios = bm._parse_ratios("general=20,chat=40,wiki=20,cot=10,tool=10")
    assert sum(ratios.values()) == 100


def test_extract_instruction_output(bm):
    text = bm._extract_text(
        {"instruction": "Hello", "output": "World"}
    )
    assert text is not None
    assert "<|im_start|>user\nHello" in text
    assert "<|im_start|>assistant\nWorld" in text


def test_extract_cot(bm):
    text = bm._extract_text(
        {
            "problem": "1+1",
            "thinking": "Let me add.",
            "solution": "2",
        }
    )
    assert text is not None
    assert "<think>\nLet me add.\n</think>" in text
    assert "\n2\n<|im_end|>" in text


def test_extract_messages(bm):
    text = bm._extract_text(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
    )
    assert text is not None
    assert "<|im_start|>user\nhi" in text
    assert "<|im_start|>assistant\nhello" in text


def test_split_long_text(bm):
    text = "word " * 1000
    chunks = bm._split_long_text(text, chunk_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)  # small slack for sentence split
    assert "".join(chunks).replace(" ", "") == "word" * 1000


def test_filter_contaminated(bm):
    needles = {"isaac newton", "william shakespeare"}
    texts = [
        "Isaac Newton was a physicist.",
        "A completely unrelated sentence.",
        "Hamlet was written by William Shakespeare.",
    ]
    kept, removed = bm._filter_contaminated(texts, needles)
    assert removed == 2
    assert kept == ["A completely unrelated sentence."]
