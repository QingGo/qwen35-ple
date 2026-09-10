"""Tests for QA-SFT loading and cache construction."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")


def _load_harness():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_phase0.py"
    return runpy.run_path(str(path), run_name="test_run_phase0")


class _FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(ch) % 97 + 1 for ch in text]


class _FakeStore:
    def fetch(self, ids):
        return np.zeros((len(ids), 2560), dtype=np.float32)


def test_load_qa_file_supports_jsonl(tmp_path: Path):
    harness = _load_harness()
    path = tmp_path / "qa.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"task": "triviaqa", "question": "q1", "answer": "a1"}),
                json.dumps({"task": "boolq", "question": "q2", "answer": "yes"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    items = harness["_load_qa_file"](path)
    assert [item["task"] for item in items] == ["triviaqa", "boolq"]
    assert items[1]["answer"] == "yes"


def test_load_qa_sft_file_supports_json_and_jsonl(tmp_path: Path):
    harness = _load_harness()
    jsonl = tmp_path / "sft.jsonl"
    jsonl.write_text(
        json.dumps({"question": "q", "answer": "a"}) + "\n", encoding="utf-8"
    )
    assert harness["_load_qa_sft_file"](jsonl)[0]["question"] == "q"

    as_json = tmp_path / "sft.json"
    as_json.write_text(
        json.dumps([{"question": "q2", "answer": "a2"}]), encoding="utf-8"
    )
    assert harness["_load_qa_sft_file"](as_json)[0]["answer"] == "a2"


class _ChatFakeTokenizer(_FakeTokenizer):
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        return [7, 8, 9]


def test_qa_prompt_ids_uses_chat_template_when_requested():
    harness = _load_harness()
    tokenizer = _ChatFakeTokenizer()
    ids = harness["_qa_prompt_ids"](
        tokenizer,
        {"task": "triviaqa", "question": "capital?", "answer": "Paris"},
        prompt_template="Question: {question}\nAnswer:",
        boolq_prompt_template=None,
        chat_template=True,
        chat_enable_thinking=True,
    )
    assert ids == [7, 8, 9]
    assert tokenizer.calls[0]["enable_thinking"] is True
    assert tokenizer.calls[0]["add_generation_prompt"] is True
    assert tokenizer.calls[0]["messages"][0]["role"] == "user"


def test_qa_prompt_ids_keeps_completion_prompt_by_default():
    harness = _load_harness()
    tokenizer = _ChatFakeTokenizer()
    ids = harness["_qa_prompt_ids"](
        tokenizer,
        {"task": "triviaqa", "question": "capital?", "answer": "Paris"},
        prompt_template="Question: {question}\nAnswer:",
        boolq_prompt_template=None,
        chat_template=False,
    )
    expected = tokenizer.encode("Question: capital?\nAnswer:")
    assert ids == expected
    assert tokenizer.calls == []


def test_build_qa_sft_cache_masks_prompt():
    harness = _load_harness()
    tokenizer = _FakeTokenizer()
    store = _FakeStore()
    items = [
        {"task": "triviaqa", "question": "capital?", "answer": "Paris"},
    ]
    cache = harness["_build_qa_sft_cache"](
        items,
        tokenizer,
        store,
        prompt_template="Question: {question}\nAnswer:",
        boolq_prompt_template="Question: {question}\nAnswer:",
        max_len=256,
        eos_id=tokenizer.eos_token_id,
    )
    assert len(cache) == 1
    entry = cache[0]
    prompt_ids = tokenizer.encode("Question: capital?\nAnswer:")
    answer_ids = tokenizer.encode("Paris") + [tokenizer.eos_token_id]
    assert entry["prompt_len"] == len(prompt_ids)
    assert len(entry["ids"]) == len(prompt_ids) + len(answer_ids)
    assert entry["e_t"].shape == (len(entry["ids"]), 2560)
    assert entry["ids"][-1] == tokenizer.eos_token_id


def test_build_qa_sft_cache_lazy_fetches_on_demand():
    harness = _load_harness()
    tokenizer = _FakeTokenizer()

    class _CountingStore:
        def __init__(self):
            self.calls = 0

        def fetch(self, ids):
            self.calls += 1
            return np.zeros((len(ids), 2560), dtype=np.float32)

    store = _CountingStore()
    entries = harness["_build_qa_sft_cache"](
        [{"task": "triviaqa", "question": "capital?", "answer": "Paris"}],
        tokenizer,
        store,
        prompt_template="Question: {question}\nAnswer:",
        boolq_prompt_template="Question: {question}\nAnswer:",
        max_len=256,
        eos_id=tokenizer.eos_token_id,
        lazy=True,
    )
    assert store.calls == 0
    assert "e_t" not in entries[0]
    cache = harness["_LazyQASFTCache"](entries, store, max_cached=1)
    item = cache[0]
    assert store.calls == 1
    assert item["e_t"].shape == (len(item["ids"]), 2560)
    _ = cache[0]
    assert store.calls == 1


def test_build_qa_sft_cache_left_truncates_long_prompt():
    harness = _load_harness()
    tokenizer = _FakeTokenizer()
    store = _FakeStore()
    items = [
        {"task": "boolq", "question": "x" * 200, "answer": "yes"},
    ]
    cache = harness["_build_qa_sft_cache"](
        items,
        tokenizer,
        store,
        prompt_template="Question: {question}\nAnswer:",
        boolq_prompt_template="Question: {question}\nAnswer:",
        max_len=32,
        eos_id=tokenizer.eos_token_id,
    )
    assert len(cache) == 1
    entry = cache[0]
    assert entry["prompt_len"] < len(tokenizer.encode("Question: " + "x" * 200 + "\nAnswer:"))
    assert len(entry["ids"]) <= 32
