"""Tests for the PLE-2 serving router / logit processor."""

from __future__ import annotations

import numpy as np

from qwen35_ple.addressable_memory import AddressableNgramMemory
from qwen35_ple.router import CalibratedNgramLogitProcessor


def _memory() -> AddressableNgramMemory:
    mem = AddressableNgramMemory(min_order=2, max_order=4)
    mem.add_sequence([1, 2, 3, 1, 2, 3, 4])
    return mem


def test_processor_disabled_returns_same() -> None:
    proc = CalibratedNgramLogitProcessor(_memory(), enabled=False)
    logits = np.zeros(10, dtype=np.float32)
    out = proc(logits, [1, 2])
    assert np.array_equal(out, logits)


def test_processor_applies_ngram_bias() -> None:
    proc = CalibratedNgramLogitProcessor(
        _memory(), scale=1.0, bias=3.0, temperature=1.0
    )
    logits = np.zeros(10, dtype=np.float32)
    out = proc(logits, [1, 2])
    # Token 3 is the dominant continuation; with positive bias it should become
    # the argmax among ngram candidates and above untouched tokens.
    assert out[3] > 0.0
    assert int(np.argmax(out)) == 3


def test_processor_state_dict() -> None:
    proc = CalibratedNgramLogitProcessor(_memory(), scale=0.5, bias=-1.0, temperature=2.0)
    state = proc.state_dict()
    assert state["scale"] == 0.5
    assert state["bias"] == -1.0
    assert state["temperature"] == 2.0
    assert state["enabled"] is True


def test_task_classifier_routes_semantic_code_number_name() -> None:
    from qwen35_ple.router import TaskClassifier

    clf = TaskClassifier()
    assert clf.classify("What is the capital of France?") == "semantic"
    assert clf.classify("def foo():\n    return 1") == "code"
    assert clf.classify("Compute the sum of 1 2 3") == "number"
    assert clf.classify("the author of this book is called Ada") == "name"
    assert clf.classify("random unrelated prose") == "general"


def test_log_density_ratio_is_positive_for_informative_memory() -> None:
    from qwen35_ple.router import log_density_ratio

    base = np.zeros(10, dtype=np.float32)
    dist = {5: 0.9, 7: 0.1}
    score = log_density_ratio(base, dist)
    assert score > 0.0
    assert log_density_ratio(base, None) == -float("inf")


def test_density_gate_modes_and_persistence() -> None:
    from qwen35_ple.router import (
        LogDensityRatioGate,
        load_fusion_router_config,
        save_fusion_router_config,
    )

    gate = LogDensityRatioGate(mode="expected_kl", threshold=0.0)
    base = np.zeros(10, dtype=np.float32)
    dist = {5: 0.9, 7: 0.1}
    info = gate.evaluate(base, dist)
    assert info["active"] is True
    assert info["expected_log_density_ratio"] > 0.0
    assert info["base_top1_prob"] > 0.0

    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "router.json"
        cfg = save_fusion_router_config(
            path,
            {
                "fusion": {"scale": 1.25, "bias": -0.5},
                "router": {"mode": "pseudo_label", "min_log_density_ratio": 0.1},
            },
        )
        loaded = load_fusion_router_config(path)
        assert loaded["fusion"]["scale"] == 1.25
        assert loaded["fusion"]["bias"] == -0.5
        assert loaded["router"]["mode"] == "pseudo_label"
        assert loaded["router"]["min_log_density_ratio"] == 0.1
        # Defaults survive partial updates.
        assert loaded["router"]["task_scale"]["code"] == 1.0
        assert json.loads(path.read_text())["schema"] == "ngram-fusion-router-v1"
        assert cfg["fusion"]["temperature"] == 0.5


def test_task_conditioned_processor_disables_semantic() -> None:
    from qwen35_ple.router import (
        LogDensityRatioGate,
        TaskConditionedNgramLogitProcessor,
    )

    proc = TaskConditionedNgramLogitProcessor(
        _memory(),
        scale=1.0,
        bias=3.0,
        temperature=1.0,
        task="semantic",
        density_gate=LogDensityRatioGate(mode="expected_kl", threshold=0.0),
    )
    logits = np.zeros(10, dtype=np.float32)
    out = proc(logits, [1, 2])
    assert np.array_equal(out, logits)


def test_task_conditioned_processor_applies_for_code() -> None:
    from qwen35_ple.router import (
        LogDensityRatioGate,
        TaskConditionedNgramLogitProcessor,
    )

    proc = TaskConditionedNgramLogitProcessor(
        _memory(),
        scale=1.0,
        bias=3.0,
        temperature=1.0,
        task="code",
        density_gate=LogDensityRatioGate(mode="expected_kl", threshold=0.0),
    )
    logits = np.zeros(10, dtype=np.float32)
    out = proc(logits, [1, 2])
    assert out[3] > 0.0
    assert int(np.argmax(out)) == 3
    state = proc.state_dict()
    assert state["task"] == "code"
    assert "density_gate" in state


def test_build_processor_from_config(tmp_path) -> None:
    import json

    from qwen35_ple.router import build_task_conditioned_processor

    path = tmp_path / "router.json"
    path.write_text(
        json.dumps(
            {
                "fusion": {"scale": 2.0, "bias": -1.0, "temperature": 0.5},
                "router": {
                    "mode": "expected_kl",
                    "min_log_density_ratio": 0.0,
                    "default_task": "code",
                },
            }
        ),
        encoding="utf-8",
    )
    proc = build_task_conditioned_processor(_memory(), path)
    assert proc.scale == 2.0
    assert proc.bias == -1.0
    assert proc.default_task == "code"
    assert proc.density_gate.mode == "expected_kl"


def test_task_router_returns_channel_weights() -> None:
    from qwen35_ple.router import TaskRouter

    router = TaskRouter()
    semantic = router.route("What is the capital of France?")
    assert semantic["task"] == "semantic"
    assert semantic["channel_weights"]["ngram_weight"] == 0.0
    assert semantic["channel_weights"]["dense_weight"] == 2.0

    code = router.route("def add(a, b):\n    return a + b")
    assert code["task"] == "code"
    assert code["channel_weights"]["ngram_weight"] == 2.0

    custom = TaskRouter(
        channel_weights={
            "semantic": {"bm25_weight": 0.5, "dense_weight": 1.0, "ngram_weight": 0.0}
        }
    )
    assert custom.route("What is the capital of France?")["channel_weights"]["bm25_weight"] == 0.5


def test_build_task_router_from_config(tmp_path) -> None:
    from qwen35_ple.router import (
        build_task_router_from_config,
        save_fusion_router_config,
    )

    router = build_task_router_from_config()
    assert router.route("What is the capital?")["task"] == "semantic"
    assert router.route("def f(): pass")["task"] == "code"

    cfg = save_fusion_router_config(
        tmp_path / "router.json",
        {
            "router": {
                "channel_weights": {
                    "semantic": {
                        "bm25_weight": 0.25,
                        "dense_weight": 3.0,
                        "ngram_weight": 0.0,
                    }
                }
            }
        },
    )
    loaded = build_task_router_from_config(cfg)
    assert loaded.route("Who is Ada?")["channel_weights"]["bm25_weight"] == 0.25


def test_serving_adapter_auto_builds_from_config(tmp_path) -> None:
    import json

    import pytest

    pytest.importorskip("torch")

    from qwen35_ple.addressable_memory import AddressableNgramMemory
    from qwen35_ple.rag import BM25Index, HybridRetriever
    from qwen35_ple.serving.rag import RAGServingAdapter

    mem = AddressableNgramMemory(min_order=2, max_order=4)
    mem.add_sequence([1, 2, 3, 4])
    config = tmp_path / "router.json"
    config.write_text(
        json.dumps(
            {
                "fusion": {"scale": 1.0, "bias": -1.0, "temperature": 0.5},
                "router": {
                    "mode": "expected_kl",
                    "min_log_density_ratio": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    class DummyModel:
        pass

    class DummyTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [1, 2]

        def decode(self, ids) -> str:
            return "dummy"

    adapter = RAGServingAdapter(
        DummyModel(),
        DummyTokenizer(),
        HybridRetriever(BM25Index(["alpha beta"])),
        ngram_memory=mem,
        fusion_config=config,
        max_new_tokens=0,
    )
    assert adapter.task_router is not None
    assert adapter.logit_processor is not None
    assert hasattr(adapter.logit_processor, "set_task")
    assert adapter.logit_processor.scale == 1.0
