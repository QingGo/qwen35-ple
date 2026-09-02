"""Tests for qwen35-ple reader registry and serving bundle helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def _args(**overrides):
    base = {
        "hc_mult": 4,
        "kernel_size": 4,
        "dilation": 3,
        "bridge_mlp": False,
        "bridge_hidden": None,
        "out_mlp": False,
        "out_hidden": None,
        "branches": 1,
        "zero_init_v": False,
        "short_conv": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_reader_config_from_args_official() -> None:
    from qwen35_ple.reader_registry import (
        OFFICIAL_SOURCE_QWEN_V1,
        reader_config_from_args,
    )

    cfg = reader_config_from_args(_args(), d_target=8, reader_name=OFFICIAL_SOURCE_QWEN_V1)
    assert cfg["d_target"] == 8
    assert cfg["hc"] == 4
    assert cfg["d_mem"] == 2560
    assert "checkpoint_path" not in cfg


def test_reader_config_from_args_engram() -> None:
    from qwen35_ple.reader_registry import ENGRAM_V1, reader_config_from_args

    cfg = reader_config_from_args(_args(hc_mult=2), d_target=16, reader_name=ENGRAM_V1)
    assert cfg["d_model"] == 16
    assert cfg["hc_mult"] == 2
    assert cfg["d_mem"] == 2560


def test_save_and_load_official_reader() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("engramdb")

    from qwen35_ple.reader import OfficialSourceQwenReader
    from qwen35_ple.reader_registry import (
        OFFICIAL_SOURCE_QWEN_V1,
        get_registry,
        load_reader,
        save_reader,
    )

    reader = OfficialSourceQwenReader(
        d_target=8,
        d_source=4,
        d_mem=4,
        hc=1,
        kernel_size=2,
        dilation=1,
        freeze_source=False,
        zero_init_out=True,
    )
    with torch.no_grad():
        for param in reader.parameters():
            param.copy_(torch.randn_like(param))

    config = {
        "d_target": 8,
        "d_source": 4,
        "d_mem": 4,
        "hc": 1,
        "kernel_size": 2,
        "dilation": 1,
        "freeze_source": False,
        "zero_init_out": True,
        "bridge_mlp": False,
        "bridge_hidden": None,
        "out_mlp": False,
        "out_hidden": None,
    }
    with tempfile.TemporaryDirectory(prefix="reader-registry-") as td:
        path = Path(td) / "reader.pt"
        save_reader(
            reader,
            path,
            name=OFFICIAL_SOURCE_QWEN_V1,
            config=config,
        )
        restored = load_reader(path, registry=get_registry())

    state = reader.state_dict()
    loaded_state = restored.state_dict()
    assert set(state) == set(loaded_state)
    for key in state:
        torch.testing.assert_close(state[key], loaded_state[key], atol=0, rtol=0)


def test_save_and_load_simple_reader_with_short_conv_extra() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("engramdb")

    from qwen35_ple.reader import EngramReader, ShortConv
    from qwen35_ple.reader_registry import (
        SIMPLE_V1,
        get_registry,
        load_reader_with_extra,
        save_reader,
    )

    reader = EngramReader(d_model=8, d_mem=8, num_branches=1)
    short_conv = ShortConv(hidden_size=8, kernel_size=2, dilation=1)
    with torch.no_grad():
        for param in reader.parameters():
            param.copy_(torch.randn_like(param))
        for param in short_conv.parameters():
            param.copy_(torch.randn_like(param))

    config = {
        "d_model": 8,
        "d_mem": 8,
        "gate_bias_init": -2.0,
        "num_branches": 1,
        "zero_init_v": False,
    }
    with tempfile.TemporaryDirectory(prefix="simple-reader-extra-") as td:
        path = Path(td) / "reader.pt"
        save_reader(
            reader,
            path,
            name=SIMPLE_V1,
            config=config,
            extra_state={"short_conv": short_conv.state_dict()},
        )
        restored, extra = load_reader_with_extra(path, registry=get_registry())
        restored_short = ShortConv(hidden_size=8, kernel_size=2, dilation=1)
        restored_short.load_state_dict(extra["short_conv"])

    for key in reader.state_dict():
        torch.testing.assert_close(reader.state_dict()[key], restored.state_dict()[key], atol=0, rtol=0)
    for key in short_conv.state_dict():
        torch.testing.assert_close(short_conv.state_dict()[key], restored_short.state_dict()[key], atol=0, rtol=0)


def test_bundle_roundtrip() -> None:
    pytest.importorskip("engramdb")

    from qwen35_ple.serving.bundle import load_bundle, make_bundle, save_bundle

    bundle = make_bundle(
        backbone_path="/tmp/model",
        memory={
            "type": "store",
            "store": {
                "path": "/tmp/rows",
                "shards": 1,
                "rows_per_shard": 1,
                "width": 160,
            },
        },
        ple={
            "ple_embed_dim": 2560,
            "num_heads": 16,
            "head_dim": 160,
            "scale": 1.0,
        },
        readers=[
            {
                "name": "official_source_qwen_v1",
                "version": "1",
                "path": "/tmp/reader.pt",
                "options": {},
            }
        ],
    )
    with tempfile.TemporaryDirectory(prefix="qwen-bundle-") as td:
        path = Path(td) / "bundle.json"
        save_bundle(bundle, path)
        loaded = load_bundle(path)
        assert loaded.schema == "engramdb-bundle-v1"
        assert loaded.validate() == []
        assert loaded.readers[0]["name"] == "official_source_qwen_v1"


def test_qwen_reader_serving_adapter_injects_e_t() -> None:
    torch = pytest.importorskip("torch")

    from qwen35_ple.serving.adapter import QwenReaderServingAdapter

    class DummyLayer(torch.nn.Module):
        def forward(self, x):
            return x

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([DummyLayer()])

        def forward(self, input_ids):
            x = input_ids.float().unsqueeze(-1).expand(-1, -1, 3)
            return self.model.layers[0](x)

    class DummyMemory:
        def __call__(self, input_ids):
            return torch.ones(input_ids.shape[0], input_ids.shape[1], 4)

    class DummyReader:
        def __call__(self, hidden, e_t):
            return e_t[..., : hidden.size(-1)]

    model = DummyModel()
    adapter = QwenReaderServingAdapter(
        model,
        DummyReader(),
        layer_index=0,
        memory_adapter=DummyMemory(),
    ).install()
    try:
        input_ids = torch.tensor([[1, 2, 3]])
        out = model(input_ids)
        expected = input_ids.float().unsqueeze(-1).expand(-1, -1, 3) + 1.0
        torch.testing.assert_close(out, expected)
        assert model._current_ple_e_t is not None
    finally:
        adapter.remove()
    assert model._current_ple_e_t is None
