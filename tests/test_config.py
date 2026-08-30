"""Tests for qwen35-ple YAML configuration loading/validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qwen35_ple.config import load_config


def test_load_sample_config(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "project": "qwen35-08b-ple",
                "base_model": "Qwen/Qwen3.5-0.8B-Base",
                "tokenizer": "Qwen/Qwen3.5-0.8B-Base",
                "engine": {
                    "engine": "qwen_ple",
                    "table_spec": "PLE_QWEN_V1",
                    "table_source": "engramdb:view",
                    "view_path": "data/views/qwen-ple.view.bin",
                },
                "engram": {
                    "ngram_sizes": [2, 3],
                    "n_head_per_ngram": 8,
                    "embedding_dim": 2560,
                    "target_layers": [1],
                    "hc_mult": 1,
                },
                "training": {
                    "train_mode": "engram_only",
                    "backbone_freeze_steps": 1000,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.project == "qwen35-08b-ple"
    assert cfg.engine.engine == "qwen_ple"
    assert cfg.engine.table_source == "engramdb:view"
    assert cfg.engram.embedding_dim == 2560
    assert cfg.training.backbone_freeze_steps == 1000


def test_rejects_missing_view_for_view_source(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "project": "bad",
                "base_model": "x",
                "tokenizer": "x",
                "engine": {
                    "engine": "qwen_ple",
                    "table_spec": "PLE_QWEN_V1",
                    "table_source": "engramdb:view",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(cfg_path)

def test_repo_sample_config_is_loadable():
    sample = Path(__file__).resolve().parents[1] / "configs" / "qwen35-08b-ple.yaml"
    cfg = load_config(sample)
    assert cfg.engine.engine == "qwen_ple"
    assert cfg.engine.table_spec == "PLE_QWEN_V1"
    assert cfg.engram.target_layers == [1]
