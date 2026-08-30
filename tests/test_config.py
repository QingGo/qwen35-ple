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


def test_load_synthetic_prime_sizes(tmp_path):
    cfg_path = tmp_path / "synthetic.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "project": "synthetic",
                "base_model": "tiny",
                "tokenizer": "tiny",
                "engine": {"engine": "qwen_ple", "table_spec": "PLE_QWEN_V1"},
                "engram": {
                    "embedding_dim": 160,
                    "prime_sizes": [
                        17, 19, 23, 29, 31, 37, 41, 43,
                        47, 53, 59, 61, 67, 71, 73, 79,
                    ],
                    "use_sparse_embeddings": False,
                },
                "training": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.engram.embedding_dim == 160
    assert cfg.engram.prime_sizes == [
        17, 19, 23, 29, 31, 37, 41, 43,
        47, 53, 59, 61, 67, 71, 73, 79,
    ]
    assert cfg.engram.use_sparse_embeddings is False


def test_rejects_bad_prime_sizes(tmp_path):
    cfg_path = tmp_path / "bad-prime.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "project": "synthetic",
                "base_model": "tiny",
                "tokenizer": "tiny",
                "engine": {"engine": "qwen_ple", "table_spec": "PLE_QWEN_V1"},
                "engram": {"prime_sizes": [17, 19]},
                "training": {},
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

def test_to_engram_config_bridges_store_fields(tmp_path):
    cfg_path = tmp_path / "store.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "project": "store-e2e",
                "base_model": "tiny",
                "tokenizer": "tiny",
                "engine": {
                    "engine": "qwen_ple",
                    "table_spec": "PLE_QWEN_V1",
                    "table_source": "engramdb:store",
                    "store_path": "/tmp/rows",
                    "model_dir": "/tmp/qwen-ple",
                    "shards": 128,
                    "rows_per_shard": 2_500_012,
                    "width": 160,
                    "scale": 0.0002,
                    "cache_size": 2048,
                },
                "engram": {
                    "embedding_dim": 2560,
                    "target_layers": [1],
                },
                "training": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    ecfg = cfg.to_engram_config(
        hidden_size=2560,
        compressed_vocab_size=248320,
        pad_id=248044,
        tokenizer_name_or_path="/tmp/tokenizer",
    )
    assert ecfg.table_source == "engramdb:store"
    assert ecfg.table_store_path == "/tmp/rows"
    assert ecfg.table_model_dir == "/tmp/qwen-ple"
    assert ecfg.table_shards == 128
    assert ecfg.table_rows_per_shard == 2_500_012
    assert ecfg.table_width == 160
    assert ecfg.table_scale == 0.0002
    assert ecfg.table_cache_size == 2048
    assert ecfg.hidden_size == 2560
    assert ecfg.engine == "qwen_ple"

