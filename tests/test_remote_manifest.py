"""Tests for scripts/remote_manifest.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "remote_manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("remote_manifest", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_manifest(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "qwen35-ple"
    repo = root / "repo"
    venv = root / "venv"
    model = root / "models" / "Qwen3.5-0.8B"
    tokenizer = root / "models" / "Qwen3.8-tokenizer"
    rows = root / "qwen38-rows"
    for directory in (repo, venv, model, tokenizer, rows):
        directory.mkdir(parents=True, exist_ok=True)
    (repo / ".git").mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    for index in range(2):
        (rows / f"shard_{index:03d}.bin").write_bytes(b"abcd")

    manifest = module.collect_manifest(
        root=root,
        repo_dir=repo,
        venv=venv,
        model_dir=model,
        tokenizer_dir=tokenizer,
        rows_dir=rows,
        expected_shards=2,
        shard_bytes=4,
        include_gpu=False,
        include_git=False,
    )
    assert manifest["schema"] == "qwen35-ple-remote-manifest-v1"
    assert manifest["rows"]["valid"] is True
    assert manifest["rows"]["num_shards"] == 2
    assert manifest["paths"]["model"]["exists"] is True
    assert manifest["paths"]["tokenizer"]["exists"] is True
    assert "torch" in manifest["packages"]
    assert manifest["gpu"] == []


def test_collect_manifest_missing_rows(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "qwen35-ple"
    root.mkdir()
    manifest = module.collect_manifest(
        root=root,
        repo_dir=root / "repo",
        venv=root / "venv",
        model_dir=root / "model",
        tokenizer_dir=root / "tokenizer",
        rows_dir=root / "rows",
        expected_shards=2,
        shard_bytes=4,
        include_gpu=False,
        include_git=False,
    )
    assert manifest["rows"]["valid"] is False
    assert manifest["rows"]["exists"] is False
