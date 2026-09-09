"""Tests for scripts/verify_qwen38_rows.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_qwen38_rows.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_qwen38_rows", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_shard(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def test_verify_valid(tmp_path: Path) -> None:
    module = _load_module()
    _write_shard(tmp_path / "shard_000.bin", b"abcd")
    _write_shard(tmp_path / "shard_001.bin", b"efgh")
    report = module.verify_rows(tmp_path, expected_shards=2, shard_bytes=4)
    assert report["valid"] is True
    assert report["num_shards"] == 2
    assert report["total_bytes"] == 8
    assert report["missing"] == []
    assert report["wrong_size"] == []
    assert report["extra"] == []


def test_verify_missing_and_wrong_size(tmp_path: Path) -> None:
    module = _load_module()
    _write_shard(tmp_path / "shard_000.bin", b"abcd")
    _write_shard(tmp_path / "shard_001.bin", b"abc")
    report = module.verify_rows(tmp_path, expected_shards=3, shard_bytes=4)
    assert report["valid"] is False
    assert report["missing"] == ["shard_002.bin"]
    assert report["wrong_size"] == [{"name": "shard_001.bin", "size": 3}]


def test_verify_extra_file(tmp_path: Path) -> None:
    module = _load_module()
    _write_shard(tmp_path / "shard_000.bin", b"abcd")
    _write_shard(tmp_path / "shard_001.bin", b"efgh")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "shard_002.bin").write_bytes(b"ijkl")
    report = module.verify_rows(tmp_path, expected_shards=2, shard_bytes=4)
    assert report["valid"] is False
    assert "notes.txt" in report["extra"]
    assert "shard_002.bin" in report["extra"]


def test_verify_sha256_manifest(tmp_path: Path) -> None:
    module = _load_module()
    hash_a = _write_shard(tmp_path / "shard_000.bin", b"abcd")
    hash_b = _write_shard(tmp_path / "shard_001.bin", b"efgh")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"shards": {"shard_000.bin": hash_a, "shard_001.bin": hash_b}}),
        encoding="utf-8",
    )
    report = module.verify_rows(
        tmp_path,
        expected_shards=2,
        shard_bytes=4,
        sha256_manifest=manifest,
    )
    assert report["valid"] is True
    assert report["sha256_verified"] == 2
    assert report["sha256_mismatches"] == []

    bad_manifest = tmp_path / "bad-manifest.json"
    bad_manifest.write_text(
        json.dumps({"shard_000.bin": "0" * 64, "shard_001.bin": hash_b}),
        encoding="utf-8",
    )
    bad = module.verify_rows(
        tmp_path,
        expected_shards=2,
        shard_bytes=4,
        sha256_manifest=bad_manifest,
    )
    assert bad["valid"] is False
    assert bad["sha256_mismatches"][0]["name"] == "shard_000.bin"


def test_verify_manifest_missing_expected_shard(tmp_path: Path) -> None:
    module = _load_module()
    hash_a = _write_shard(tmp_path / "shard_000.bin", b"abcd")
    _write_shard(tmp_path / "shard_001.bin", b"efgh")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"shard_000.bin": hash_a}), encoding="utf-8")
    report = module.verify_rows(
        tmp_path,
        expected_shards=2,
        shard_bytes=4,
        sha256_manifest=manifest,
    )
    assert report["valid"] is False
    assert report["sha256_manifest_missing"] == ["shard_001.bin"]


def test_metadata_only_manifest_is_not_required(tmp_path: Path) -> None:
    module = _load_module()
    _write_shard(tmp_path / "shard_000.bin", b"abcd")
    _write_shard(tmp_path / "shard_001.bin", b"efgh")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "shards": [
                    {"name": "shard_000.bin", "size": 4},
                    {"name": "shard_001.bin", "size": 4},
                ]
            }
        ),
        encoding="utf-8",
    )
    report = module.verify_rows(
        tmp_path,
        expected_shards=2,
        shard_bytes=4,
        sha256_manifest=manifest,
    )
    assert report["valid"] is True
    assert report["sha256_manifest_required"] is False
    assert report["sha256_manifest_missing"] == []


def test_verify_compute_sha256(tmp_path: Path) -> None:
    module = _load_module()
    _write_shard(tmp_path / "shard_000.bin", b"abcd")
    _write_shard(tmp_path / "shard_001.bin", b"efgh")
    report = module.verify_rows(
        tmp_path,
        expected_shards=2,
        shard_bytes=4,
        compute_sha256=True,
    )
    assert report["valid"] is True
    assert len(report["sha256_checked"]) == 2
    assert all("sha256" in shard for shard in report["shards"])
