"""Tests for the qwen35-ple table-asset orchestration helpers."""

from __future__ import annotations

import json

from qwen35_ple.table_assets import ViewManifest, manifest_path_for_view, read_key_file


def test_view_manifest_parses_engramdb_format(tmp_path):
    manifest = {
        "grans": 123,
        "heads": 16,
        "slot_bytes": 2560,
        "record_bytes": 2560,
        "build_seconds": 1.5,
        "build_mb_s": 200.0,
        "rows": 123 * 16,
        "source": "shards=128",
    }
    path = tmp_path / "qwen-ple.view.manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    info = ViewManifest.from_file(path)
    assert info.grans == 123
    assert info.heads == 16
    assert info.slot_bytes == 2560
    assert info.record_bytes == 2560
    assert info.rows == 123 * 16
    assert info.expected_bytes == 123 * 2560


def test_read_key_file_skips_blank_lines(tmp_path):
    keys = tmp_path / "keys.txt"
    keys.write_text("1\n2\n\n3\n", encoding="utf-8")
    assert read_key_file(keys) == [1, 2, 3]


def test_manifest_path_matches_engramdb_convention():
    assert manifest_path_for_view("data/views/qwen-ple.view.bin").name == "qwen-ple.view.manifest.json"
