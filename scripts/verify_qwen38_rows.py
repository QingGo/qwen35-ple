#!/usr/bin/env python3
"""Verify EngramDB Store-I ``qwen38-rows`` shard files.

The Qwen3.8-Flash-Next PLE table is extracted as 128 fixed-size shard files::

    shard_000.bin ... shard_127.bin
    400001920 bytes each
    51,200,245,760 bytes total

This script checks presence, exact size and optional SHA-256 manifests.  It is
used by the remote bootstrap path to decide whether ``/dev/shm/qwen38-rows`` or
the persistent data-disk copy can be trusted.

Usage::

    python scripts/verify_qwen38_rows.py \
      --rows-dir /dev/shm/qwen38-rows \
      --write-manifest /dev/shm/qwen38-rows/manifest.json

    python scripts/verify_qwen38_rows.py \
      --rows-dir /root/autodl-tmp/qwen35-ple/qwen38-rows \
      --sha256-manifest artifacts/qwen38-rows-sha256.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SHARD_RE = re.compile(r"^shard_(\d{3})\.bin$")
METADATA_RE = re.compile(
    r"^(?:manifest(?:[._-].*)?\.json|.*\.sha256)$", flags=re.IGNORECASE
)
DEFAULT_SHARDS = 128
DEFAULT_SHARD_BYTES = 400_001_920


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, str]:
    """Load a sha256 manifest in several common shapes."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("shards"), dict):
        return {str(key): str(value) for key, value in data["shards"].items()}
    if isinstance(data, dict) and isinstance(data.get("shards"), list):
        out: dict[str, str] = {}
        for item in data["shards"]:
            if isinstance(item, dict) and "name" in item and "sha256" in item:
                out[str(item["name"])] = str(item["sha256"])
        return out
    if isinstance(data, dict):
        return {
            str(key): str(value)
            for key, value in data.items()
            if str(key).endswith(".bin")
        }
    raise ValueError(f"unsupported manifest format: {path}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def verify_rows(
    rows_dir: str | Path,
    *,
    expected_shards: int = DEFAULT_SHARDS,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    sha256_manifest: str | Path | None = None,
    compute_sha256: bool = False,
    sample_sha256: int = 0,
) -> dict[str, Any]:
    """Return a machine-readable verification report."""
    rows_path = Path(rows_dir)
    report: dict[str, Any] = {
        "generated_at": _now_iso(),
        "rows_dir": str(rows_path),
        "expected_shards": int(expected_shards),
        "expected_shard_bytes": int(shard_bytes),
        "exists": rows_path.is_dir(),
        "valid": False,
        "num_shards": 0,
        "total_bytes": 0,
        "missing": [],
        "wrong_size": [],
        "extra": [],
        "shards": [],
        "sha256_manifest": str(sha256_manifest) if sha256_manifest else None,
        "sha256_manifest_required": False,
        "sha256_manifest_missing": [],
        "sha256_checked": [],
        "sha256_mismatches": [],
        "sha256_verified": 0,
    }
    if not rows_path.is_dir():
        report["error"] = "rows_dir does not exist"
        return report

    expected_names = [f"shard_{index:03d}.bin" for index in range(expected_shards)]
    found: dict[int, Path] = {}
    extra: list[str] = []
    for path in sorted(rows_path.iterdir()):
        if not path.is_file():
            continue
        if METADATA_RE.match(path.name):
            continue
        match = SHARD_RE.match(path.name)
        if match:
            found[int(match.group(1))] = path
        else:
            extra.append(path.name)

    expected_indices = set(range(expected_shards))
    missing_indices = sorted(expected_indices - set(found))
    extra_indices = sorted(set(found) - expected_indices)
    report["missing"] = [f"shard_{index:03d}.bin" for index in missing_indices]
    report["extra"] = extra + [
        f"shard_{index:03d}.bin" for index in extra_indices
    ]

    for index in sorted(found):
        path = found[index]
        size = path.stat().st_size
        entry: dict[str, Any] = {"name": path.name, "size": size}
        if size != shard_bytes:
            report["wrong_size"].append(entry)
        report["shards"].append(entry)
    report["num_shards"] = len(found)
    report["total_bytes"] = sum(int(entry["size"]) for entry in report["shards"])

    manifest: dict[str, str] = {}
    if sha256_manifest:
        manifest = _load_manifest(Path(sha256_manifest))
        report["sha256_manifest_required"] = True
        report["sha256_manifest_missing"] = [
            name for name in expected_names if name not in manifest
        ]

    checked_names: list[str] = []
    if manifest:
        checked_names = [
            entry["name"] for entry in report["shards"] if entry["name"] in manifest
        ]
    elif compute_sha256:
        checked_names = [entry["name"] for entry in report["shards"]]
    elif sample_sha256 > 0:
        checked_names = [
            entry["name"] for entry in report["shards"][:sample_sha256]
        ]

    sha_by_name: dict[str, str] = {}
    for name in checked_names:
        path = rows_path / name
        digest = _sha256(path)
        sha_by_name[name] = digest
        report["sha256_checked"].append({"name": name, "sha256": digest})
        expected = manifest.get(name)
        if expected is not None and expected != digest:
            report["sha256_mismatches"].append(
                {"name": name, "expected": expected, "actual": digest}
            )
    report["sha256_verified"] = (
        len(report["sha256_checked"]) - len(report["sha256_mismatches"])
    )
    for entry in report["shards"]:
        digest = sha_by_name.get(str(entry["name"]))
        if digest is not None:
            entry["sha256"] = digest

    report["valid"] = bool(
        report["exists"]
        and not report["missing"]
        and not report["wrong_size"]
        and not report["extra"]
        and not report["sha256_mismatches"]
        and not report["sha256_manifest_missing"]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-dir", required=True)
    parser.add_argument("--expected-shards", type=int, default=DEFAULT_SHARDS)
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument(
        "--sha256-manifest",
        default=None,
        help="JSON manifest with expected sha256 per shard; full manifest required",
    )
    parser.add_argument(
        "--compute-sha256",
        action="store_true",
        help="compute sha256 for every shard (slow, useful when writing a manifest)",
    )
    parser.add_argument(
        "--sample-sha256",
        type=int,
        default=0,
        help="compute sha256 for the first N shards only",
    )
    parser.add_argument("--write-manifest", default=None)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only print the final valid/invalid line",
    )
    args = parser.parse_args()

    report = verify_rows(
        args.rows_dir,
        expected_shards=args.expected_shards,
        shard_bytes=args.shard_bytes,
        sha256_manifest=args.sha256_manifest,
        compute_sha256=args.compute_sha256,
        sample_sha256=args.sample_sha256,
    )
    if args.write_manifest:
        manifest_path = Path(args.write_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[verify-rows] wrote {manifest_path}")
    if args.quiet:
        print("valid" if report["valid"] else "invalid")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
