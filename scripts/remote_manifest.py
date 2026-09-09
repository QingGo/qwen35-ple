#!/usr/bin/env python3
"""Write a reproducibility manifest for the remote PLE environment.

The manifest records the exact assets, versions and checksums used by the
Phase 2 diagnostic run so a rebooted instance can be audited before launching
GPU work.

Usage::

    python scripts/remote_manifest.py \
      --output /root/autodl-tmp/qwen35-ple/remote-manifest.json \
      --root /root/autodl-tmp/qwen35-ple
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SHARDS = 128
DEFAULT_SHARD_BYTES = 400_001_920

PACKAGES = (
    "torch",
    "transformers",
    "tokenizers",
    "numpy",
    "engramdb-python",
    "engram-peft",
    "modelscope",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_verify_rows():
    """Load the sibling verifier without relying on sys.path."""
    path = Path(__file__).resolve().parent / "verify_qwen38_rows.py"
    spec = importlib.util.spec_from_file_location("verify_qwen38_rows", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load verifier from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_rows


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _path_info(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _git_info(repo_dir: str | Path) -> dict[str, Any]:
    repo = Path(repo_dir)
    if not (repo / ".git").exists():
        return {"available": False, "reason": "not a git checkout"}
    out: dict[str, Any] = {"available": True}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        out.update(
            {
                "commit": commit,
                "dirty": bool(status),
                "status_porcelain": status.splitlines()[:50],
            }
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        out.update({"available": False, "reason": str(exc)})
    return out


def _gpu_info() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    out: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts:
            continue
        out.append(
            {
                "name": parts[0] if len(parts) > 0 else "",
                "memory_total": parts[1] if len(parts) > 1 else "",
                "driver_version": parts[2] if len(parts) > 2 else "",
                "compute_cap": parts[3] if len(parts) > 3 else "",
            }
        )
    return out


def _dir_files(path: str | Path, limit: int = 20) -> dict[str, Any]:
    directory = Path(path)
    if not directory.is_dir():
        return {"exists": False, "files": []}
    files = sorted(item.name for item in directory.iterdir() if item.is_file())
    return {
        "exists": True,
        "file_count": len(files),
        "files": files[:limit],
    }


def collect_manifest(
    *,
    root: str | Path,
    repo_dir: str | Path,
    venv: str | Path,
    model_dir: str | Path,
    tokenizer_dir: str | Path,
    rows_dir: str | Path,
    rows_manifest: str | Path | None = None,
    expected_shards: int = DEFAULT_SHARDS,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    include_gpu: bool = True,
    include_git: bool = True,
) -> dict[str, Any]:
    """Collect environment, asset and row-verification metadata."""
    root_path = Path(root)
    rows_report = _load_verify_rows()(
        rows_dir,
        expected_shards=expected_shards,
        shard_bytes=shard_bytes,
        sha256_manifest=rows_manifest,
    )
    manifest: dict[str, Any] = {
        "schema": "qwen35-ple-remote-manifest-v1",
        "generated_at": _now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "packages": _package_versions(),
        "gpu": _gpu_info() if include_gpu else [],
        "disk": {
            "root": str(root_path),
            "total_bytes": shutil.disk_usage(root_path).total
            if root_path.exists()
            else None,
            "free_bytes": shutil.disk_usage(root_path).free
            if root_path.exists()
            else None,
        },
        "paths": {
            "root": _path_info(root_path),
            "repo": _path_info(repo_dir),
            "venv": _path_info(venv),
            "model": _path_info(model_dir),
            "tokenizer": _path_info(tokenizer_dir),
            "rows": _path_info(rows_dir),
        },
        "files": {
            "repo": _dir_files(repo_dir),
            "model": _dir_files(model_dir),
            "tokenizer": _dir_files(tokenizer_dir),
        },
        "rows": rows_report,
    }
    if include_git:
        manifest["git"] = _git_info(repo_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default="/root/autodl-tmp/qwen35-ple")
    parser.add_argument("--repo-dir", default=None)
    parser.add_argument("--venv", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--rows-dir", default=None)
    parser.add_argument("--rows-manifest", default=None)
    parser.add_argument("--expected-shards", type=int, default=DEFAULT_SHARDS)
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--no-git", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    manifest = collect_manifest(
        root=root,
        repo_dir=args.repo_dir or root / "repo",
        venv=args.venv or root / "venv",
        model_dir=args.model_dir or root / "models" / "Qwen3.5-0.8B",
        tokenizer_dir=args.tokenizer_dir
        or root / "models" / "Qwen3.8-Flash-Next-FP8-tokenizer",
        rows_dir=args.rows_dir or Path("/dev/shm/qwen38-rows"),
        rows_manifest=args.rows_manifest,
        expected_shards=args.expected_shards,
        shard_bytes=args.shard_bytes,
        include_gpu=not args.no_gpu,
        include_git=not args.no_git,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[remote-manifest] wrote {output_path}")
    print(
        json.dumps(
            {
                "rows_valid": manifest["rows"]["valid"],
                "num_shards": manifest["rows"]["num_shards"],
                "git_commit": manifest.get("git", {}).get("commit"),
                "gpu": manifest["gpu"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if manifest["rows"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
