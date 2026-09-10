#!/usr/bin/env python3
"""Download a ModelScope snapshot with curl resume, falling back to HF.

ModelScope is preferred (China network); Hugging Face via ``HF_ENDPOINT`` is
the fallback.  Only model/config/tokenizer files are downloaded; GGUF and
original-format duplicates are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

WANTED_SUFFIXES = (
    ".safetensors",
    ".json",
    ".jinja",
    ".txt",
    ".model",
)
SKIP_NAMES = {".gitattributes", "README.md", "LICENSE", "LICENSE.txt"}


def _modelscope_files(model: str) -> dict[str, int]:
    url = (
        f"https://modelscope.cn/api/v1/models/{model}/repo/files"
        "?Revision=master&Root="
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        payload: dict[str, Any] = json.load(response)
    files = payload.get("Data", {}).get("Files", [])
    out: dict[str, int] = {}
    for item in files:
        path = str(item.get("Path", ""))
        if not path or path in SKIP_NAMES:
            continue
        if not path.endswith(WANTED_SUFFIXES):
            continue
        out[path] = int(item.get("Size", 0))
    if "config.json" not in out:
        raise RuntimeError(f"ModelScope file list for {model} has no config.json")
    return out


def _download_one(
    model: str, path: str, size: int, dest: Path, attempt: int = 0
) -> tuple[str, bool]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if dest.exists() and size and dest.stat().st_size == size:
        return (path, True)
    url = f"https://modelscope.cn/models/{model}/resolve/master/{path}"
    base = [
        "curl",
        "--http1.1",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--silent",
        "--show-error",
    ]
    if attempt == 0:
        cmd = base + ["-C", "-", "-o", str(part), url]
    else:
        part.unlink(missing_ok=True)
        cmd = base + ["-o", str(part), url]
    rc = subprocess.call(cmd)
    if rc != 0:
        if attempt == 0:
            return _download_one(model, path, size, dest, attempt=1)
        raise RuntimeError(f"curl failed for {path} rc={rc}")
    if size and part.stat().st_size != size:
        raise RuntimeError(
            f"size mismatch for {path}: got {part.stat().st_size}, expected {size}"
        )
    part.replace(dest)
    return (path, True)


def _download_modelscope(
    model: str, local_dir: Path, max_workers: int
) -> None:
    files = _modelscope_files(model)
    total = sum(files.values())
    print(f"[modelscope] {model}: {len(files)} files, {total / 1e9:.2f} GB")
    pending: list[tuple[str, int, Path]] = []
    for path, size in files.items():
        dest = local_dir / path
        if dest.exists() and size and dest.stat().st_size == size:
            continue
        pending.append((path, size, dest))
    if not pending:
        print("[modelscope] all files already present")
        return
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_one, model, path, size, dest): path
            for path, size, dest in pending
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                future.result()
                print(f"[modelscope] done {path}", flush=True)
            except (RuntimeError, OSError) as exc:  # pragma: no cover - network path
                failures.append(path)
                print(f"[modelscope] FAILED {path}: {exc}", file=sys.stderr)
    if failures:
        raise RuntimeError(f"failed files: {failures}")


def _download_hf(model: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"[hf] {model} -> {local_dir} via {endpoint}")
    snapshot_download(
        repo_id=model,
        local_dir=str(local_dir),
        max_workers=8,
        ignore_patterns=["*.gguf", "*.pth", "*.msgpack", "*.h5", "original/*"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument(
        "--force-hf", action="store_true", help="skip ModelScope and use HF"
    )
    args = parser.parse_args(argv)
    args.local_dir.mkdir(parents=True, exist_ok=True)

    if not args.force_hf:
        try:
            _download_modelscope(args.model, args.local_dir, args.max_workers)
        except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
            print(f"[modelscope] failed: {exc}; falling back to HF", file=sys.stderr)

    config_path = args.local_dir / "config.json"
    if not config_path.exists():
        _download_hf(args.model, args.local_dir)

    if not config_path.exists():
        raise SystemExit(f"config.json missing after download: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    print(
        f"[verify] {args.model}: hidden_size={text_config.get('hidden_size')} "
        f"layers={text_config.get('num_hidden_layers')} "
        f"vocab={text_config.get('vocab_size')}"
    )
    shards = list(args.local_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors shards in {args.local_dir}")
    print(f"[verify] {len(shards)} shards, {sum(p.stat().st_size for p in shards) / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
