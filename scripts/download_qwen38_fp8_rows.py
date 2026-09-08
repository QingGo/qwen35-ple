#!/usr/bin/env python3
"""Extract EngramDB Store-I PLE rows from the Qwen3.8-Flash-Next-FP8 checkpoint.

The FP8 checkpoint stores the PLE n-gram table as 128 tensors::

    model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{N}.weight

Each tensor is byte-identical to an EngramDB Store-I ``shard_{N:03d}.bin`` file
(FP8 E4M3, shape ``[2500012, 160]``).  This script downloads only the
checkpoint files that contain those tensors, extracts the raw bytes, and
optionally removes the downloaded checkpoint shards.

Example::

    python scripts/download_qwen38_fp8_rows.py \
        --out-dir /dev/shm/qwen38-rows \
        --download-dir /root/autodl-tmp/qwen35-ple/tmp \
        --keep-source

The default download URL points at ModelScope's FP8 mirror.  Pass
``--base-url`` to use a different mirror.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "Qwen/Qwen3.8-Flash-Next-FP8"
DEFAULT_BASE_URL = "https://modelscope.cn/models/{model_id}/resolve/master"
TARGET_SUFFIX = "ngram_embedding.shard_"


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest*, resuming a partial file when possible."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "--http1.1",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "-C",
        "-",
        "-o",
        str(dest),
        url,
    ]
    subprocess.run(cmd, check=True)


def _load_json_url(url: str) -> dict[str, Any]:
    """Fetch a small JSON document with curl and parse it."""
    out = subprocess.check_output(
        ["curl", "--http1.1", "-L", "-sS", "--fail", "--retry", "3", url],
        text=True,
    )
    return json.loads(out)


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    """Return ``(header, header_len)`` for the safetensors file *path*."""
    with path.open("rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
    return header, header_len


def _copy_tensor(
    source: Path,
    header_len: int,
    tensor: dict[str, Any],
    dest: Path,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> None:
    """Copy one raw safetensors tensor into *dest*."""
    start, end = tensor["data_offsets"]
    expected = int(end) - int(start)
    shape = tensor.get("shape", [])
    count = 1
    for dim in shape:
        count *= int(dim)
    if expected != count:
        raise ValueError(
            f"tensor byte size mismatch: offsets={expected} shape_count={count}"
        )

    tmp = dest.with_suffix(dest.suffix + ".part")
    base = 8 + header_len
    with source.open("rb") as src, tmp.open("wb") as dst:
        src.seek(base + int(start))
        remaining = expected
        while remaining:
            block = src.read(min(chunk_bytes, remaining))
            if not block:
                raise OSError(f"unexpected EOF while reading {source}")
            dst.write(block)
            remaining -= len(block)
    if tmp.stat().st_size != expected:
        raise OSError(
            f"short tensor write for {dest}: {tmp.stat().st_size} != {expected}"
        )
    tmp.replace(dest)


def _target_tensors(index: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Map checkpoint filename -> list of PLE shard tensors in that file."""
    by_file: dict[str, list[dict[str, Any]]] = {}
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise TypeError("model.safetensors.index.json has no weight_map object")
    for name, filename in weight_map.items():
        if TARGET_SUFFIX not in name:
            continue
        if not name.endswith(".weight"):
            continue
        _, rest = name.split(TARGET_SUFFIX, 1)
        shard_text = rest.split(".", 1)[0]
        try:
            shard_idx = int(shard_text)
        except ValueError as exc:
            raise ValueError(f"cannot parse shard index from {name!r}") from exc
        by_file.setdefault(str(filename), []).append({"name": name, "shard": shard_idx})
    if not by_file:
        raise ValueError("no ngram_embedding.shard_*.weight tensors in index")
    return by_file


def _bf16_to_float(bits: int) -> float:
    """Convert a raw BF16 bit pattern to Python float."""
    sign = (bits >> 15) & 1
    exp = (bits >> 7) & 0xFF
    mant = bits & 0x7F
    if exp == 0:
        value = mant * (2.0**-7) * (2.0**-126)
    elif exp == 0xFF:
        value = float("inf") if mant == 0 else float("nan")
    else:
        value = (1.0 + mant / 128.0) * (2.0 ** (exp - 127))
    return -value if sign else value


def _extract_scalar(
    source: Path,
    header_len: int,
    header: dict[str, Any],
    name: str,
) -> float | None:
    """Read a scalar F32/BF16/F16 tensor if present in *header*."""
    entry = header.get(name)
    if entry is None:
        return None
    dtype = entry.get("dtype")
    start, end = entry["data_offsets"]
    expected = 4 if dtype == "F32" else 2 if dtype in {"BF16", "F16"} else None
    if expected is None or int(end) - int(start) != expected:
        return None
    with source.open("rb") as fh:
        fh.seek(8 + header_len + int(start))
        raw = fh.read(expected)
    if dtype == "F32":
        return float(struct.unpack("<f", raw)[0])
    bits = struct.unpack("<H", raw)[0]
    if dtype == "BF16":
        return _bf16_to_float(bits)
    return float(struct.unpack("<e", raw)[0])


def _download_index(args: argparse.Namespace) -> Path:
    """Return a local index path, downloading it when necessary."""
    index_path = Path(args.index)
    if index_path.exists():
        return index_path
    url = f"{args.base_url.format(model_id=args.model_id)}/model.safetensors.index.json"
    print(f"[extract] downloading index {url}", flush=True)
    data = _load_json_url(url)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(data), encoding="utf-8")
    return index_path


def _itemsize(dtype: str | None) -> int:
    """Return the number of bytes per element for the supported safetensors dtypes."""
    if dtype in {"F8_E4M3", "F8_E5M2", "I8", "U8"}:
        return 1
    if dtype in {"F16", "BF16", "I16", "U16"}:
        return 2
    if dtype in {"F32", "I32", "U32"}:
        return 4
    if dtype in {"F64", "I64", "U64"}:
        return 8
    raise NotImplementedError(f"unsupported safetensors dtype {dtype!r}")


def _should_skip_shard(path: Path, expected_bytes: int) -> bool:
    return path.exists() and path.stat().st_size == expected_bytes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="ModelScope model id containing the FP8 checkpoint",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="resolve base URL template; use {model_id} as placeholder",
    )
    parser.add_argument(
        "--index",
        default="",
        help="local model.safetensors.index.json path; downloaded if missing",
    )
    parser.add_argument("--out-dir", required=True, help="Store-I rows output dir")
    parser.add_argument(
        "--download-dir",
        default="",
        help="directory for temporary checkpoint shards (default: out-dir/../checkpoint-tmp)",
    )
    parser.add_argument(
        "--expected-shards",
        type=int,
        default=128,
        help="expected number of PLE ngram shards",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="keep downloaded checkpoint shards after extraction",
    )
    parser.add_argument(
        "--meta-out",
        default="",
        help="optional JSON path for extracted PLE metadata",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    download_dir = (
        Path(args.download_dir)
        if args.download_dir
        else out_dir.parent / "checkpoint-tmp"
    )
    download_dir.mkdir(parents=True, exist_ok=True)
    if not args.index:
        args.index = str(out_dir.parent / "model.safetensors.index.json")

    index_path = _download_index(args)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    by_file = _target_tensors(index)

    all_targets = sorted(
        (t for targets in by_file.values() for t in targets),
        key=lambda item: int(item["shard"]),
    )
    expected_total = len(all_targets)
    if expected_total != args.expected_shards:
        print(
            f"[extract] warning: index has {expected_total} PLE shards, "
            f"expected {args.expected_shards}",
            file=sys.stderr,
        )

    print(
        f"[extract] {expected_total} PLE shards across {len(by_file)} checkpoint files",
        flush=True,
    )

    started = time.time()
    done_shards = 0
    meta: dict[str, Any] = {
        "model_id": args.model_id,
        "num_shards": expected_total,
        "files": sorted(by_file),
    }

    for file_index, (filename, targets) in enumerate(sorted(by_file.items()), 1):
        source = download_dir / filename
        url = f"{args.base_url.format(model_id=args.model_id)}/{filename}"
        output_paths = [
            out_dir / f"shard_{int(target['shard']):03d}.bin" for target in targets
        ]

        if all(path.exists() for path in output_paths):
            print(
                f"[extract] {file_index}/{len(by_file)} skip {filename}: "
                "all outputs already exist",
                flush=True,
            )
            if source.exists() and not args.keep_source:
                source.unlink()
            done_shards += len(targets)
            continue

        print(
            f"[extract] {file_index}/{len(by_file)} download {filename} "
            f"({source.stat().st_size if source.exists() else 0} bytes present)",
            flush=True,
        )
        _download(url, source)
        header, header_len = _read_header(source)

        for target, dest in zip(targets, output_paths, strict=True):
            entry = header.get(target["name"])
            if entry is None:
                raise KeyError(f"{target['name']!r} missing from {filename}")
            shape = entry.get("shape", [])
            expected_bytes = _itemsize(entry.get("dtype"))
            for dim in shape:
                expected_bytes *= int(dim)
            if _should_skip_shard(dest, expected_bytes):
                print(f"[extract]   shard {target['shard']:03d} already complete")
                continue
            _copy_tensor(source, header_len, entry, dest)
            done_shards += 1
            print(
                f"[extract]   shard {target['shard']:03d} -> {dest} "
                f"({dest.stat().st_size} bytes)",
                flush=True,
            )

        scale = _extract_scalar(
            source,
            header_len,
            header,
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale",
        )
        if scale is not None:
            meta["weight_scale"] = scale
        if not args.keep_source:
            source.unlink()

    missing = [
        int(target["shard"])
        for target in all_targets
        if not _should_skip_shard(
            out_dir / f"shard_{int(target['shard']):03d}.bin",
            2500012 * 160,
        )
    ]
    if missing:
        raise SystemExit(f"missing or incomplete PLE shards: {missing[:20]}")
    total_bytes = sum(
        (out_dir / f"shard_{int(target['shard']):03d}.bin").stat().st_size
        for target in all_targets
    )
    meta["total_bytes"] = total_bytes
    meta["elapsed_seconds"] = round(time.time() - started, 1)
    if args.meta_out:
        Path(args.meta_out).write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"[extract] done: {done_shards} new shards, "
        f"{total_bytes / 1024**3:.2f} GiB, {meta['elapsed_seconds']}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
