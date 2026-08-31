#!/usr/bin/env python3
"""Low-resource real-PLE-row oracle.

This script verifies that EngramDB Store-I rows are byte-identical to the rows
stored in the original Qwen3.8/Qwen4Exp checkpoint, without loading either the
48GB Store or the full PLE embedding into RAM.

It only:

1. reads PLE metadata (scale/multipliers/prime table) from the checkpoint index;
2. computes rowids for a small fixed token sequence;
3. reads only the touched FP8 rows from the original safetensors shards via raw
   header/data-offset slicing;
4. fetches the same rowids from the EngramDB Store;
5. compares the raw bytes.

Usage:

    PYTHONPATH=src:../EngramDB/python \\
    python scripts/sparse_real_row_oracle.py \\
        --checkpoint "/Volumes/My Passport/qwen38-ple" \\
        --store "/Volumes/My Passport/qwen38-rows"
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import engramdb
from engramdb.ple_adapter import (
    PLE_BASE,
    PLE_EOS,
    head_offsets,
    head_vocab_sizes,
    ple_rowids,
)

# A small fixed sequence; contains EOS to exercise segment resets.
TOKENS = [100, 200, 300, 400, 500, PLE_EOS, 600, 700, 1000]


def _read_tensor_metadata(path: Path, tensor_name: str) -> tuple[int, int, int]:
    """Return (data_offset, n_rows, row_bytes) without loading tensor data."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        entry = header[tensor_name]
        start, end = entry["data_offsets"]
        shape = entry["shape"]
        return start, shape[0], shape[1]


def _read_fp8_rows(
    path: Path,
    tensor_name: str,
    local_rows: list[int],
    row_bytes: int,
) -> dict[int, bytes]:
    """Read selected FP8 rows from a safetensors shard using byte ranges."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        entry = header[tensor_name]
        start, _end = entry["data_offsets"]
        base = 8 + header_len + start
        out: dict[int, bytes] = {}
        for local in local_rows:
            f.seek(base + local * row_bytes)
            out[local] = f.read(row_bytes)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/Volumes/My Passport/qwen38-ple")
    parser.add_argument("--store", default="/Volumes/My Passport/qwen38-rows")
    parser.add_argument("--shards", type=int, default=128)
    parser.add_argument("--rows-per-shard", type=int, default=2_500_012)
    parser.add_argument("--width", type=int, default=160)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    info = engramdb.discover_ple(ckpt)
    if info is None:
        raise SystemExit(f"no PLE metadata in {ckpt}")

    multipliers = info.get("layer_multipliers") or info.get("rowid_multipliers")
    if not multipliers:
        raise SystemExit("PLE layer_multipliers not found")
    base = int(info.get("ngram_vocab_size_base") or PLE_BASE)
    eos = int(info.get("eos_token_id") or PLE_EOS)
    # Discovery does not currently return eos; the real Qwen PLE EOS is known.
    eos = PLE_EOS if eos == PLE_EOS else eos
    sizes = head_vocab_sizes(base=base, heads=16)
    offsets = head_offsets(sizes)

    rows = ple_rowids(
        TOKENS,
        multipliers,
        eos=eos,
        sizes=sizes,
        offsets=offsets,
        ngram_size=3,
        heads_per_ngram=8,
    )
    unique = sorted({r for row in rows for r in row})
    print(f"[oracle] tokens={len(TOKENS)} unique_rows={len(unique)}")

    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    prefix = (
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    )
    rows_per_shard = args.rows_per_shard

    # Group touched rows by shard to read each safetensors file once.
    by_shard: dict[int, list[int]] = {}
    for rowid in unique:
        shard = rowid // rows_per_shard
        local = rowid % rows_per_shard
        by_shard.setdefault(shard, []).append(local)

    ckpt_rows: dict[int, bytes] = {}
    for shard, locals_ in by_shard.items():
        tensor_name = f"{prefix}.shard_{shard}.weight"
        shard_file = ckpt / weight_map[tensor_name]
        _, n_rows, row_bytes = _read_tensor_metadata(shard_file, tensor_name)
        assert row_bytes == args.width, f"unexpected row width {row_bytes}"
        assert all(0 <= local < n_rows for local in locals_)
        data = _read_fp8_rows(shard_file, tensor_name, locals_, row_bytes)
        for local, raw in data.items():
            ckpt_rows[shard * rows_per_shard + local] = raw

    store = engramdb.Store(
        args.store,
        shards=args.shards,
        rows_per_shard=rows_per_shard,
        width=args.width,
    )
    try:
        raw = store.fetch(unique)
    finally:
        store.close()

    assert len(raw) == len(unique) * args.width
    store_rows: dict[int, bytes] = {}
    for i, rowid in enumerate(unique):
        store_rows[rowid] = raw[i * args.width:(i + 1) * args.width]

    mismatches = []
    for rowid in unique:
        if ckpt_rows[rowid] != store_rows[rowid]:
            mismatches.append(rowid)
            if len(mismatches) >= 5:
                break

    if mismatches:
        print(f"[oracle] MISMATCH rows: {mismatches}")
        return 1

    print(f"[oracle] all {len(unique)} real rows byte-identical")
    print("SPARSE_REAL_ROW_ORACLE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
