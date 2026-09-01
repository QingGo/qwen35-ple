#!/usr/bin/env python3
"""Batch/incremental Store-P view builder for full tables (V132).

This script splits a large flat rowid keys file into chunks, builds each chunk
as a small Store-P view through ``engramdb view build --keys-stream``, then
concatenates the chunk views into one final access-order view.  With
``--resume`` already-built chunks are skipped, so an interrupted full-table
build can continue without starting over.

Usage:

    PYTHONPATH=src:/path/to/EngramDB/python \
    python scripts/build_full_store_p_batch.py \
        --rows-dir "/Volumes/My Passport/qwen38-rows" \
        --keys-file /path/to/full.keys.txt \
        --view /path/to/full.view \
        --work-dir /tmp/full-storep-batch \
        --chunk-grams 200000 \
        --resume
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

HEADS = 16


def _run(cmd: list[str]) -> None:
    print("[batch] running:", " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    print(f"[batch] done in {time.perf_counter() - t0:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-dir", required=True)
    parser.add_argument("--keys-file", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--engramdb-bin", default="engramdb")
    parser.add_argument("--chunk-grams", type=int, default=200_000)
    parser.add_argument("--slot-bytes", type=int, default=2560)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-sample", type=int, default=0)
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    chunk_dir = work / "chunks"
    chunk_dir.mkdir(exist_ok=True)

    total_grams = 0
    chunk_views: list[Path] = []
    grams_per_chunk = max(1, int(args.chunk_grams))
    rows_per_chunk = grams_per_chunk * HEADS

    with open(args.keys_file, "r", encoding="utf-8") as src:
        chunk_index = 0
        while True:
            rows: list[str] = []
            for _ in range(rows_per_chunk):
                line = src.readline()
                if not line:
                    break
                if line.strip():
                    rows.append(line.strip())
            if not rows:
                break
            if len(rows) % HEADS != 0:
                raise SystemExit(
                    f"keys file ended with incomplete chunk: {len(rows)} rows"
                )
            n_chunk = len(rows) // HEADS
            chunk_view = chunk_dir / f"chunk-{chunk_index:06d}.view"
            chunk_keys = chunk_dir / f"chunk-{chunk_index:06d}.keys.txt"
            if args.resume and chunk_view.exists() and chunk_view.stat().st_size > 0:
                print(f"[batch] resume: skipping chunk {chunk_index}")
            else:
                chunk_keys.write_text("\n".join(rows) + "\n", encoding="utf-8")
                _run(
                    [
                        args.engramdb_bin,
                        "view",
                        "build",
                        args.rows_dir,
                        str(n_chunk),
                        str(chunk_view),
                        str(chunk_keys),
                        "--keys-stream",
                        str(chunk_keys),
                        "--slot",
                        str(args.slot_bytes),
                    ]
                )
            chunk_views.append(chunk_view)
            total_grams += n_chunk
            chunk_index += 1

    if not chunk_views:
        print("[batch] no chunks built")
        return 1

    final_view = Path(args.view)
    final_view.parent.mkdir(parents=True, exist_ok=True)
    print(f"[batch] concatenating {len(chunk_views)} chunks -> {final_view}")
    with open(final_view, "wb") as out:
        for p in chunk_views:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)

    final_bytes = final_view.stat().st_size
    expected_bytes = total_grams * args.slot_bytes
    if final_bytes != expected_bytes:
        print(
            f"[batch] ERROR size mismatch: {final_bytes} != {expected_bytes}"
        )
        return 1

    manifest = {
        "grans": total_grams,
        "heads": HEADS,
        "slot_bytes": args.slot_bytes,
        "record_bytes": HEADS * 160,
        "rows": total_grams * HEADS,
        "source": f"batched:{len(chunk_views)} chunks",
        "layout": "access-order",
        "keys_out": str(Path(args.keys_file).resolve()),
        "batched": True,
    }
    manifest_path = final_view.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[batch] total_grams={total_grams} final={final_bytes}B manifest={manifest_path}")

    if args.verify_sample > 0:
        # Lightweight sample verification: build a small sample keys file and ask
        # EngramDB to read/verify those slot records against the final view.
        sample_rows: list[str] = []
        with open(args.keys_file, "r", encoding="utf-8") as src:
            for i, line in enumerate(src):
                if i >= args.verify_sample * HEADS:
                    break
                if line.strip():
                    sample_rows.append(line.strip())
        if len(sample_rows) == args.verify_sample * HEADS:
            sample_keys = work / "sample.keys.txt"
            sample_keys.write_text("\n".join(sample_rows) + "\n", encoding="utf-8")
            _run(
                [
                    args.engramdb_bin,
                    "view",
                    "verify",
                    args.rows_dir,
                    str(final_view),
                    "--keys",
                    str(sample_keys),
                    "--sub",
                    str(args.verify_sample),
                ]
            )

    print("BATCH_STORE_P_BUILD_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
