#!/usr/bin/env python3
"""Build an access-order Store-P view for a fixed token corpus.

This is the missing semantic link for Track B/C:

* Generate official PLE rowids for every token in the corpus.
* Write a flat keys file (16 rowids per token, in token order).
* Build a Store-P view with ``engramdb view build --keys IN_KEYS``.
* Because the view slot order equals the input token order, the semantic
  mapping is simply ``slot_index = token_index``:
    ``LiveETViewStore(view, np.arange(T), ...)`` can then serve the exact e_t
  for each token as a sequential Store-P read.

Usage:

    PYTHONPATH=src:../EngramDB/python \\
    python scripts/build_corpus_store_p_view.py \\
        --rows-dir "/Volumes/My Passport/qwen38-rows" \\
        --tokens-npy /tmp/tokens.npy \\
        --model-dir "/Volumes/My Passport/qwen38-ple" \\
        --output-view /tmp/corpus.view \\
        --engramdb-bin /path/to/engramdb

Outputs:

    /tmp/corpus.view                Store-P view (2560-byte slots)
    /tmp/corpus.keys                flat rowid keys used to build the view
    /tmp/corpus.slot_indices.npy    arange(T), i.e. token -> view slot
    /tmp/corpus.slot_index.npz      generic rowid-tuple -> slot semantic index
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from qwen35_ple.real_ple import resolve_ple_weight_scale, rowids_from_tokens
from qwen35_ple.slot_index import SlotIndex


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-dir", default="/Volumes/My Passport/qwen38-rows")
    parser.add_argument("--tokens-npy", default=None)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--model-dir", default="/Volumes/My Passport/qwen38-ple")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--output-view", required=True)
    parser.add_argument("--keys-out", default=None)
    parser.add_argument("--slot-indices-out", default=None)
    parser.add_argument("--slot-index-out", default=None)
    parser.add_argument("--slot-index-dir", default=None)
    parser.add_argument("--slot-index-buckets", type=int, default=16384)
    parser.add_argument("--skip-slot-index-npz", action="store_true")
    parser.add_argument("--engramdb-bin", default="engramdb")
    parser.add_argument("--slot-bytes", type=int, default=2560)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--stream", action="store_true", help="use engramdb view build --keys-stream (CLI streams keys file)")
    parser.add_argument("--keep-temp-keys", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    if args.tokens_npy:
        tokens = np.load(args.tokens_npy).astype(np.int64)
    elif args.tokens is not None:
        tokens = np.arange(args.tokens, dtype=np.int64)
    else:
        raise SystemExit("must provide --tokens-npy or --tokens")

    if args.max_tokens is not None:
        tokens = tokens[: args.max_tokens]
    print(f"[build-view] tokens={len(tokens)}")

    scale = resolve_ple_weight_scale(model_dir=args.model_dir, scale=args.scale)
    print("[build-view] generating rowids ...")
    t0 = time.perf_counter()
    rowids = rowids_from_tokens(tokens)
    print(
        f"[build-view] rowids={rowids.shape} scale={scale:.6g} "
        f"rowid_s={time.perf_counter() - t0:.2f}s"
    )

    flat = rowids.reshape(-1).tolist()
    n = len(rowids)

    with tempfile.TemporaryDirectory(prefix="store-p-keys-") as td:
        keys_in = Path(td) / "keys_in.txt"
        keys_in.write_text("\n".join(map(str, flat)) + "\n", encoding="utf-8")

        keys_out = Path(args.keys_out) if args.keys_out else Path(td) / "keys_out.txt"
        view_path = Path(args.output_view)
        view_path.parent.mkdir(parents=True, exist_ok=True)

        key_arg = "--keys-stream" if args.stream else "--keys"
        cmd = [
            args.engramdb_bin,
            "view",
            "build",
            args.rows_dir,
            str(n),
            str(view_path),
            str(keys_out),
            key_arg,
            str(keys_in),
            "--slot",
            str(args.slot_bytes),
        ]
        if args.verify:
            cmd.append("--verify")

        print("[build-view] running:", " ".join(cmd))
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        build_s = time.perf_counter() - t0
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise SystemExit(f"engramdb view build failed with {proc.returncode}")

        print(proc.stdout.strip())
        print(f"[build-view] build_s={build_s:.2f}s")

        # Access-order mapping: slot i corresponds to token i.
        slot_out = Path(args.slot_indices_out) if args.slot_indices_out else (
            view_path.with_suffix(".slot_indices.npy")
        )
        np.save(slot_out, np.arange(n, dtype=np.int64))
        print(f"[build-view] slot_indices -> {slot_out}")

        # Generic rowid -> slot semantic index.  Even for an access-order view
        # this makes the view usable by arbitrary later token streams (the
        # same rowid tuple can be resolved to a representative physical slot).
        slot_index_out = Path(args.slot_index_out) if args.slot_index_out else (
            view_path.with_suffix(".slot_index.npz")
        )
        if args.skip_slot_index_npz:
            print("[build-view] skipping in-memory slot_index.npz")
        else:
            SlotIndex.from_rowids(rowids, np.arange(n, dtype=np.int64)).save(slot_index_out)
            print(f"[build-view] slot_index -> {slot_index_out}")

        # Optional disk-backed index for full 320M-scale tables.  The native
        # CLI streams the keys file and emits the Python-compatible v2 format,
        # so this path does not need to hold the whole rowid matrix in RAM.
        slot_index_dir = None
        if args.slot_index_dir:
            slot_index_dir = Path(args.slot_index_dir)
            slot_index_dir.mkdir(parents=True, exist_ok=True)
            disk_cmd = [
                args.engramdb_bin,
                "slot-index",
                "build",
                str(keys_out),
                str(slot_index_dir),
                "--buckets",
                str(args.slot_index_buckets),
            ]
            print("[build-view] running:", " ".join(disk_cmd))
            t0 = time.perf_counter()
            proc = subprocess.run(disk_cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(proc.stdout.strip())
                print(proc.stderr.strip())
                print(
                    "[build-view] native slot-index unavailable; trying "
                    "Python DiskSlotIndex fallback"
                )
                try:
                    from qwen35_ple.slot_index import DiskSlotIndex as PyDisk
                except Exception:  # noqa: BLE001 - optional dependency
                    PyDisk = None
                if PyDisk is None:
                    raise SystemExit(
                        f"engramdb slot-index build failed with {proc.returncode} "
                        "and no Python DiskSlotIndex fallback"
                    )
                PyDisk.build_from_keys_file(
                    keys_out,
                    slot_index_dir,
                    num_buckets=args.slot_index_buckets,
                    hash_name="fnv1a-64",
                )
            else:
                print(proc.stdout.strip())
            print(
                f"[build-view] disk slot_index -> {slot_index_dir} "
                f"({time.perf_counter() - t0:.2f}s)"
            )

        if args.keys_out:
            print(f"[build-view] keys_out -> {keys_out}")

        manifest_path = view_path.with_suffix(".manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not args.skip_slot_index_npz:
                manifest["slot_index"] = str(slot_index_out)
            manifest["slot_indices"] = str(slot_out)
            if slot_index_dir is not None:
                manifest["slot_index_disk"] = str(slot_index_dir)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"[build-view] manifest grans={manifest.get('grans')} "
                  f"slot={manifest.get('slot_bytes')} "
                  f"slot_index={slot_index_out.name}")
        else:
            print("[build-view] manifest not found")

        if not args.keep_temp_keys and not args.keys_out:
            # Temporary directory cleans up automatically.
            pass

    print("BUILD_CORPUS_STORE_P_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
