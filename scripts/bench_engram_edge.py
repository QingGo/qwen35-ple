#!/usr/bin/env python3
"""G3: EngramDB PLE storage microbenchmark (disk-first edge serving).

Measures the real cost of fetching PLE rows from the store, which our
training/eval path currently hides by using ``/dev/shm``.  Two regimes:

1. ``warm``   - the OS page cache may serve rows (steady-state decode).
2. ``cold``   - the page cache is dropped for the shard files first
                (``posix_fadvise(POSIX_FADV_DONTNEED)``), approximating the
                first touch after boot / cache eviction (long-context prefill).

Reports p50/p90/p99 per fetch, effective rows/s and tokens/s, plus the process
RSS and bytes read from disk, so the edge budget in docs/round-149 section 1.2
can be checked with measured numbers instead of assumptions.

Usage:
    python scripts/bench_engram_edge.py --rows-dir /path/to/rows \
        --json out.json --markdown out.md
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

POSIX_FADV_DONTNEED = 4


def _fadvise_dontneed(paths: list[Path]) -> bool:
    """Best-effort page-cache drop for the shard files (Linux only)."""
    libc_name = ctypes.util.find_library("c")
    if libc_name is None:
        return False
    libc = ctypes.CDLL(libc_name, use_errno=True)
    ok = False
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            if libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) == 0:
                ok = True
        finally:
            os.close(fd)
    return ok


def _proc_status() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    parts = line.split()
                    out[parts[0].rstrip(":")] = float(parts[1]) / 1024.0  # MiB
    except OSError:
        pass
    try:
        with open("/proc/self/io", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("read_bytes:"):
                    out["read_bytes"] = float(line.split()[1])
    except OSError:
        pass
    return out


def _percentiles(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {}
    ordered = sorted(samples)

    def pick(q: float) -> float:
        idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
        return ordered[idx]

    return {
        "min": ordered[0],
        "p50": pick(0.50),
        "p90": pick(0.90),
        "p99": pick(0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-dir", required=True)
    parser.add_argument("--label", default=None, help="label for the report")
    parser.add_argument("--batch-sizes", default="1,16,64,256,1024,2048")
    parser.add_argument("--tokens", type=int, default=4096, help="token pool for rowids")
    parser.add_argument("--reps", type=int, default=30, help="fetches per batch size")
    parser.add_argument("--cold-reps", type=int, default=5)
    parser.add_argument("--cold", action="store_true", help="also measure cold cache")
    parser.add_argument("--shards", type=int, default=128)
    parser.add_argument("--rows-per-shard", type=int, default=2_500_012)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--scale", type=float, default=0.00019931793212890625)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--target-prefill-tokens", type=int, default=2048)
    parser.add_argument("--target-decode-tok-s", type=float, default=20.0)
    args = parser.parse_args()

    import engramdb
    import numpy as np

    rows_dir = Path(args.rows_dir)
    label = args.label or str(rows_dir)
    shard_paths = sorted(p for p in rows_dir.iterdir() if p.is_file() and p.suffix != ".json")
    requested_batches = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    # engramdb fetches whole head groups: len(flat) must be a multiple of the
    # head count, so silently truncating a batch would measure the wrong size.
    batch_sizes: list[int] = []
    for batch in requested_batches:
        if batch % args.num_heads != 0:
            rounded = max(args.num_heads, (batch // args.num_heads) * args.num_heads)
            print(
                f"[g3] batch {batch} is not a multiple of num_heads="
                f"{args.num_heads}; using {rounded}"
            )
            batch = rounded
        if batch not in batch_sizes:
            batch_sizes.append(batch)
    if not batch_sizes:
        raise SystemExit("no valid batch sizes")

    store = engramdb.Store(
        str(rows_dir),
        shards=args.shards,
        rows_per_shard=args.rows_per_shard,
        width=args.width,
    )
    rng = np.random.default_rng(args.seed)
    report: dict[str, Any] = {
        "label": label,
        "rows_dir": str(rows_dir),
        "shards": args.shards,
        "rows_per_shard": args.rows_per_shard,
        "width": args.width,
        "scale": args.scale,
        "tokens": args.tokens,
        "reps": args.reps,
        "batch_sizes": batch_sizes,
        "batch_sizes_requested": requested_batches,
        "results": {},
        "environment": {},
    }
    try:
        tokens = list(range(100, 100 + args.tokens))
        t0 = time.perf_counter()
        rowids = engramdb.rowids_for_seq(tokens)
        report["rowid_seconds"] = time.perf_counter() - t0
        flat_all = [r for row in rowids for r in row]

        def fetch(flat: list[int]):
            return engramdb.fetch_e_t_tensor(
                store,
                flat,
                scale=args.scale,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
            )

        # Warm-up: take first-touch and lazy-init effects out of the warm series.
        fetch(flat_all[: max(batch_sizes)])

        for batch in batch_sizes:
            warm: list[float] = []
            for rep in range(args.reps):
                start = int(rng.integers(0, max(1, len(flat_all) - batch)))
                flat = flat_all[start : start + batch]
                t = time.perf_counter()
                fetch(flat)
                warm.append(time.perf_counter() - t)
            stats = _percentiles(warm)
            rows_per_s = batch / stats["p50"] if stats["p50"] > 0 else None
            report["results"][str(batch)] = {
                "batch": batch,
                "warm": stats,
                "rows_per_s_p50": rows_per_s,
                "tokens_per_s_p50": rows_per_s / args.num_heads if rows_per_s else None,
            }
            print(
                f"[g3:{label}] batch={batch:5d} warm p50={stats['p50'] * 1e3:8.3f}ms "
                f"p99={stats['p99'] * 1e3:8.3f}ms rows/s={rows_per_s:,.0f}"
            )

        if args.cold and shard_paths:
            cache_ok = _fadvise_dontneed(shard_paths)
            report["environment"]["fadvise_dontneed"] = cache_ok
            io_before = _proc_status()
            for batch in batch_sizes:
                cold: list[float] = []
                for rep in range(args.cold_reps):
                    if cache_ok:
                        _fadvise_dontneed(shard_paths)
                    start = int(rng.integers(0, max(1, len(flat_all) - batch)))
                    flat = flat_all[start : start + batch]
                    t = time.perf_counter()
                    fetch(flat)
                    cold.append(time.perf_counter() - t)
                stats = _percentiles(cold)
                rows_per_s = batch / stats["p50"] if stats["p50"] > 0 else None
                entry = report["results"][str(batch)]
                entry["cold"] = stats
                entry["cold_rows_per_s_p50"] = rows_per_s
                print(
                    f"[g3:{label}] batch={batch:5d} cold p50={stats['p50'] * 1e3:8.3f}ms "
                    f"p99={stats['p99'] * 1e3:8.3f}ms rows/s={rows_per_s:,.0f}"
                )
            io_after = _proc_status()
            if "read_bytes" in io_before and "read_bytes" in io_after:
                report["environment"]["bench_read_bytes"] = (
                    io_after["read_bytes"] - io_before["read_bytes"]
                )
            report["environment"].update(io_after)

        # Edge budget projection using the prefill-sized batch latency.
        prefill_key = str(
            min(
                batch_sizes,
                key=lambda b: abs(b - args.target_prefill_tokens),
            )
        )
        entry = report["results"].get(prefill_key, {})
        base = entry.get("warm", {})
        if base:
            # rows/s from the measured batch, then convert rows -> tokens using
            # num_heads rows per token (a 2048-token prefill = 32,768 rows).
            rows_per_s = float(prefill_key) / base["p50"]
            prefill_rows = args.target_prefill_tokens * args.num_heads
            report["projection"] = {
                "batch": int(prefill_key),
                "measured_rows": int(prefill_key),
                "measured_tokens": int(prefill_key) // args.num_heads,
                "prefill_rows_2048": prefill_rows,
                "prefill_rows_per_s": rows_per_s,
                "prefill_seconds_2048": prefill_rows / rows_per_s,
                "prefill_tokens_per_s_2048": rows_per_s / args.num_heads,
                "notes": (
                    "fetch-only throughput extrapolated linearly from the "
                    "nearest measured batch; end-to-end decode also pays model "
                    "compute, so this is an upper bound on the storage share"
                ),
            }
        report["environment"].update(
            {k: v for k, v in _proc_status().items() if k not in report["environment"]}
        )
    finally:
        store.close()

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    if args.markdown:
        lines = [
            f"# G3 EngramDB storage benchmark: {label}",
            "",
            f"- rows_dir: `{rows_dir}`",
            f"- shards: {args.shards} x {args.rows_per_shard:,} rows, width {args.width}",
            f"- tokens in pool: {args.tokens}, reps: {args.reps}",
            "",
            "| batch | warm p50 ms | warm p99 ms | warm rows/s | cold p50 ms | cold rows/s |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for batch, entry in report["results"].items():
            warm = entry.get("warm", {})
            cold = entry.get("cold", {})
            lines.append(
                f"| {batch} | {warm.get('p50', float('nan')) * 1e3:.3f} | "
                f"{warm.get('p99', float('nan')) * 1e3:.3f} | "
                f"{entry.get('rows_per_s_p50') or 0:,.0f} | "
                f"{(cold.get('p50', float('nan')) * 1e3) if cold else float('nan'):.3f} | "
                f"{entry.get('cold_rows_per_s_p50') or 0:,.0f} |"
            )
        proj = report.get("projection")
        if proj:
            lines += [
                "",
                (
                    f"Prefill projection: one token needs {args.num_heads} rows, so a "
                    f"2048-token prefill needs {proj['prefill_rows_2048']:,} rows.  At "
                    f"the measured p50 of the nearest measured batch that is "
                    f"{proj['prefill_seconds_2048'] * 1e3:.2f} ms of row fetching per "
                    f"2048-token forward, i.e. {proj['prefill_rows_per_s']:,.0f} rows/s "
                    f"= {proj['prefill_tokens_per_s_2048']:,.0f} tokens/s fetch-only."
                ),
            ]
        env = report.get("environment", {})
        if env:
            lines += [
                "",
                "Environment: "
                + ", ".join(f"{k}={v}" for k, v in sorted(env.items())),
            ]
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[g3] wrote {args.markdown}")
    print("ENGRAM_EDGE_BENCH_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
