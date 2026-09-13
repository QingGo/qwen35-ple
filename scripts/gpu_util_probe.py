#!/usr/bin/env python3
"""Sample GPU utilisation and classify the binding constraint.

Round 167 asked for a standing rule: an experiment below ~50% GPU utilisation is
wasting the instance, and low utilisation must be diagnosed rather than tolerated.
This probe is the "measure first" half of that rule.  It samples utilisation and
memory for a fixed window and maps the result onto the four cases that actually
have different fixes.

The classification is deliberately simple because the four cases need different
remedies, and guessing wrong wastes more time than the sample costs:

``OVERHEAD_BOUND``
    Low utilisation, low memory.  A single step does too little GPU work, so
    Python/kernel-launch overhead dominates.  Fixes: increase the work per
    optimiser step; remove per-step Python work; strip unused heads.
``UNDER_BATCHED``
    Low utilisation but memory is the reason you cannot simply raise the batch
    count.  Fixes: batch by *token* budget rather than item count, so
    variable-length inputs fill the same memory; or lower precision.
``MEMORY_BOUND``
    High memory, moderate utilisation.  The job is close to the memory ceiling.
    Fixes: gradient checkpointing, smaller sequence, activation offload.
``OCCUPANCY_BOUND``
    Moderate utilisation with lots of free memory.  A *single* job cannot use
    more of the card, but several can: run independent arms concurrently.  This
    is the most common case in this repo and the cheapest to fix.
``SATURATED``
    At or above the target.  Nothing to do.

Usage::

    python scripts/gpu_util_probe.py --seconds 30
    python scripts/gpu_util_probe.py --seconds 30 --json outputs/gpu-probe.json
    python scripts/gpu_util_probe.py --seconds 30 --require 50   # exit 1 if below
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

__all__ = [
    "MEMORY_BOUND",
    "OCCUPANCY_BOUND",
    "OVERHEAD_BOUND",
    "SATURATED",
    "UNDER_BATCHED",
    "classify",
    "sample",
    "summarise_trace",
]

OVERHEAD_BOUND = "OVERHEAD_BOUND"
UNDER_BATCHED = "UNDER_BATCHED"
MEMORY_BOUND = "MEMORY_BOUND"
OCCUPANCY_BOUND = "OCCUPANCY_BOUND"
SATURATED = "SATURATED"

#: Fraction of total memory above which a job is treated as memory-limited.
MEMORY_FRACTION_HIGH = 0.75
#: Fraction of total memory below which spare capacity is assumed to exist.
MEMORY_FRACTION_LOW = 0.50


def _nvidia_smi() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip()


def sample(seconds: float, interval: float = 2.0) -> dict:
    """Sample utilisation/memory for ``seconds`` and summarise."""
    util: list[float] = []
    used: list[float] = []
    total: float | None = None
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        line = _nvidia_smi()
        if line:
            # One line per GPU; report the busiest so a probe is never fooled by
            # an idle second card.
            best: tuple[float, float, float] | None = None
            for part in line.splitlines():
                fields = [p.strip() for p in part.split(",")]
                if len(fields) < 3:
                    continue
                try:
                    u, m, t = float(fields[0]), float(fields[1]), float(fields[2])
                except ValueError:
                    continue
                if best is None or u > best[0]:
                    best = (u, m, t)
            if best is not None:
                util.append(best[0])
                used.append(best[1])
                total = best[2]
        time.sleep(interval)

    if not util:
        return {"available": False, "error": "no nvidia-smi samples"}
    return {
        "available": True,
        "n_samples": len(util),
        "util_median": float(statistics.median(util)),
        "util_mean": float(statistics.mean(util)),
        "util_p10": float(sorted(util)[max(0, len(util) // 10)]),
        "memory_used_mib_median": float(statistics.median(used)),
        "memory_total_mib": total,
        "memory_fraction_median": (
            float(statistics.median(used) / total) if total else None
        ),
        "samples": util,
    }


def summarise_trace(path: str | Path) -> dict:
    """Summarise a ``utilization.gpu,memory.used`` CSV trace.

    Sampling *during* a queue is the only way to measure it: a probe taken
    between phases sees an idle GPU and always reports 0%.
    """
    util: list[float] = []
    used: list[float] = []
    for line in Path(path).read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            util.append(float(parts[0]))
            used.append(float(parts[1]))
        except ValueError:
            continue
    if not util:
        return {"available": False, "error": "empty utilisation trace"}
    ordered = sorted(util)
    return {
        "available": True,
        "n_samples": len(util),
        "util_median": float(statistics.median(util)),
        "util_mean": float(statistics.mean(util)),
        "util_p10": float(ordered[len(ordered) // 10]),
        "util_max": float(ordered[-1]),
        "memory_used_mib_peak": max(used) if used else None,
    }


def classify(stats: dict, target: float = 50.0) -> dict:
    """Map a sample onto the case that determines the fix."""
    if not stats.get("available"):
        return {"verdict": "UNKNOWN", "reason": "no GPU visible", "fixes": []}

    u = stats["util_median"]
    frac = stats.get("memory_fraction_median")

    if u >= target:
        return {
            "verdict": SATURATED,
            "reason": f"median utilisation {u:.0f}% >= target {target:.0f}%",
            "fixes": [],
        }

    if frac is None:
        return {"verdict": "UNKNOWN", "reason": "no memory reading", "fixes": []}

    if frac >= MEMORY_FRACTION_HIGH:
        return {
            "verdict": MEMORY_BOUND,
            "reason": (
                f"utilisation {u:.0f}% with {frac * 100:.0f}% of memory used: the "
                "card is near its memory ceiling, so more concurrency is not available"
            ),
            "fixes": [
                "gradient checkpointing or smaller sequence length",
                "lower precision (bf16/fp8) for activations",
                "batch by token budget so a long input cannot blow the batch",
            ],
        }

    if frac <= MEMORY_FRACTION_LOW:
        # Plenty of spare memory: either the single job is too small to use the
        # card (overhead/under-batched) or several jobs would fit.
        return {
            "verdict": OCCUPANCY_BOUND,
            "reason": (
                f"utilisation {u:.0f}% while only {frac * 100:.0f}% of memory is "
                "used: one job cannot fill the card, but several can"
            ),
            "fixes": [
                (
                    "run independent arms concurrently (about "
                    f"{int(1 / max(frac, 0.01))} fit in memory)"
                ),
                "increase tokens per optimiser step (batch by token budget)",
                "remove per-step Python/kernel-launch overhead",
                "do not run the LM head when only hidden states are needed",
            ],
        }

    return {
        "verdict": UNDER_BATCHED,
        "reason": (
            f"utilisation {u:.0f}% at {frac * 100:.0f}% memory: some headroom "
            "remains, so raise the work per step"
        ),
        "fixes": [
            "increase tokens per step within the memory headroom",
            "batch by token budget rather than item count",
            "run one more arm concurrently",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--require", type=float, default=None, help="exit 1 below this %%")
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--trace",
        default=None,
        help="summarise an existing utilisation trace CSV instead of sampling",
    )
    args = ap.parse_args()

    if args.trace:
        stats = summarise_trace(args.trace)
        verdict = classify(stats, target=args.require if args.require is not None else 50.0)
        if stats.get("available"):
            peak = stats.get("memory_used_mib_peak") or 0.0
            print(
                f"utilisation trace: n={stats['n_samples']} "
                f"median={stats['util_median']:.0f}% mean={stats['util_mean']:.0f}% "
                f"p10={stats['util_p10']:.0f}% peak_mem={peak / 1024:.1f}GiB "
                f"=> {verdict['verdict']}"
            )
        else:
            print(f"utilisation trace: {stats.get('error')}")
        if args.json:
            pth = Path(args.json)
            pth.parent.mkdir(parents=True, exist_ok=True)
            pth.write_text(json.dumps({"stats": stats, "diagnosis": verdict}, indent=2) + "\n")
        if args.require is not None and stats.get("available") and (
            stats["util_median"] < args.require
        ):
            print(
                f"UTILISATION GATE: {stats['util_median']:.0f}% < {args.require:.0f}%",
                file=sys.stderr,
            )
            return 1
        return 0

    stats = sample(args.seconds, args.interval)
    verdict = classify(stats, target=args.require if args.require is not None else 50.0)

    print(json.dumps({"stats": stats, "diagnosis": verdict}, indent=2))
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"stats": stats, "diagnosis": verdict}, indent=2) + "\n")

    if not stats.get("available"):
        print("no GPU visible; skipping the utilisation gate", file=sys.stderr)
        return 0

    print(
        f"\nmedian utilisation {stats['util_median']:.0f}% "
        f"(p10 {stats['util_p10']:.0f}%), "
        f"memory {stats['memory_fraction_median'] * 100:.0f}%\n"
        f"=> {verdict['verdict']}: {verdict['reason']}",
        file=sys.stderr,
    )
    for f in verdict["fixes"]:
        print(f"   fix: {f}", file=sys.stderr)

    if args.require is not None and stats["util_median"] < args.require:
        print(
            f"UTILISATION GATE: {stats['util_median']:.0f}% < {args.require:.0f}%",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
