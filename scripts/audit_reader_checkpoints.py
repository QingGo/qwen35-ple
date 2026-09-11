#!/usr/bin/env python3
"""Assert that trained reader checkpoints actually moved, and by how much.

Round-153 discipline #1: *no silently-invalid arm*.  A LoRA/reader arm that
never reached the optimizer produces an adapter indistinguishable from its
initialisation, and every downstream number is then meaningless -- a failure
mode this project already hit once (round 152: LoRA adapters stayed exactly zero
for 500 steps and the "trained" arm was bit-identical to the frozen baseline).

This audits saved reader checkpoints along three axes:

* **frozen source tensors** (``key_proj``/``value_proj``/``norm_*``/``conv1d``)
  must be *bit-identical* across arms -- if they differ, ``freeze_source``
  leaked and the arms are not comparable;
* **trainable adapters** (``query_bridge``/``out_proj``) must have *moved* away
  from their initialisation, measured against the pristine official checkpoint;
* **arm-to-arm magnitude** -- if one arm's adapter norm is orders of magnitude
  larger than another's, that arm injects a much bigger vector into the residual
  stream and a real-vs-control comparison is not interpretable as a content
  effect.  This is the check that decides whether a control arm "poisoned the
  stream" or merely carried different content.

Usage::

    python scripts/audit_reader_checkpoints.py \
        --reference real=outputs/round152g0/reader-4b-real-seed0.pt \
        --arm control=outputs/round152g0/reader-4b-control-seed0.pt \
        --init data/official_ple_reader.pt

``--init`` is optional; without it the "moved from init" column is skipped.
Exit status is non-zero when a hard invariant is violated (frozen tensors
differ between arms, or an arm's adapters did not move at all).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

FROZEN_PREFIXES = (
    "key_proj.",
    "value_proj.",
    "norm_key.",
    "norm_query.",
    "norm_conv.",
    "conv1d.",
)
ADAPTER_PREFIXES = ("query_bridge.", "out_proj.")
# The official reader checkpoint uses a ``model.language_model.layers.1.ple.``
# prefix on the source tensors; accept both spellings.
INIT_PREFIX = "model.language_model.layers.1.ple."


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a dict checkpoint, got {type(payload).__name__}")
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise SystemExit(f"{path}: checkpoint has no usable state_dict")
    return state


def _init_lookup(init: dict[str, torch.Tensor], name: str) -> torch.Tensor | None:
    return init.get(name) or init.get(INIT_PREFIX + name)


def _classify(name: str) -> str:
    if name.startswith(FROZEN_PREFIXES):
        return "frozen"
    if name.startswith(ADAPTER_PREFIXES):
        return "adapter"
    return "other"


def _norm(t: torch.Tensor) -> float:
    return float(t.float().norm()) if torch.is_floating_point(t) else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        required=True,
        metavar="LABEL=PATH",
        help="the arm every other arm is compared against",
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="further arm to audit (repeatable)",
    )
    parser.add_argument(
        "--init",
        type=Path,
        default=None,
        help="pristine official reader checkpoint, to measure adapter drift",
    )
    parser.add_argument(
        "--min-drift",
        type=float,
        default=0.0,
        help="fail when an arm's adapter drift (relative L2 vs init) is at or "
        "below this value; 0 only fails on an exact no-op",
    )
    parser.add_argument(
        "--max-norm-ratio",
        type=float,
        default=3.0,
        help="fail when a non-reference arm's adapter norm exceeds the reference "
        "arm's by this factor; such a contrast is confounded by injection scale",
    )
    args = parser.parse_args(argv)

    specs = [args.reference, *args.arm]
    arms: dict[str, dict[str, torch.Tensor]] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--reference/--arm must be LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        arms[label] = _load_state(Path(path))

    ref_label = specs[0].split("=", 1)[0]
    ref = arms[ref_label]
    init = _load_state(args.init) if args.init else None

    print(f"reference arm: {ref_label}  ({len(ref)} tensors)")
    if init is not None:
        print(f"init:          {args.init}  ({len(init)} tensors)")
    print()

    failures: list[str] = []

    for label, state in arms.items():
        if label == ref_label:
            continue
        missing = sorted(set(ref) - set(state))
        extra = sorted(set(state) - set(ref))
        if missing or extra:
            failures.append(f"{label}: tensor set differs from {ref_label} (missing={missing}, extra={extra})")
        shared = sorted(set(ref) & set(state))
        moved = [(k, float((state[k].float() - ref[k].float()).abs().max())) for k in shared]
        moved.sort(key=lambda kv: kv[1], reverse=True)
        worst = moved[0] if moved else ("", 0.0)
        print(f"arm {label}: {len(shared)} shared tensors, max |Δ| vs {ref_label} = {worst[1]:.6g} ({worst[0]})")

    print()
    other_labels = [lbl for lbl in arms if lbl != ref_label]
    cols = [ref_label, *other_labels]
    header = f"{'param':<27}{'kind':<9}{'shape':>16}" + "".join(f"{'|W| '+c:>16}" for c in cols)
    header += f"{'Δ vs '+ref_label:>14}"
    print(header)
    print("-" * len(header))

    def _drift_from_init(t: torch.Tensor, name: str) -> float | None:
        if init is None:
            return None
        base = _init_lookup(init, name)
        if base is None:
            return None
        denom = _norm(base)
        if denom <= 0:
            return float("inf")
        return float((t.float() - base.float()).norm()) / denom

    for name in sorted(ref):
        t = ref[name]
        if not torch.is_floating_point(t):
            continue
        kind = _classify(name)
        row = f"{name:<27}{kind:<9}{tuple(t.shape)!s:>16}"
        for c in cols:
            row += f"{_norm(arms[c][name]):>16.5f}"
        row += f"{float((arms[other_labels[0]][name].float() - t.float()).abs().max()) if other_labels else 0.0:>14.6g}"
        drift = _drift_from_init(t, name)
        if drift is not None:
            row += f"   init relΔ={drift:.5f}"
            if kind == "adapter" and drift <= args.min_drift:
                failures.append(f"{ref_label} adapter {name} did not move (relΔ={drift:.3g})")
        print(row)

    print()
    for label, state in arms.items():
        adapters = [(n, _norm(state[n])) for n in sorted(state) if _classify(n) == "adapter"]
        frozen = [n for n in state if _classify(n) == "frozen"]
        print(f"[{label}] adapter norms: " + ", ".join(f"{n}={v:.4g}" for n, v in adapters))
        print(f"[{label}] frozen tensors: {len(frozen)} (must match across arms)")
        if adapters and all(v == 0 for _, v in adapters):
            failures.append(f"{label}: every adapter tensor has zero norm (untrained arm)")

    # The decisive cross-arm invariant: a control arm whose adapter norm dwarfs
    # the reference arm's injects a much larger vector, so "real beats control"
    # would not be evidence of content.
    for label in other_labels:
        ratios = []
        for name in sorted(ref):
            if _classify(name) != "adapter" or not torch.is_floating_point(ref[name]):
                continue
            a, b = _norm(ref[name]), _norm(arms[label][name])
            if a > 0:
                ratios.append((name, b / a))
        if ratios:
            worst = max(ratios, key=lambda kv: kv[1])
            summary = ", ".join(f"{n}={r:.3f}x" for n, r in ratios)
            print(f"[{label}/{ref_label}] adapter norm ratios: {summary}")
            if worst[1] >= args.max_norm_ratio:
                failures.append(
                    f"{label}: adapter {worst[0]} norm is {worst[1]:.2f}x the reference arm's "
                    f"(>= {args.max_norm_ratio}x); the contrast is confounded by injection scale"
                )

    print()
    if failures:
        print("AUDIT FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("audit passed: frozen tensors consistent, adapters moved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
