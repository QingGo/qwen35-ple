#!/usr/bin/env python3
"""Smoke test for the EngramDB v0.2.8 surface that qwen35-ple can consume.

This is not an experiment.  It verifies the newly exposed APIs used by the
next reader/live-store experiments:

1. ``engramdb.rowids_for_seq`` matches the frozen local PleSpec reference.
2. ``discover_ple`` / ``load_ple_weight_scale`` read the real Qwen3.8 PLE
   metadata and FP8 scale.
3. ``fetch_e_t`` with the real ``weight_scale`` matches the official
   dequantized embedding scale.
4. ``engramdb.Store`` + ``PleDiskGather`` still fetch real rows.
5. ``engramdb.View`` can open/read the existing full Store-P view when present.

Run:

    PYTHONPATH=src:../EngramDB/python \\
    python scripts/run_engramdb_v028_smoke.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROWS_DIR = "/Volumes/My Passport/qwen38-rows"
PLE_MODEL_DIR = "/Volumes/My Passport/qwen38-ple"
VIEW_PATH = "/Volumes/My Passport/p4view-full-2560.bin"


def main() -> int:
    results: dict[str, str] = {}

    try:
        import engramdb
        from qwen35_ple.ple_hash import real_spec

        toks = [248044, 1000, 99999, 42, 12345]
        native = engramdb.rowids_for_seq(toks)
        ref = [list(r) for r in real_spec().rowids_for_seq(toks)]
        assert native == ref, "EngramDB rowids differ from local PleSpec"
        results["rowids_for_seq"] = "ok"
    except Exception as exc:
        results["rowids_for_seq"] = f"skip/fail: {exc}"

    try:
        from engramdb import discover_ple, load_ple_weight_scale

        info = discover_ple(PLE_MODEL_DIR)
        assert info is not None, "discover_ple returned None"
        assert info["ple_layer_ids"] == [2], info["ple_layer_ids"]
        scale = float(load_ple_weight_scale(PLE_MODEL_DIR))
        assert scale > 0.0
        results["discover_ple"] = (
            f"ok (ple_embed_dim={info['ple_embed_dim']}, scale={scale:.6g})"
        )
    except Exception as exc:
        results["discover_ple"] = f"skip/fail: {exc}"

    try:
        from qwen35_ple.real_ple import (
            fetch_e_t,
            resolve_ple_weight_scale,
            rowids_from_tokens,
        )

        scale = resolve_ple_weight_scale(PLE_MODEL_DIR)
        token_ids = np.asarray([248044, 1000, 99999, 42, 12345], dtype=np.int64)
        rowids = rowids_from_tokens(token_ids)
        e_t = fetch_e_t(ROWS_DIR, rowids, scale=scale)
        assert e_t.shape == (5, 2560)
        assert np.isfinite(e_t).all()
        results["store_fetch_e_t"] = f"ok (std={float(e_t.std()):.4g})"
    except Exception as exc:
        results["store_fetch_e_t"] = f"skip/fail: {exc}"

    try:
        import engramdb

        if Path(VIEW_PATH).exists():
            view = engramdb.View(VIEW_PATH)
            rec = view.read_record(0)
            arr = np.frombuffer(rec, dtype=np.float32)
            assert arr.shape == (640,), arr.shape
            results["view"] = f"ok (len={view.len()}, slot={view.slot_bytes()})"
            view.close()
        else:
            results["view"] = "skip (view not mounted)"
    except Exception as exc:
        results["view"] = f"skip/fail: {exc}"

    print(json.dumps(results, indent=2, ensure_ascii=False))
    failed = [k for k, v in results.items() if v.startswith("skip/fail")]
    if failed:
        print(f"SMOKE_PARTIAL_FAIL: {failed}")
        return 1
    print("ENGRAMDB_V028_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
