#!/usr/bin/env python3
"""Extract the small official Qwen3.8 PLE reader weights into a portable file.

The official Qwen3.8-Flash-Next checkpoint is ~49GB.  The PLE *reader* (not the
51B row table) is only ~66MB and can be copied/used independently:

    model.language_model.layers.1.ple.key_proj.weight
    model.language_model.layers.1.ple.value_proj.weight
    model.language_model.layers.1.ple.norm_key.weight
    model.language_model.layers.1.ple.norm_query.weight
    model.language_model.layers.1.ple.norm_conv.weight
    model.language_model.layers.1.ple.conv1d.weight
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


SOURCE_KEYS = [
    "model.language_model.layers.1.ple.key_proj.weight",
    "model.language_model.layers.1.ple.value_proj.weight",
    "model.language_model.layers.1.ple.norm_key.weight",
    "model.language_model.layers.1.ple.norm_query.weight",
    "model.language_model.layers.1.ple.norm_conv.weight",
    "model.language_model.layers.1.ple.conv1d.weight",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/Volumes/My Passport/qwen38-ple")
    parser.add_argument("--output", default="data/official_ple_reader.pt")
    args = parser.parse_args()

    root = Path(args.model_dir)
    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"missing {index_path}")
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]

    state: dict[str, torch.Tensor] = {}
    for key in SOURCE_KEYS:
        if key not in weight_map:
            raise SystemExit(f"missing tensor in index: {key}")
        shard = root / weight_map[key]
        with safe_open(shard, framework="pt") as f:
            state[key] = f.get_tensor(key)
        print(f"  loaded {key}: {tuple(state[key].shape)} {state[key].dtype}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"saved -> {out}  ({out.stat().st_size / 1e6:.1f} MB, sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
