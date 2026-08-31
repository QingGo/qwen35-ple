#!/usr/bin/env python3
"""Phase B custom loader: official Qwen4Exp model with disk-backed PLE shards.

The real full-model load needs a big-memory machine with enough RAM/disk for all
non-PLE weights, plus a transformers version that ships the Qwen4-Exp model
class.  This script is the reproducible entry point:

1. Discover the real PLE metadata (shards, scale, multipliers).
2. Show exactly which checkpoint keys would be skipped.
3. Install ``engramdb.official_loader.install_disk_ple_in_official_model`` on a
   constructed official model instance.

Usage (dry run first):

    PYTHONPATH=src:../EngramDB/python \\
    python scripts/qwen4_ple_custom_loader.py \\
        --model-dir "/Volumes/My Passport/qwen38-ple" \\
        --rows-dir "/Volumes/My Passport/qwen38-rows"

Full model load (big-memory environment):

    PYTHONPATH=src:../EngramDB/python \\
    python scripts/qwen4_ple_custom_loader.py \\
        --model-dir "/Volumes/My Passport/qwen38-ple" \\
        --rows-dir "/Volumes/My Passport/qwen38-rows" \\
        --load-model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import engramdb
from engramdb.official_loader import (
    filter_ngram_shard_state_dict,
    install_disk_ple_in_official_model,
    load_official_checkpoint_without_ngram_shards,
    patch_official_ngram_embedding_for_disk_load,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/Volumes/My Passport/qwen38-ple")
    parser.add_argument("--rows-dir", default="/Volumes/My Passport/qwen38-rows")
    parser.add_argument("--layer-ids", type=int, nargs="*", default=None)
    parser.add_argument("--cache-size", type=int, default=4096)
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()

    info = engramdb.discover_ple(args.model_dir)
    if info is None:
        raise SystemExit(f"no PLE metadata found in {args.model_dir}")

    print(json.dumps({
        "model_dir": args.model_dir,
        "rows_dir": args.rows_dir,
        "ple_embed_dim": info.get("ple_embed_dim"),
        "ngram_size": info.get("ngram_size"),
        "heads_per_ngram": info.get("heads_per_ngram"),
        "shard_count": info.get("ngram_embedding_shard_count"),
        "weight_scale": info.get("weight_scale"),
        "layer_multipliers": info.get("layer_multipliers"),
        "layer_ids": args.layer_ids,
    }, indent=2, ensure_ascii=False))

    # Show the keys that would be excluded from a full state_dict load.
    try:
        import json as _json
        index = _json.loads(
            (Path(args.model_dir) / "model.safetensors.index.json").read_text()
        )
        all_keys = list(index["weight_map"])
        filtered = filter_ngram_shard_state_dict({k: None for k in all_keys})
        skipped = len(all_keys) - len(filtered)
        print(
            f"[loader] would keep {len(filtered)} non-PLE tensors, "
            f"skip {skipped} PLE ngram tensors"
        )
    except Exception as exc:
        print(f"[loader] could not read checkpoint key map: {exc}")

    if not args.load_model:
        print("DRY_RUN_OK (pass --load-model on a big-memory machine to actually load)")
        return 0

    # Full load path.  This requires:
    #   * transformers with Qwen4ExpForCausalLM
    #   * a machine with enough RAM for the non-PLE weights
    try:
        import torch  # noqa: F401
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - big-machine dependency
        raise SystemExit(
            "full model load requires torch + a transformers build with Qwen4Exp"
        ) from exc

    config = AutoConfig.from_pretrained(args.model_dir)
    _tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # Phase B1: patch the official ngram embedding constructor so from_config
    # allocates a one-row placeholder instead of the multi-hundred-GB PLE table.
    # The real rows stay in EngramDB and are installed after non-PLE loading.
    with patch_official_ngram_embedding_for_disk_load():
        model = AutoModelForCausalLM.from_config(config)

    load_result = load_official_checkpoint_without_ngram_shards(
        model,
        args.model_dir,
        strict=False,
    )
    print(
        f"[loader] loaded {load_result.loaded_tensors} non-PLE tensors, "
        f"skipped {load_result.skipped_ngram_tensors} PLE ngram tensors; "
        f"missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}"
    )

    store = engramdb.Store(
        args.rows_dir,
        shards=(
            128
            if info["ngram_embedding_shard_count"] is None
            else info["ngram_embedding_shard_count"]
        ),
        rows_per_shard=2_500_012,
        width=160,
    )
    try:
        replaced = install_disk_ple_in_official_model(
            model,
            store,
            info=info,
            scale=info.get("weight_scale"),
            cache_size=args.cache_size,
            layer_ids=args.layer_ids,
        )
        print(f"[loader] replaced PLE modules: {replaced}")
        model.eval()
        print("OFFICIAL_MODEL_DISK_PLE_READY")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
