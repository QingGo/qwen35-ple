#!/usr/bin/env python3
"""Download the ModelScope source datasets used by M1-M5 mix construction.

This is the reproducibility companion to ``scripts/build_mix.py``.  It downloads
only the small/medium files needed for the current 1M-token experiments, not the
full large corpora.

Usage:

    python scripts/download_mix_sources.py \
      --output-dir data/sources \
      --cache-dir /tmp/modelscope-cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SOURCES = [
    {
        "repo": "AI-ModelScope/alpaca-cleaned",
        "file": "alpaca_data_cleaned.json",
        "category": "chat",
    },
    {
        "repo": "Salesforce/wikitext",
        "file": "wikitext-2-raw-v1/train-00000-of-00001.parquet",
        "category": "wiki",
    },
    {
        "repo": "nohurry/Opus-4.6-Reasoning-3000x-filtered",
        "file": "distilled_corpus_400k_with_cot-filtered.jsonl",
        "category": "cot",
    },
    {
        "repo": "iic/MSAgent-Bench",
        "file": "dev.jsonl",
        "category": "tool",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/sources")
    parser.add_argument("--cache-dir", default="/tmp/modelscope-cache")
    parser.add_argument("--force", action="store_true", help="redownload even if a file already exists")
    args = parser.parse_args()

    try:
        from modelscope_hub import HubApi, RepoType
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "modelscope-hub is required; install with "
            "`pip install --only-binary :all: modelscope-hub`"
        ) from exc

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    api = HubApi()
    for source in SOURCES:
        repo = source["repo"]
        file = source["file"]
        dest = out / Path(file).name
        if not args.force and dest.exists() and dest.stat().st_size > 0:
            print(f"[source] exists, skip {dest}")
            continue
        print(f"[source] downloading {repo}/{file}")
        path = api.download_file(
            repo,
            RepoType.DATASET,
            file,
            cache_dir=args.cache_dir,
            local_dir=str(out),
        )
        print(f"[source] -> {path} ({path.stat().st_size} bytes)")

    # Convert the small WikiText parquet to JSONL if pyarrow is available; the
    # builder can also read the parquet directly, but JSONL keeps the local
    # pipeline usable without platform-specific parquet dependencies.
    wiki_parquet = out / "train-00000-of-00001.parquet"
    wiki_jsonl = out / "wikitext.jsonl"
    if wiki_parquet.exists() and not wiki_jsonl.exists():
        try:
            import pyarrow.parquet as pq
        except Exception:
            print("[source] pyarrow not available; keep parquet as-is")
            return 0
        table = pq.read_table(wiki_parquet)
        with wiki_jsonl.open("w", encoding="utf-8") as f:
            for row in table.to_pylist():
                text = row.get("text")
                if text:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        print(f"[source] converted {wiki_parquet} -> {wiki_jsonl}")

    print("[source] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
