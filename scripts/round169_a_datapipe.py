#!/usr/bin/env python
"""Round 169 (A-prep): the pretraining data pipeline prototype and its benchmark.

A needs a token stream big enough that a *from-scratch* pair of models (with and
without a conditional-memory axis, matched on tokens and FLOPs) can be trained and
compared across scales.  This script builds that pipeline and, more importantly,
MEASURES it, because the three quantities that decide A's feasibility are all
empirical:

1. **tokens per second** -- how long 2B tokens takes to tokenize on this box;
2. **tokens per byte** -- the ratio that converts a text budget into a token
   budget and therefore into a disk budget;
3. **thread scaling** -- whether the fast tokenizer actually uses the cores, or
   whether the pipeline is bound somewhere else (I/O, parquet decode, GIL).

Output format: a ``.npy`` of ``uint32``.  ``uint32`` is required, not chosen:
the Qwen vocabulary is 248320, so ``uint16`` cannot represent a token id.
``.npy`` is used rather than raw binary because it carries its own dtype and
shape, so a training run cannot silently misread the width -- and it still
memory-maps, which is what the pretraining loader will do.

One ``<eos>`` is appended between documents.  That is a modelling decision, not a
detail: without it the model cannot tell a document boundary from a topic shift,
and every one of A's arms would share the flaw, so it would cancel -- but the
absolute loss would be wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def log(msg: str) -> None:
    print(f"[a-datapipe] {msg}", flush=True)


def iter_batches(parquet_path: Path, batch_docs: int, limit_docs: int = 0):
    """Yield lists of document strings from a parquet, one batch at a time."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    col = "text" if "text" in pf.schema_arrow.names else pf.schema_arrow.names[0]
    seen = 0
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=[col])
        values = table.column(col).to_pylist()
        for s in range(0, len(values), batch_docs):
            batch = [v for v in values[s : s + batch_docs] if v]
            if not batch:
                continue
            if limit_docs:
                batch = batch[: max(0, limit_docs - seen)]
                if not batch:
                    return
            seen += len(batch)
            yield batch
            if limit_docs and seen >= limit_docs:
                return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-parquet", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-tokens", required=True)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--batch-docs", type=int, default=1000)
    ap.add_argument("--limit-docs", type=int, default=0, help="benchmark only; 0 = all")
    ap.add_argument("--eos-id", type=int, default=-1, help="-1 = read from the tokenizer")
    ap.add_argument("--meta-out", default="")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok_path = Path(args.tokenizer)
    tok_file = tok_path / "tokenizer.json" if tok_path.is_dir() else tok_path
    tokenizer = Tokenizer.from_file(str(tok_file))
    eos = int(args.eos_id)
    if eos < 0:
        eos = int(tokenizer.token_to_id("<|endoftext|>") or tokenizer.token_to_id("<|im_end|>") or 0)
    log(f"tokenizer {tok_file} vocab={tokenizer.get_vocab_size()} eos={eos}")

    raw = Path(args.raw_parquet)
    raw_bytes = raw.stat().st_size
    log(f"raw {raw.name} {raw_bytes / 1e9:.2f} GB; threads={args.threads} batch_docs={args.batch_docs}")

    chunks: list[np.ndarray] = []
    n_docs = n_tokens = n_chars = 0
    t0 = time.time()

    def encode(batch: list[str]) -> list[list[int]]:
        return [e.ids for e in tokenizer.encode_batch(batch)]

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        inflight = []
        for batch in iter_batches(raw, args.batch_docs, args.limit_docs):
            inflight.append(pool.submit(encode, batch))
            n_docs += len(batch)
            n_chars += sum(len(x) for x in batch)
            # Drain in order so memory stays bounded; encode_batch releases the
            # GIL, so this is real parallelism rather than thread interleaving.
            if len(inflight) >= args.threads * 4:
                for fut in inflight:
                    ids = fut.result()
                    for seq in ids:
                        chunks.append(np.asarray(seq, dtype=np.uint32))
                        chunks.append(np.asarray([eos], dtype=np.uint32))
                        n_tokens += len(seq) + 1
                inflight = []
                el = time.time() - t0
                log(
                    f"  docs={n_docs:,} tokens={n_tokens:,} "
                    f"{n_tokens / el:,.0f} tok/s {n_chars / el / 1e6:.1f} Mchar/s"
                )
        for fut in inflight:
            for seq in fut.result():
                chunks.append(np.asarray(seq, dtype=np.uint32))
                chunks.append(np.asarray([eos], dtype=np.uint32))
                n_tokens += len(seq) + 1

    elapsed = time.time() - t0
    log(f"tokenized {n_docs:,} docs -> {n_tokens:,} tokens in {elapsed:.0f}s")
    flat = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint32)
    del chunks
    out = Path(args.out_tokens)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, flat)
    out_bytes = out.stat().st_size
    log(f"wrote {out} {out_bytes / 1e9:.2f} GB")

    meta = {
        "raw_parquet": str(raw), "raw_bytes": raw_bytes,
        "out_tokens": str(out), "out_bytes": out_bytes,
        "docs": n_docs, "tokens": int(flat.size), "chars": n_chars,
        "eos_id": eos, "threads": args.threads, "batch_docs": args.batch_docs,
        "limit_docs": args.limit_docs,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (flat.size / elapsed if elapsed else None),
        "chars_per_second": (n_chars / elapsed if elapsed else None),
        # tokens-per-RAW-BYTE is only meaningful for a whole file.  Under
        # --limit-docs the tokens come from a prefix while raw_bytes is the whole
        # file, which deflated this ratio ~6x on the first benchmark run.
        # tokens-per-CHAR is always valid, so the per-byte figures are only
        # reported when the file was actually complete.
        "tokens_per_char": (flat.size / n_chars if n_chars else None),
        "tokens_per_raw_byte": (
            flat.size / raw_bytes if (raw_bytes and not args.limit_docs) else None
        ),
        "tokenized_bytes_per_raw_byte": (
            out_bytes / raw_bytes if (raw_bytes and not args.limit_docs) else None
        ),
        "whole_file": not args.limit_docs,
    }
    if args.meta_out:
        Path(args.meta_out).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    log(
        f"throughput {meta['tokens_per_second']:,.0f} tok/s | "
        f"tokens/char {meta['tokens_per_char']:.4f}"
        + (
            f" | tokens/raw-byte {meta['tokens_per_raw_byte']:.4f}"
            f" | disk ratio {meta['tokenized_bytes_per_raw_byte']:.2f}x"
            if meta["whole_file"]
            else " | (partial file: per-byte ratios not reported)"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
