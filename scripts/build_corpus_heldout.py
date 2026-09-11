#!/usr/bin/env python3
"""Build a document-disjoint held-out slice for any build_mix corpus (Round 159).

``scripts/build_wikitext_heldout.py`` does this for PURE_WIKI only.  The
round-159 cross-distribution question needs the same treatment for PURE_CODE and
PURE_STEM, so this script is a thin generalisation: it reads the corpus's own
``manifest.json`` (category, sources, target, seed, chunking, QA filter) and
replays exactly that build, keeping the records it did **not** consume, then
applies the same token-level overlap filter against the corpus's own
``tokens.npy``.

Everything is imported from the round-158 builder so there is one implementation
of the replay, the overlap filter and the verification:

  * ``select_with_indices``      -- verbatim replay of build_mix._select_for_budget
  * ``token_overlap_filter``     -- drop candidates sharing a >=N-token span
  * ``verify_disjoint``          -- post-filter check against corpus.txt

PURE_FINEWEB is *not* supported and must not be faked: its manifest records a
pre-tokenized dump ("first N tokens" of a 30M-token stream), so there are no
source records to hold out; for that corpus only an in-stream split exists.

Usage::

    python scripts/build_corpus_heldout.py \
        --corpus-dir data/phase1/PURE_CODE \
        --tokenizer data/models/Qwen3.5-0.8B \
        --output data/phase1/PURE_CODE-heldout-decon \
        --max-tokens 1152891 --overlap-tokens 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_mix as bm  # noqa: E402
import build_wikitext_heldout as bh  # noqa: E402


def log(msg: str) -> None:
    print("[corpus-heldout] {}".format(msg), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", required=True,
                    help="a build_mix corpus dir containing manifest.json + tokens.npy")
    ap.add_argument("--tokenizer", default="data/models/Qwen3.5-0.8B")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=1_152_891,
                    help="cap the held-out stream so corpora can be compared at "
                         "identical eval size")
    ap.add_argument("--overlap-mode", choices=["token", "char"], default="token")
    ap.add_argument("--overlap-chars", type=int, default=32)
    ap.add_argument("--overlap-tokens", type=int, default=16)
    ap.add_argument("--overlap-sensitivity", default="8,16,32,64")
    ap.add_argument("--verify-needles", default="32,128")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    cdir = Path(args.corpus_dir)
    man = json.loads((cdir / "manifest.json").read_text())
    if man.get("schema") != "qwen35-ple-mix-v1":
        raise SystemExit("{} is not a build_mix corpus (schema {!r}); refusing to guess"
                         .format(cdir, man.get("schema")))
    ratios = man["ratios"]
    cats = [c for c, v in ratios.items() if v > 0]
    if len(cats) != 1:
        raise SystemExit("expected exactly one non-zero ratio, got {}".format(ratios))
    cat = cats[0]
    sources = [Path(p) for p in man["sources"].get(cat, [])]
    if not sources:
        raise SystemExit("manifest lists no source for category {}".format(cat))
    for s in sources:
        if not s.exists():
            raise SystemExit("source not found locally: {}".format(s))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    log("corpus={} category={} sources={} target={} seed={} chunk={}".format(
        cdir.name, cat, [str(s) for s in sources], man["target_tokens"], man["seed"],
        man["chunk_tokens"]))

    texts, lengths = bm._load_category(sources, cat, tokenizer, None,
                                       chunk_chars=man["chunk_chars"])
    log("loaded {} records ({} tokens)".format(len(texts), sum(lengths)))
    needles = bm._load_qa_needles(man["exclude_qa"])
    kept_texts, removed = bm._filter_contaminated(texts, needles)
    log("QA filter removed {} records -> kept {} (manifest said {})".format(
        removed, len(kept_texts), man.get("excluded_records", {}).get(cat)))

    rng = random.Random(man["seed"])
    used_idx, n_chunks, total = bh.select_with_indices(
        kept_texts, lengths, man["target_tokens"], rng, man["chunk_tokens"], tokenizer)
    rng2 = random.Random(man["seed"])
    sel_texts, _sl, used_ref = bm._select_for_budget(
        kept_texts, lengths, man["target_tokens"], rng2, man["chunk_tokens"], tokenizer)
    if len(sel_texts) != n_chunks or used_ref != total:
        raise SystemExit("replay mismatch: chunks {}/{} tokens {}/{}".format(
            len(sel_texts), n_chunks, used_ref, total))
    log("replay verified: {} chunks / {} tokens (manifest selected_tokens {})".format(
        n_chunks, total, man.get("selected_tokens")))

    used = set(used_idx)
    candidate = [i for i in range(len(kept_texts)) if i not in used]
    rec_ids = {i: np.asarray(tokenizer.encode(kept_texts[i], add_special_tokens=False),
                             dtype=np.int64) for i in candidate}
    log("complement: {} records / {} tokens".format(
        len(candidate), int(sum(r.shape[0] for r in rec_ids.values()))))

    if args.overlap_mode == "char":
        cf = bh.char_overlap_filter(kept_texts, candidate,
                                    (cdir / "corpus.txt").read_text(
                                        encoding="utf-8", errors="ignore"),
                                    args.overlap_chars, args.seed)
        bad = set(cf["excluded"])
        filt_summary = {"mode": "char", "chars": args.overlap_chars,
                        "excluded": len(cf["excluded"]),
                        "untestable_records_too_short": len(cf["untestable"]),
                        "untestable_tokens": int(sum(rec_ids[i].shape[0]
                                                     for i in cf["untestable"]))}
        log("char-overlap filter vs {}/corpus.txt: {} of {} candidates share a >= "
            "{}-char window -> dropped; {} too short to test ({:,} tokens, kept)".format(
                cdir.name, len(cf["excluded"]), len(candidate), args.overlap_chars,
                len(cf["untestable"]), filt_summary["untestable_tokens"]))
    else:
        levels = sorted({args.overlap_tokens} | {
            int(p) for p in args.overlap_sensitivity.split(",") if p.strip()})
        ref_tokens = np.load(cdir / "tokens.npy").astype(np.int64).reshape(-1)
        filt = bh.token_overlap_filter([rec_ids[i] for i in candidate], ref_tokens,
                                       levels, args.seed)
        bad = {candidate[p] for p in filt["excluded_by_ngram"][args.overlap_tokens]}
        filt_summary = {"mode": "token", "tokens": args.overlap_tokens,
                        "sensitivity": filt["summary"]["per_ngram"]}
        log("token-overlap filter vs {}/tokens.npy: {} of {} complement records dropped "
            "(sensitivity {})".format(cdir.name, len(bad), len(candidate),
                                      filt["summary"]["per_ngram"]))

    token_ids: List[int] = []
    held: List[int] = []
    for i in candidate:
        if i in bad:
            continue
        token_ids.extend(rec_ids[i].tolist())
        if bm.EOS_SEPARATOR and tokenizer.eos_token_id is not None:
            token_ids.append(tokenizer.eos_token_id)
        held.append(i)
        if args.max_tokens and len(token_ids) >= args.max_tokens:
            break
    tokens = np.asarray(token_ids, dtype=np.int64)
    log("held-out stream: {} records / {} tokens".format(len(held), tokens.shape[0]))

    keep = [i for i in held if args.verify_needles]
    needles_len = [int(p) for p in args.verify_needles.split(",") if p.strip()]
    verification = bh.verify_disjoint(tokenizer, kept_texts, held,
                                      cdir / "corpus.txt", needles_len) if keep \
        else {"skipped": True}
    if verification.get("checked"):
        log("post-filter text check vs corpus.txt: {} records, {} match -> {}".format(
            verification["records_checked"], verification["records_matching_any_needle"],
            verification["per_needle"]))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "tokens.npy", tokens)
    h = hashlib.sha256()
    with (out / "tokens.npy").open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    manifest = {
        "schema": "qwen35-ple-heldout-v1",
        "name": out.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "purpose": "{} records NOT consumed by {}; held-out stream for the "
                   "count-based reference models".format(cat, cdir.name),
        "source_corpus_dir": str(cdir),
        "source_corpus_manifest": str(cdir / "manifest.json"),
        "source_manifest_selected_tokens": man.get("selected_tokens"),
        "category": cat,
        "sources": [str(s) for s in sources],
        "tokenizer": args.tokenizer,
        "qa_filter": man["exclude_qa"],
        "records_total": len(texts),
        "records_removed_by_qa_filter": removed,
        "records_kept": len(kept_texts),
        "records_used_by_corpus": len(used),
        "records_in_complement": len(candidate),
        "records_held_out": len(held),
        "held_out_tokens": int(tokens.shape[0]),
        "max_tokens_cap": args.max_tokens,
        "replayed_selected_tokens": int(used_ref),
        "seed": man["seed"],
        "chunk_tokens": man["chunk_tokens"],
        "chunk_chars": man["chunk_chars"],
        "eos_separator": bm.EOS_SEPARATOR,
        "tokens_sha256": h.hexdigest(),
        "overlap_filter": dict(filt_summary, **{
            "chosen_tokens": args.overlap_tokens,
            "reference_stream": str(cdir / "tokens.npy"),
            "excluded_records": len(bad),
            "excluded_tokens": int(sum(rec_ids[i].shape[0] for i in bad)),
            "sensitivity": filt_summary.get("sensitivity", {}),
            "candidate_records": len(candidate),
            "candidate_tokens": int(sum(r.shape[0] for r in rec_ids.values())),
        }),
        "disjointness_verification": verification,
        "decontamination_caveat": (
            "two levels against the corpora the count model trains on: (1) records "
            "build_mix did not select, (2) no {}-token verbatim span shared with "
            "{}/tokens.npy. Neither says anything about the Qwen backbone.".format(
                args.overlap_tokens, cdir.name)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                       encoding="utf-8")
    log("wrote {} ({:,} tokens, {:.1f}s)".format(out / "tokens.npy", tokens.shape[0],
                                                 time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
