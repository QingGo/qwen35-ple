#!/usr/bin/env python3
"""Build a WikiText slice disjoint from the PURE_WIKI CPT corpus (Round 158).

``data/phase1/PURE_WIKI`` was produced by ``scripts/build_mix.py`` from
``data/sources/wikitext.jsonl`` with

    --target-tokens 1000000 --ratios wiki=100 --seed 0
    --chunk-tokens 512 --chunk-chars 4000 --allow-shortfall
    --exclude-qa data/qa-expanded-150.json

so 9,552 of the 23,767 WikiText records were consumed and ~14k were not.  The
records that were **not** consumed are a held-out slice that the graft's table
never saw either, which is what makes them usable as an uncontaminated
evaluation stream for the count-based reference models.

Rather than re-implementing the selection (which would silently drift from the
real corpus), this script *imports* ``build_mix`` and drives its own helper
functions:

  * ``_load_category``          -- same record loading / token-length counting
  * ``_load_qa_needles`` /
    ``_filter_contaminated``    -- same QA decontamination pass
  * a byte-for-byte replay of ``_select_for_budget``'s loop that additionally
    records *which* record indices were touched, cross-checked against
    ``_select_for_budget``'s own output (asserted, not assumed).

Note the replay must reproduce one quirk of the real build: ``build_mix.main``
filters ``texts`` for QA contamination but passes the **unfiltered** ``lengths``
list to ``_select_for_budget``, so record i is budgeted with the token length of
a different record.  That is reproduced here on purpose -- the goal is the same
split the corpus actually used, not the split the code intended.

Decontamination caveat (stated for the record): a disjoint *record* slice only
guarantees the count models and the graft's table did not train on these tokens.
It says nothing about the Qwen3.5/Qwen3.8 backbone, which was pretrained on web
text that almost certainly contains WikiText/Wikipedia; the backbone is
therefore contaminated on both the train and the held-out slice, and no
WikiText-based number here is a clean measure of "knowledge not in the weights".

Usage::

    python scripts/build_wikitext_heldout.py \
        --wikitext data/sources/wikitext.jsonl \
        --pure-wiki-dir data/phase1/PURE_WIKI \
        --qa data/qa-expanded-150.json \
        --tokenizer /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B \
        --output data/phase1/wikitext-heldout
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Sequence, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_mix as bm  # noqa: E402  (sibling script, deliberately reused)
from bench_ngram_reference import ngram_hash_keys  # noqa: E402  (single canonical copy)


def log(msg: str) -> None:
    print("[heldout] {}".format(msg), flush=True)


def select_with_indices(texts: Sequence[str], lengths: Sequence[int], budget: int,
                        rng: random.Random, chunk_tokens: int, tokenizer):
    """Verbatim replay of ``build_mix._select_for_budget`` that tracks indices."""
    order = list(range(len(texts)))
    rng.shuffle(order)
    used: List[int] = []
    n_chunks = 0
    total = 0
    for i in order:
        if total >= budget:
            break
        n = lengths[i]
        if n > chunk_tokens:
            ids = tokenizer.encode(texts[i], add_special_tokens=False)
            touched = False
            for start in range(0, len(ids), chunk_tokens):
                if total >= budget:
                    break
                shard_len = len(ids[start : start + chunk_tokens])
                total += shard_len + (1 if bm.EOS_SEPARATOR else 0)
                n_chunks += 1
                touched = True
            if touched:
                used.append(i)
            continue
        used.append(i)
        n_chunks += 1
        total += n + (1 if bm.EOS_SEPARATOR else 0)
    return used, n_chunks, total


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _text_keys(text: str, n: int, mults: np.ndarray) -> np.ndarray:
    b = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    return ngram_hash_keys(b, n, 256, mults)


def _byte_keys(buf: bytes, n: int, mults: np.ndarray) -> np.ndarray:
    return ngram_hash_keys(np.frombuffer(buf, dtype=np.uint8), n, 256, mults)


def token_overlap_filter(rec_ids: Sequence[np.ndarray], reference: np.ndarray,
                         n_tokens: Sequence[int], seed: int) -> dict:
    """Drop candidates sharing a verbatim token span with the real training stream.

    This is the criterion that matters: ``PURE_WIKI/tokens.npy`` is what the graft
    trained on, so a held-out record may not contain any span that also occurs
    there.  The check is done in *token* space (not text) because the stream
    stores ``encode(decode(shard))`` and its decoded spacing/punctuation differs
    from the source text, which makes text-level matching both noisy and
    unreliable.

    Why the complement of ``build_mix``'s selection is not enough on its own:
    WikiText-2-raw repeats records (2,575 of 23,767 sit in duplicate groups, the
    largest with 129 copies) and it also carries nested granularities, so a
    paragraph in the complement can be a verbatim substring of an article that
    PURE_WIKI consumed.  Measured here rather than assumed.
    """
    ref = np.asarray(reference, dtype=np.int64).reshape(-1)
    top = int(ref.max()) if ref.size else 0
    for ids in rec_ids:
        if ids.size:
            top = max(top, int(ids.max()))
    vocab = top + 2          # one past the largest real id
    sentinel = top + 1       # separator that cannot occur in data
    mults = np.random.default_rng(seed).integers(
        1, 2 ** 63, size=512, dtype=np.uint64) | np.uint64(1)

    lengths = np.array([ids.size for ids in rec_ids], dtype=np.int64)
    joined = np.empty(int(lengths.sum()) + len(rec_ids), dtype=np.int64)
    starts = np.zeros(len(rec_ids) + 1, dtype=np.int64)
    pos = 0
    for i, ids in enumerate(rec_ids):
        starts[i] = pos
        joined[pos : pos + ids.size] = ids
        pos += ids.size
        joined[pos] = sentinel
        pos += 1
    starts[len(rec_ids)] = pos
    seppos = np.flatnonzero(joined == sentinel)

    result = {"reference": "PURE_WIKI/tokens.npy (the graft's training stream)",
              "reference_tokens": int(ref.shape[0]), "candidate_records": len(rec_ids),
              "candidate_tokens": int(lengths.sum()), "per_ngram": {}}
    excluded: Dict[int, List[int]] = {}
    for n in sorted(set(int(x) for x in n_tokens)):
        if n < 2:
            continue
        ref_keys = np.unique(ngram_hash_keys(ref, n, vocab, mults))
        keys = ngram_hash_keys(joined, n, vocab, mults)
        N = keys.shape[0]
        if N:
            w = np.arange(N, dtype=np.int64)
            owner = np.clip(np.searchsorted(starts, w, side="right") - 1,
                            0, len(rec_ids) - 1)
            spans_sep = (np.searchsorted(seppos, w, side="right")
                         != np.searchsorted(seppos, w + n, side="right"))
            valid = (~spans_sep) & (w + n <= starts[owner + 1])
            hit = np.isin(keys, ref_keys) & valid
            bad = sorted({int(b) for b in np.unique(owner[hit]).tolist()
                          if b < len(rec_ids)})
            del w, owner, spans_sep, valid, hit
        else:
            bad = []
        excluded[n] = bad
        result["per_ngram"][str(n)] = {
            "records_excluded": len(bad),
            "tokens_excluded": int(lengths[bad].sum()) if bad else 0,
        }
        del ref_keys, keys
    del joined, seppos
    return {"summary": result, "excluded_by_ngram": excluded}
    """Drop candidate records that share a long verbatim span with used records.

    WikiText-2-raw is not a set of disjoint documents: it repeats records (2,575
    of 23,767 sit in duplicate groups, the largest with 129 copies) and it also
    carries nested granularities, so a paragraph in the complement can be a
    verbatim substring of an article PURE_WIKI consumed.  The complement of
    ``build_mix``'s selection is therefore NOT a clean held-out set.

    This compares the candidate records against the normalised text of the
    records PURE_WIKI actually consumed and drops every candidate sharing an
    ``n``-character span with it.  All offsets are **bytes** (the corpus is
    heavily non-ASCII, so character offsets would misalign the record ranges).

    Character level rather than token level because PURE_WIKI/tokens.npy stores
    ``encode(decode(shard))`` -- a re-encoding that loses about one token per
    512-token chunk -- so exact token n-gram matching misses real duplicates
    whose text is unchanged.
    """
    mults = np.random.default_rng(seed).integers(
        1, 2 ** 63, size=512, dtype=np.uint64) | np.uint64(1)
    ref_text = _normalize(" \n ".join(kept_texts[i] for i in used))
    enc = [_normalize(kept_texts[i]).encode("utf-8") for i in candidate]
    joined = b"\x00".join(enc)
    starts = np.zeros(len(candidate) + 1, dtype=np.int64)
    pos = 0
    for idx, e in enumerate(enc):
        starts[idx] = pos
        pos += len(e) + 1
    starts[len(candidate)] = pos
    sep_idx = np.flatnonzero(np.frombuffer(joined, dtype=np.uint8) == 0)
    result = {"reference": "normalised concatenation of the {} records PURE_WIKI "
                           "consumed".format(len(used)),
              "reference_chars": len(ref_text), "candidate_records": len(candidate),
              "per_ngram": {}}
    per_ngram_excluded = {}
    for n in sorted(set(int(x) for x in n_chars)):
        ref_keys = np.unique(_text_keys(ref_text, n, mults))
        keys = _byte_keys(joined, n, mults)
        N = keys.shape[0]
        if N:
            wpos = np.arange(N, dtype=np.int64)
            owner = np.clip(np.searchsorted(starts, wpos, side="right") - 1,
                            0, len(candidate) - 1)
            contains_sep = np.zeros(N, dtype=bool)
            if sep_idx.size:
                contains_sep = (np.searchsorted(sep_idx, wpos, side="right")
                                != np.searchsorted(sep_idx, wpos + n, side="right"))
            valid = (~contains_sep) & (wpos + n <= starts[owner + 1])
            hit = np.isin(keys, ref_keys) & valid
            bad = {candidate[b] for b in np.unique(owner[hit]).tolist()
                   if b < len(candidate)}
            del wpos, owner, contains_sep, valid, hit
        else:
            bad = set()
        per_ngram_excluded[n] = sorted(bad)
        result["per_ngram"][str(n)] = {"records_excluded": len(bad)}
        del ref_keys, keys
    del enc, joined, sep_idx
    return {"summary": result, "excluded_by_ngram": per_ngram_excluded}


def verify_disjoint(tokenizer, kept_texts: Sequence[str], held: Sequence[int],
                    corpus_path: Path, needles: Sequence[int]) -> dict:
    """Check the held-out records really are absent from the built corpus.

    ``data/phase1/PURE_WIKI/corpus.txt`` stores the *decoded* shards, so the
    comparison is made the same way: decode the first N tokens of each held-out
    record and look for that string in the whitespace-normalised corpus.  A
    record that PURE_WIKI consumed would match at every needle length.
    """
    if not corpus_path.exists():
        return {"corpus": str(corpus_path), "checked": False,
                "reason": "corpus.txt not present"}
    haystack = _normalize(corpus_path.read_text(encoding="utf-8", errors="ignore"))
    out = {"corpus": str(corpus_path), "checked": True, "haystack_chars": len(haystack),
           "per_needle": {}, "matched_record_indices": []}
    matched_any = set()
    for n_tok in needles:
        hits = 0
        for i in held:
            ids = tokenizer.encode(kept_texts[i], add_special_tokens=False)[:n_tok]
            if len(ids) < n_tok:
                continue
            needle = _normalize(tokenizer.decode(ids, skip_special_tokens=False))
            if len(needle) < 20:
                continue
            if needle in haystack:
                hits += 1
                matched_any.add(i)
        out["per_needle"][str(n_tok)] = hits
    out["records_checked"] = len(held)
    out["records_matching_any_needle"] = len(matched_any)
    out["matched_record_indices"] = sorted(matched_any)[:50]
    out["interpretation"] = (
        "0 matches means the held-out slice is verifiably absent from the corpus "
        "PURE_WIKI actually built; a small number of matches would indicate "
        "boundary records where the replayed selection drifted from the real one.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wikitext", default="data/sources/wikitext.jsonl")
    ap.add_argument("--pure-wiki-dir", default="data/phase1/PURE_WIKI")
    ap.add_argument("--qa", default="data/qa-expanded-150.json")
    ap.add_argument("--tokenizer", default="data/models/Qwen3.5-0.8B")
    ap.add_argument("--output", default="data/phase1/wikitext-heldout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-tokens", type=int, default=1_000_000)
    ap.add_argument("--chunk-tokens", type=int, default=bm.DEFAULT_CHUNK_TOKENS)
    ap.add_argument("--chunk-chars", type=int, default=bm.DEFAULT_CHUNK_CHARS)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="cap the held-out stream length (0 = all held-out records)")
    ap.add_argument("--verify-against", default=None,
                    help="corpus.txt to verify disjointness against "
                         "(default: <pure-wiki-dir>/corpus.txt; empty string skips)")
    ap.add_argument("--verify-needles", default="32,128",
                    help="token-prefix lengths decoded and searched for")
    ap.add_argument("--decontaminate", dest="decontaminate", action="store_true",
                    default=True,
                    help="drop candidate records that share a long verbatim span "
                         "with the records PURE_WIKI consumed (default: on)")
    ap.add_argument("--no-decontaminate", dest="decontaminate", action="store_false")
    ap.add_argument("--decontaminate-against", default=None,
                    help="training token stream to verify against "
                         "(default: <pure-wiki-dir>/tokens.npy)")
    ap.add_argument("--overlap-tokens", type=int, default=16,
                    help="length of the verbatim token span that disqualifies "
                         "a record")
    ap.add_argument("--overlap-sensitivity", default="8,16,32,64",
                    help="also report exclusion counts at these span lengths")
    args = ap.parse_args()

    t0 = time.time()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    log("tokenizer loaded (vocab={}, eos={})".format(
        tokenizer.vocab_size, tokenizer.eos_token_id))

    wiki_path = Path(args.wikitext)
    if not wiki_path.exists():
        raise SystemExit("wikitext source not found: {}".format(wiki_path))
    texts, lengths = bm._load_category(
        [wiki_path], "wiki", tokenizer, None, chunk_chars=args.chunk_chars)
    log("loaded {} wiki records ({} tokens)".format(len(texts), sum(lengths)))

    needles = bm._load_qa_needles(args.qa)
    kept_texts, removed = bm._filter_contaminated(texts, needles)
    log("QA filter removed {} records -> kept {}".format(removed, len(kept_texts)))

    # faithful replay: real build passes the UNFILTERED lengths list
    rng = random.Random(args.seed)
    used_idx, n_chunks, total = select_with_indices(
        kept_texts, lengths, args.target_tokens, rng, args.chunk_tokens, tokenizer)

    # cross-check against the real helper (same rng state -> same split)
    rng2 = random.Random(args.seed)
    sel_texts, sel_lengths, used_ref = bm._select_for_budget(
        kept_texts, lengths, args.target_tokens, rng2, args.chunk_tokens, tokenizer)
    if len(sel_texts) != n_chunks or used_ref != total:
        raise SystemExit(
            "replay mismatch: chunks {}/{} tokens {}/{}".format(
                len(sel_texts), n_chunks, used_ref, total))
    log("replay verified against build_mix: {} chunks, {} tokens".format(n_chunks, total))

    used_set: Set[int] = set(used_idx)
    candidate = [i for i in range(len(kept_texts)) if i not in used_set]
    log("records: {} used, {} in the complement".format(len(used_set), len(candidate)))

    # tokenise every complement record once; assembly and reporting use the ids
    rec_ids = {i: np.asarray(tokenizer.encode(kept_texts[i], add_special_tokens=False),
                             dtype=np.int64) for i in candidate}
    log("tokenised {} complement records ({} tokens)".format(
        len(rec_ids), int(sum(r.shape[0] for r in rec_ids.values()))))

    if args.decontaminate:
        levels = sorted({args.overlap_tokens} | {
            int(p) for p in args.overlap_sensitivity.split(",") if p.strip()})
        ref_path = Path(args.decontaminate_against) if args.decontaminate_against \
            else Path(args.pure_wiki_dir) / "tokens.npy"
        if not ref_path.exists():
            raise SystemExit("decontamination reference not found: {}".format(ref_path))
        ref_tokens = np.load(ref_path).astype(np.int64).reshape(-1)
        filt = token_overlap_filter([rec_ids[i] for i in candidate], ref_tokens,
                                    levels, args.seed)
        # the filter indexes the positional candidate list; map back to record ids
        bad = {candidate[p] for p in filt["excluded_by_ngram"][args.overlap_tokens]}
        overlap = {
            "method": "verbatim token {}-gram overlap against {}: a candidate "
                      "record is dropped if any {}-token span of it also occurs in "
                      "the stream the graft trained on".format(
                          args.overlap_tokens, ref_path, args.overlap_tokens),
            "chosen_tokens": args.overlap_tokens,
            "reference_stream": str(ref_path),
            "reference_sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest(),
            "excluded_records": len(bad),
            "excluded_tokens": int(sum(rec_ids[i].shape[0] for i in bad)),
            "sensitivity": filt["summary"]["per_ngram"],
            "candidate_records": filt["summary"]["candidate_records"],
            "candidate_tokens": filt["summary"]["candidate_tokens"],
            "excluded_record_indices": sorted(bad),
            "excluded_record_indices_sha256": sha256_text(
                ",".join(str(i) for i in sorted(bad))),
        }
        log("token-overlap filter vs {}: {} of {} complement records share a "
            ">= {}-token span with the training stream -> dropped (sensitivity {})".format(
                ref_path, len(bad), len(candidate), args.overlap_tokens,
                filt["summary"]["per_ngram"]))
    else:
        bad = set()
        overlap = {"skipped": True,
                   "reason": "--no-decontaminate: complement slice used as-is"}

    kept_pairs = [(i, rec_ids[i]) for i in candidate if i not in bad]
    token_ids: List[int] = []
    held: List[int] = []
    for i, ids in kept_pairs:
        token_ids.extend(ids.tolist())
        if bm.EOS_SEPARATOR and tokenizer.eos_token_id is not None:
            token_ids.append(tokenizer.eos_token_id)
        held.append(i)
        if args.max_tokens and len(token_ids) >= args.max_tokens:
            break
    tokens = np.asarray(token_ids, dtype=np.int64)
    log("held-out stream: {} records, {} tokens".format(len(held), int(tokens.shape[0])))

    verify_path = Path(args.verify_against) if args.verify_against is not None \
        else Path(args.pure_wiki_dir) / "corpus.txt"
    needles = [int(p) for p in args.verify_needles.split(",") if p.strip()]
    verification = {"skipped": True} if not needles else verify_disjoint(
        tokenizer, kept_texts, held, verify_path, needles)
    if verification.get("checked"):
        log("post-filter text check vs {}: {} records checked, {} match -> {}".format(
            verify_path, verification["records_checked"],
            verification["records_matching_any_needle"],
            verification["per_needle"]))
    else:
        log("post-filter text check skipped ({})".format(
            verification.get("reason", "disabled")))

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
        "purpose": "WikiText records NOT consumed by data/phase1/PURE_WIKI; "
                   "evaluation stream for the count-based reference models.",
        "source": str(wiki_path),
        "source_sha256": hashlib.sha256(wiki_path.read_bytes()).hexdigest(),
        "tokenizer": args.tokenizer,
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "qa_filter": args.qa,
        "records_total": len(texts),
        "records_removed_by_qa_filter": removed,
        "records_kept": len(kept_texts),
        "records_used_by_pure_wiki": len(used_set),
        "records_in_complement": len(candidate),
        "records_held_out": len(held),
        "held_out_tokens": int(len(tokens)),
        "pure_wiki_selected_tokens_replayed": int(used_ref),
        "pure_wiki_target_tokens": args.target_tokens,
        "seed": args.seed,
        "chunk_tokens": args.chunk_tokens,
        "chunk_chars": args.chunk_chars,
        "eos_separator": bm.EOS_SEPARATOR,
        "tokens_sha256": h.hexdigest(),
        "heldout_record_indices": held,
        "heldout_record_indices_sha256": sha256_text(
            ",".join(str(i) for i in held)),
        "used_record_indices_sha256": sha256_text(
            ",".join(str(i) for i in sorted(used_set))),
        "overlap_filter": overlap,
        "disjointness_verification": verification,
        "decontamination_caveat": (
            "two levels of decontamination against the corpora the graft trained "
            "on: (1) records build_mix did not select, (2) no {}-character "
            "verbatim span shared with any record PURE_WIKI did consume (WikiText-2 "
            "repeats 2,575 of 23,767 records, so the complement alone leaks). "
            "Neither level says anything about the Qwen3.5/Qwen3.8 backbone, which "
            "was pretrained on web text that very likely includes WikiText; these "
            "positions are NOT clean w.r.t. the backbone.".format(args.overlap_tokens)),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log("wrote {} ({} tokens, {} records, {:.1f}s)".format(
        out / "tokens.npy", len(tokens), len(held), time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
