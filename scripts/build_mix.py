#!/usr/bin/env python3
"""Build a reproducible token-mixed 1M-scale corpus for Phase 0 PLE experiments.

The builder is intentionally corpus-agnostic and provenance-first:

* Each category (general / chat / wiki / cot / tool) can come from one or more
  local files in common text/JSON/JSONL shapes.
* Records are converted to plain text (or Qwen-style chat text when the source
  contains ``messages`` / ``conversations`` / ``instruction``+``output``).
* Each category is sampled to a requested token budget.
* If a Qwen tokenizer path is supplied, the selected records are tokenized and
  written as ``tokens.npy`` (the format consumed by ``precompute_real_ple_features.py``
  and ``run_phase0.py``).
* A JSON manifest records every path, ratio, selected record count, token count
  and shortfall so the mix can be recreated exactly.

Typical production use:

    python scripts/build_mix.py \\
      --output data/mixes/M1 \\
      --target-tokens 1000000 \\
      --ratios general=50,chat=20,wiki=20,cot=6,tool=4 \\
      --tokenizer data/models/Qwen3.5-0.8B \\
      --general data/wet-1m-one.txt \\
      --chat data/chat.jsonl \\
      --wiki data/wiki.jsonl \\
      --cot data/cot.jsonl \\
      --tool data/tool.jsonl \\
      --seed 0

For pipeline smoke testing without external datasets, use ``--demo-synthetic``;
it builds small chat/CoT/tool examples by reusing the general text corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

CATEGORIES = ("general", "chat", "wiki", "cot", "tool")
DEFAULT_RATIOS = {
    "general": 50,
    "chat": 20,
    "wiki": 20,
    "cot": 6,
    "tool": 4,
}
DEFAULT_CHUNK_TOKENS = 512
DEFAULT_CHUNK_CHARS = 4000
EOS_SEPARATOR = True


def _log(msg: str) -> None:
    print(f"[build-mix] {msg}", flush=True)


def _parse_ratios(raw: str | None) -> dict[str, float]:
    ratios = dict(DEFAULT_RATIOS)
    if not raw:
        return ratios
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--ratios expects key=value pairs, got: {part!r}")
        key, val = part.split("=", 1)
        key = key.strip()
        if key not in CATEGORIES:
            raise SystemExit(f"unknown ratio category: {key!r} (valid: {CATEGORIES})")
        ratios[key] = float(val.strip())
    total = sum(ratios.values())
    if total <= 0:
        raise SystemExit("sum of ratios must be positive")
    return ratios


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


def _extract_text(record: Any) -> str | None:
    """Extract a single training text from a loose record schema."""
    if isinstance(record, str):
        return record.strip()
    if not isinstance(record, dict):
        return None

    if isinstance(record.get("text"), str) and record["text"].strip():
        return record["text"].strip()

    instr = (
        record.get("instruction")
        or record.get("query")
        or record.get("prompt")
        or record.get("problem")
        or record.get("question")
    )
    out = (
        record.get("output")
        or record.get("response")
        or record.get("answer")
        or record.get("solution")
    )
    thinking = record.get("thinking") or record.get("reasoning_content")
    if isinstance(instr, str) and isinstance(thinking, str) and thinking.strip():
        sol = out.strip() if isinstance(out, str) else ""
        return (
            "<|im_start|>user\n"
            + instr.strip()
            + "\n<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n"
            + thinking.strip()
            + "\n</think>\n\n"
            + sol
            + "\n<|im_end|>\n"
        )
    if isinstance(instr, str) and isinstance(out, str) and (instr.strip() or out.strip()):
        return (
            "<|im_start|>user\n"
            + instr.strip()
            + "\n<|im_end|>\n"
            + "<|im_start|>assistant\n"
            + out.strip()
            + "\n<|im_end|>\n"
        )

    messages = record.get("messages") or record.get("conversations")
    if isinstance(messages, list) and messages:
        text = _format_messages(messages)
        if text:
            return text

    for value in record.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _format_messages(messages: list[Any]) -> str:
    """Format a lightweight messages list using Qwen chat markers."""
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("from") or "user"
        content = msg.get("content") or msg.get("value") or ""
        if isinstance(content, list):
            content = " ".join(
                str(x.get("text", "")) if isinstance(x, dict) else str(x)
                for x in content
            )
        content = str(content).strip()
        if not content and not msg.get("tool_calls"):
            continue
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}\n<|im_end|>\n")
        elif role in ("tool", "function"):
            parts.append(
                f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response>\n<|im_end|>\n"
            )
        elif role == "assistant":
            text = f"<|im_start|>assistant\n{content}"
            tokens = msg.get("tool_calls")
            if tokens:
                for tc in tokens:
                    fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                    name = fn.get("name") or "unknown"
                    args = fn.get("arguments") or {}
                    if isinstance(args, dict):
                        arg_text = "".join(
                            f"<parameter={k}>\n{v}\n</parameter>\n"
                            for k, v in args.items()
                        )
                    else:
                        arg_text = str(args)
                    text += (
                        f"\n<tool_call>\n<function={name}>\n"
                        f"{arg_text}</function>\n</tool_call>"
                    )
            text += "\n<|im_end|>\n"
            parts.append(text)
        else:
            parts.append(f"<|im_start|>user\n{content}\n<|im_end|>\n")
    return "".join(parts)


def _split_long_text(text: str, chunk_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    """Split an over-long single-line text into bounded chunks.

    This is primarily for raw web dumps that arrive as one giant line; splitting
    before contamination filtering prevents one hit from removing the whole
    source.
    """
    text = text.strip()
    if len(text) <= chunk_chars:
        return [text] if text else []
    # Split on sentence punctuation, keeping chunks up to chunk_chars.
    pieces = re.split(r"(?<=[。！？!?.;])\s*", text)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if len(current) + len(piece) <= chunk_chars:
            current += piece
        else:
            if current:
                chunks.append(current.strip())
            # If a single piece is already too long, hard-split it.
            while len(piece) > chunk_chars:
                chunks.append(piece[:chunk_chars].strip())
                piece = piece[chunk_chars:]
            current = piece
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def _load_category(
    paths: list[Path],
    category: str,
    tokenizer: Any,
    max_records: int | None = None,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
) -> tuple[list[str], list[int]]:
    """Load records and return (texts, token_lengths)."""
    texts: list[str] = []
    lengths: list[int] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"{category} source not found: {path}")
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            data = _read_json(path)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and isinstance(data.get("data"), list):
                records = data["data"]
            else:
                records = [data]
        elif suffix == ".parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            records = table.to_pylist()
        else:
            lines = [
                line.strip()
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip()
            ]
            records = []
            for line in lines:
                records.extend(_split_long_text(line, chunk_chars))

        for record in records:
            text = _extract_text(record)
            if not text:
                continue
            if max_records is not None and len(texts) >= max_records:
                break
            if tokenizer is not None:
                try:
                    n = len(tokenizer.encode(text, add_special_tokens=False))
                except Exception as exc:  # noqa: BLE001
                    _log(f"warning: tokenize failed for {category}: {exc}; using approx")
                    n = max(1, len(text) // 4)
            else:
                n = max(1, len(text.split()))
            texts.append(text)
            lengths.append(n)
    return texts, lengths


def _select_for_budget(
    texts: list[str],
    lengths: list[int],
    budget: int,
    rng: random.Random,
    chunk_tokens: int,
    tokenizer: Any,
) -> tuple[list[str], list[int], int]:
    """Deterministically sample full records up to a token budget."""
    order = list(range(len(texts)))
    rng.shuffle(order)
    selected_texts: list[str] = []
    selected_lengths: list[int] = []
    total = 0
    for i in order:
        if total >= budget:
            break
        n = lengths[i]
        if n > chunk_tokens:
            if tokenizer is None:
                # Without a tokenizer we cannot split at token boundaries; keep
                # the whole record and let the approximate count drive the mix.
                selected_texts.append(texts[i])
                selected_lengths.append(n)
                total += n + (1 if EOS_SEPARATOR else 0)
            else:
                ids = tokenizer.encode(texts[i], add_special_tokens=False)
                for start in range(0, len(ids), chunk_tokens):
                    if total >= budget:
                        break
                    shard_ids = ids[start : start + chunk_tokens]
                    shard_text = tokenizer.decode(shard_ids, skip_special_tokens=False)
                    shard_len = len(shard_ids)
                    selected_texts.append(shard_text)
                    selected_lengths.append(shard_len)
                    total += shard_len + (1 if EOS_SEPARATOR else 0)
            continue
        selected_texts.append(texts[i])
        selected_lengths.append(n)
        total += n + (1 if EOS_SEPARATOR else 0)
    return selected_texts, selected_lengths, total


def _demo_synthetic(
    general_texts: list[str], rng: random.Random, count_per_type: int = 80
) -> dict[str, list[str]]:
    """Create tiny chat/CoT/tool examples from the general corpus for smoke runs."""
    samples = general_texts[:]
    rng.shuffle(samples)
    chat: list[str] = []
    cot: list[str] = []
    tool: list[str] = []
    for text in samples[:count_per_type]:
        snippet = text[:200].strip()
        chat.append(
            "<|im_start|>user\nPlease explain briefly: "
            + snippet
            + "\n<|im_end|>\n<|im_start|>assistant\n"
            + snippet
            + "\n<|im_end|>\n"
        )
        cot.append(
            "<|im_start|>user\n"
            + snippet
            + "\n<|im_end|>\n<|im_start|>assistant\n<think>\n"
            + "Let me reason step by step.\n"
            + snippet
            + "\n</think>\nThe answer is based on the above.\n<|im_end|>\n"
        )
        tool.append(
            "<|im_start|>user\nUse a tool to look up: "
            + snippet
            + "\n<|im_end|>\n<|im_start|>assistant\n<tool_call>\n<function=lookup>\n"
            + f"<parameter=query>\n{snippet}\n</parameter>\n</function>\n</tool_call>\n"
            + "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
            + snippet
            + "\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n"
            + snippet
            + "\n<|im_end|>\n"
        )
    return {"chat": chat, "cot": cot, "tool": tool}


def _contamination_norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def _load_qa_needles(qa_path: str | Path) -> set[str]:
    """Build normalized QA needles for record-level contamination filtering."""
    data = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("--exclude-qa requires a JSON list of {question, answer}")
    needles: set[str] = set()
    for item in data:
        q = _contamination_norm(str(item.get("question", "")))
        a = _contamination_norm(str(item.get("answer", "")))
        qa = _contamination_norm(
            str(item.get("question", "")) + " " + str(item.get("answer", ""))
        )
        # Phrase answers remove the strongest direct-memorization signal while
        # tolerating unavoidable one-word common words.  Full question and
        # question+answer pairs are always filtered.
        if len(a.split()) >= 2:
            needles.add(a)
        if len(q.split()) >= 6:
            needles.add(q)
        if len(qa.split()) >= 4:
            needles.add(qa)
    return needles


def _filter_contaminated(
    texts: list[str], needles: set[str]
) -> tuple[list[str], int]:
    kept: list[str] = []
    removed = 0
    for text in texts:
        norm = _contamination_norm(text)
        if needles and any(needle in norm for needle in needles):
            removed += 1
        else:
            kept.append(text)
    return kept, removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reproducible mixed token corpus")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--target-tokens", type=int, default=1_000_000)
    parser.add_argument("--ratios", default=None, help="e.g. general=50,chat=20,wiki=20,cot=6,tool=4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default=None, help="Qwen tokenizer/model dir; enables tokens.npy output")
    parser.add_argument("--general", action="append", default=[], help="general text/jsonl source (repeatable)")
    parser.add_argument("--chat", action="append", default=[], help="chat/instruction source (repeatable)")
    parser.add_argument("--wiki", action="append", default=[], help="wiki/encyclopedia source (repeatable)")
    parser.add_argument("--cot", action="append", default=[], help="chain-of-thought source (repeatable)")
    parser.add_argument("--tool", action="append", default=[], help="tool/agent source (repeatable)")
    parser.add_argument("--max-records-per-category", type=int, default=None)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--demo-synthetic", action="store_true", help="synthesize missing task-format sources from --general")
    parser.add_argument("--no-corpus-txt", action="store_true", help="do not write corpus.txt")
    parser.add_argument("--allow-shortfall", action="store_true", help="continue even if desired ratio cannot be met")
    parser.add_argument("--exclude-qa", default=None, help="JSON QA file; filter records containing QA answers/questions")
    args = parser.parse_args()

    ratios = _parse_ratios(args.ratios)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
            _log(f"tokenizer loaded from {args.tokenizer}")
        except Exception as exc:  # noqa: BLE001
            _log(f"warning: cannot load tokenizer {args.tokenizer}: {exc}")
            tokenizer = None

    sources: dict[str, list[Path]] = {}
    for cat in CATEGORIES:
        paths = [Path(p) for p in getattr(args, cat)]
        if not paths and cat == "general":
            default = Path("data/wet-1m-one.txt")
            if default.exists():
                paths = [default]
        sources[cat] = paths

    general_texts: list[str] = []
    if sources["general"]:
        general_texts, _lengths = _load_category(
            sources["general"],
            "general",
            tokenizer,
            args.max_records_per_category,
            chunk_chars=args.chunk_chars,
        )
        _log(f"loaded general: {len(general_texts)} records")

    if args.demo_synthetic:
        demo = _demo_synthetic(general_texts, random.Random(args.seed ^ 0x9E3779B9))
        for cat in ("chat", "cot", "tool"):
            if not sources[cat]:
                extra_dir = out / "synthetic"
                extra_dir.mkdir(parents=True, exist_ok=True)
                p = extra_dir / f"{cat}.txt"
                p.write_text("\n\n".join(demo[cat]) + "\n", encoding="utf-8")
                sources[cat] = [p]
                _log(f"created synthetic {cat}: {len(demo[cat])} records -> {p}")

    needles: set[str] = set()
    if args.exclude_qa:
        needles = _load_qa_needles(args.exclude_qa)
        _log(f"loaded {len(needles)} contamination needles from {args.exclude_qa}")

    all_texts: dict[str, list[str]] = {}
    all_lengths: dict[str, list[int]] = {}
    excluded_counts: dict[str, int] = {}
    for cat in CATEGORIES:
        if not sources[cat]:
            all_texts[cat] = []
            all_lengths[cat] = []
            excluded_counts[cat] = 0
            continue
        texts, lengths = _load_category(
            sources[cat],
            cat,
            tokenizer,
            args.max_records_per_category,
            chunk_chars=args.chunk_chars,
        )
        if needles:
            texts, removed = _filter_contaminated(texts, needles)
            excluded_counts[cat] = removed
            _log(
                f"{cat}: contamination-filter removed {removed} / "
                f"{len(texts) + removed} records, kept {len(texts)}"
            )
        else:
            excluded_counts[cat] = 0
        _log(f"loaded {cat}: {len(texts)} records")
        all_texts[cat] = texts
        all_lengths[cat] = lengths

    rng = random.Random(args.seed)
    selected: dict[str, list[str]] = {}
    selected_tokens: dict[str, int] = {}
    selected_records: dict[str, int] = {}
    shortfalls: dict[str, int] = {}
    total_selected = 0

    for cat in CATEGORIES:
        budget = int(round(args.target_tokens * ratios[cat] / sum(ratios.values())))
        texts, lengths = all_texts[cat], all_lengths[cat]
        if not texts:
            _log(f"{cat}: no source, budget={budget}, selected=0")
            selected[cat] = []
            selected_tokens[cat] = 0
            selected_records[cat] = 0
            shortfalls[cat] = budget
            continue
        sel_texts, sel_lengths, used = _select_for_budget(
            texts,
            lengths,
            budget,
            rng,
            args.chunk_tokens,
            tokenizer,
        )
        selected[cat] = sel_texts
        selected_tokens[cat] = used
        selected_records[cat] = len(sel_texts)
        shortfalls[cat] = max(0, budget - used)
        total_selected += used
        _log(
            f"{cat}: budget={budget}, selected_records={len(sel_texts)}, "
            f"selected_tokens={used}, shortfall={shortfalls[cat]}"
        )
        if shortfalls[cat] > 0 and not args.allow_shortfall:
            print(
                f"[build-mix] WARNING: {cat} short by {shortfalls[cat]} tokens. "
                "Use --allow-shortfall to continue with a reduced mix.",
                file=sys.stderr,
            )

    required = args.target_tokens
    actual = total_selected
    if actual < required and not args.allow_shortfall:
        raise SystemExit(
            f"shortfall too large: selected {actual} < target {required}; "
            "add more source data or use --allow-shortfall"
        )

    corpus_lines: list[str] = []
    token_ids: list[int] = []
    for cat in CATEGORIES:
        for text in selected[cat]:
            corpus_lines.append(text)
            if tokenizer is not None:
                ids = tokenizer.encode(text, add_special_tokens=False)
                token_ids.extend(ids)
                if EOS_SEPARATOR and tokenizer.eos_token_id is not None:
                    token_ids.append(tokenizer.eos_token_id)
    corpus_text = "\n".join(line.replace("\n", " ") for line in corpus_lines if line.strip()) + "\n"

    if not args.no_corpus_txt:
        corpus_path = out / "corpus.txt"
        corpus_path.write_text(corpus_text, encoding="utf-8")
        _log(f"wrote corpus.txt: {corpus_path} ({len(corpus_lines)} records)")

    if tokenizer is not None:
        import numpy as np

        tokens = np.asarray(token_ids, dtype=np.int64)
        tokens_path = out / "tokens.npy"
        np.save(tokens_path, tokens)
        _log(f"wrote tokens.npy: {tokens_path} ({len(tokens)} tokens)")
    else:
        tokens_path = None

    manifest = {
        "schema": "qwen35-ple-mix-v1",
        "name": out.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "target_tokens": args.target_tokens,
        "selected_tokens": actual,
        "seed": args.seed,
        "ratios": ratios,
        "chunk_tokens": args.chunk_tokens,
        "chunk_chars": args.chunk_chars,
        "tokenizer": args.tokenizer,
        "exclude_qa": args.exclude_qa,
        "excluded_records": excluded_counts,
        "sources": {cat: [str(p) for p in sources[cat]] for cat in CATEGORIES},
        "per_category": {
            cat: {
                "records": selected_records.get(cat, 0),
                "tokens": selected_tokens.get(cat, 0),
                "shortfall": shortfalls.get(cat, 0),
            }
            for cat in CATEGORIES
        },
        "outputs": {
            "corpus.txt": str((out / "corpus.txt").resolve()) if not args.no_corpus_txt else None,
            "tokens.npy": str(tokens_path.resolve()) if tokens_path else None,
        },
    }
    if tokens_path is not None and tokens_path.exists():
        h = hashlib.sha256()
        with tokens_path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        manifest["tokens_sha256"] = h.hexdigest()
    if not args.no_corpus_txt:
        h = hashlib.sha256()
        with (out / "corpus.txt").open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        manifest["corpus_sha256"] = h.hexdigest()

    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _log(f"wrote manifest: {manifest_path}")
    _log(f"done: selected_tokens={actual} (target={required})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
