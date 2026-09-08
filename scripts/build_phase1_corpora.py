#!/usr/bin/env python3
"""Build the Phase 1 pure/matched 1M-token corpus matrix.

The Phase 1 matrix is deliberately domain-first rather than a general chat mix:

    PURE_WIKI   : WikiText
    PURE_FINEWEB: local FineWeb text/stream
    PURE_STEM   : local STEM/mathematical QA + CoT
    PURE_CODE   : local Python code
    FW_CODE     : FineWeb 70% + Code 30%
    FW_STEM     : FineWeb 60% + STEM 40%

Pure corpora are built with ``scripts/build_mix.py`` (or directly from an
existing pre-tokenized FineWeb stream).  Mixed corpora are constructed at the
token level from the pure token streams, which avoids re-tokenizing the large
FineWeb file for every condition.

This script only writes under ``data/phase1`` (git-ignored) by default.

Usage::

    python scripts/build_phase1_corpora.py \
      --tokenizer data/models/Qwen3.5-0.8B \
      --fineweb-tokens "/Volumes/My Passport/engramdb-data/p2-work/tokens/fineweb/fineweb.txt.u32.npy" \
      --output-root data/phase1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REF_FINEWEB_TOKENS = (
    "/Volumes/My Passport/engramdb-data/p2-work/tokens/fineweb/fineweb.txt.u32.npy"
)


def _log(msg: str) -> None:
    print(f"[phase1-corpora] {msg}", flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _run_build_mix(
    output: Path,
    ratios: str,
    *,
    tokenizer: str | None,
    extra_args: list[str],
    qa_file: str | None,
    target_tokens: int,
    seed: int,
    allow_shortfall: bool = True,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_mix.py"),
        "--output",
        str(output),
        "--target-tokens",
        str(target_tokens),
        "--ratios",
        ratios,
        "--seed",
        str(seed),
        "--allow-shortfall" if allow_shortfall else "--no-allow-shortfall",
    ]
    if tokenizer:
        cmd += ["--tokenizer", tokenizer]
    if qa_file:
        cmd += ["--exclude-qa", qa_file]
    cmd += extra_args
    _log("run: " + " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    tokens_path = output / "tokens.npy"
    if not tokens_path.exists():
        raise SystemExit(
            f"{output.name}: build_mix did not produce tokens.npy; "
            "check that the tokenizer can be loaded in this environment"
        )
    _log(f"built {output.name} in {time.time() - t0:.1f}s")


def _write_direct_tokens(
    output: Path,
    tokens_path: Path,
    target_tokens: int,
    source_label: str,
) -> None:
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    arr = np.load(tokens_path, mmap_mode="r")
    if len(arr) < target_tokens:
        raise SystemExit(
            f"{tokens_path} has {len(arr)} tokens, need {target_tokens}"
        )
    selected = np.asarray(arr[:target_tokens], dtype=np.int64).copy()
    dest = output / "tokens.npy"
    np.save(dest, selected)
    manifest = {
        "schema": "qwen35-ple-phase1-v1",
        "name": output.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source_label,
        "source_tokens_path": str(tokens_path),
        "source_total_tokens": len(arr),
        "target_tokens": target_tokens,
        "selected_tokens": len(selected),
        "selection": "first N tokens",
        "tokenizer": "pre-tokenized source",
        "tokens_sha256": _sha256(dest),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _log(f"wrote direct tokens -> {dest} ({len(selected)} tokens)")


def _mix_token_streams(
    output: Path,
    parts: list[tuple[str, int]],
    *,
    target_tokens: int,
) -> None:
    """Concatenate token slices from already-built pure corpora."""
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    arrays: list[np.ndarray] = []
    source_parts: list[dict[str, object]] = []
    for name, count in parts:
        token_path = output.parent / name / "tokens.npy"
        if not token_path.exists():
            raise SystemExit(f"missing pure corpus tokens: {token_path}")
        arr = np.load(token_path, mmap_mode="r")
        if len(arr) < count:
            raise SystemExit(
                f"{name} has {len(arr)} tokens, need {count} for mix"
            )
        arrays.append(np.asarray(arr[:count], dtype=np.int64))
        source_parts.append(
            {
                "source": name,
                "requested_tokens": count,
                "actual_tokens": count,
                "path": str(token_path),
            }
        )
        _log(f"mix part {name}: {count} tokens")

    selected = np.concatenate(arrays).astype(np.int64)
    if len(selected) != target_tokens:
        _log(
            f"WARNING: mixed stream has {len(selected)} tokens "
            f"(target {target_tokens})"
        )
    dest = output / "tokens.npy"
    np.save(dest, selected)
    manifest = {
        "schema": "qwen35-ple-phase1-v1",
        "name": output.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target_tokens": target_tokens,
        "selected_tokens": len(selected),
        "parts": source_parts,
        "tokens_sha256": _sha256(dest),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _log(f"wrote mixed tokens -> {dest} ({len(selected)} tokens)")


def _ensure_code_corpus(code_corpus: Path, code_roots: list[str], max_files: int) -> None:
    if code_corpus.exists():
        _log(f"using existing code corpus: {code_corpus}")
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_code_corpus.py"),
        "--output",
        str(code_corpus),
        "--max-files",
        str(max_files),
    ]
    for root in code_roots:
        cmd += ["--root", root]
    _log("run: " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/phase1")
    parser.add_argument(
        "--tokenizer", default="data/models/Qwen3.5-0.8B"
    )
    parser.add_argument("--target-tokens", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qa-file", default="data/qa-expanded-150.json")
    parser.add_argument("--fineweb-tokens", default=REF_FINEWEB_TOKENS)
    parser.add_argument("--fineweb-token-source-label", default="local FineWeb tokens")
    parser.add_argument(
        "--stem-corpus",
        default="data/sources/distilled_corpus_400k_with_cot-filtered.jsonl",
    )
    parser.add_argument("--code-corpus", default="data/code-corpus-phase1.jsonl")
    parser.add_argument(
        "--code-root",
        action="append",
        default=None,
        help="root to scan for Python files (repeatable; defaults to repo + sibling repos)",
    )
    parser.add_argument("--max-code-files", type=int, default=20000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    root = Path(REPO_ROOT)
    code_roots = list(args.code_root or [])
    if not code_roots:
        code_roots = ["src", "scripts", "tests"]
        for sibling in ["../engram-peft/src", "../EngramDB/python", "../LLM-CompileForge"]:
            if (root / sibling).exists():
                code_roots.append(sibling)
    code_corpus = Path(args.code_corpus)

    # 1. Pure corpora.
    pure_specs = [
        ("PURE_WIKI", "wiki=100", ["--wiki", "data/sources/wikitext.jsonl"]),
        ("PURE_STEM", "stem=100", ["--stem", str(args.stem_corpus)]),
    ]
    for name, ratios, extra in pure_specs:
        dest = out_root / name
        if not args.force and (dest / "tokens.npy").exists():
            _log(f"skip existing {name}")
            continue
        _run_build_mix(
            dest,
            ratios,
            tokenizer=args.tokenizer,
            extra_args=extra,
            qa_file=args.qa_file,
            target_tokens=args.target_tokens,
            seed=args.seed,
            allow_shortfall=True,
        )

    # 2. FineWeb direct from pre-tokenized stream (fast path).
    fw_dest = out_root / "PURE_FINEWEB"
    fw_tokens = Path(args.fineweb_tokens)
    if not args.force and (fw_dest / "tokens.npy").exists():
        _log(f"skip existing {fw_dest.name}")
    else:
        if not fw_tokens.exists():
            raise SystemExit(f"--fineweb-tokens not found: {fw_tokens}")
        _write_direct_tokens(
            fw_dest,
            fw_tokens,
            args.target_tokens,
            args.fineweb_token_source_label,
        )

    # 3. Code pure corpus; needs generated JSONL first.
    _ensure_code_corpus(code_corpus, code_roots, args.max_code_files)
    code_dest = out_root / "PURE_CODE"
    if args.force or not (code_dest / "tokens.npy").exists():
        _run_build_mix(
            code_dest,
            "code=100",
            tokenizer=args.tokenizer,
            extra_args=["--code", str(code_corpus)],
            qa_file=args.qa_file,
            target_tokens=args.target_tokens,
            seed=args.seed,
            allow_shortfall=True,
        )

    # 4. Token-level mixed corpora.
    mix_specs = [
        ("FW_CODE", [("PURE_FINEWEB", int(0.70 * args.target_tokens)), ("PURE_CODE", int(0.30 * args.target_tokens))]),
        ("FW_STEM", [("PURE_FINEWEB", int(0.60 * args.target_tokens)), ("PURE_STEM", int(0.40 * args.target_tokens))]),
    ]
    for name, parts in mix_specs:
        dest = out_root / name
        if not args.force and (dest / "tokens.npy").exists():
            _log(f"skip existing {name}")
            continue
        _mix_token_streams(dest, parts, target_tokens=args.target_tokens)

    _log(f"done: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
