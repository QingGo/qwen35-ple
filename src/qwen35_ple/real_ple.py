"""Helpers for reading real PLE FP8 table and assembling e_t features.

These helpers are shared by:
- ``scripts/precompute_real_ple_features.py`` (offline feature dumps)
- ``scripts/run_ple_knowledge_probe.py`` (fast knowledge-signal probe)

The rowid semantics come from EngramDB's official ``rowids_for_seq`` (with a
fallback to ``qwen35_ple.ple_hash.real_spec``), and the row IO comes from
EngramDB's Python bindings (``Store`` / ``PleDiskGather``).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from qwen35_ple.ple_hash import real_spec


def resolve_ple_weight_scale(
    model_dir: str | Path | None = None,
    scale: float | None = None,
) -> float:
    """Resolve the real Qwen PLE FP8 ``weight_scale``.

    Priority:
    1. explicit ``scale`` argument
    2. ``engramdb.load_ple_weight_scale(model_dir)`` when a model dir is given
    3. the known Qwen3.8-Flash-Next fallback used by EngramDB (``0.0002``)
    """
    if scale is not None:
        return float(scale)
    if model_dir is not None:
        try:
            from engramdb import load_ple_weight_scale

            return float(load_ple_weight_scale(str(model_dir)))
        except Exception:
            pass
    return 0.0002


def tokenize_texts(tokenizer_path: str, texts: list[str]) -> np.ndarray:
    """Tokenize a list of text segments, inserting EOS between segments."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True
    )
    ids: list[int] = []
    for i, text in enumerate(texts):
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        if i + 1 < len(texts):
            ids.append(tokenizer.eos_token_id)
    return np.asarray(ids, dtype=np.int64)


def tokenize_texts_with_offsets(
    tokenizer_path: str, texts: list[str]
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Tokenize segments and return (ids, [(start, end), ...])."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True
    )
    ids: list[int] = []
    offsets: list[tuple[int, int]] = []
    for i, text in enumerate(texts):
        start = len(ids)
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        if i + 1 < len(texts):
            ids.append(tokenizer.eos_token_id)
        offsets.append((start, len(ids)))
    return np.asarray(ids, dtype=np.int64), offsets


def rowids_from_tokens(tokens: np.ndarray) -> np.ndarray:
    """Return [T, 16] official PLE rowids for each token.

    Prefers EngramDB's native/PyO3/C-ABI rowid implementation; falls back to
    the local frozen reference.
    """
    try:
        from engramdb import rowids_for_seq

        rows = rowids_for_seq(tokens.tolist())
        return np.asarray(rows, dtype=np.int64)
    except Exception:
        spec = real_spec()
        rows = spec.rowids_for_seq(tokens.tolist())
        return np.asarray(rows, dtype=np.int64)


def fetch_e_t(
    rows_dir: str,
    rowids: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Fetch FP8 rows from EngramDB Store and return [T, 2560] float32 e_t.

    The returned vectors are dequantized with the PLE ``weight_scale`` so they
    match EngramDB's real ``DiskPleNGramEmbedding`` / official PLE path.
    """
    import engramdb
    from engramdb.vllm import PleDiskGather

    spec = real_spec()
    store = engramdb.Store(
        rows_dir,
        shards=spec.shards,
        rows_per_shard=spec.rows_per_shard,
        width=160,
    )

    try:
        gather = PleDiskGather(store, row_bytes=160)
        flat = rowids.reshape(-1)
        raw = gather.fetch(flat.tolist())
        arr = torch.frombuffer(bytearray(raw), dtype=torch.float8_e4m3fn)
        fp8 = arr.float().numpy()
        return (fp8 * scale).reshape(len(rowids), 16, 160).reshape(len(rowids), 2560)
    finally:
        store.close()


def precompute_e_t(
    rows_dir: str,
    tokenizer_path: str,
    texts: list[str],
    scale: float = 1.0,
    model_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """One-call precompute: tokenize, hash, fetch and assemble e_t.

    If ``scale`` is not given, tries to read it from ``model_dir`` via
    EngramDB's discovery helpers.
    """
    scale = resolve_ple_weight_scale(model_dir=model_dir, scale=scale)
    tokens = tokenize_texts(tokenizer_path, texts)
    rowids = rowids_from_tokens(tokens)
    t0 = time.time()
    e_t = fetch_e_t(rows_dir, rowids, scale=scale)
    elapsed = time.time() - t0
    return tokens, rowids, e_t, {
        "num_tokens": len(tokens),
        "num_segments": len(texts),
        "fetch_seconds": float(elapsed),
        "e_t_shape": list(e_t.shape),
        "weight_scale": scale,
    }


def save_precomputed(
    output_dir: str | Path,
    tokens: np.ndarray,
    rowids: np.ndarray,
    e_t: np.ndarray,
    meta: dict,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "tokens.npy", tokens)
    np.save(out / "keys.npy", rowids)
    np.save(out / "e_t.npy", e_t)
    import json

    (out / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
