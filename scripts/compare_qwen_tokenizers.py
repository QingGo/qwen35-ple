#!/usr/bin/env python3
"""Compare two HuggingFace tokenizers at raw-file and runtime levels.

This was written for the Qwen3.5-0.8B vs Qwen3.8-Flash-Next PLE audit:
the PLE rowid mapping hashes raw token ids, so any tokenizer difference can
silently change the external-memory rows.

It reports:

* file-level SHA256 and raw ``tokenizer.json`` / ``tokenizer_config.json`` diffs;
* runtime ``len`` / vocab / special-token ids / added vocab;
* token ids for a fixed set of English, Chinese, code, whitespace, Unicode and
  special-token test strings.

Usage::

    python scripts/compare_qwen_tokenizers.py \
      --tokenizer-a data/models/Qwen3.5-0.8B \
      --tokenizer-b data/models/Qwen3.8-Flash-Next-FP8-tokenizer \
      --output outputs/tokenizer-compare.json \
      --markdown outputs/tokenizer-compare.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_TEXTS = [
    "The capital of France is Paris.",
    "Hello, world! How are you?",
    "你好，世界！今天天气怎么样？",
    "def foo(x):\n    return x + 1  # code",
    "1234567890 3.14159 -42",
    "café ﬁ ＡＢＣ ｈｅｌｌｏ",
    "emoji 😀🚀🔥 and symbols ©®™",
    "multiple   spaces\tand\nnewlines",
    "naïve résumé coöperate",
    "<think>\nreasoning\n</think> answer",
    '<tool_response>{"x": 1}</tool_response>',
    "<|im_start|>user\nhi<|im_end|>",
    "<|endoftext|>",
    "<|audio_start|>abc<|audio_end|>",
]

RAW_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_diffs(a: list[Any], b: list[Any], limit: int = 10) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            out.append({"index": index, "a": left, "b": right})
            if len(out) >= limit:
                break
    return out


def _added_token_diff(
    a_tokens: list[dict[str, Any]], b_tokens: list[dict[str, Any]]
) -> dict[str, Any]:
    a_by_id = {int(item["id"]): item for item in a_tokens}
    b_by_id = {int(item["id"]): item for item in b_tokens}
    common = sorted(set(a_by_id) & set(b_by_id))
    changed = [
        {"id": token_id, "a": a_by_id[token_id], "b": b_by_id[token_id]}
        for token_id in common
        if a_by_id[token_id] != b_by_id[token_id]
    ]
    return {
        "a_count": len(a_tokens),
        "b_count": len(b_tokens),
        "a_only": [a_by_id[token_id] for token_id in sorted(set(a_by_id) - set(b_by_id))],
        "b_only": [b_by_id[token_id] for token_id in sorted(set(b_by_id) - set(a_by_id))],
        "changed": changed,
    }


def _raw_tokenizer_diff(a_path: Path, b_path: Path) -> dict[str, Any]:
    a = _load_json(a_path / "tokenizer.json")
    b = _load_json(b_path / "tokenizer.json")
    a_model = a.get("model", {})
    b_model = b.get("model", {})
    return {
        "top_level_keys": {
            "a": sorted(a.keys()),
            "b": sorted(b.keys()),
        },
        "model_keys": {
            "a": sorted(a_model.keys()),
            "b": sorted(b_model.keys()),
        },
        "model_non_vocab_diff": {
            key: {"a": a_model.get(key), "b": b_model.get(key)}
            for key in sorted(set(a_model) | set(b_model))
            if key not in {"vocab", "merges"} and a_model.get(key) != b_model.get(key)
        },
        "vocab_equal": a_model.get("vocab") == b_model.get("vocab"),
        "merges_equal": a_model.get("merges") == b_model.get("merges"),
        "normalizer_equal": a.get("normalizer") == b.get("normalizer"),
        "pre_tokenizer_equal": a.get("pre_tokenizer") == b.get("pre_tokenizer"),
        "post_processor_equal": a.get("post_processor") == b.get("post_processor"),
        "decoder_equal": a.get("decoder") == b.get("decoder"),
        "added_tokens": _added_token_diff(
            a.get("added_tokens", []), b.get("added_tokens", [])
        ),
    }


def _tokenizer_config_diff(a_path: Path, b_path: Path) -> dict[str, Any]:
    a = _load_json(a_path / "tokenizer_config.json")
    b = _load_json(b_path / "tokenizer_config.json")
    keys = sorted(set(a) | set(b))
    diffs: dict[str, Any] = {}
    for key in keys:
        if a.get(key) == b.get(key):
            continue
        if key in {"chat_template"}:
            diffs[key] = {
                "a_len": len(str(a.get(key, ""))),
                "b_len": len(str(b.get(key, ""))),
            }
        elif key == "added_tokens_decoder":
            diffs[key] = _added_token_diff(
                list(a.get(key, {}).values()), list(b.get(key, {}).values())
            )
        else:
            diffs[key] = {"a": a.get(key), "b": b.get(key)}
    return diffs


def _runtime_summary(tokenizer: Any) -> dict[str, Any]:
    return {
        "class": type(tokenizer).__name__,
        "len": len(tokenizer),
        "vocab_size": int(tokenizer.vocab_size),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
        "all_special_tokens": list(tokenizer.all_special_tokens),
        "all_special_ids": list(tokenizer.all_special_ids),
        "additional_special_tokens": list(
            getattr(tokenizer, "additional_special_tokens", [])
        ),
        "added_vocab_count": len(tokenizer.get_added_vocab()),
        "added_vocab": tokenizer.get_added_vocab(),
    }


def _compare_encoding(tokenizer_a: Any, tokenizer_b: Any, text: str) -> dict[str, Any]:
    ids_a = tokenizer_a.encode(text, add_special_tokens=False)
    ids_b = tokenizer_b.encode(text, add_special_tokens=False)
    ids_a_special = tokenizer_a.encode(text, add_special_tokens=True)
    ids_b_special = tokenizer_b.encode(text, add_special_tokens=True)
    out: dict[str, Any] = {
        "text": text,
        "a_len": len(ids_a),
        "b_len": len(ids_b),
        "equal": ids_a == ids_b,
        "equal_with_special_tokens": ids_a_special == ids_b_special,
    }
    if not out["equal"]:
        first = next(
            (
                index
                for index, (left, right) in enumerate(zip(ids_a, ids_b))
                if left != right
            ),
            min(len(ids_a), len(ids_b)),
        )
        out["first_diff_index"] = first
        out["a_ids_head"] = ids_a[:first + 3]
        out["b_ids_head"] = ids_b[:first + 3]
        out["a_tokens_head"] = tokenizer_a.convert_ids_to_tokens(ids_a[: first + 3])
        out["b_tokens_head"] = tokenizer_b.convert_ids_to_tokens(ids_b[: first + 3])
    return out


def _compare_vocab(tokenizer_a: Any, tokenizer_b: Any) -> dict[str, Any]:
    vocab_a = tokenizer_a.get_vocab()
    vocab_b = tokenizer_b.get_vocab()
    only_a = sorted(set(vocab_a) - set(vocab_b))
    only_b = sorted(set(vocab_b) - set(vocab_a))
    changed = sorted(
        token
        for token in set(vocab_a) & set(vocab_b)
        if vocab_a[token] != vocab_b[token]
    )
    return {
        "equal": vocab_a == vocab_b,
        "a_size": len(vocab_a),
        "b_size": len(vocab_b),
        "a_only_count": len(only_a),
        "b_only_count": len(only_b),
        "changed_count": len(changed),
        "a_only_sample": only_a[:20],
        "b_only_sample": only_b[:20],
        "changed_sample": changed[:20],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen tokenizer comparison",
        "",
        f"- A: `{report['tokenizer_a']}`",
        f"- B: `{report['tokenizer_b']}`",
        "",
        "## Runtime summary",
        "",
        "| Field | A | B |",
        "|---|---|---|",
    ]
    a_runtime = report["runtime"]["a"]
    b_runtime = report["runtime"]["b"]
    for key in (
        "class",
        "len",
        "vocab_size",
        "eos_token",
        "eos_token_id",
        "pad_token",
        "pad_token_id",
        "added_vocab_count",
    ):
        lines.append(f"| {key} | {a_runtime.get(key)} | {b_runtime.get(key)} |")
    lines += [
        "",
        "## Raw files",
        "",
        "| File | A sha256 | B sha256 | Equal |",
        "|---|---|---|---|",
    ]
    for name, entry in report["raw_files"].items():
        lines.append(
            f"| {name} | {str(entry['a_sha256'])[:12]} | "
            f"{str(entry['b_sha256'])[:12]} | {entry['equal']} |"
        )
    raw = report["raw_tokenizer"]
    lines += [
        "",
        "## Raw tokenizer.json",
        "",
        f"- model.vocab equal: `{raw['vocab_equal']}`",
        f"- model.merges equal: `{raw['merges_equal']}`",
        f"- normalizer equal: `{raw['normalizer_equal']}`",
        f"- pre_tokenizer equal: `{raw['pre_tokenizer_equal']}`",
        f"- post_processor equal: `{raw['post_processor_equal']}`",
        f"- decoder equal: `{raw['decoder_equal']}`",
        (
            f"- added_tokens A/B: `{raw['added_tokens']['a_count']}` / "
            f"`{raw['added_tokens']['b_count']}`"
        ),
        "",
        "## Encoding",
        "",
        "| Text | A len | B len | Equal no-special | Equal with special |",
        "|---|---:|---:|---|---|",
    ]
    for entry in report["encoding"]:
        text = str(entry["text"]).replace("|", "\\|").replace("\n", "\\n")
        lines.append(
            f"| `{text[:80]}` | {entry['a_len']} | {entry['b_len']} | "
            f"{entry['equal']} | {entry['equal_with_special_tokens']} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
    ]
    for key, value in report["verdict"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-a", required=True)
    parser.add_argument("--tokenizer-b", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="pass trust_remote_code=True when loading tokenizers",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    a_path = Path(args.tokenizer_a)
    b_path = Path(args.tokenizer_b)
    tokenizer_a = AutoTokenizer.from_pretrained(
        str(a_path),
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer_b = AutoTokenizer.from_pretrained(
        str(b_path),
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )

    report: dict[str, Any] = {
        "tokenizer_a": str(a_path),
        "tokenizer_b": str(b_path),
        "runtime": {
            "a": _runtime_summary(tokenizer_a),
            "b": _runtime_summary(tokenizer_b),
        },
        "vocab": _compare_vocab(tokenizer_a, tokenizer_b),
        "raw_files": {
            name: {
                "a_sha256": _sha256(a_path / name),
                "b_sha256": _sha256(b_path / name),
                "equal": (a_path / name).is_file()
                and (b_path / name).is_file()
                and _sha256(a_path / name) == _sha256(b_path / name),
            }
            for name in RAW_FILES
        },
        "raw_tokenizer": _raw_tokenizer_diff(a_path, b_path),
        "tokenizer_config_diff": _tokenizer_config_diff(a_path, b_path),
        "encoding": [
            _compare_encoding(tokenizer_a, tokenizer_b, text) for text in DEFAULT_TEXTS
        ],
    }
    report["verdict"] = {
        "base_vocab_identical": report["raw_tokenizer"]["vocab_equal"],
        "merges_identical": report["raw_tokenizer"]["merges_equal"],
        "normalizer_identical": report["raw_tokenizer"]["normalizer_equal"],
        "pre_tokenizer_identical": report["raw_tokenizer"]["pre_tokenizer_equal"],
        "post_processor_identical": report["raw_tokenizer"]["post_processor_equal"],
        "decoder_identical": report["raw_tokenizer"]["decoder_equal"],
        "runtime_vocab_identical": report["vocab"]["equal"],
        "runtime_eos_ids_equal": (
            tokenizer_a.eos_token_id == tokenizer_b.eos_token_id
        ),
        "ordinary_text_encoding_identical": all(
            entry["equal"] for entry in report["encoding"]
        ),
        "all_text_encoding_identical_with_special_tokens": all(
            entry["equal_with_special_tokens"] for entry in report["encoding"]
        ),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[tokenizer-compare] wrote {output_path}")

    if args.markdown:
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        print(f"[tokenizer-compare] wrote {markdown_path}")

    print(json.dumps(report["verdict"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
