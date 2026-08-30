#!/usr/bin/env python3
"""Generate src/qwen35_ple/official_ple_snapshot.py from the frozen upstream ref.

The upstream file is a full transformers-generated modeling module.  This
script extracts only the PLE-related definitions by AST and writes a standalone
torch-only reference snapshot.  It also supports --check to verify that the
committed snapshot is up to date.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "refs" / "qwen4_exp_modeling.py"
MANIFEST = ROOT / "refs" / "qwen4_exp_modeling.manifest.json"
SNAPSHOT = ROOT / "src" / "qwen35_ple" / "official_ple_snapshot.py"

# Names extracted from the upstream modeling module.
SELECTED = {
    # constants
    "_MASK64",
    "_SPLITMIX_GAMMA",
    "_SPLITMIX_M1",
    "_SPLITMIX_M2",
    "_PRIME_1",
    # functions
    "_splitmix64",
    "_build_layer_multipliers",
    "_is_prime",
    "_find_nth_prime_after",
    "apply_mask_to_padding_states",
    # PLE classes
    "Qwen4ExpTextRMSNorm",
    "Qwen4ExpTextNGramEmbedding",
    "Qwen4ExpTextPLELayer",
}

HEADER = '''# Auto-generated from refs/qwen4_exp_modeling.py by
# scripts/generate_official_ple_snapshot.py -- DO NOT EDIT.
#
# Frozen, torch-only extraction of the upstream Qwen4-Exp PLE layer.  The
# upstream file is Apache-2.0 and is kept in refs/ only for reference.
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# Placeholders only for the generated reference module.  The full upstream
# module belongs to transformers; these symbols are not used at runtime in the
# frozen PLE-only snapshot because annotations are strings under PEP 563.
Qwen4ExpTextConfig = Any
Cache = Any

if not hasattr(nn, "Buffer"):
    class Buffer(torch.nn.Parameter):
        """Compatibility shim for torch<2.4; used by the reference snapshot."""

        def __new__(cls, data, requires_grad=False):
            return super().__new__(cls, data.detach().clone(), requires_grad=False)

    nn.Buffer = Buffer

'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_source() -> str:
    src = REFS.read_text(encoding="utf-8")
    tree = ast.parse(src)
    parts: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names: list[str] = []
            if isinstance(node, ast.Assign):
                names = [
                    t.id for t in node.targets if isinstance(t, ast.Name)
                ]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            if any(name in SELECTED for name in names):
                segment = ast.get_source_segment(src, node)
                if segment:
                    parts.append(segment)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in SELECTED:
            segment = ast.get_source_segment(src, node)
            if segment:
                parts.append(segment)

    if not parts:
        raise RuntimeError("no selected PLE nodes found in upstream reference")
    return HEADER + "\n\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify committed snapshot is current")
    args = parser.parse_args()

    if not REFS.exists():
        print(f"missing upstream reference: {REFS}", file=sys.stderr)
        return 2
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        expected = manifest["sha256"]
        actual = sha256(REFS)
        if actual != expected:
            print(
                f"refs sha256 mismatch: expected {expected}, got {actual}\n"
                "update refs/README.md and refs/qwen4_exp_modeling.manifest.json",
                file=sys.stderr,
            )
            return 3

    generated = build_source()
    if args.check:
        if not SNAPSHOT.exists():
            print(f"snapshot missing: {SNAPSHOT}", file=sys.stderr)
            return 4
        current = SNAPSHOT.read_text(encoding="utf-8")
        if current != generated:
            print("official_ple_snapshot.py is stale; run without --check to regenerate", file=sys.stderr)
            return 5
        print("official PLE snapshot is up to date")
        return 0

    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(generated, encoding="utf-8")
    print(f"wrote {SNAPSHOT} ({len(generated)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
