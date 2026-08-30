"""Data/rowid pipeline helpers for qwen35-ple.

The authoritative rowid implementation is EngramDB/engram-peft.  This package
contains thin orchestration around token streams and the golden reference.
"""

from __future__ import annotations

from qwen35_ple.ple_hash import PleSpec, real_spec

__all__ = ["PleSpec", "real_spec"]
