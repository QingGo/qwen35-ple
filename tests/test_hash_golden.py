"""Golden test for the PLE_QWEN_V1 rowid mapping.

The golden vector is produced by EngramDB's keygen implementation and is the
cross-repository contract fixture.  The test intentionally uses the pure-Python
reference in ``qwen35_ple.ple_hash`` so an implementation can be shaken out without
requiring a Rust/PyO3 build; the same fixture must later be used against
``engram-peft``'s production mapping (contract C2.3).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from qwen35_ple.ple_hash import real_spec

ENGRAMDB_ROOT = Path(__file__).resolve().parents[2] / "EngramDB"
GOLDEN_PATH = ENGRAMDB_ROOT / "crates" / "engramdb-keygen" / "tests" / "golden.json"


def _load_golden() -> dict:
    if not GOLDEN_PATH.exists():
        _require_or_skip(f"golden not found: {GOLDEN_PATH}")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_ple_spec_matches_engramdb_golden():
    golden = _load_golden()
    spec = real_spec()

    assert spec.padded == 320_001_536
    assert spec.rows_per_shard * spec.shards == spec.padded
    assert list(spec.multipliers) == golden["multipliers"]
    assert list(spec.prime_sizes) == golden["prime_sizes"]

    tokens: list[int] = golden["tokens"]
    expected: list[list[int]] = golden["rowids"]
    actual = [list(row) for row in spec.rowids_for_seq(tokens)]

    assert len(actual) == len(expected)
    for i, (got, want) in enumerate(zip(actual, expected)):
        assert got == want, f"rowid mismatch at token position {i}"


def test_rowids_are_in_padded_table_space():
    spec = real_spec()
    rows = spec.rowids_for_seq([1000, 99_999, 42])
    assert len(rows) == 3
    for row in rows:
        assert all(0 <= r < spec.padded for r in row)


def _require_or_skip(message: str) -> None:
    """Skip locally, FAIL when the environment says the fixture must be present.

    These golden fixtures are the cross-repository contract: they pin
    ``real_spec()`` to EngramDB's Rust keygen output and to engram-peft's
    production mapping.  They skip when the sibling repos are not checked out,
    which is correct for a laptop and dangerous in CI -- a green run then cannot
    be told apart from a run where the contract was never checked.  Round 161 hit
    exactly that: the experiments ran on a box where all four skipped, and
    "171 passed, 7 skipped" looked healthy.  CI sets QWEN35_REQUIRE_GOLDEN=1 so
    a missing fixture is an error there.
    """
    if os.environ.get("QWEN35_REQUIRE_GOLDEN") == "1":
        raise AssertionError(
            f"{message} -- and QWEN35_REQUIRE_GOLDEN=1, so a missing cross-repo "
            "golden fixture is a failure rather than a skip"
        )
    pytest.skip(message)
