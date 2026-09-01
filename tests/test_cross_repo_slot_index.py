"""Cross-repo contract test for SlotIndex.

If EngramDB is installed and exposes the canonical ``SlotIndex``, this test
verifies that qwen35-ple's re-exported SlotIndex behaves identically on the
same rowid input.  It skips cleanly in dependency-light environments.
"""

from __future__ import annotations

import numpy as np
import pytest

from qwen35_ple.slot_index import SlotIndex


def test_slot_index_matches_engramdb_canonical() -> None:
    engramdb = pytest.importorskip("engramdb")
    canonical = getattr(engramdb, "SlotIndex", None)
    if canonical is None:
        pytest.skip("engramdb.SlotIndex unavailable (numpy not installed)")

    rowids = np.arange(64, dtype=np.int64).reshape(4, 16)
    slots = np.array([10, 20, 30, 40], dtype=np.int64)

    local = SlotIndex.from_rowids(rowids, slots)
    canonical_idx = canonical.from_rowids(rowids, slots)

    np.testing.assert_array_equal(local.to_slots(rowids), canonical_idx.to_slots(rowids))
    for row in rowids:
        assert local.lookup(tuple(int(x) for x in row)) == canonical_idx.lookup(
            tuple(int(x) for x in row)
        )

    # Duplicate rowids must resolve to the same representative slots.
    dup = np.stack([rowids[0], rowids[2], rowids[0]], axis=0)
    np.testing.assert_array_equal(
        local.to_slots(dup),
        canonical_idx.to_slots(dup),
    )


def test_disk_slot_index_matches_in_memory() -> None:
    import tempfile
    from pathlib import Path

    engramdb = pytest.importorskip("engramdb")
    if getattr(engramdb, "DiskSlotIndex", None) is None:
        pytest.skip("engramdb.DiskSlotIndex unavailable (numpy not installed)")

    rowids = np.arange(64, dtype=np.int64).reshape(4, 16)
    slots = np.array([10, 20, 30, 40], dtype=np.int64)

    mem = engramdb.SlotIndex.from_rowids(rowids, slots)
    with tempfile.TemporaryDirectory(prefix="disk-slot-contract-") as td:
        disk = engramdb.DiskSlotIndex.build(
            rowids,
            td,
            num_buckets=8,
            slots=iter(slots.tolist()),
        )
        try:
            np.testing.assert_array_equal(disk.to_slots(rowids), mem.to_slots(rowids))
            assert disk.lookup(tuple(int(x) for x in rowids[2])) == 30
        finally:
            disk.close()
