#!/usr/bin/env python3
"""Run a child command, then record its wall time and peak RSS as JSON lines.

``/usr/bin/time -v`` is not installed on every box this project runs on, and the
round-162 deliverable asks for per-arm wall time and peak memory, so the
measurement cannot depend on it.  This wrapper forks the child, waits for it,
and reads the child's peak resident set size from
``resource.getrusage(RUSAGE_CHILDREN)`` (Linux reports ``ru_maxrss`` in KiB).

Usage::

    python scripts/with_rusage.py <stats.jsonl> <command> [args...]

The child's exit status is propagated unchanged, and one JSON line is appended
to ``<stats.jsonl>``::

    {"cmd": [...], "rc": 0, "wall_s": 611.4, "peak_rss_kib": 5418236}
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import time


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    stats_path, cmd = argv[0], argv[1:]
    t0 = time.time()
    rc = subprocess.call(cmd)
    wall = time.time() - t0
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = {
        "cmd": cmd,
        "rc": int(rc),
        "wall_s": round(wall, 1),
        "peak_rss_kib": int(usage.ru_maxrss),
    }
    with open(stats_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(
        f"[with-rusage] rc={rc} wall={record['wall_s']}s "
        f"peak_rss={record['peak_rss_kib'] / 1048576:.2f}GiB",
        flush=True,
    )
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
