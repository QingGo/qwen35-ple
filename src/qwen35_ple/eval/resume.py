"""Fine-grained Phase 0/2 result resume helpers.

Phase 2 originally wrote one JSON only after a whole corpus finished.  A reboot
in the middle of a corpus could therefore lose up to ~48 minutes of work.  These
helpers write one partial JSON per ``(mode, seed)`` and merge them back into the
final result document.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PARTIAL_RE = re.compile(
    r"^(?P<stem>.+)-(?P<mode>real|control|no-reader)-seed(?P<seed>\d+)\.json$"
)


def partial_result_path(
    partial_dir: str | Path,
    output_path: str | Path,
    mode: str,
    seed: int,
) -> Path:
    """Return the partial JSON path for one ``(mode, seed)`` run."""
    stem = Path(output_path).stem
    return Path(partial_dir) / f"{stem}-{mode}-seed{int(seed)}.json"


def write_partial_result(
    partial_dir: str | Path,
    output_path: str | Path,
    result: dict[str, Any],
) -> Path:
    """Atomically write one partial result."""
    path = partial_result_path(
        partial_dir,
        output_path,
        str(result["mode"]),
        int(result["seed"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def load_partial_results(
    partial_dir: str | Path,
    output_path: str | Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Load all partial results belonging to ``output_path``.

    Corrupt/foreign files are ignored so a single bad temporary file cannot
    block a resume.
    """
    directory = Path(partial_dir)
    if not directory.is_dir():
        return {}
    stem = Path(output_path).stem
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(directory.glob(f"{stem}-*-seed*.json")):
        match = PARTIAL_RE.match(path.name)
        if not match or match.group("stem") != stem:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mode = str(data.get("mode", match.group("mode")))
        try:
            seed = int(data.get("seed", match.group("seed")))
        except (TypeError, ValueError):
            continue
        if mode not in {"real", "control", "no-reader"}:
            continue
        out[(mode, seed)] = data
    return out


def merge_results(
    existing: dict[tuple[str, int], dict[str, Any]],
    new_results: list[dict[str, Any]],
    *,
    modes: list[str] | tuple[str, ...],
    seeds: list[int] | tuple[int, ...],
) -> list[dict[str, Any]]:
    """Merge existing partials and newly computed results in matrix order."""
    merged = dict(existing)
    for result in new_results:
        merged[(str(result["mode"]), int(result["seed"]))] = result
    ordered: list[dict[str, Any]] = []
    for seed in seeds:
        for mode in modes:
            result = merged.get((mode, int(seed)))
            if result is not None:
                ordered.append(result)
    return ordered
