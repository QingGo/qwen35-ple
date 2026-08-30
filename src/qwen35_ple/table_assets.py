"""Helpers for qwen35-ple table/view assets.

The production storage operations are owned by EngramDB.  This module only
provides the thin orchestration layer needed by qwen35-ple: locating the
EngramDB CLI, invoking the Store-P view builder, and reading/validating the
manifest that EngramDB writes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENGRAMDB_CANDIDATES = (
    REPO_ROOT.parent / "EngramDB" / "target" / "release" / "engramdb",
    REPO_ROOT.parent / "EngramDB" / "target" / "debug" / "engramdb",
)


@dataclass(frozen=True)
class ViewManifest:
    """Frozen view of an EngramDB Store-P ``.manifest.json`` file."""

    path: Path
    grans: int
    heads: int
    slot_bytes: int
    record_bytes: int
    build_seconds: float | None
    build_mb_s: float | None
    rows: int
    source: str

    @classmethod
    def from_file(cls, manifest_path: str | Path) -> ViewManifest:
        path = Path(manifest_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            grans=int(data["grans"]),
            heads=int(data["heads"]),
            slot_bytes=int(data["slot_bytes"]),
            record_bytes=int(data["record_bytes"]),
            build_seconds=float(data["build_seconds"]),
            build_mb_s=float(data["build_mb_s"]),
            rows=int(data["rows"]),
            source=str(data.get("source", "")),
        )

    @property
    def expected_bytes(self) -> int:
        return self.grans * self.slot_bytes




def manifest_path_for_view(view_path: str | Path) -> Path:
    """Return the manifest path EngramDB writes for a given view file."""
    return Path(view_path).with_suffix(".manifest.json")

def find_engramdb_cli() -> str | None:
    """Return an EngramDB CLI path if one is available."""
    env = os.environ.get("ENGRAMDB_CLI")
    if env:
        path = Path(env).expanduser()
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"ENGRAMDB_CLI does not exist: {path}")
    for candidate in DEFAULT_ENGRAMDB_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("engramdb")
    return found


def build_store_p_view(
    rows_dir: str | Path,
    output_dir: str | Path,
    view_name: str = "qwen-ple.view.bin",
    n_grams: int | None = None,
    keys_path: str | Path | None = None,
    slot_bytes: int = 2560,
    verify: bool = True,
    engramdb_cli: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Build a Store-P view by delegating to ``engramdb view build``.

    If ``keys_path`` is provided, the view is built in that exact access order
    (``build_view_from_keys`` semantics).  Otherwise a random LCG sample of
    ``n_grams`` is used.
    """
    cli = engramdb_cli or find_engramdb_cli()
    if cli is None:
        raise FileNotFoundError(
            "engramdb CLI not found; set ENGRAMDB_CLI or build EngramDB first"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    view_out = out_dir / view_name
    keys_out = out_dir / f"{Path(view_name).stem}.keys.txt"

    cmd = [cli, "view", "build", str(rows_dir)]
    if keys_path is not None:
        # With --keys, EngramDB expects the *input* key file; the CLI also
        # writes a copy to keys_out.
        key_file = Path(keys_path)
        if not key_file.is_file():
            raise FileNotFoundError(f"keys file not found: {key_file}")
        cmd += [str(len(read_key_file(key_file)) // 16), str(view_out), str(keys_out), "--keys", str(key_file)]
    else:
        if n_grams is None:
            raise ValueError("n_grams is required when keys_path is not provided")
        cmd += [str(n_grams), str(view_out), str(keys_out)]
    cmd += ["--slot", str(slot_bytes)]
    if verify:
        cmd.append("--verify")

    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def read_key_file(path: str | Path) -> list[int]:
    """Read a rowids keys file (one u64 per line)."""
    out: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(int(line))
    return out
