"""Bundle manifest helpers for qwen35-ple serving/deployment.

A bundle is the unified deployment artifact:

* backbone/local model path,
* PLE memory description (Store-I or Store-P + slot index),
* PLE rowid/scale metadata,
* one or more target-reader entries (name/version/path/options).

The on-disk schema follows EngramDB ``BundleManifest`` (``engramdb-bundle-v1``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BUNDLE_SCHEMA = "engramdb-bundle-v1"


def make_bundle(
    *,
    backbone_path: str | Path,
    memory: dict[str, Any],
    ple: dict[str, Any],
    readers: list[dict[str, Any]],
    bundle_id: str | None = None,
) -> dict[str, Any]:
    """Create a bundle manifest dictionary compatible with EngramDB v1."""
    return {
        "schema": BUNDLE_SCHEMA,
        "id": bundle_id or f"qwen35-ple-{Path(backbone_path).name}",
        "backbone": {"path": str(backbone_path)},
        "memory": memory,
        "ple": ple,
        "readers": readers,
    }


def save_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    """Write a bundle manifest as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_bundle(path: str | Path) -> Any:
    """Load an EngramDB ``BundleManifest`` (requires engramdb-python)."""
    try:
        from engramdb import BundleManifest
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "loading a bundle requires engramdb-python>=0.2.12"
        ) from exc
    return BundleManifest.load(path)


def open_bundle_memory(bundle: Any) -> Any:
    """Open the PLE memory described by a loaded ``BundleManifest``."""
    if hasattr(bundle, "open_memory"):
        return bundle.open_memory()
    raise TypeError("bundle must be an EngramDB BundleManifest")
