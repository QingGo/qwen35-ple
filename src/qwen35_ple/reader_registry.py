"""Target-side reader registry and checkpoint helpers for qwen35-ple.

This module is intentionally a thin layer over EngramDB's canonical
``TargetReaderRegistry`` / ``BundleManifest`` protocol.  qwen35-ple only
registers concrete reader constructors and adds checkpoint save/load helpers;
it does not implement a second registry protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CHECKPOINT_FORMAT = "qwen35-ple-reader-v1"

# Stable reader names used in bundles and checkpoints.
OFFICIAL_SOURCE_QWEN_V1 = "official_source_qwen_v1"
ENGRAM_V1 = "engram_v1"
SIMPLE_V1 = "simple_v1"

DEFAULT_VERSION = "1"

_REGISTRY: Any | None = None


def _load_path_payload(path: str | Path) -> dict[str, Any]:
    """Read a reader path for registry-manifest construction.

    Supports both qwen35-ple checkpoint files (``CHECKPOINT_FORMAT``) and raw
    source-state dicts produced by the official-reader extraction scripts.
    """
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and payload.get("format") == CHECKPOINT_FORMAT:
        return {"state_dict": payload.get("state_dict"), "source_state": None}
    if isinstance(payload, dict):
        return {"state_dict": None, "source_state": payload}
    return {"state_dict": None, "source_state": None}


class _LocalRegistry:
    """Minimal fallback target-reader registry.

    Used only when EngramDB's canonical ``TargetReaderRegistry`` is unavailable
    (for example an older engramdb-python on the WSL machine).  It provides the
    small part of the protocol needed by qwen35-ple save/load and bundle helpers.
    """

    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], Any] = {}
        self._latest: dict[str, str] = {}

    def register(
        self,
        name: str,
        factory: Any | None = None,
        *,
        version: str = DEFAULT_VERSION,
        override: bool = False,
    ) -> Any:
        name = str(name)
        version = str(version)
        key = (name, version)

        def _register(fn: Any) -> Any:
            if key in self._factories and not override:
                raise ValueError(f"reader {name!r} v{version!r} already registered")
            self._factories[key] = fn
            self._latest[name] = version
            return fn

        if factory is not None:
            return _register(factory)
        return _register

    def unregister(self, name: str, version: str | None = None) -> None:
        name = str(name)
        if version is None:
            for key in [k for k in self._factories if k[0] == name]:
                del self._factories[key]
            self._latest.pop(name, None)
            return
        key = (name, str(version))
        self._factories.pop(key, None)
        if self._latest.get(name) == str(version):
            remaining = [k[1] for k in self._factories if k[0] == name]
            if remaining:
                self._latest[name] = remaining[-1]
            else:
                self._latest.pop(name, None)

    def available(self) -> list[dict[str, str]]:
        return [
            {"name": name, "version": version}
            for name, version in sorted(self._latest.items())
        ]

    def has(self, name: str, version: str | None = None) -> bool:
        name = str(name)
        if version is None:
            return name in self._latest
        return (name, str(version)) in self._factories

    def create(
        self,
        name: str,
        *,
        version: str | None = None,
        path: str | Path | None = None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        name = str(name)
        if version is None:
            version = self._latest.get(name)
        if version is None:
            raise KeyError(f"no target reader registered: {name!r}")
        factory = self._factories.get((name, str(version)))
        if factory is None:
            raise KeyError(f"no target reader registered: {name!r} v{version}")
        merged = dict(options or {})
        merged.update(kwargs)
        if path is not None:
            merged.setdefault("path", str(path))
        return factory(**merged)

    def create_from_manifest(self, manifest: Any, *, index: int = 0) -> Any:
        if hasattr(manifest, "readers"):
            readers = list(manifest.readers)
        else:
            readers = list(manifest.get("readers", []))
        if not readers:
            raise LookupError("bundle manifest has no readers")
        entry = readers[index]
        options = dict(entry.get("options") or {})
        if entry.get("path"):
            options.setdefault("path", entry["path"])
        return self.create(
            entry["name"],
            version=entry.get("version", DEFAULT_VERSION),
            **options,
        )


def get_registry() -> Any:
    """Return the cached target-reader registry.

    Prefers EngramDB's canonical ``TargetReaderRegistry``; falls back to a small
    local registry when an older/absent engramdb-python is present.
    """
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    try:
        from engramdb import TargetReaderRegistry as _TargetReaderRegistry
    except ImportError:  # pragma: no cover - depends on environment
        _TargetReaderRegistry = _LocalRegistry  # type: ignore[assignment,misc]

    registry = _TargetReaderRegistry()

    @registry.register(OFFICIAL_SOURCE_QWEN_V1, version=DEFAULT_VERSION)
    def _official_source_qwen(
        d_target: int,
        *,
        d_source: int = 2560,
        d_mem: int = 2560,
        hc: int = 4,
        kernel_size: int = 4,
        dilation: int = 3,
        freeze_source: bool = True,
        zero_init_out: bool = True,
        bridge_mlp: bool = False,
        bridge_hidden: int | None = None,
        out_mlp: bool = False,
        out_hidden: int | None = None,
        eps: float = 1e-6,
        source_state: dict[str, Any] | None = None,
        path: str | Path | None = None,
        **kwargs: Any,
    ) -> Any:
        from qwen35_ple.reader import OfficialSourceQwenReader

        if path is not None:
            loaded = _load_path_payload(path)
            if loaded["state_dict"] is not None:
                reader = OfficialSourceQwenReader(
                    d_target=d_target,
                    d_source=d_source,
                    d_mem=d_mem,
                    hc=hc,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    freeze_source=freeze_source,
                    zero_init_out=zero_init_out,
                    bridge_mlp=bridge_mlp,
                    bridge_hidden=bridge_hidden,
                    out_mlp=out_mlp,
                    out_hidden=out_hidden,
                    eps=eps,
                )
                reader.load_state_dict(loaded["state_dict"])
                return reader
            if loaded["source_state"] is not None:
                source_state = loaded["source_state"]

        return OfficialSourceQwenReader(
            d_target=d_target,
            d_source=d_source,
            d_mem=d_mem,
            hc=hc,
            kernel_size=kernel_size,
            dilation=dilation,
            freeze_source=freeze_source,
            zero_init_out=zero_init_out,
            bridge_mlp=bridge_mlp,
            bridge_hidden=bridge_hidden,
            out_mlp=out_mlp,
            out_hidden=out_hidden,
            eps=eps,
            source_state=source_state,
            **kwargs,
        )

    @registry.register(ENGRAM_V1, version=DEFAULT_VERSION)
    def _engram(
        d_model: int,
        *,
        d_mem: int = 2560,
        hc_mult: int = 4,
        kernel_size: int = 4,
        dilation: int = 3,
        zero_init: bool = True,
        path: str | Path | None = None,
        **kwargs: Any,
    ) -> Any:
        from qwen35_ple.reader import QwenEngramReader

        if path is not None:
            loaded = _load_path_payload(path)
            if loaded["state_dict"] is not None:
                reader = QwenEngramReader(
                    d_model=d_model,
                    d_mem=d_mem,
                    hc_mult=hc_mult,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    zero_init=zero_init,
                )
                reader.load_state_dict(loaded["state_dict"])
                return reader

        return QwenEngramReader(
            d_model=d_model,
            d_mem=d_mem,
            hc_mult=hc_mult,
            kernel_size=kernel_size,
            dilation=dilation,
            zero_init=zero_init,
            **kwargs,
        )

    @registry.register(SIMPLE_V1, version=DEFAULT_VERSION)
    def _simple(
        d_model: int,
        *,
        d_mem: int = 2560,
        gate_bias_init: float = -2.0,
        num_branches: int = 1,
        zero_init_v: bool = False,
        path: str | Path | None = None,
        **kwargs: Any,
    ) -> Any:
        from qwen35_ple.reader import EngramReader

        if path is not None:
            loaded = _load_path_payload(path)
            if loaded["state_dict"] is not None:
                reader = EngramReader(
                    d_model=d_model,
                    d_mem=d_mem,
                    gate_bias_init=gate_bias_init,
                    num_branches=num_branches,
                    zero_init_v=zero_init_v,
                )
                reader.load_state_dict(loaded["state_dict"])
                return reader

        return EngramReader(
            d_model=d_model,
            d_mem=d_mem,
            gate_bias_init=gate_bias_init,
            num_branches=num_branches,
            zero_init_v=zero_init_v,
            **kwargs,
        )

    _REGISTRY = registry
    return registry


def build_reader(name: str, **config: Any) -> Any:
    """Construct a reader through the canonical registry."""
    return get_registry().create(name, **config)


def save_reader(
    reader: Any,
    path: str | Path,
    *,
    name: str = OFFICIAL_SOURCE_QWEN_V1,
    version: str = DEFAULT_VERSION,
    config: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    """Save a reader plus its constructor config and full ``state_dict``.

    ``extra_state`` is an optional dict of auxiliary module state dicts, e.g.
    ``{"short_conv": short_conv.state_dict()}`` for the simple reader path.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "reader": name,
        "version": version,
        "config": config or {},
        "state_dict": reader.state_dict(),
        "extra_state": extra_state or {},
    }
    torch.save(payload, path)
    return path


def load_reader_with_extra(
    path: str | Path,
    *,
    device: str | None = None,
    registry: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a reader checkpoint and return ``(reader, extra_state)``."""
    import torch

    path = Path(path)
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported reader checkpoint format: {payload.get('format')!r}"
        )
    name = str(payload.get("reader") or "")
    if not name:
        raise ValueError("reader checkpoint is missing reader name")
    version = str(payload.get("version") or DEFAULT_VERSION)
    config = dict(payload.get("config") or {})
    state_dict = payload.get("state_dict")
    if state_dict is None:
        raise ValueError("reader checkpoint is missing state_dict")

    if registry is None:
        registry = get_registry()
    reader = registry.create(name, version=version, **config)
    reader.load_state_dict(state_dict, strict=True)
    if device:
        reader.to(device)
    return reader, dict(payload.get("extra_state") or {})


def load_reader(
    path: str | Path,
    *,
    device: str | None = None,
    registry: Any | None = None,
) -> Any:
    """Load a reader checkpoint produced by :func:`save_reader`.

    The checkpoint stores the reader name/version and constructor config, so a
    deployment only needs the checkpoint file (plus EngramDB registry support),
    not the original training command.
    """
    reader, _extra = load_reader_with_extra(
        path,
        device=device,
        registry=registry,
    )
    return reader


def reader_config_from_args(
    args: Any,
    d_target: int,
    reader_name: str = OFFICIAL_SOURCE_QWEN_V1,
) -> dict[str, Any]:
    """Build a reader constructor config from the Phase 0 CLI args.

    The returned config intentionally does not include the official source
    checkpoint path: after training, all needed weights live in the saved
    ``state_dict`` and deployment should not depend on the original source file.
    """
    if reader_name == OFFICIAL_SOURCE_QWEN_V1:
        # run_phase0's official path currently uses the OfficialSourceQwenReader
        # defaults for hc/kernel/dilation; keep checkpoint config in sync.
        return {
            "d_target": d_target,
            "d_source": 2560,
            "d_mem": 2560,
            "hc": 4,
            "kernel_size": 4,
            "dilation": 3,
            "freeze_source": True,
            "zero_init_out": True,
            "bridge_mlp": bool(getattr(args, "bridge_mlp", False)),
            "bridge_hidden": getattr(args, "bridge_hidden", None),
            "out_mlp": bool(getattr(args, "out_mlp", False)),
            "out_hidden": getattr(args, "out_hidden", None),
        }
    if reader_name == ENGRAM_V1:
        return {
            "d_model": d_target,
            "d_mem": 2560,
            "hc_mult": int(getattr(args, "hc_mult", 4) or 4),
            "kernel_size": int(getattr(args, "kernel_size", 4) or 4),
            "dilation": int(getattr(args, "dilation", 3) or 3),
            "zero_init": bool(getattr(args, "zero_init_v", False)),
        }
    if reader_name == SIMPLE_V1:
        return {
            "d_model": d_target,
            "d_mem": 2560,
            "gate_bias_init": -2.0,
            "num_branches": int(getattr(args, "branches", 1) or 1),
            "zero_init_v": bool(getattr(args, "zero_init_v", False)),
        }
    raise ValueError(f"unknown reader name: {reader_name}")
