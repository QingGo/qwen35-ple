"""P1 memory-interface prototypes for using the frozen PLE table.

This package contains the three low-cost pieces called out in the systematic
plan:

* exact longest-match n-gram PLE bank;
* TokenMem-style independent cross-attention memory channel;
* distribution-level memory head plus router fusion.

The PyTorch modules are imported lazily so the exact bank (NumPy-only) can be
used in lightweight/no-torch environments.
"""

from __future__ import annotations

from qwen35_ple.memory.bank import ExactNgramBank

_LAZY = {
    "MemoryLogitFusion",
    "MemoryLogitHead",
    "MemoryRouter",
    "P1MemoryModule",
    "PureLogitMemoryModule",
    "TokenMemCrossAttention",
}


def __getattr__(name: str):
    if name in _LAZY:
        from qwen35_ple.memory import token_mem

        return getattr(token_mem, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ExactNgramBank",
    "MemoryLogitFusion",
    "MemoryLogitHead",
    "MemoryRouter",
    "P1MemoryModule",
    "PureLogitMemoryModule",
    "TokenMemCrossAttention",
]
