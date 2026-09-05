"""Serving integration helpers for qwen35-ple.

This package is a thin consumer of EngramDB's optional serving layer
(``BundleManifest`` / ``PleMemory`` / ``TargetReaderRegistry``).  It does not
implement another bundle protocol.
"""

from __future__ import annotations

from qwen35_ple.serving.adapter import (
    QwenReaderServingAdapter,
    install_qwen_reader_adapter,
    install_qwen_reader_adapter_from_bundle,
    install_sglang_reader_from_bundle,
    install_vllm_reader_from_bundle,
)
from qwen35_ple.serving.bundle import (
    load_bundle,
    make_bundle,
    open_bundle_memory,
    save_bundle,
)
from qwen35_ple.serving.rag import RAGServingAdapter

__all__ = [
    "QwenReaderServingAdapter",
    "RAGServingAdapter",
    "install_qwen_reader_adapter",
    "install_qwen_reader_adapter_from_bundle",
    "install_sglang_reader_from_bundle",
    "install_vllm_reader_from_bundle",
    "load_bundle",
    "make_bundle",
    "open_bundle_memory",
    "save_bundle",
]
