"""Serving adapter bridging EngramDB PLE memory to qwen35 target readers.

This is a thin PyTorch layer that:

* optionally uses an EngramDB ``PleMemoryAdapter`` to fetch incremental ``e_t``
  for each forward call (per sequence history),
* attaches the existing qwen35 reader post-hook at the chosen transformer layer,
* keeps the model API unchanged for callers that pass ``input_ids`` through the
  standard Transformers forward path.

It is intended as the first building block for vLLM/SGLang/CompileForge
integration; engine-specific code can replace the memory adapter or hooks later.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def _extract_input_ids(args: tuple[Any, ...], kwargs: dict[str, Any] | None) -> Any:
    """Extract the input_ids tensor from a model forward call."""
    if kwargs and "input_ids" in kwargs:
        return kwargs["input_ids"]
    if args:
        first = args[0]
        if torch is not None and isinstance(first, torch.Tensor):
            return first
    return None


class QwenReaderServingAdapter:
    """Install and manage e_t injection plus target-reader layer hooks."""

    def __init__(
        self,
        model: Any,
        reader: Any,
        layer_index: int,
        memory_adapter: Any | None = None,
    ) -> None:
        self.model = model
        self.reader = reader
        self.layer_index = int(layer_index)
        self.memory_adapter = memory_adapter
        self._pre_handle: Any = None
        self._post_handle: Any = None

    def _pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None) -> None:
        if self.memory_adapter is None:
            return
        input_ids = _extract_input_ids(args, kwargs)
        if input_ids is None:
            return
        e_t = self.memory_adapter(input_ids)
        if torch is not None and isinstance(e_t, torch.Tensor) and e_t.dim() == 4:
            e_t = e_t.flatten(2)
        self.model._current_ple_e_t = e_t

    def install(self) -> "QwenReaderServingAdapter":
        if self._pre_handle is not None or self._post_handle is not None:
            return self
        self._pre_handle = self.model.register_forward_pre_hook(self._pre_hook)

        from qwen35_ple.reader import install_reader_hook

        self._post_handle = install_reader_hook(
            self.model,
            self.layer_index,
            self.reader,
        )
        return self

    def remove(self) -> None:
        if self._pre_handle is not None:
            self._pre_handle.remove()
            self._pre_handle = None
        if self._post_handle is not None:
            self._post_handle.remove()
            self._post_handle = None
        if hasattr(self.model, "_current_ple_e_t"):
            self.model._current_ple_e_t = None

    def __enter__(self) -> "QwenReaderServingAdapter":
        return self.install()

    def __exit__(self, *exc_info: Any) -> None:
        self.remove()


def install_qwen_reader_adapter(
    model: Any,
    reader: Any,
    layer_index: int,
    memory_adapter: Any | None = None,
) -> QwenReaderServingAdapter:
    """Convenience wrapper for :class:`QwenReaderServingAdapter`."""
    return QwenReaderServingAdapter(
        model,
        reader,
        layer_index,
        memory_adapter=memory_adapter,
    ).install()


def install_qwen_reader_adapter_from_bundle(
    model: Any,
    bundle: Any,
    registry: Any,
    layer_index: int,
    *,
    reader_index: int = 0,
) -> QwenReaderServingAdapter:
    """Install a reader from an EngramDB bundle manifest.

    The bundle supplies PLE storage metadata and the reader entry.  This helper
    creates an EngramDB ``PleMemoryAdapter`` when PyTorch is available.
    """
    from engramdb import PleMemoryAdapter

    memory = bundle.open_memory()
    reader = registry.create_from_manifest(bundle, index=reader_index)
    memory_adapter = PleMemoryAdapter(memory)
    return install_qwen_reader_adapter(
        model,
        reader,
        layer_index,
        memory_adapter=memory_adapter,
    )


def install_vllm_reader_from_bundle(
    model: Any,
    bundle: Any,
    registry: Any,
    layer_index: int,
    **kwargs: Any,
) -> QwenReaderServingAdapter:
    """Alias for installing a qwen35 reader from a bundle in vLLM-style models."""
    return install_qwen_reader_adapter_from_bundle(
        model,
        bundle,
        registry,
        layer_index,
        **kwargs,
    )


def install_sglang_reader_from_bundle(
    model: Any,
    bundle: Any,
    registry: Any,
    layer_index: int,
    **kwargs: Any,
) -> QwenReaderServingAdapter:
    """Alias for installing a qwen35 reader from a bundle in SGLang-style models."""
    return install_qwen_reader_adapter_from_bundle(
        model,
        bundle,
        registry,
        layer_index,
        **kwargs,
    )
