"""PLE-2 serving router: calibrated n-gram logit processor.

This module connects the calibrated n-gram fusion to actual generation.  A
``CalibratedNgramLogitProcessor`` is a drop-in logit processor for
``RAGServingAdapter``: it looks up the current n-gram distribution from the
addressable memory and applies the calibrated ``scale / bias / temperature``
transformation to the base model logits.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qwen35_ple.fusion import fuse_ngram_logits


class CalibratedNgramLogitProcessor:
    """Apply a calibrated n-gram log-prior during generation.

    Parameters
    ----------
    memory
        Any object with ``continuation_distribution(context)`` returning either
        ``(dict, order)`` or ``None`` (for example ``AddressableNgramMemory``).
    scale, bias, temperature
        Calibrated fusion hyperparameters.
    """

    def __init__(
        self,
        memory,
        *,
        scale: float = 1.0,
        bias: float = 0.0,
        temperature: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self.memory = memory
        self.scale = float(scale)
        self.bias = float(bias)
        self.temperature = float(temperature)
        self.enabled = bool(enabled)

    def __call__(self, logits: Any, context_ids: Any) -> Any:
        """Return fused logits (same type as input)."""
        if not self.enabled:
            return logits

        # Accept torch tensor or numpy array.
        is_torch = hasattr(logits, "detach") and hasattr(logits, "to")
        if is_torch:
            import torch

            logits_np = logits.detach().cpu().float().numpy()
            context = (
                context_ids.detach().cpu().tolist()
                if hasattr(context_ids, "detach")
                else list(context_ids)
            )
        else:
            logits_np = np.asarray(logits, dtype=np.float32)
            context = list(context_ids)

        result = self.memory.continuation_distribution(context)
        if result is None:
            dist = None
        else:
            dist = result[0]

        fused_np = fuse_ngram_logits(
            logits_np,
            dist,
            scale=self.scale,
            bias=self.bias,
            temperature=self.temperature,
        )

        if is_torch:
            return torch.as_tensor(fused_np, dtype=logits.dtype, device=logits.device)
        return fused_np

    def state_dict(self) -> dict[str, float]:
        return {
            "scale": self.scale,
            "bias": self.bias,
            "temperature": self.temperature,
            "enabled": self.enabled,
        }


__all__ = ["CalibratedNgramLogitProcessor"]
