"""Reference PLE-lite forward for M1 golden checks.

This module mirrors the Qwen ``Qwen4ExpTextPLELayer`` mathematics for the
``hc=1`` (one residual branch) case.  It is intentionally independent from the
production engram-peft implementation so the two can be diffed numerically in
``tests/test_ple_forward_golden.py``.

The reference consumes the already-initialized hash mapping, embedding table,
gating projections/norms and short-conv module, and returns the full
``hidden_states + PLE output`` tensor produced by an EngramLayer with one
branch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def qwen_ple_forward_reference(
    *,
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    hash_mapping,
    multi_head_embedding: torch.nn.Module,
    value_proj: torch.nn.Module,
    key_proj: torch.nn.Module,
    norm_key: torch.nn.Module,
    norm_query: torch.nn.Module,
    norm_conv: torch.nn.Module,
    conv1d: torch.nn.Module,
    hidden_size: int,
    conv_kernel_size: int,
    conv_dilation: int,
) -> torch.Tensor:
    """Compute the expected PLE-lite output for one hash-mapping layer.

    Args:
        input_ids: ``[B, T]`` original token ids.
        hidden_states: ``[B, T, hidden_size]`` backbone states.
        hash_mapping: an engram-peft ``QwenPleHashMapping`` instance.
        multi_head_embedding: the embedding module (in-memory or disk).
        value_proj / key_proj / norms / conv1d: the layer's own submodules.
        hidden_size: model hidden dimension (also Qwen PLE hidden size).
        conv_kernel_size / conv_dilation: short-conv hyperparameters.

    Returns:
        ``[B, T, hidden_size]`` matching ``EngramLayer.forward`` with
        ``hc_mult=1``.
    """
    batch_size, seq_len = input_ids.shape
    with torch.no_grad():
        # 1. Hash to local per-head indices, then gather from the same
        #    MultiHeadEmbedding used by the production layer.
        hash_np = hash_mapping.hash(input_ids.cpu().numpy())[1]
        hash_indices = torch.from_numpy(np.asarray(hash_np, dtype=np.int64))
        embeddings = multi_head_embedding(hash_indices).flatten(-2)

        # 2. Qwen PLE gating (hc=1).
        value = value_proj(embeddings)  # [B, T, D]
        key = key_proj(embeddings)  # [B, T, D]
        normed_key = norm_key(key)
        normed_query = norm_query(hidden_states)
        gate_score = (normed_key * normed_query).sum(-1, keepdim=True)
        gate = gate_score / (hidden_size**0.5)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = torch.sigmoid(gate)  # [B, T, 1]
        gated_value = (
            gate.unsqueeze(-1) * value.unsqueeze(-2)
        ).squeeze(2)  # [B, T, D]

        # 3. Short conv: RMSNorm -> causal dilated depthwise conv -> SiLU.
        normed = norm_conv(gated_value)
        conv_input = normed.transpose(1, 2)  # [B, D, T]
        pad_len = (conv_kernel_size - 1) * conv_dilation
        conv_input = F.pad(conv_input, (pad_len, 0))
        conv_out = F.conv1d(
            conv_input,
            conv1d.weight,
            groups=hidden_size,
            dilation=conv_dilation,
        )
        short_conv = F.silu(conv_out).transpose(1, 2)  # [B, T, D]

        # 4. Residual: hidden + gated value + short conv.
        return hidden_states + gated_value + short_conv
