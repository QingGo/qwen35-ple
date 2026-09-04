"""TokenMem-style memory channel and distribution-level fusion modules.

These are the P1 trainable components.  The backbone is intended to stay frozen;
the modules below are deliberately small so the whole P1 prototype can be
trained on CPU or a single GPU.

Design notes
------------

* :class:`TokenMemCrossAttention` is an *independent* memory channel.  It uses
  its own query/key/value projections and a separate softmax over a small number
  of memory slots.  It does not modify the backbone self-attention weights.

* :class:`MemoryLogitHead` is the distribution-level memory head.  It maps the
  frozen PLE/memory feature to a vocabulary distribution, matching the
  MLP-Memory / MemSFT idea that memory contributes a distribution rather than
  only a hidden residual.

* :class:`MemoryRouter` learns a per-token mixing weight between the backbone
  distribution and the memory distribution.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class TokenMemCrossAttention(torch.nn.Module):
    """Small multi-slot cross-attention over exact-bank memory candidates.

    The forward signature is ``(h, mem)``:

    * ``h``: ``[B, T, d_model]`` backbone hidden state;
    * ``mem``: ``[B, T, K, d_mem]`` per-token memory slots (for example from
      ``ExactNgramBank.lookup_multi``).

    The output is a residual in the backbone hidden space, gated by a learned
    per-token confidence.
    """

    def __init__(
        self,
        d_model: int,
        d_mem: int = 2560,
        *,
        n_heads: int = 4,
        dropout: float = 0.0,
        zero_init_out: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.n_heads = n_heads
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.head_dim = d_model // n_heads

        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj = torch.nn.Linear(d_mem, d_model, bias=False)
        self.v_proj = torch.nn.Linear(d_mem, d_model, bias=False)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = torch.nn.Linear(d_model + d_mem, 1, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

        if zero_init_out:
            torch.nn.init.zeros_(self.out_proj.weight)

    def forward(self, h: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        # h: [B,T,d_model], mem: [B,T,K,d_mem]
        b, t, k, _ = mem.shape
        q = self.q_proj(h)
        # Multi-head reshape: [B,T,H,D] after splitting.
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(mem)  # [B,T,K,d_model]
        key = key.view(b, t, k, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        value = self.v_proj(mem)
        value = value.view(b, t, k, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        scores = torch.matmul(q.unsqueeze(-2), key.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        scores = scores.squeeze(-2)  # [B,H,T,K]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn.unsqueeze(-2), value)  # [B,H,T,1,D]
        out = out.squeeze(-2).transpose(1, 2).contiguous().view(b, t, self.d_model)
        out = self.out_proj(out)

        # Per-token confidence gate from the hidden state and the mean memory.
        mem_mean = mem.mean(dim=2)
        gate = torch.sigmoid(self.gate_proj(torch.cat([h, mem_mean], dim=-1)))
        return gate * out


class MemoryLogitHead(torch.nn.Module):
    """Distribution-level memory head: memory feature -> vocabulary logits.

    The head can be trained on next-token prediction using only frozen memory
    features, before or while the router learns when to trust it.
    """

    def __init__(
        self,
        d_mem: int,
        vocab_size: int,
        *,
        d_model: int = 1024,
        hidden: int | None = None,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        hidden = hidden or d_model
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_mem, hidden, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, vocab_size, bias=False),
        )
        if zero_init:
            last = self.mlp[-1]
            torch.nn.init.zeros_(last.weight)

    def forward(self, mem: torch.Tensor) -> torch.Tensor:
        return self.mlp(mem)


class MemoryRouter(torch.nn.Module):
    """Learns a per-token scalar mixing weight for memory logits.

    The router sees both the backbone hidden state and the memory feature.  The
    output is a sigmoid in ``[0, 1]``.  During P1 the backbone is frozen, so this
    is a very small parameterization.
    """

    def __init__(
        self,
        d_model: int,
        d_mem: int,
        *,
        hidden: int = 64,
        bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model + d_mem, hidden, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 1, bias=False),
        )
        self.bias = torch.nn.Parameter(torch.full((1,), bias_init))

    def forward(
        self, h: torch.Tensor, mem: torch.Tensor
    ) -> torch.Tensor:
        # mem may be [B,T,d_mem] or [B,T,K,d_mem]; reduce slots for router.
        if mem.ndim == 4:
            mem = mem.mean(dim=2)
        x = torch.cat([h, mem], dim=-1)
        return torch.sigmoid(self.net(x) + self.bias)


class MemoryLogitFusion(torch.nn.Module):
    """Fuse backbone logits with memory logits using a learned per-token router.

    ``alpha`` is the weight placed on the memory distribution:

    .. code-block:: python

        final = (1 - alpha) * base_logits + alpha * memory_logits
    """

    def __init__(
        self,
        d_model: int,
        d_mem: int,
        *,
        hidden: int = 64,
        bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.router = MemoryRouter(
            d_model, d_mem, hidden=hidden, bias_init=bias_init
        )

    def forward(
        self,
        h: torch.Tensor,
        mem: torch.Tensor,
        base_logits: torch.Tensor,
        memory_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.router(h, mem)  # [B,T,1]
        fused = (1.0 - alpha) * base_logits + alpha * memory_logits
        return fused, alpha


class P1MemoryModule(torch.nn.Module):
    """End-to-end P1 memory interface: channel + distribution head + router.

    This is the trainable wrapper used by the P1 scripts.  The backbone is
    intended to remain frozen; only this module is optimized.
    """

    def __init__(
        self,
        d_model: int,
        d_mem: int,
        vocab_size: int,
        *,
        n_heads: int = 4,
        head_hidden: int = 256,
        router_hidden: int = 64,
        zero_init_head: bool = True,
    ) -> None:
        super().__init__()
        self.channel = TokenMemCrossAttention(
            d_model=d_model, d_mem=d_mem, n_heads=n_heads
        )
        self.mem_head = MemoryLogitHead(
            d_model,
            vocab_size,
            d_model=d_model,
            hidden=head_hidden,
            zero_init=zero_init_head,
        )
        self.fusion = MemoryLogitFusion(
            d_model=d_model, d_mem=d_mem, hidden=router_hidden
        )

    def forward(
        self,
        h: torch.Tensor,
        mem: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mem_hidden = self.channel(h, mem)
        mem_rep = h + mem_hidden
        memory_logits = self.mem_head(mem_rep)
        fused, alpha = self.fusion(h, mem, base_logits, memory_logits)
        return fused, memory_logits, alpha


__all__ = [
    "MemoryLogitFusion",
    "MemoryLogitHead",
    "MemoryRouter",
    "P1MemoryModule",
    "TokenMemCrossAttention",
]
