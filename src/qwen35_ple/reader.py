"""Target-side PLE reader modules used by experiment harnesses.

These are intentionally lightweight and self-contained so they can run under
the current CPU/torch-2.2 compatibility shims.  The math is the XMemTransfer /
DeepSeek-Engram-style reader; Phase 1 will replace or augment this with the
full official Qwen/Engram ``ContextAwareGating + ShortConv`` path.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class ShortConv(torch.nn.Module):
    """Depthwise causal short conv used by DeepSeek Engram / Qwen PLE."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 2,
        dilation: int = 2,
        zero_init: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.norm = RMSNorm(hidden_size)
        self.conv = torch.nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            dilation=dilation,
            bias=False,
        )
        if zero_init:
            torch.nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        normed = self.norm(x)
        conv_in = normed.transpose(1, 2)
        pad_len = (self.kernel_size - 1) * self.dilation
        conv_in = F.pad(conv_in, (pad_len, 0))
        out = F.silu(self.conv(conv_in)).transpose(1, 2)
        return out + x


class EngramReader(torch.nn.Module):
    """XMemTransfer-style target-side reader with optional multi-branch keys.

    This is the Phase 0 reader used to establish the formal evaluation protocol.
    It is intentionally kept simple; Phase 1 will swap in the faithful
    engram-peft / official PLE gating.
    """

    def __init__(
        self,
        d_model: int,
        d_mem: int = 2560,
        gate_bias_init: float = -2.0,
        num_branches: int = 1,
        zero_init_v: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_branches = num_branches
        self.w_v = torch.nn.Linear(d_mem, d_model, bias=False)
        if zero_init_v:
            torch.nn.init.zeros_(self.w_v.weight)
        else:
            torch.nn.init.normal_(self.w_v.weight, mean=0.0, std=0.02)

        if num_branches == 1:
            self.w_k = torch.nn.Linear(d_mem, d_model, bias=False)
            self.norm_k = RMSNorm(d_model)
            torch.nn.init.normal_(self.w_k.weight, mean=0.0, std=0.02)
        else:
            self.w_k = torch.nn.ModuleList(
                [torch.nn.Linear(d_mem, d_model, bias=False) for _ in range(num_branches)]
            )
            self.norm_k = torch.nn.ModuleList(
                [RMSNorm(d_model) for _ in range(num_branches)]
            )
            for proj in self.w_k:
                torch.nn.init.normal_(proj.weight, mean=0.0, std=0.02)

        self.norm_h = RMSNorm(d_model)
        self.gate_bias = torch.nn.Parameter(
            torch.full((num_branches,), gate_bias_init)
        )

    def forward(self, h, e_t):
        v = self.w_v(e_t)
        norm_h = self.norm_h(h)
        if self.num_branches == 1:
            k = self.w_k(e_t)
            gate_logit = (norm_h * self.norm_k(k)).sum(-1) / math.sqrt(self.d_model)
            gate = torch.sigmoid(gate_logit + self.gate_bias[0])
            return gate.unsqueeze(-1) * v
        contributions = []
        for branch_idx, (proj_k, norm_k) in enumerate(
            zip(self.w_k, self.norm_k)
        ):
            k = proj_k(e_t)
            gate_logit = (norm_h * norm_k(k)).sum(-1) / math.sqrt(self.d_model)
            gate = torch.sigmoid(gate_logit + self.gate_bias[branch_idx])
            contributions.append(gate.unsqueeze(-1) * v)
        return torch.stack(contributions, dim=0).mean(dim=0)


def install_reader_hook(
    model: torch.nn.Module,
    layer_index: int,
    reader: torch.nn.Module,
    short_conv: torch.nn.Module | None = None,
):
    """Install a post-forward hook injecting PLE reader output into a model layer."""
    layer = model.model.layers[layer_index]

    def post_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        current = getattr(model, "_current_ple_e_t", None)
        if current is not None and current.shape[1] == hidden.shape[1]:
            contribution = reader(hidden, current)
            if short_conv is not None:
                contribution = short_conv(contribution)
            new_hidden = hidden + contribution
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden
        return output

    handle = layer.register_forward_hook(post_hook)
    return handle


class QwenShortConv(torch.nn.Module):
    """Faithful multi-branch ShortConv from engram-peft / official PLE."""

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int = 4,
        kernel_size: int = 4,
        dilation: int = 3,
        zero_init: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.norms = torch.nn.ModuleList(
            [RMSNorm(hidden_size) for _ in range(hc_mult)]
        )
        total_channels = hc_mult * hidden_size
        self.conv = torch.nn.Conv1d(
            total_channels,
            total_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=total_channels,
            bias=False,
        )
        if zero_init:
            torch.nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        # x: [B, T, hc_mult, hidden_size]
        b, t, hc, d = x.shape
        normed = torch.stack(
            [self.norms[i](x[:, :, i, :]) for i in range(hc)], dim=2
        )
        conv_in = normed.permute(0, 2, 3, 1).reshape(b, hc * d, t)
        pad_len = (self.kernel_size - 1) * self.dilation
        conv_in = F.pad(conv_in, (pad_len, 0))
        conv_out = F.silu(self.conv(conv_in))
        out = conv_out.view(b, hc, d, t).permute(0, 3, 1, 2).contiguous()
        return out + x


class QwenEngramReader(torch.nn.Module):
    """Faithful engram-peft / official-PLE-style target-side reader.

    This implements the multi-branch ``ContextAwareGating`` and multi-branch
    ``ShortConv`` that are used by DeepSeek Engram and Qwen PLE.  For a
    single-stream target model, hidden states are expanded logically to
    ``hc_mult`` branches and summed back after ShortConv.
    """

    def __init__(
        self,
        d_model: int,
        d_mem: int = 2560,
        hc_mult: int = 4,
        kernel_size: int = 4,
        dilation: int = 3,
        zero_init: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.hc_mult = hc_mult
        self.w_v = torch.nn.Linear(d_mem, d_model, bias=False)
        if zero_init:
            torch.nn.init.zeros_(self.w_v.weight)
        else:
            torch.nn.init.normal_(self.w_v.weight, mean=0.0, std=0.02)

        self.w_k = torch.nn.ModuleList(
            [torch.nn.Linear(d_mem, d_model, bias=False) for _ in range(hc_mult)]
        )
        self.norm_h = torch.nn.ModuleList(
            [RMSNorm(d_model) for _ in range(hc_mult)]
        )
        self.norm_k = torch.nn.ModuleList(
            [RMSNorm(d_model) for _ in range(hc_mult)]
        )
        self.short_conv = QwenShortConv(
            hidden_size=d_model,
            hc_mult=hc_mult,
            kernel_size=kernel_size,
            dilation=dilation,
            zero_init=zero_init,
        )

    def forward(self, h, e_t):
        value = self.w_v(e_t)  # [B, T, d_model]
        gates = []
        for m in range(self.hc_mult):
            key = self.w_k[m](e_t)
            normed_key = self.norm_k[m](key)
            normed_query = self.norm_h[m](h)
            score = (normed_key * normed_query).sum(-1, keepdim=True)
            score = score / math.sqrt(self.d_model)
            score = score.abs().clamp_min(1e-6).sqrt() * score.sign()
            gate = torch.sigmoid(score)
            gates.append(gate)
        gate = torch.stack(gates, dim=2)  # [B, T, hc_mult, 1]
        gated_value = gate * value.unsqueeze(2)  # [B, T, hc_mult, d_model]
        y = self.short_conv(gated_value)  # [B, T, hc_mult, d_model]
        return y.sum(dim=2)
