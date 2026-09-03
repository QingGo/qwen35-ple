"""Target-side PLE reader modules used by experiment harnesses.

These are intentionally lightweight and self-contained so they can run under
the current CPU/torch-2.2 compatibility shims.  The math is the XMemTransfer /
DeepSeek-Engram-style reader; Phase 1 will replace or augment this with the
full official Qwen/Engram ``ContextAwareGating + ShortConv`` path.
"""

from __future__ import annotations

import math
import os

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


class OfficialQwenRMSNorm(torch.nn.Module):
    """Qwen4ExpTextRMSNorm-compatible grouped RMSNorm.

    The official Qwen PLE reader uses:
        output = norm(x) * (1 + weight)
    with ``weight`` initialized to zero.  This implementation mirrors that
    behaviour so we can directly load the checkpoint tensors.
    """

    def __init__(self, dim: int, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.group_size = group_size
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        orig_dtype = x.dtype
        out = x.float()
        if self.group_size is not None:
            out = out.reshape(*out.shape[:-1], -1, self.group_size)
            out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
            out = out.reshape(*x.shape[:-1], self.dim)
        else:
            out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float())
        return out.to(orig_dtype)


class MLPValueReader(torch.nn.Module):
    """Nonlinear target-side reader with E_perp value path.

    This is the theory-guided prototype:

        E_perp = E - W_he H
        v      = MLP(E_perp)
        g      = sigmoid(gate)
        output = g * v

    Unlike the official reader, the value path is a trainable MLP (nonlinear),
    and the value input is orthogonalized against H to avoid injecting
    redundant information.
    """

    def __init__(
        self,
        d_model: int,
        d_mem: int = 2560,
        hidden: int = 256,
        gate_bias_init: float = -2.0,
        zero_init_v: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.hidden = hidden

        # H -> E projection used to estimate E_parallel and form E_perp.
        self.h_to_e = torch.nn.Linear(d_model, d_mem, bias=False)
        torch.nn.init.zeros_(self.h_to_e.weight)

        # Nonlinear value path.  It sees both H and E_perp, matching the
        # oracle MLP which used H+E_perp as input.
        self.value_mlp = torch.nn.Sequential(
            torch.nn.Linear(d_mem + d_model, hidden, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, d_model, bias=False),
        )
        if zero_init_v:
            torch.nn.init.zeros_(self.value_mlp[-1].weight)

        # Simple gate.
        self.key_proj = torch.nn.Linear(d_mem, d_model, bias=False)
        torch.nn.init.normal_(self.key_proj.weight, mean=0.0, std=0.02)
        self.norm_h = RMSNorm(d_model)
        self.norm_k = RMSNorm(d_model)
        self.gate_bias = torch.nn.Parameter(torch.full((1,), gate_bias_init))

    def forward(self, h, e_t):
        e_perp = e_t - self.h_to_e(h)
        value_input = torch.cat([h, e_perp], dim=-1)
        v = self.value_mlp(value_input)

        # Differential / E-specific value: subtract the value that would be
        # produced with no E content, so we inject only memory-specific info.
        zero_e_perp = torch.zeros_like(e_perp)
        base_input = torch.cat([h, zero_e_perp], dim=-1)
        v_base = self.value_mlp(base_input)
        v_diff = v - v_base

        k = self.key_proj(e_t)
        norm_h = self.norm_h(h)
        norm_k = self.norm_k(k)
        gate_logit = (norm_h * norm_k).sum(-1, keepdim=True) / math.sqrt(self.d_model)
        gate = torch.sigmoid(gate_logit + self.gate_bias)
        return gate * v_diff


class OfficialSourceQwenReader(torch.nn.Module):
    """Best-effort reuse of the official Qwen3.8 PLE reader.

    The official ``key_proj`` / ``value_proj`` / norms / ``conv1d`` are reused
    as a frozen *source-space* reader.  Only two small trainable adapters are
    added:

    * ``query_bridge``: target hidden (e.g. 1024) -> source query space (4x2560)
    * ``out_proj``: source PLE output (2560) -> target hidden (e.g. 1024)

    The official source weights are kept frozen by default so that training is
    cheap and the Qwen3.8 reader semantics are preserved.
    """

    def __init__(
        self,
        d_target: int,
        d_source: int = 2560,
        d_mem: int = 2560,
        hc: int = 4,
        kernel_size: int = 4,
        dilation: int = 3,
        source_state: dict[str, torch.Tensor] | str | None = None,
        freeze_source: bool = True,
        zero_init_out: bool = True,
        bridge_mlp: bool = False,
        bridge_hidden: int | None = None,
        out_mlp: bool = False,
        out_hidden: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_target = d_target
        self.d_source = d_source
        self.d_mem = d_mem
        self.hc = hc
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.src_dim = hc * d_source

        # Trainable adapters.
        bridge_hidden = bridge_hidden or d_target
        out_hidden = out_hidden or d_target
        if bridge_mlp:
            self.query_bridge = torch.nn.Sequential(
                torch.nn.Linear(d_target, bridge_hidden, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(bridge_hidden, self.src_dim, bias=False),
            )
        else:
            self.query_bridge = torch.nn.Linear(d_target, self.src_dim, bias=False)

        if out_mlp:
            self.out_proj = torch.nn.Sequential(
                torch.nn.Linear(d_source, out_hidden, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(out_hidden, d_target, bias=False),
            )
            if zero_init_out:
                last = self.out_proj[-1]
                torch.nn.init.zeros_(last.weight)
        else:
            self.out_proj = torch.nn.Linear(d_source, d_target, bias=False)
            if zero_init_out:
                torch.nn.init.zeros_(self.out_proj.weight)

        # Official source-space reader.
        self.key_proj = torch.nn.Linear(d_mem, self.src_dim, bias=False)
        self.value_proj = torch.nn.Linear(d_mem, d_source, bias=False)
        self.norm_key = OfficialQwenRMSNorm(self.src_dim, group_size=d_source, eps=eps)
        self.norm_query = OfficialQwenRMSNorm(self.src_dim, group_size=d_source, eps=eps)
        self.norm_conv = OfficialQwenRMSNorm(self.src_dim, group_size=d_source, eps=eps)
        self.conv1d = torch.nn.Conv1d(
            self.src_dim,
            self.src_dim,
            kernel_size=kernel_size,
            groups=self.src_dim,
            dilation=dilation,
            bias=False,
        )

        if source_state is not None:
            self.load_source_state(source_state, strict=True)

        if freeze_source:
            for name, p in self.named_parameters():
                if name.startswith(("key_proj.", "value_proj.", "norm_", "conv1d.")):
                    p.requires_grad_(False)

    @classmethod
    def from_official_checkpoint(
        cls,
        checkpoint_path: str,
        d_target: int,
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
    ) -> OfficialSourceQwenReader:
        """Create the reader and load official source tensors from a .pt/.bin file.

        The file must contain the reader keys produced by
        ``scripts/extract_official_reader.py``.
        """
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return cls(
            d_target=d_target,
            d_source=d_source,
            d_mem=d_mem,
            hc=hc,
            kernel_size=kernel_size,
            dilation=dilation,
            source_state=state,
            freeze_source=freeze_source,
            zero_init_out=zero_init_out,
            bridge_mlp=bridge_mlp,
            bridge_hidden=bridge_hidden,
            out_mlp=out_mlp,
            out_hidden=out_hidden,
        )

    def load_source_state(
        self, source_state: dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        if isinstance(source_state, (str, os.PathLike)):
            source_state = torch.load(source_state, map_location="cpu")
        prefix = "model.language_model.layers.1.ple."
        target = {
            "key_proj.weight": self.key_proj.weight,
            "value_proj.weight": self.value_proj.weight,
            "norm_key.weight": self.norm_key.weight,
            "norm_query.weight": self.norm_query.weight,
            "norm_conv.weight": self.norm_conv.weight,
            "conv1d.weight": self.conv1d.weight,
        }
        missing = []
        unexpected = []
        for name, param in target.items():
            src = source_state.get(name) or source_state.get(prefix + name)
            if src is not None:
                with torch.no_grad():
                    param.copy_(src.to(param.dtype))
            else:
                missing.append(name)
        for name in source_state:
            if name in target or name == prefix + name:
                continue
            if name.startswith(prefix):
                short = name[len(prefix):]
                if short in target:
                    continue
            unexpected.append(name)
        if strict and missing:
            raise KeyError(f"missing official source tensors: {missing}")
        if strict and unexpected:
            # Allow extra bookkeeping keys but raise for truly unknown weights.
            known_extra = {
                "layer_multipliers",
                "ngram_heads_offsets",
                "ngram_heads_vocab_sizes",
                "ple_embedding.layer_multipliers",
                "ple_embedding.ngram_heads_offsets",
                "ple_embedding.ngram_heads_vocab_sizes",
            }
            real_unexpected = [k for k in unexpected if k not in known_extra]
            if real_unexpected:
                raise KeyError(f"unexpected source tensors: {real_unexpected}")

    def forward(self, h, e_t):
        # h: [B, T, d_target], e_t: [B, T, d_mem]
        b, t, _ = h.shape

        key = self.key_proj(e_t)                       # [B,T,4*2560]
        key_normed = self.norm_key(key).view(b, t, self.hc, self.d_source)

        query = self.query_bridge(h)                   # [B,T,4*2560]
        query_normed = self.norm_query(query).view(b, t, self.hc, self.d_source)

        score = (key_normed * query_normed).sum(-1, keepdim=True) / math.sqrt(self.d_source)
        score = score.abs().clamp_min(1e-6).sqrt() * score.sign()
        gate = torch.sigmoid(score)                    # [B,T,4,1]

        value = self.value_proj(e_t)                   # [B,T,2560]
        gated = gate * value.unsqueeze(2)              # [B,T,4,2560]
        gated_flat = gated.reshape(b, t, self.src_dim)

        gated_normed = self.norm_conv(gated_flat)
        conv_in = gated_normed.transpose(1, 2)
        pad_len = (self.kernel_size - 1) * self.dilation
        conv_in = F.pad(conv_in, (pad_len, 0))
        conv_out = F.silu(self.conv1d(conv_in)).transpose(1, 2)

        source_output = gated_flat + conv_out          # [B,T,4*2560]
        branch_sum = source_output.view(b, t, self.hc, self.d_source).sum(dim=2)
        return self.out_proj(branch_sum)               # [B,T,d_target]
