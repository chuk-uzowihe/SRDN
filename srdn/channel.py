"""The SHARED channel mixer (second sublayer), identical for SRDN and every baseline.

So "swap only the token mixer" is enforced structurally: every architecture wraps
its mixer (ops/*.py) in the same Block (core.py) with this exact channel mixer.
Default is a dense SwiGLU FFN; an optional state-routed MoE is available (same for
all archs) for capacity-matched MoE comparisons.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import RMSNorm, StateRoutedMoE


class ChannelMixer(nn.Module):
    """pre-norm -> (SwiGLU FFN | state-routed MoE) -> residual. Shared by all archs.

    ffn_mult > 0 enables the dense SwiGLU FFN. moe_ffn=True enables a state-routed
    MoE sublayer instead (or in addition). The standard comparison uses the dense
    FFN for everyone; MoE is opt-in and identical across architectures.
    """

    def __init__(self, d_model: int, ffn_mult: float, *, moe_ffn: bool = False,
                 n_experts: int = 4, top_k: int = 2, expert_mult: float = 2.3,
                 gamma_o_init: float = 0.0) -> None:
        super().__init__()
        d = int(d_model)
        self.use_moe = bool(moe_ffn)
        if self.use_moe:
            self.moe_norm = RMSNorm(d)
            self.moe = StateRoutedMoE(d, d, d_route=d, n_experts=n_experts, top_k=top_k,
                                      d_hidden=int(round(expert_mult * d)))
            self.gamma_ffn = nn.Parameter(torch.tensor(float(gamma_o_init)))
        self.use_ffn = float(ffn_mult) > 0.0
        if self.use_ffn:
            hidden = int(math.ceil((d * float(ffn_mult)) / 64.0) * 64)
            self.ffn_norm = RMSNorm(d)
            self.ffn_in = nn.Linear(d, 2 * hidden, bias=False)
            self.ffn_out = nn.Linear(hidden, d, bias=False)
        self._router_logits: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_moe:
            h = self.moe_norm(x); lead = h.shape[:-1]; d = h.shape[-1]
            out, lg = self.moe(h.reshape(-1, d), h.reshape(-1, d))
            self._router_logits.append(lg)
            x = x + (self.gamma_ffn.float() * out).reshape(*lead, d).float()
        if self.use_ffn:
            h = self.ffn_norm(x)
            up, gate = self.ffn_in(h.to(self.ffn_in.weight.dtype)).chunk(2, dim=-1)
            x = x + self.ffn_out(up * F.silu(gate)).float()
        return x

    def pop_router_logits(self) -> list[torch.Tensor]:
        out, self._router_logits = self._router_logits, []
        return out


__all__ = ["ChannelMixer"]
