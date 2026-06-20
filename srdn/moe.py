"""State-routed mixture-of-experts (the SRDN channel mixer for the moe_ffn sublayer).

The state must enter an MoE via ROUTING, not via the additive write path:
`StateRoutedMoE` keeps the two inputs separate -- experts see `x_in`, the router
sees `route_in`. Which expert fires is a function of what the recurrence has
accumulated -- a magnitude-robust (softmax over logits) handle for
state-conditioned behavior.

Experts are SwiGLUs with separated per-expert down-projections, computed densely
(E small) and mixed by the top-k routing weights -- bit-exact forward/step parity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateRoutedMoE(nn.Module):
    """Top-k mixture of SwiGLU experts; expert/router inputs decoupled.

    forward(x_in [B,d_in], route_in [B,d_route]) -> (out [B,d_out], logits [B,E]).
    Router logits are surfaced for the load-balance aux loss (`moe_aux_loss`)."""

    def __init__(self, d_in: int, d_out: int, d_route: int,
                 n_experts: int, top_k: int, d_hidden: int = 0) -> None:
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.E, self.k = int(n_experts), int(top_k)
        self.d_hidden = int(d_hidden)
        self.router = nn.Linear(d_route, self.E, bias=False)
        inner = self.d_hidden if self.d_hidden > 0 else d_out
        # d_hidden > 0: standard two-layer experts with SEPARATED per-expert
        # down_projs (w_down) -- no shared projection couples expert gradients.
        self.w_gate = nn.Parameter(torch.empty(self.E, d_in, inner))
        self.w_up = nn.Parameter(torch.empty(self.E, d_in, inner))
        bound = d_in ** -0.5                       # nn.Linear default init scale
        nn.init.uniform_(self.w_gate, -bound, bound)
        nn.init.uniform_(self.w_up, -bound, bound)
        if self.d_hidden > 0:
            self.w_down = nn.Parameter(torch.empty(self.E, self.d_hidden, d_out))
            nn.init.uniform_(self.w_down, -(self.d_hidden ** -0.5), self.d_hidden ** -0.5)

    def forward(self, x_in: torch.Tensor, route_in: torch.Tensor):
        logits = self.router(route_in).float()                      # [B,E]
        topv, topi = logits.topk(self.k, dim=-1)
        topw = F.softmax(topv, dim=-1)                              # normalize over chosen
        weights = torch.zeros_like(logits).scatter(-1, topi, topw)  # 0 off-support
        g = torch.einsum("bd,edw->bew", x_in, self.w_gate.float())
        u = torch.einsum("bd,edw->bew", x_in, self.w_up.float())
        act = F.silu(g) * u                                         # [B,E,inner]
        if self.d_hidden > 0:
            act = torch.einsum("beh,ehd->bed", act, self.w_down.float())
        out = torch.einsum("be,bew->bw", weights, act)
        return out, logits


def moe_aux_loss(router_logits: list[torch.Tensor], top_k: int) -> torch.Tensor:
    """Switch-transformer load-balance loss over collected [.,E] logit tensors:
    E * sum_e f_e*P_e (f = dispatch fraction, P = mean prob); 0 if list empty."""
    if not router_logits:
        return torch.zeros(())
    total = 0.0
    for lg in router_logits:
        flat = lg.reshape(-1, lg.shape[-1])
        E = flat.shape[-1]
        P = F.softmax(flat, dim=-1).mean(0)
        topi = flat.topk(top_k, dim=-1).indices
        f = F.one_hot(topi, E).sum(1).clamp(max=1).float().mean(0)
        total = total + E * (f * P).sum()
    return total / len(router_logits)


@torch.no_grad()
def router_stats(router_logits: list[torch.Tensor], top_k: int) -> dict:
    """Per-expert load fraction + routing entropy (interpretability)."""
    if not router_logits:
        return {}
    loads, ents = [], []
    for lg in router_logits:
        flat = lg.reshape(-1, lg.shape[-1])
        E = flat.shape[-1]
        topi = flat.topk(top_k, dim=-1).indices
        loads.append(F.one_hot(topi, E).sum(1).clamp(max=1).float().mean(0))
        probs = F.softmax(flat, dim=-1)
        ents.append((-(probs * probs.clamp_min(1e-9).log()).sum(-1)).mean())
    load = torch.stack(loads).mean(0)
    return {
        "load_frac": [round(x, 4) for x in load.tolist()],
        "load_cv": round((load.std() / load.mean().clamp_min(1e-9)).item(), 4),
        "route_entropy": round(torch.stack(ents).mean().item(), 4),
    }


__all__ = ["StateRoutedMoE", "moe_aux_loss", "router_stats"]
