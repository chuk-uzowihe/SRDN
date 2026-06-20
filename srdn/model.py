"""SRDN: a gated delta-rule linear-attention cell with STATE-CONDITIONED projections.

One gated delta-rule core per head, with a square state S in R^{dh x dh},
S = sum_i k_i v_i^T (indexed S[key, value]). The write, per token:

    retrieved = k . S                  # contract key axis -> value space
    b         = 2 * sigmoid(blogit)    # per-channel write gate (allow negative eigvals)
    u         = b * (v - retrieved)    # delta-rule write
    S         = a . S + k (x) u        # per-channel Mamba-2 decay a, outer-product write
    o         = q . S                  # read after the write -> straight to the residual

What makes SRDN *recurrence-complete* (not chunk-parallelizable): the projections
are STATE-CONDITIONED, so step-t inputs depend nonlinearly on the step-(t-1) state.
  - q reads a diagonal state summary  s0 = rms(diag S)         (the read direction)
  - k, v and the decay/write gates a,b read the content read   sx = rms(q . S)
All conditioning is an additive ReZero path (gamma=0 at init -> exactly GDN at
init, the state coupling grows in). q/k features are L2-normalized; v is not.

This is the srdn5 cell, fixed (no rung registry / out-mode / decay-param knobs):
Mamba-2 multi-timescale decay, output straight to the residual (n_heads*d_head ==
d_model), optional fla-style short conv, and either a dense SwiGLU FFN (graph-RL)
or a state-routed MoE sublayer (enwik8) as the channel mixer.

forward / hidden / logits        : teacher-forced full-sequence pass.
chunked_logits                   : sqrt-exact BPTT (carry S + conv ring-buffer
                                   across chunk boundaries, checkpoint/remat).
init_states / step               : incremental rollout (one token, optional mask).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .moe import StateRoutedMoE
from .norm import RMSNorm


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _rmsnorm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


@dataclass
class SRDNConfig:
    vocab_size: int = 0
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_head: int = 32              # n_heads * d_head MUST equal d_model (o -> residual)
    ffn_mult: float = 2.0         # dense SwiGLU FFN; 0 disables it
    short_conv: bool = False      # fla-style depthwise causal short conv on q,k,v
    conv_size: int = 4
    # channel mixer: dense SwiGLU FFN (default, graph-RL) OR a state-routed MoE (enwik8)
    moe_ffn: bool = False
    n_experts: int = 4
    top_k: int = 2
    expert_mult: float = 2.3      # per-expert hidden = expert_mult * d_model
    gamma_o_init: float = 0.0     # ReZero init for the moe_ffn sublayer
    rezero_init: float = 0.0      # ReZero init for the state-conditioning paths

    def to_dict(self) -> dict:
        return asdict(self)


class CausalDWConv1d(nn.Module):
    """Depthwise causal short conv, LINEAR (activation applied separately). Exact
    forward / chunk-continuation / single-step parity via a (K-1)-frame ring buffer."""

    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.kernel = int(kernel)
        self.channels = int(channels)
        self.conv = nn.Conv1d(channels, channels, self.kernel, groups=channels,
                              padding=self.kernel - 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # [B,T,C]
        T = x.shape[1]
        return self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)

    def forward_cont(self, x: torch.Tensor, hist: torch.Tensor):
        """Chunk-correct continuation: prepend the carried (K-1)-frame history as
        real left-context, valid-conv to exactly T outputs, return the new history.
        With a zero hist this equals `forward` on the first chunk."""
        xin = torch.cat([hist, x], dim=1)                      # [B, K-1+T, C]
        y = F.conv1d(xin.transpose(1, 2), self.conv.weight, self.conv.bias,
                     padding=0, groups=self.channels)          # [B, C, T]
        return y.transpose(1, 2), xin[:, -(self.kernel - 1):, :]

    def init_hist(self, B: int, device) -> torch.Tensor:
        return torch.zeros(B, self.kernel - 1, self.channels, device=device)

    def step(self, x_t: torch.Tensor, hist: torch.Tensor):     # x_t [B,C], hist [B,K-1,C]
        window = torch.cat([hist, x_t.unsqueeze(1)], dim=1)    # [B,K,C]
        y = torch.einsum("bkc,ck->bc", window, self.conv.weight[:, 0, :]) + self.conv.bias
        return y, window[:, 1:, :]


class QKVFeature(nn.Module):
    """Per-projection front-end: Linear -> optional depthwise causal short conv.
    Returns LINEAR float features [..., H, dh]; SiLU + L2 + state-conditioning are
    applied downstream in the cell step (so x and the state interact nonlinearly)."""

    def __init__(self, d_in: int, H: int, dh: int, *, short_conv: bool, conv_size: int) -> None:
        super().__init__()
        self.H, self.dh = H, dh
        self.proj = nn.Linear(d_in, H * dh, bias=False)
        self.conv = CausalDWConv1d(H * dh, conv_size) if short_conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # [B,T,d] -> [B,T,H,dh]
        z = self.proj(x)
        if self.conv is not None:
            z = self.conv(z)
        return z.view(*x.shape[:-1], self.H, self.dh).float()

    def forward_with_hist(self, x: torch.Tensor, hist):
        z = self.proj(x)
        if self.conv is not None:
            z, hist = self.conv.forward_cont(z, hist)
        return z.view(*x.shape[:-1], self.H, self.dh).float(), hist

    def init_hist(self, B: int, device):
        return None if self.conv is None else self.conv.init_hist(B, device)

    def step(self, x_t: torch.Tensor, hist):                   # [B,d] -> [B,H,dh]
        z = self.proj(x_t)
        if self.conv is not None:
            z, hist = self.conv.step(z, hist)
        return z.view(x_t.shape[0], self.H, self.dh).float(), hist


class SRDNBlock(nn.Module):
    """RMSNorm -> SRDN cell (state-conditioned gated delta rule) -> residual ;
    then the channel mixer (dense SwiGLU FFN or a state-routed MoE sublayer)."""

    def __init__(self, cfg: SRDNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, H, dh = cfg.d_model, cfg.n_heads, cfg.d_head
        self.H, self.dh = H, dh
        inner = H * dh
        assert inner == d, "SRDN feeds o straight to the residual: n_heads*d_head must == d_model"

        self.norm = RMSNorm(d)
        fk = dict(short_conv=cfg.short_conv, conv_size=cfg.conv_size)
        self.qfeat = QKVFeature(d, H, dh, **fk)
        self.kfeat = QKVFeature(d, H, dh, **fk)
        self.vfeat = QKVFeature(d, H, dh, **fk)
        self.a_proj = nn.Linear(d, inner, bias=False)          # per-channel decay logit
        self.b_proj = nn.Linear(d, inner, bias=False)          # per-channel write-gate logit

        # Mamba-2 / GDN-2 multi-timescale decay: per-HEAD log-rate + per-channel dt bias.
        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1.0, 16.0)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(torch.rand(inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
        self.dt_bias._no_weight_decay = True

        # state conditioning (additive ReZero). q reads s0=rms(diag S); k,v,a,b read sx=rms(q.S).
        rz = float(cfg.rezero_init)
        self.W_qs = nn.Parameter(torch.randn(H, dh, dh) * dh ** -0.5)
        self.gamma_q = nn.Parameter(torch.full((1,), rz))
        self.W_kx = nn.Parameter(torch.randn(H, dh, dh) * dh ** -0.5)
        self.W_vx = nn.Parameter(torch.randn(H, dh, dh) * dh ** -0.5)
        self.gamma_kx = nn.Parameter(torch.full((1,), rz))
        self.gamma_vx = nn.Parameter(torch.full((1,), rz))
        self.W_ax = nn.Parameter(torch.randn(H, dh, dh) * dh ** -0.5)
        self.W_bx = nn.Parameter(torch.randn(H, dh, dh) * dh ** -0.5)
        self.gamma_ax = nn.Parameter(torch.full((1,), rz))
        self.gamma_bx = nn.Parameter(torch.full((1,), rz))

        self.use_ffn = float(cfg.ffn_mult) > 0.0
        if self.use_ffn:
            hidden = int(round(d * float(cfg.ffn_mult)))
            self.ffn_norm = RMSNorm(d)
            self.ffn_gate = nn.Linear(d, hidden, bias=False)
            self.ffn_up = nn.Linear(d, hidden, bias=False)
            self.ffn_down = nn.Linear(hidden, d, bias=False)
        self.use_moe_ffn = bool(cfg.moe_ffn)
        if self.use_moe_ffn:
            self.moe_ffn_norm = RMSNorm(d)
            self.moe_ffn = StateRoutedMoE(d, d, d_route=d, n_experts=cfg.n_experts,
                                          top_k=cfg.top_k, d_hidden=int(round(cfg.expert_mult * d)))
            self.gamma_ffn = nn.Parameter(torch.tensor(float(cfg.gamma_o_init)))
        self._router_logits: list[torch.Tensor] = []

    # ---- core ----
    def _decay(self, alogit: torch.Tensor) -> torch.Tensor:
        A = self.A_log.float().exp().view(self.H, 1)
        dtb = self.dt_bias.float().view(self.H, self.dh)
        return torch.exp(-A * F.softplus(alogit + dtb))

    def _recur(self, q, k, v, alogit, blogit, S):
        """One token of the state-conditioned gated delta rule. q,k,v [B,H,dh]."""
        H, dh = self.H, self.dh
        s0 = _rmsnorm(torch.diagonal(S, dim1=-2, dim2=-1))                       # [B,H,dh]
        q = F.silu(q + self.gamma_q.float() * torch.einsum("bhs,hsq->bhq", s0, self.W_qs.float()))
        sx = _rmsnorm((S * _l2norm(q)[..., :, None]).sum(dim=-2))                # content read (pre-write)
        alogit = alogit + self.gamma_ax.float() * torch.einsum("bhs,hsc->bhc", sx, self.W_ax.float())
        blogit = blogit + self.gamma_bx.float() * torch.einsum("bhs,hsc->bhc", sx, self.W_bx.float())
        a = self._decay(alogit)
        k = F.silu(k + self.gamma_kx.float() * torch.einsum("bhs,hsk->bhk", sx, self.W_kx.float()))
        v = F.silu(v + self.gamma_vx.float() * torch.einsum("bhs,hsv->bhv", sx, self.W_vx.float()))
        q = _l2norm(q) * (dh ** -0.5)
        k = _l2norm(k)
        retrieved = (S * k[..., :, None]).sum(dim=-2)                            # k . S
        b = 2.0 * torch.sigmoid(blogit)
        u = b * (v - retrieved)
        S = a[..., None, :] * S + k[..., :, None] * u[..., None, :]
        o = (S * q[..., :, None]).sum(dim=-2)                                    # q . S (post-write)
        return o, S

    def _init_S(self, B: int, device):
        return torch.zeros(B, self.H, self.dh, self.dh, device=device, dtype=torch.float32)

    def _gates(self, xn):
        lead = xn.shape[:-1]
        return (self.a_proj(xn).view(*lead, self.H, self.dh).float(),
                self.b_proj(xn).view(*lead, self.H, self.dh).float())

    def _channel_mix(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_moe_ffn:
            h = self.moe_ffn_norm(x); lead = h.shape[:-1]; d = h.shape[-1]
            out, lg = self.moe_ffn(h.reshape(-1, d), h.reshape(-1, d))
            self._router_logits.append(lg)
            x = x + (self.gamma_ffn.float() * out).reshape(*lead, d).float()
        if self.use_ffn:
            h = self.ffn_norm(x)
            x = x + self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h)).float()
        return x

    # ---- full sequence ----
    def _mix(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        xn = self.norm(x)
        q, k, v = self.qfeat(xn), self.kfeat(xn), self.vfeat(xn)
        alogit, blogit = self._gates(xn)
        S = self._init_S(B, x.device)
        outs = []
        for t in range(T):
            o, S = self._recur(q[:, t], k[:, t], v[:, t], alogit[:, t], blogit[:, t], S)
            outs.append(o)
        return torch.stack(outs, dim=1).reshape(B, T, self.H * self.dh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() + self._mix(x)
        return self._channel_mix(x)

    # ---- chunked (carry S + conv ring-buffer; sqrt-exact BPTT) ----
    def init_chunk_state(self, B: int, device):
        hists = {"q": self.qfeat.init_hist(B, device),
                 "k": self.kfeat.init_hist(B, device),
                 "v": self.vfeat.init_hist(B, device)}
        return (self._init_S(B, device), hists)

    def forward_with_state(self, x: torch.Tensor, state):
        S, hists = state
        hists = dict(hists)
        B, T, _ = x.shape
        xn = self.norm(x)
        q, hists["q"] = self.qfeat.forward_with_hist(xn, hists["q"])
        k, hists["k"] = self.kfeat.forward_with_hist(xn, hists["k"])
        v, hists["v"] = self.vfeat.forward_with_hist(xn, hists["v"])
        alogit, blogit = self._gates(xn)
        outs = []
        for t in range(T):
            o, S = self._recur(q[:, t], k[:, t], v[:, t], alogit[:, t], blogit[:, t], S)
            outs.append(o)
        mix = torch.stack(outs, dim=1).reshape(B, T, self.H * self.dh)
        x = x.float() + mix
        return self._channel_mix(x), (S, hists)

    # ---- single-token rollout ----
    def step(self, x_t: torch.Tensor, state):
        S, hists = state
        hists = dict(hists)
        xn = self.norm(x_t)
        q, hists["q"] = self.qfeat.step(xn, hists["q"])
        k, hists["k"] = self.kfeat.step(xn, hists["k"])
        v, hists["v"] = self.vfeat.step(xn, hists["v"])
        alogit, blogit = self._gates(xn)
        o, S = self._recur(q, k, v, alogit, blogit, S)
        x = x_t.float() + o.reshape(x_t.shape[0], self.H * self.dh)
        return self._channel_mix(x), (S, hists)


class SRDNLM(nn.Module):
    """SRDN language model + RL-policy surface (logits / chunked_logits / init_states /
    step). Weights are tied to the embedding."""

    def __init__(self, cfg: SRDNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = int(cfg.vocab_size)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(SRDNBlock(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model)
        nn.init.normal_(self.embed.weight, std=cfg.d_model ** -0.5)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        for b in self.blocks:
            h = b(h)
        return self.final_norm(h)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.hidden(tokens)

    def logits(self, tokens: torch.Tensor, *, use_xma: bool = False) -> torch.Tensor:
        del use_xma
        return self.hidden(tokens) @ self.embed.weight.float().T

    def chunked_logits(self, tokens: torch.Tensor, *, use_xma: bool = False, chunk_size: int = 0,
                       detach_boundaries: bool = False, remat_chunks: bool = False) -> torch.Tensor:
        """sqrt-exact BPTT: split the sequence, carry each block's state S (and conv
        ring-buffer) across boundaries, checkpoint/remat each chunk. detach_boundaries
        False is bit-for-bit equal (fwd + grad) to `logits`; True truncates (TBPTT)."""
        del use_xma
        T = int(tokens.shape[1])
        if int(chunk_size) <= 0 or int(chunk_size) >= T:
            return self.logits(tokens)
        B = int(tokens.shape[0])
        states = [b.init_chunk_state(B, tokens.device) for b in self.blocks]
        keys = ("q", "k", "v")
        has_conv = states[0][1]["q"] is not None

        def flatten(sts):
            flat = []
            for S, h in sts:
                flat.append(S)
                if has_conv:
                    flat.extend(h[k] for k in keys)
            return flat

        def unflatten(flat):
            out, i = [], 0
            for _ in self.blocks:
                S = flat[i]; i += 1
                h = {k: flat[i + j] for j, k in enumerate(keys)} if has_conv else {"q": None, "k": None, "v": None}
                i += len(keys) if has_conv else 0
                out.append((S, h))
            return out

        def run_chunk(chunk_tokens, *flat):
            x = self.embed(chunk_tokens)
            nxt = []
            for b, st in zip(self.blocks, unflatten(flat)):
                x, st = b.forward_with_state(x, st)
                nxt.append(st)
            logits = self.final_norm(x) @ self.embed.weight.float().T
            return (logits, *flatten(nxt))

        outs = []
        for start in range(0, T, int(chunk_size)):
            chunk = tokens[:, start:start + int(chunk_size)]
            flat = flatten(states)
            if remat_chunks and torch.is_grad_enabled():
                result = checkpoint(run_chunk, chunk, *flat, use_reentrant=False)
            else:
                result = run_chunk(chunk, *flat)
            outs.append(result[0])
            states = unflatten(list(result[1:]))
            if detach_boundaries:
                states = [(S.detach(), {k: (None if h[k] is None else h[k].detach()) for k in keys}) for S, h in states]
        self.pop_router_logits()
        return torch.cat(outs, dim=1)

    def pop_router_logits(self) -> list[torch.Tensor]:
        out = []
        for b in self.blocks:
            out.extend(b._router_logits)
            b._router_logits = []
        return out

    # ---- rollout ----
    @torch.no_grad()
    def init_states(self, batch_size: int, device):
        return [b.init_chunk_state(int(batch_size), device) for b in self.blocks]

    @torch.no_grad()
    def step(self, token: torch.Tensor, states, update_mask: torch.Tensor | None = None):
        h = self.embed(token)
        new_states = []
        for b, st in zip(self.blocks, states):
            h, nst = b.step(h, st)
            if update_mask is not None:
                nst = _merge_state(update_mask, nst, st)
            new_states.append(nst)
        logits = self.final_norm(h) @ self.embed.weight.float().T
        self.pop_router_logits()
        if update_mask is not None:
            logits = torch.where(update_mask[:, None], logits, torch.zeros_like(logits))
        return logits, new_states

    def num_params(self) -> int:
        seen, n = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p)); n += p.numel()
        return n


def _merge_state(mask: torch.Tensor, new, old):
    """Keep `new` rows where mask is True, else `old`, through the (S, hists) state."""
    S_new, h_new = new
    S_old, h_old = old
    m = mask.view(-1, 1, 1, 1)
    S = torch.where(m, S_new, S_old)
    h = {}
    for k in h_new:
        if h_new[k] is None:
            h[k] = None
        else:
            mh = mask.view(-1, *([1] * (h_new[k].dim() - 1)))
            h[k] = torch.where(mh, h_new[k], h_old[k])
    return (S, h)


__all__ = ["SRDNConfig", "SRDNBlock", "SRDNLM"]
