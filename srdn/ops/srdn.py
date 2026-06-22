"""SRDN token mixer: gated delta rule with STATE-CONDITIONED projections.

Per head a square state S in R^{dh x dh}, S = sum_i k_i v_i^T. Per token:

    retrieved = k . S
    b         = 2 * sigmoid(blogit)            # per-channel write gate, neg eigvals
    u         = b * (v - retrieved)            # delta-rule write
    S         = a . S + k (x) u                # per-channel Mamba-2 decay a
    o         = q . S                          # read after the write -> residual

Recurrence-completeness: the projections read the state, so step-t inputs depend
nonlinearly on step-(t-1) state (breaks the chunk-parallel form).
  q reads s0 = rms(diag S)         (the read direction)
  k,v,a,b read sx = rms(q . S)     (the content read, pre-write)
All conditioning is additive ReZero (gamma=0 at init -> exactly GDN at init).

Owns its pre-norm and returns the residual DELTA (core does x = x + mixer(x)).
chunkable: state-carry BPTT (S + conv ring-buffer) -> exact gradient across chunks.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from srdn.core import RMSNorm
from .conv import QKVFeature


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _rmsnorm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class SRDNMixer(nn.Module):
    chunkable = True

    def __init__(self, d_model: int, n_heads: int, d_head: int, *,
                 short_conv: bool = False, conv_size: int = 4, rezero_init: float = 0.0) -> None:
        super().__init__()
        H, dh, d = int(n_heads), int(d_head), int(d_model)
        self.H, self.dh = H, dh
        inner = H * dh
        assert inner == d, "SRDN feeds o straight to the residual: n_heads*d_head must == d_model"

        self.norm = RMSNorm(d)
        fk = dict(short_conv=short_conv, conv_size=conv_size)
        self.qfeat = QKVFeature(d, H, dh, **fk)
        self.kfeat = QKVFeature(d, H, dh, **fk)
        self.vfeat = QKVFeature(d, H, dh, **fk)
        self.a_proj = nn.Linear(d, inner, bias=False)
        self.b_proj = nn.Linear(d, inner, bias=False)

        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1.0, 16.0)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(torch.rand(inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
        self.dt_bias._no_weight_decay = True

        rz = float(rezero_init)
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

    # ---- core math ----
    def _decay(self, alogit):
        A = self.A_log.float().exp().view(self.H, 1)
        dtb = self.dt_bias.float().view(self.H, self.dh)
        return torch.exp(-A * F.softplus(alogit + dtb))

    def _recur(self, q, k, v, alogit, blogit, S):
        dh = self.dh
        s0 = _rmsnorm(torch.diagonal(S, dim1=-2, dim2=-1))
        q = F.silu(q + self.gamma_q.float() * torch.einsum("bhs,hsq->bhq", s0, self.W_qs.float()))
        sx = _rmsnorm((S * _l2norm(q)[..., :, None]).sum(dim=-2))
        alogit = alogit + self.gamma_ax.float() * torch.einsum("bhs,hsc->bhc", sx, self.W_ax.float())
        blogit = blogit + self.gamma_bx.float() * torch.einsum("bhs,hsc->bhc", sx, self.W_bx.float())
        a = self._decay(alogit)
        k = F.silu(k + self.gamma_kx.float() * torch.einsum("bhs,hsk->bhk", sx, self.W_kx.float()))
        v = F.silu(v + self.gamma_vx.float() * torch.einsum("bhs,hsv->bhv", sx, self.W_vx.float()))
        q = _l2norm(q) * (dh ** -0.5)
        k = _l2norm(k)
        retrieved = (S * k[..., :, None]).sum(dim=-2)
        b = 2.0 * torch.sigmoid(blogit)
        u = b * (v - retrieved)
        S = a[..., None, :] * S + k[..., :, None] * u[..., None, :]
        o = (S * q[..., :, None]).sum(dim=-2)
        return o, S

    def _init_S(self, B, device):
        return torch.zeros(B, self.H, self.dh, self.dh, device=device, dtype=torch.float32)

    def _gates(self, xn):
        lead = xn.shape[:-1]
        return (self.a_proj(xn).view(*lead, self.H, self.dh).float(),
                self.b_proj(xn).view(*lead, self.H, self.dh).float())

    # ---- interface: forward (full seq), returns residual delta ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    # ---- interface: chunked (carry S + conv ring-buffer) ----
    def init_state(self, B, device):
        hists = {"q": self.qfeat.init_hist(B, device),
                 "k": self.kfeat.init_hist(B, device),
                 "v": self.vfeat.init_hist(B, device)}
        return (self._init_S(B, device), hists)

    def forward_with_state(self, x, state):
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
        return torch.stack(outs, dim=1).reshape(B, T, self.H * self.dh), (S, hists)

    _KEYS = ("q", "k", "v")

    def flatten_state(self, state):
        S, h = state
        if h["q"] is None:
            return [S]
        return [S, h["q"], h["k"], h["v"]]

    def unflatten_state(self, flat):
        if len(flat) == 1:
            return (flat[0], {"q": None, "k": None, "v": None})
        S, q, k, v = flat
        return (S, {"q": q, "k": k, "v": v})

    # ---- interface: rollout ----
    def step(self, x_t, state):
        S, hists = state
        hists = dict(hists)
        xn = self.norm(x_t)
        q, hists["q"] = self.qfeat.step(xn, hists["q"])
        k, hists["k"] = self.kfeat.step(xn, hists["k"])
        v, hists["v"] = self.vfeat.step(xn, hists["v"])
        alogit, blogit = self._gates(xn)
        o, S = self._recur(q, k, v, alogit, blogit, S)
        return o.reshape(x_t.shape[0], self.H * self.dh), (S, hists)


__all__ = ["SRDNMixer"]
