"""Causal multi-head attention token mixer (sinusoidal positions, KV-cache rollout).

Parallelizable / TC0-limited reference. chunkable=False -> core trains it full-seq;
rollout uses a per-mixer KV cache. Owns its pre-norm + positional embedding and
returns the residual delta.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from srdn.core import RMSNorm


class AttentionMixer(nn.Module):
    chunkable = False

    def __init__(self, d_model: int, n_heads: int, *, max_seq_len: int = 4096) -> None:
        super().__init__()
        d = int(d_model)
        self.d_model, self.heads = d, int(n_heads)
        if d % self.heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.head_dim = d // self.heads
        self.max_seq_len = int(max_seq_len)
        self.norm = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.register_buffer("pos_embed", self._sinusoidal(self.max_seq_len, d), persistent=False)

    @staticmethod
    def _sinusoidal(max_seq_len, d_model):
        pos = torch.arange(int(max_seq_len), dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, int(d_model), 2, dtype=torch.float32) * (-math.log(10000.0) / int(d_model)))
        table = torch.zeros((int(max_seq_len), int(d_model)), dtype=torch.float32)
        table[:, 0::2] = torch.sin(pos * div)
        table[:, 1::2] = torch.cos(pos * div[: table[:, 1::2].shape[1]])
        return table

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, _ = x.shape
        if steps > self.max_seq_len:
            raise ValueError(f"sequence length {steps} exceeds max_seq_len {self.max_seq_len}")
        xn = self.norm(x) + self.pos_embed[:steps].to(x.device)[None]
        q, k, v = self.qkv(xn.to(self.qkv.weight.dtype)).chunk(3, dim=-1)
        hd = self.head_dim
        q = q.view(bsz, steps, self.heads, hd).transpose(1, 2)
        k = k.view(bsz, steps, self.heads, hd).transpose(1, 2)
        v = v.view(bsz, steps, self.heads, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(bsz, steps, self.d_model)
        return self.out_proj(y.to(self.out_proj.weight.dtype)).float()

    # rollout: state = (lengths [B], k_cache, v_cache)
    def init_state(self, B, device):
        lengths = torch.zeros((int(B),), device=device, dtype=torch.long)
        kc = torch.zeros((int(B), self.heads, self.max_seq_len, self.head_dim), device=device)
        vc = torch.zeros((int(B), self.heads, self.max_seq_len, self.head_dim), device=device)
        return (lengths, kc, vc)

    def step(self, x_t, state):
        lengths, kc, vc = state
        bsz = int(x_t.shape[0])
        lengths = lengths + 1
        if int(lengths.max().item()) > self.max_seq_len:
            raise ValueError(f"rollout length exceeds max_seq_len {self.max_seq_len}")
        pos = (lengths - 1).clamp_min(0)
        xn = self.norm(x_t) + self.pos_embed[pos].to(x_t.device)
        q, k, v = self.qkv(xn.to(self.qkv.weight.dtype)).chunk(3, dim=-1)
        hd = self.head_dim
        q = q.view(bsz, self.heads, 1, hd)
        kc = kc.clone(); vc = vc.clone()
        rows = torch.arange(bsz, device=x_t.device)
        kc[rows, :, pos, :] = k.view(bsz, self.heads, hd)
        vc[rows, :, pos, :] = v.view(bsz, self.heads, hd)
        max_len = max(1, int(lengths.max().item()))
        key_pos = torch.arange(max_len, device=x_t.device)
        attn_mask = key_pos.view(1, 1, 1, max_len) < lengths.view(bsz, 1, 1, 1)
        y = F.scaled_dot_product_attention(q, kc[:, :, :max_len, :], vc[:, :, :max_len, :],
                                           attn_mask=attn_mask, is_causal=False)
        y = y.reshape(bsz, self.d_model)
        return self.out_proj(y.to(self.out_proj.weight.dtype)).float(), (lengths, kc, vc)


__all__ = ["AttentionMixer"]
