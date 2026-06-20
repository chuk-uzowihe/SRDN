"""Small causal Transformer baseline (sinusoidal positions, KV-cache rollout).

A parallelizable (TC0-limited) reference: it does not get sequence-state
checkpointing because attention is not a recurrent state-carry -- chunked_logits
is the full-sequence pass. Included on every task as the standard baseline.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_mult: float) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.heads = int(heads)
        if self.d_model % self.heads != 0:
            raise ValueError("d_model must be divisible by heads")
        hidden = int(math.ceil((self.d_model * float(ffn_mult)) / 64.0) * 64)
        self.attn_norm = RMSNorm(self.d_model)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.ffn_norm = RMSNorm(self.d_model)
        self.ffn_in = nn.Linear(self.d_model, 2 * hidden, bias=False)
        self.ffn_out = nn.Linear(hidden, self.d_model, bias=False)

    def forward(self, x, causal_mask):
        bsz, steps, _ = x.shape
        residual = x
        q, k, v = self.qkv(self.attn_norm(x).to(self.qkv.weight.dtype)).chunk(3, dim=-1)
        hd = self.d_model // self.heads
        q = q.view(bsz, steps, self.heads, hd).transpose(1, 2)
        k = k.view(bsz, steps, self.heads, hd).transpose(1, 2)
        v = v.view(bsz, steps, self.heads, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, is_causal=False)
        y = y.transpose(1, 2).reshape(bsz, steps, self.d_model)
        x = residual.float() + self.out_proj(y.to(self.out_proj.weight.dtype)).float()
        ff = self.ffn_norm(x)
        up, gate = self.ffn_in(ff.to(self.ffn_in.weight.dtype)).chunk(2, dim=-1)
        return x + self.ffn_out(up * F.silu(gate)).float()

    def step(self, x, k_cache, v_cache, lengths, update_mask):
        bsz = int(x.shape[0])
        residual = x
        q, k, v = self.qkv(self.attn_norm(x).to(self.qkv.weight.dtype)).chunk(3, dim=-1)
        hd = self.d_model // self.heads
        q = q.view(bsz, self.heads, 1, hd)
        k = k.view(bsz, self.heads, hd)
        v = v.view(bsz, self.heads, hd)
        rows = torch.nonzero(update_mask, as_tuple=False).flatten()
        if rows.numel() > 0:
            pos = lengths[rows] - 1
            k_cache[rows, :, pos, :] = k[rows]
            v_cache[rows, :, pos, :] = v[rows]
        max_len = max(1, int(lengths.max().item()))
        key_pos = torch.arange(max_len, device=x.device)
        attn_mask = key_pos.view(1, 1, 1, max_len) < lengths.view(bsz, 1, 1, 1)
        y = F.scaled_dot_product_attention(q, k_cache[:, :, :max_len, :], v_cache[:, :, :max_len, :],
                                           attn_mask=attn_mask, is_causal=False)
        y = y.reshape(bsz, self.d_model)
        x = residual.float() + self.out_proj(y.to(self.out_proj.weight.dtype)).float()
        ff = self.ffn_norm(x)
        up, gate = self.ffn_in(ff.to(self.ffn_in.weight.dtype)).chunk(2, dim=-1)
        return x + self.ffn_out(up * F.silu(gate)).float(), k_cache, v_cache


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, layers, heads, ffn_mult, *, max_seq_len=4096) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)
        self.heads = int(heads)
        if self.d_model % self.heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.head_dim = self.d_model // self.heads
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.register_buffer("pos_embed", self._sinusoidal(self.max_seq_len, self.d_model), persistent=False)
        self.layers = nn.ModuleList(TransformerBlock(d_model, heads, ffn_mult) for _ in range(int(layers)))
        self.final_norm = RMSNorm(self.d_model)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _sinusoidal(max_seq_len, d_model):
        pos = torch.arange(int(max_seq_len), dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, int(d_model), 2, dtype=torch.float32) * (-math.log(10000.0) / int(d_model)))
        table = torch.zeros((int(max_seq_len), int(d_model)), dtype=torch.float32)
        table[:, 0::2] = torch.sin(pos * div)
        table[:, 1::2] = torch.cos(pos * div[: table[:, 1::2].shape[1]])
        return table

    def forward(self, tokens, *, use_xma=False):
        del use_xma
        steps = int(tokens.shape[1])
        if steps > self.max_seq_len:
            raise ValueError(f"sequence length {steps} exceeds max_seq_len {self.max_seq_len}")
        x = self.embed(tokens) + self.pos_embed[:steps].to(tokens.device)[None, :, :]
        causal_mask = torch.ones((steps, steps), device=tokens.device, dtype=torch.bool).tril()
        for layer in self.layers:
            x = layer(x, causal_mask)
        return self.final_norm(x.float())

    def logits(self, tokens, *, use_xma=False):
        return self.forward(tokens, use_xma=use_xma) @ self.embed.weight.float().T

    def chunked_logits(self, tokens, *, use_xma=False, chunk_size=0,
                       detach_boundaries=False, remat_chunks=False):
        del chunk_size, detach_boundaries, remat_chunks
        return self.logits(tokens, use_xma=use_xma)

    @torch.no_grad()
    def init_states(self, batch_size, device):
        lengths = torch.zeros((int(batch_size),), device=device, dtype=torch.long)
        states = [lengths]
        for _ in self.layers:
            states.append(torch.zeros((int(batch_size), self.heads, self.max_seq_len, self.head_dim), device=device))
            states.append(torch.zeros((int(batch_size), self.heads, self.max_seq_len, self.head_dim), device=device))
        return states

    @torch.no_grad()
    def step(self, token, states, update_mask):
        lengths = states[0]
        next_lengths = lengths.clone()
        rows = torch.nonzero(update_mask, as_tuple=False).flatten()
        if rows.numel() > 0:
            if int(next_lengths[rows].max().item()) >= self.max_seq_len:
                raise ValueError(f"rollout sequence length exceeds max_seq_len {self.max_seq_len}")
            next_lengths[rows] += 1
        pos = next_lengths.clamp_min(1) - 1
        x = self.embed(token) + self.pos_embed[pos].to(token.device)
        next_states = [next_lengths]
        si = 1
        for layer in self.layers:
            x, kc, vc = layer.step(x, states[si], states[si + 1], next_lengths.clamp_min(1), update_mask)
            next_states.extend([kc, vc])
            si += 2
        logits = self.final_norm(x.float()) @ self.embed.weight.float().T
        return torch.where(update_mask[:, None], logits, torch.zeros_like(logits)), next_states

    def num_params(self) -> int:
        seen, n = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p)); n += p.numel()
        return n


__all__ = ["TransformerBlock", "TransformerLM"]
