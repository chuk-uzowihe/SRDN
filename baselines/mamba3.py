"""Mamba-3 baseline (fla.layers.Mamba3).

The most recent Mamba. Parallelizable (TC0-limited) like the Transformer, so it is
a "recurrence-incomplete" reference: it has an efficient internal-chunk kernel and
O(1) state, so chunked_logits is the full-sequence pass (no exact state-carry BPTT
is needed -- that machinery is for the recurrence-complete models). Rollout uses
the FLA Cache (recurrent + conv state) like GDN-2.

Requires fla pinned to the commit with Mamba3 (see README; PyPI 0.5.0 lacks fla.ops).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers import Mamba3
from fla.models.utils import Cache

from .common import RMSNorm
from .gdn2 import _blend_caches, _reset_cache_rows


class Mamba3Block(nn.Module):
    def __init__(self, d_model, ffn_mult, *, state_size, expand, head_dim, n_groups,
                 chunk_size, layer_idx) -> None:
        super().__init__()
        self.norm = RMSNorm(int(d_model))
        self.mixer = Mamba3(hidden_size=int(d_model), state_size=int(state_size), expand=int(expand),
                            head_dim=int(head_dim), n_groups=int(n_groups), chunk_size=int(chunk_size),
                            layer_idx=int(layer_idx))
        self.use_ffn = float(ffn_mult) > 0.0
        if self.use_ffn:
            hidden = int(round(int(d_model) * float(ffn_mult)))
            self.ffn_norm = RMSNorm(int(d_model))
            self.ffn_in = nn.Linear(int(d_model), 2 * hidden, bias=False)
            self.ffn_out = nn.Linear(hidden, int(d_model), bias=False)

    def forward(self, x, *, past_key_values=None, use_cache=False, cu_seqlens=None):
        dtype = next(self.mixer.parameters()).dtype
        y, _, past_key_values = self.mixer(self.norm(x).to(dtype), past_key_values=past_key_values,
                                           use_cache=use_cache, cu_seqlens=cu_seqlens)
        x = x.float() + y.float()
        if self.use_ffn:
            ff = self.ffn_norm(x)
            up, gate = self.ffn_in(ff.to(self.ffn_in.weight.dtype)).chunk(2, dim=-1)
            x = x + self.ffn_out(up * F.silu(gate)).float()
        return x, past_key_values


class Mamba3LM(nn.Module):
    def __init__(self, vocab_size, d_model, layers, ffn_mult, *, state_size=128, expand=2,
                 head_dim=64, n_groups=1, chunk_size=64) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.layers = nn.ModuleList(
            Mamba3Block(d_model, ffn_mult, state_size=state_size, expand=expand, head_dim=head_dim,
                        n_groups=n_groups, chunk_size=chunk_size, layer_idx=i) for i in range(int(layers)))
        self.final_norm = RMSNorm(self.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, tokens, *, cu_seqlens=None):
        x = self.embed(tokens)
        cache = Cache()
        for layer in self.layers:
            x, cache = layer(x, past_key_values=cache, use_cache=False, cu_seqlens=cu_seqlens)
        return self.final_norm(x.float())

    def logits(self, tokens, *, use_xma=False, cu_seqlens=None):
        del use_xma
        return self.forward(tokens, cu_seqlens=cu_seqlens) @ self.embed.weight.float().T

    def chunked_logits(self, tokens, *, use_xma=False, chunk_size=0,
                       detach_boundaries=False, remat_chunks=False):
        # parallelizable + memory-efficient kernel: full-seq (like the Transformer baseline).
        del use_xma, chunk_size, detach_boundaries, remat_chunks
        return self.logits(tokens)

    @torch.no_grad()
    def init_states(self, batch_size, device) -> Cache:
        del batch_size, device
        return Cache()

    @torch.no_grad()
    def step(self, token, cache: Cache, update_mask, reset_mask=None):
        if token.ndim != 1:
            raise ValueError("token must have shape [batch]")
        update = update_mask.to(device=token.device, dtype=torch.bool)
        reset = torch.zeros_like(update) if reset_mask is None else reset_mask.to(device=token.device, dtype=torch.bool)
        old_cache = _reset_cache_rows(cache, reset)
        new_cache = Cache.from_legacy_cache(old_cache.to_legacy_cache())
        x = self.embed(token).unsqueeze(1)
        for layer in self.layers:
            x, new_cache = layer(x, past_key_values=new_cache, use_cache=True)
        logits = self.final_norm(x[:, -1].float()) @ self.embed.weight.float().T
        blended = _blend_caches(old_cache, new_cache, update)
        return torch.where(update[:, None], logits, torch.zeros_like(logits)), blended

    def num_params(self) -> int:
        seen, n = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p)); n += p.numel()
        return n


__all__ = ["Mamba3LM", "Mamba3Block"]
