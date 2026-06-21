"""Mamba-3 token mixer (fla.layers.Mamba3; needs the mamba-ssm SISO kernels).

Parallelizable / TC0-limited like attention. chunkable=False -> core trains it
full-seq; rollout uses the FLA Cache. Owns its pre-norm, returns the residual delta.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from fla.layers import Mamba3
from fla.models.utils import Cache

from srdn.norm import RMSNorm


def _one_layer_cache(state) -> Cache:
    return Cache() if state is None else Cache.from_legacy_cache((state,))


class Mamba3Mixer(nn.Module):
    chunkable = False

    def __init__(self, d_model, *, state_size=128, expand=2, head_dim=64, n_groups=1, chunk_size=64) -> None:
        super().__init__()
        self.norm = RMSNorm(int(d_model))
        self.mixer = Mamba3(hidden_size=int(d_model), state_size=int(state_size), expand=int(expand),
                            head_dim=int(head_dim), n_groups=int(n_groups), chunk_size=int(chunk_size),
                            layer_idx=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = next(self.mixer.parameters()).dtype
        y, _, _ = self.mixer(self.norm(x).to(dtype), past_key_values=Cache(), use_cache=False)
        return y.float()

    def init_state(self, B, device):
        del B, device
        return None

    def step(self, x_t, state):
        dtype = next(self.mixer.parameters()).dtype
        cache = _one_layer_cache(state)
        x = self.norm(x_t).unsqueeze(1).to(dtype)
        y, _, cache = self.mixer(x, past_key_values=cache, use_cache=True)
        return y[:, 0].float(), cache[0]


__all__ = ["Mamba3Mixer"]
