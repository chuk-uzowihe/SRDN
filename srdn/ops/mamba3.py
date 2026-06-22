"""Mamba-3 token mixer (fla.layers.Mamba3; needs the mamba-ssm SISO kernels).

Parallelizable / TC0-limited like attention. chunkable=False -> core trains it
full-seq; rollout uses the FLA Cache. Owns its pre-norm, returns the residual delta.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from fla.layers import Mamba3
from fla.models.utils import Cache

from srdn.core import RMSNorm


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
        # fla's Mamba-3 single-token decode routes to mamba3_step_fn (the cute decode
        # kernel), which is broken at our pin: it cannot consume the state the combined
        # (prefill) kernel produces -- mixed bf16/fp32, then a stride mismatch when coerced.
        # Verified independent of our cache handling (canonical persistent-Cache fails
        # identically) and across fla history. Forward/training (the combined kernel) is
        # fine, so Mamba-3 still runs on the full-sequence tasks (FRJT, enwik8); only the
        # incremental-rollout task (graph-RL) is affected, and Mamba-3 is parallelizable
        # so it is not the protagonist there. Routing decode through the combined kernel
        # is possible in principle (Input_States=) but its state contract is undocumented.
        raise NotImplementedError(
            "Mamba-3 incremental decode is unavailable (fla mamba3_step_fn kernel bug at "
            "the pinned fla commit). Mamba-3 supports the full-sequence tasks (FRJT, enwik8) "
            "but not the streaming-rollout task (graph-RL)."
        )


__all__ = ["Mamba3Mixer"]
