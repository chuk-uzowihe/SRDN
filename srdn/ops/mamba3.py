"""Mamba-3 token mixer (fla.layers.Mamba3; needs the mamba-ssm SISO kernels).

Parallelizable / TC0-limited like attention. Full-sequence forward/logits work
(FRJT + enwik8); chunkable=False so core trains it full-seq.

ROLLOUT (.step) IS UNAVAILABLE: incremental single-token decode routes to
mamba-ssm's cute step kernel, which errors on multi-step (arg-#2 conv-state dtype)
at fp32/bf16/fp16 alike -- an upstream mamba-ssm bug, independent of the fla version
(reproduced on fla git-main too). fla's chunk path can't carry an initial recurrent
state, so there's no force-chunk decode (the trick that fixes GDN-2). Mamba-3 is
therefore a forward-only baseline here (FRJT/enwik8); the Transformer covers the
parallelizable baseline on graph-RL, which needs rollout.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from fla.layers import Mamba3
from fla.models.utils import Cache

from srdn.core import RMSNorm


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
        raise NotImplementedError(
            "Mamba-3 single-token rollout is unavailable: mamba-ssm's cute step kernel "
            "errors on multi-step decode (upstream bug, fla-version-independent). "
            "Mamba-3 is a forward-only baseline (FRJT/enwik8); it does not run graph-RL "
            "rollout. Use the Transformer as the parallelizable graph-RL baseline."
        )


__all__ = ["Mamba3Mixer"]
