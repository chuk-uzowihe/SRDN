"""Gated DeltaNet (GDN-1) token mixer (fla.layers.GatedDeltaNet).

Unlike GDN-2 (NVIDIA NC-licensed, loaded from refs/), GDN-1 ships in fla proper at
our pin -- a plain import, no shim. Owns its pre-norm, returns the residual delta.

chunkable=True: state = a single-layer FLA Cache (recurrent_state + ShortConvolution
conv_state, both chunk-correct via initial_state/output_final_state). Rollout forces
the `chunk` kernel like gdn2 does: fla's eval-mode S<=64 fused_recurrent route is a
different kernel than the training forward, and forcing train mode keeps step ==
full-seq to chunk-kernel tolerance (verified by tests/test_step_parallel.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from fla.layers import GatedDeltaNet
from fla.models.utils import Cache

from srdn.core import RMSNorm


def _one_layer_cache(state) -> Cache:
    """Wrap a single layer's (recurrent_state, conv_state) dict as a layer-0 Cache."""
    if state is None:
        return Cache()
    return Cache.from_legacy_cache((state,))


class GDN1Mixer(nn.Module):
    chunkable = True

    def __init__(self, d_model, n_heads, head_dim, *, expand_v=1.0, use_short_conv=True,
                 allow_neg_eigval=True) -> None:
        super().__init__()
        self.norm = RMSNorm(int(d_model))
        self.mixer = GatedDeltaNet(hidden_size=int(d_model), expand_v=float(expand_v),
                                   head_dim=int(head_dim), num_heads=int(n_heads), mode="chunk",
                                   use_short_conv=bool(use_short_conv),
                                   allow_neg_eigval=bool(allow_neg_eigval), layer_idx=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.mixer.q_proj.weight.dtype
        y, _, _ = self.mixer(self.norm(x).to(dtype), past_key_values=Cache(), use_cache=False)
        return y.float()

    # chunked: state IS the single-layer legacy cache dict (or None at start)
    def init_state(self, B, device):
        del B, device
        return None

    def forward_with_state(self, x, state):
        dtype = self.mixer.q_proj.weight.dtype
        cache = _one_layer_cache(state)
        old_training = self.training
        self.train()                                     # force chunk kernel
        y, _, cache = self.mixer(self.norm(x).to(dtype), past_key_values=cache, use_cache=True)
        self.train(old_training)
        return y.float(), cache[0]

    def flatten_state(self, state):
        if state is None:
            return [None, None, None, None]
        conv = state.get("conv_state")
        cq, ck, cv = (None, None, None) if conv is None else conv
        return [state.get("recurrent_state"), cq, ck, cv]

    def unflatten_state(self, flat):
        rs, cq, ck, cv = flat
        if rs is None and cq is None:
            return None
        return {"recurrent_state": rs, "conv_state": None if cq is None else (cq, ck, cv)}

    def step(self, x_t, state):
        dtype = self.mixer.q_proj.weight.dtype
        cache = _one_layer_cache(state)
        old_training = self.training
        self.train()                                     # force chunk kernel
        x = self.norm(x_t).unsqueeze(1).to(dtype)
        y, _, cache = self.mixer(x, past_key_values=cache, use_cache=True)
        self.train(old_training)
        return y[:, 0].float(), cache[0]


__all__ = ["GDN1Mixer"]
