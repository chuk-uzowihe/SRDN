"""Mamba-2 token mixer (official mamba_ssm.modules.mamba2.Mamba2).

Parallelizable / TC0-limited like attention. chunkable=False -> core trains it
full-seq; rollout uses incremental decode. Owns its pre-norm, returns the residual delta.

use_mem_eff_path=False always: the fused prefill path hard-requires the causal_conv1d
package (not installed -- fla's Mamba2 layer has the same requirement with no fallback,
which is why we use the official cell). The non-fused path falls back to F.conv1d and
runs the same mamba_chunk_scan_combined kernel; at our scale the speed gap is noise.

Decode (step): drives Mamba2.step (Triton selective_state_update + conv-buffer roll)
directly on a (conv_state, ssm_state) pair, skipping mamba_ssm's InferenceParams
plumbing. Mamba2.step mutates both states IN PLACE, so we clone before stepping --
core._merge_state needs old and new to stay distinct for masked rollout. Verified
equal to full-seq forward by tests/test_step_parallel.py (--arch mamba2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm.modules.mamba2 import Mamba2

from srdn.core import RMSNorm


class Mamba2Mixer(nn.Module):
    chunkable = False

    def __init__(self, d_model, *, state_size=128, expand=2, head_dim=64, n_groups=1,
                 chunk_size=64) -> None:
        super().__init__()
        self.norm = RMSNorm(int(d_model))
        self.mixer = Mamba2(d_model=int(d_model), d_state=int(state_size), expand=int(expand),
                            headdim=int(head_dim), ngroups=int(n_groups),
                            chunk_size=int(chunk_size), use_mem_eff_path=False, layer_idx=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.mixer.in_proj.weight.dtype
        return self.mixer(self.norm(x).to(dtype)).float()

    def init_state(self, B, device):
        # None == "no recurrent state yet"; the first step() allocates zeros (the same
        # contract core._merge_state documents for library caches). Zero conv/ssm states
        # match the full-seq forward's implicit zero left-pad / zero initial SSM state.
        del B, device
        return None

    def step(self, x_t, state):
        """Single-token decode. x_t: [B, D]. state: (conv_state, ssm_state) or None."""
        m = self.mixer
        dtype = m.in_proj.weight.dtype
        if state is None:
            conv, ssm = m.allocate_inference_cache(x_t.shape[0], 1, dtype=dtype)
        else:
            conv, ssm = state[0].clone(), state[1].clone()  # m.step mutates in place
        y, _, _ = m.step(self.norm(x_t).to(dtype).unsqueeze(1), conv, ssm)
        return y[:, 0].float(), (conv, ssm)


__all__ = ["Mamba2Mixer"]
