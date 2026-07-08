"""Mamba-3 token mixer (fla.layers.Mamba3; needs the mamba-ssm SISO kernels).

Parallelizable / TC0-limited like attention. chunkable=False -> core trains it
full-seq; rollout uses incremental decode. Owns its pre-norm, returns the residual delta.

Decode (step): fla's own single-token path routes to the broken CUTE kernel
(mamba3_step_fn) at our pin -- it can't consume the state the combined (prefill)
kernel emits. We bypass it: incremental decode runs the *working* combined kernel
`mamba3_siso_combined` at seq-len 1 with `Input_States=` (the carried recurrent
state) + `return_final_states=True`. Verified equal to full-seq forward by
tests/test_step_parallel.py (--arch mamba3).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.layers import Mamba3
from fla.layers.mamba3 import mamba3_siso_combined
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
        if self.mixer.is_mimo:
            raise NotImplementedError("Mamba3Mixer decode supports the SISO path only (is_mimo=False).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = next(self.mixer.parameters()).dtype
        y, _, _ = self.mixer(self.norm(x).to(dtype), past_key_values=Cache(), use_cache=False)
        return y.float()

    def init_state(self, B, device):
        # None == "no recurrent state yet"; the first step() populates it (the same
        # contract core._merge_state documents for library caches). Passing Input_States=None
        # on the first token is also the correct cold-start (no prior context).
        del B, device
        return None

    def step(self, x_t, state):
        """Single-token decode via the combined kernel + Input_States. x_t: [B, D].

        Mirrors fla.Mamba3.cuda_kernels_forward's SISO combined branch at seq-len 1,
        but feeds the carried state in (fla's own combined call hard-codes Input_States=None
        and routes decode to the broken CUTE step kernel instead).
        """
        m = self.mixer
        dtype = next(m.parameters()).dtype
        hs = self.norm(x_t).to(dtype).unsqueeze(1)  # [B, 1, D]

        z, x, B, C, dd_dt, dd_A, trap, angles = m._project_and_split(hs)
        z = rearrange(z, "b l (h p) -> b l h p", p=m.head_dim)
        x = rearrange(x, "b l (h p) -> b l h p", p=m.head_dim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=m.mimo_rank, g=m.n_groups)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=m.mimo_rank, g=m.n_groups)
        trap = rearrange(trap, "b l h -> b h l")

        A = -F.softplus(dd_A.to(torch.float32)).clamp(max=-m.A_floor)
        DT = F.softplus(dd_dt + m.dt_bias)
        ADT = A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")
        angles = angles.unsqueeze(-2).expand(-1, -1, m.num_heads, -1).to(torch.float32)
        B = m.B_norm(B)
        C = m.C_norm(C)

        # cache K-state carries a rank dim ([B, rank, H, N]); the kernel wants [B, H, N].
        in_states = None
        if state is not None:
            angle_s, ssm_s, k_s, v_s = state
            in_states = (angle_s, ssm_s, k_s.squeeze(1), v_s)

        y, last_angle, last_ssm, last_k, last_v = mamba3_siso_combined(
            Q=C.squeeze(2), K=B.squeeze(2), V=x, ADT=ADT, DT=DT, Trap=trap,
            Q_bias=m.C_bias.squeeze(1), K_bias=m.B_bias.squeeze(1), Angles=angles,
            D=m.D, Z=z if not m.is_outproj_norm else None,
            chunk_size=m.chunk_size, Input_States=in_states,
            return_final_states=True, cu_seqlens=None,
        )
        last_k = last_k.unsqueeze(1)  # re-add rank dim for the cache layout
        y = rearrange(y, "b l h p -> b l (h p)")
        if m.is_outproj_norm:
            y = m.norm(y, rearrange(z, "b l h p -> b l (h p)"))
        out = m.out_proj(y.to(x.dtype)).squeeze(1).float()  # [B, D]
        return out, (last_angle, last_ssm, last_k, last_v)


__all__ = ["Mamba3Mixer"]
