"""M2RNN token mixer: Mayank Mishra's full stacked M2RNN cell, loaded from refs/lm-engine
(Apache-2.0; pinned commit in the README) -- NOT a reimplementation. SoftplusDecayGate,
identity-init state_weight, short causal conv, D skip, g-gating, xma triton kernel.

Full-sequence forward only (chunkable=False, no rollout step): lm-engine's
causal_convolution hard-asserts S==1 once a conv cache is passed, so the reference cell
cannot carry state across chunks as a black box, and nothing in the tasks needs it to.
"""
from __future__ import annotations

import contextlib
import sys
from functools import cache
from pathlib import Path

import torch
import torch.nn as nn

from srdn.core import RMSNorm

_REFS = Path(__file__).resolve().parents[2] / "refs"


@cache
def _load_m2rnn():
    """Import the reference cell (plus its optional xma kernel) from refs/. Returns
    (M2RNN class, kernel context factory)."""
    lm_engine = _REFS / "lm-engine"
    if not (lm_engine / "lm_engine").exists():
        raise FileNotFoundError(f"{lm_engine} missing -- clone lm-engine to refs/ (README has the pin).")
    xma = _REFS / "xma"
    if (xma / "xma").exists() and str(xma) not in sys.path:
        sys.path.insert(0, str(xma))                     # so is_xma_available() is True
    if str(lm_engine) not in sys.path:
        sys.path.insert(0, str(lm_engine))
    from lm_engine.hf_models.modeling_utils.sequence_mixer_blocks.m2rnn import M2RNN
    from lm_engine.kernels import enable_kernels
    from lm_engine.enums import Kernel
    from lm_engine.utils import is_xma_available

    if is_xma_available():
        def kernel_ctx(x):
            return enable_kernels([Kernel.m2rnn]) if x.is_cuda else contextlib.nullcontext()
    else:
        def kernel_ctx(x):
            return contextlib.nullcontext()
    return M2RNN, kernel_ctx


class M2RNNMixer(nn.Module):
    chunkable = False

    def __init__(self, d_model, n_heads, head_dim, *, kernel_size=4, gradient_clipping=1.0,
                 n_layers=1, layer_idx=0) -> None:
        super().__init__()
        M2RNN, self._kernel_ctx = _load_m2rnn()
        d = int(d_model)
        self.norm = RMSNorm(d)                            # input pre-norm (lm-engine g_norm is on output)
        self.mixer = M2RNN(
            input_size=d, k_head_dim=int(head_dim), v_head_dim=int(head_dim), output_size=d,
            num_q_heads=1, num_k_heads=1, num_v_heads=int(n_heads), num_f_heads=int(n_heads),
            num_g_heads=int(n_heads), num_weight_heads=int(n_heads),
            use_residual=True, kernel_size=int(kernel_size), activation_function="silu",
            add_bias=False, gradient_clipping=float(gradient_clipping),
            initializer_range=0.02, m_width=1.0, init_method="normal", normalization_function="rmsnorm",
            A_init_min=0, A_init_max=16, dt_init_min=1e-3, dt_init_max=0.1, dt_init_floor=1e-4,
            num_layers=int(n_layers), layer_idx=int(layer_idx),
            use_depth_scaled_init=False, use_padding_free_transformer=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        with self._kernel_ctx(xn):
            return self.mixer(xn, cache_params=None, attention_mask=None,
                              cu_seqlens=None, max_seqlen=None).float()


__all__ = ["M2RNNMixer"]
