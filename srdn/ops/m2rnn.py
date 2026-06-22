"""Canonical M2RNN token mixer (lm-engine cell) + a parity-verified chunkable twin.

The cell is Mayank Mishra's full stacked M2RNN block, loaded from refs/lm-engine
(Apache-2.0; pinned commit in the README) -- NOT a reimplementation. SoftplusDecayGate,
identity-init state_weight, short causal conv, D skip, g-gating, xma triton kernel.

lm-engine's causal_convolution hard-asserts S==1 once a conv cache is passed, so the
canonical cell cannot be sequence-checkpointed as a black box:
  forward            -> canonical path (kernel ctx)
  step               -> canonical decode via a ShimCache (S==1 conv path is fine)
  forward_with_state -> TWIN: reuses the cell's exact submodules/params but does the
                        conv as a prepend-history valid conv so it continues across
                        chunks. Parity-tested bit-exact vs the canonical forward.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from srdn.core import RMSNorm

_REFS = Path(__file__).resolve().parents[2] / "refs"
_LM_ENGINE = _REFS / "lm-engine"
_XMA = _REFS / "xma"

_ENABLE_KERNELS = None
_KERNEL_M2RNN = None
_XMA_OK = False
_IS_KERNEL_ALLOWED = None
_M2RNN_KERNEL = None
_M2RNN_CLS = None


def _load_m2rnn():
    global _ENABLE_KERNELS, _KERNEL_M2RNN, _XMA_OK, _IS_KERNEL_ALLOWED, _M2RNN_KERNEL, _M2RNN_CLS
    if _M2RNN_CLS is not None:
        return _M2RNN_CLS
    if not (_LM_ENGINE / "lm_engine").exists():
        raise FileNotFoundError(f"{_LM_ENGINE} missing -- clone lm-engine to refs/ (README has the pin).")
    if (_XMA / "xma").exists() and str(_XMA) not in sys.path:
        sys.path.insert(0, str(_XMA))                    # so is_xma_available() is True
    if str(_LM_ENGINE) not in sys.path:
        sys.path.insert(0, str(_LM_ENGINE))
    from lm_engine.hf_models.modeling_utils.sequence_mixer_blocks.m2rnn import M2RNN
    from lm_engine.kernels import enable_kernels, is_kernel_allowed
    from lm_engine.enums import Kernel
    from lm_engine.utils import is_xma_available
    _ENABLE_KERNELS, _KERNEL_M2RNN, _XMA_OK = enable_kernels, Kernel.m2rnn, is_xma_available()
    _IS_KERNEL_ALLOWED = is_kernel_allowed
    if _XMA_OK:
        from xma import m2rnn as _k
        _M2RNN_KERNEL = _k
    _M2RNN_CLS = M2RNN
    return M2RNN


def _kernel_ctx(x):
    if x.is_cuda and _XMA_OK and _ENABLE_KERNELS is not None:
        return _ENABLE_KERNELS([_KERNEL_M2RNN])
    return contextlib.nullcontext()


class M2RNNMixer(nn.Module):
    chunkable = True

    def __init__(self, d_model, n_heads, head_dim, *, kernel_size=4, gradient_clipping=1.0,
                 n_layers=1, layer_idx=0) -> None:
        super().__init__()
        M2RNN = _load_m2rnn()
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

    # canonical forward (lm-engine owns the cell's pre-norm internally via g_norm; the
    # mixer expects already-normalized input -> we feed x directly, as in the baseline)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        with _kernel_ctx(xn):
            return self.mixer(xn, cache_params=None, attention_mask=None,
                              cu_seqlens=None, max_seqlen=None).float()

    # ---- twin (chunk-correct conv) ----
    def _conv_cont(self, x, tail):
        m = self.mixer
        xin = torch.cat([tail, x], dim=1)
        y = F.conv1d(xin.transpose(1, 2), m.conv1d.weight, m.conv1d.bias,
                     padding=0, groups=m.conv_dim).transpose(1, 2)
        return F.silu(y), xin[:, -(m.kernel_size - 1):, :]

    def _twin(self, x, tail, h):
        m = self.mixer
        proj = m.input_projection(x)
        z, f, g = proj.split((m.conv_dim, m.num_f_heads, m.g_shape), dim=-1)
        f = m.decay_gate(f, final_exponential=True, output_dtype=f.dtype)
        if m.kernel_size is not None:
            z, tail = self._conv_cont(z, tail)
        q, k, v = z.split((m.q_shape, m.k_shape, m.v_shape), dim=-1)
        q = q.view(*q.shape[:-1], m.num_q_heads, m.k_head_dim)
        k = k.view(*k.shape[:-1], m.num_k_heads, m.k_head_dim)
        v = v.view(*v.shape[:-1], m.num_v_heads, m.v_head_dim)
        use_k = (x.is_cuda and _M2RNN_KERNEL is not None and _IS_KERNEL_ALLOWED is not None
                 and _IS_KERNEL_ALLOWED(_KERNEL_M2RNN))
        if use_k:
            y, h = _M2RNN_KERNEL(query=q, key=k, value=v, weight=m.state_weight,
                                 forget_input=f, input_state=h, gradient_clipping=m.gradient_clipping)
        else:
            y, h = m._torch_forward(q=q, k=k, v=v, xf=f, h0=h,
                                    gradient_clipping=m.gradient_clipping, cu_seqlens=None, max_seqlen=None)
        if m.use_residual:
            y = y + v * m.D
        y = y.flatten(-2, -1)
        g = g.repeat_interleave(m.num_heads // m.num_g_heads, dim=-1)
        y = m.g_norm(y * F.silu(g))
        return m.output_projection(y), tail, h

    def init_state(self, B, device):
        m = self.mixer
        tail = torch.zeros(B, m.kernel_size - 1, m.conv_dim, device=device,
                           dtype=m.input_projection.weight.dtype)
        return (tail, None)

    def forward_with_state(self, x, state):
        tail, h = state
        xn = self.norm(x)
        with _kernel_ctx(xn):
            y, tail, h = self._twin(xn, tail, h)
        return y.float(), (tail, h)

    def flatten_state(self, state):
        tail, h = state
        return [tail, h]

    def unflatten_state(self, flat):
        return (flat[0], flat[1])

    # ---- rollout: the twin with T=1 (same state format as chunking; the twin is
    #      parity-equal to the canonical forward, so step stays consistent with both
    #      logits() and chunked_logits()). ----
    def step(self, x_t, state):
        tail, h = state
        xn = self.norm(x_t).unsqueeze(1)
        with _kernel_ctx(xn):
            y, tail, h = self._twin(xn, tail, h)
        return y[:, 0].float(), (tail, h)


__all__ = ["M2RNNMixer"]
