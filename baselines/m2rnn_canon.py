"""Canonical M2RNN baseline (Mayank Mishra's lm-engine layer) + a chunkable twin.

The cell is the AUTHOR'S full stacked M2RNN block loaded at runtime from the
gitignored `refs/lm-engine` checkout (Apache-2.0; pinned commit in the README) --
NOT a reimplementation. SoftplusDecayGate, identity-init state_weight, short causal
conv, D skip, g-gating, and the xma triton kernel when on CUDA.

lm-engine's `causal_convolution` hard-asserts S==1 once a conv cache is passed, so
the canonical cell cannot be sequence-checkpointed as a black box. For chunked
training we use a TWIN that reuses the canonical module's exact submodules/params
but reorganizes forward so the conv carries a ring-buffer across chunks
(prepend-history valid conv). The twin is parity-tested bit-exact vs the canonical
forward (tests/test_chunk_equivalence.py): logits = canonical, chunked_logits = twin.
"""
from __future__ import annotations

import contextlib
import math
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .common import RMSNorm, REFS

_LM_ENGINE = REFS / "lm-engine"
_XMA = REFS / "xma"

_ENABLE_KERNELS = None
_KERNEL_M2RNN = None
_XMA_OK = False
_IS_KERNEL_ALLOWED = None
_M2RNN_KERNEL = None


def _load_m2rnn_block():
    global _ENABLE_KERNELS, _KERNEL_M2RNN, _XMA_OK, _IS_KERNEL_ALLOWED, _M2RNN_KERNEL
    if not (_LM_ENGINE / "lm_engine").exists():
        raise FileNotFoundError(
            f"{_LM_ENGINE} missing -- the canonical M2RNN baseline needs refs/lm-engine "
            "(see README for the pinned commit).")
    if (_XMA / "xma").exists() and str(_XMA) not in sys.path:
        sys.path.insert(0, str(_XMA))            # so lm-engine's is_xma_available() is True
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
    return M2RNN


def _kernel_ctx(x: torch.Tensor):
    if x.is_cuda and _XMA_OK and _ENABLE_KERNELS is not None:
        return _ENABLE_KERNELS([_KERNEL_M2RNN])
    return contextlib.nullcontext()


class ShimCache:
    """Duck-typed stand-in for lm-engine GenerationCache: per-layer (conv, ssm)."""

    def __init__(self) -> None:
        self.data: dict[int, tuple] = {}

    def get_cache(self, layer_idx: int):
        return self.data.get(int(layer_idx), (None, None))

    def update(self, *, conv_state=None, ssm_state=None, num_tokens_added=0, layer_idx=0):
        self.data[int(layer_idx)] = (conv_state, ssm_state)
        return self.data[int(layer_idx)]

    def snapshot(self):
        return {li: (None if c is None else c.clone(), None if h is None else h.clone())
                for li, (c, h) in self.data.items()}

    def blend(self, old, mask):
        for li, (c_new, h_new) in self.data.items():
            c_old, h_old = old.get(li, (None, None))
            self.data[li] = (_blend(mask, c_new, c_old), _blend(mask, h_new, h_old))


def _blend(mask, new, old):
    if new is None or old is None:
        return new
    m = mask.view(-1, *([1] * (new.dim() - 1)))
    return torch.where(m, new, old)


@dataclass
class M2RNNCanonConfig:
    vocab_size: int = 256
    d_model: int = 128
    n_layers: int = 2
    heads: int = 4              # num_v/f/g/weight heads (num_q=num_k=1)
    head_dim: int = 50          # k_head_dim = v_head_dim
    ffn_mult: float = 2.0
    kernel_size: int = 4
    gradient_clipping: float = 1.0


class M2RNNCanonBlock(nn.Module):
    def __init__(self, cfg: M2RNNCanonConfig, M2RNN, layer_idx: int) -> None:
        super().__init__()
        d = cfg.d_model
        self.layer_idx = int(layer_idx)
        self.attn_norm = RMSNorm(d)
        self.mixer = M2RNN(
            input_size=d, k_head_dim=cfg.head_dim, v_head_dim=cfg.head_dim, output_size=d,
            num_q_heads=1, num_k_heads=1, num_v_heads=cfg.heads, num_f_heads=cfg.heads,
            num_g_heads=cfg.heads, num_weight_heads=cfg.heads,
            use_residual=True, kernel_size=int(cfg.kernel_size), activation_function="silu",
            add_bias=False, gradient_clipping=float(cfg.gradient_clipping),
            initializer_range=0.02, m_width=1.0, init_method="normal", normalization_function="rmsnorm",
            A_init_min=0, A_init_max=16, dt_init_min=1e-3, dt_init_max=0.1, dt_init_floor=1e-4,
            num_layers=int(cfg.n_layers), layer_idx=int(layer_idx),
            use_depth_scaled_init=False, use_padding_free_transformer=False)
        hidden = int(math.ceil((d * cfg.ffn_mult) / 64.0) * 64)
        self.ffn_norm = RMSNorm(d)
        self.ffn_in = nn.Linear(d, 2 * hidden, bias=False)
        self.ffn_out = nn.Linear(hidden, d, bias=False)

    def _ffn(self, x):
        up, gate = self.ffn_in(x).chunk(2, dim=-1)
        return self.ffn_out(up * F.silu(gate))

    def forward(self, x, cache=None):
        xn = self.attn_norm(x)
        with _kernel_ctx(xn):
            y = self.mixer(xn, cache_params=cache, attention_mask=None, cu_seqlens=None, max_seqlen=None)
        x = x.float() + y.float()
        return x + self._ffn(self.ffn_norm(x)).float()

    # ---- chunkable twin (reuses self.mixer's exact submodules) ----
    def _conv_cont(self, x, tail):
        m = self.mixer
        xin = torch.cat([tail, x], dim=1)
        y = F.conv1d(xin.transpose(1, 2), m.conv1d.weight, m.conv1d.bias,
                     padding=0, groups=m.conv_dim).transpose(1, 2)
        return F.silu(y), xin[:, -(m.kernel_size - 1):, :]

    def _mixer_with_state(self, xn, state, *, use_xma):
        m = self.mixer
        conv_tail, h = state
        proj = m.input_projection(xn)
        x, f, g = proj.split((m.conv_dim, m.num_f_heads, m.g_shape), dim=-1)
        f = m.decay_gate(f, final_exponential=True, output_dtype=f.dtype)
        if m.kernel_size is not None:
            x, conv_tail = self._conv_cont(x, conv_tail)
        q, k, v = x.split((m.q_shape, m.k_shape, m.v_shape), dim=-1)
        q = q.view(*q.shape[:-1], m.num_q_heads, m.k_head_dim)
        k = k.view(*k.shape[:-1], m.num_k_heads, m.k_head_dim)
        v = v.view(*v.shape[:-1], m.num_v_heads, m.v_head_dim)
        if (use_xma and xn.is_cuda and _M2RNN_KERNEL is not None
                and _IS_KERNEL_ALLOWED is not None and _IS_KERNEL_ALLOWED(_KERNEL_M2RNN)):
            y, h = _M2RNN_KERNEL(query=q, key=k, value=v, weight=m.state_weight,
                                 forget_input=f, input_state=h, gradient_clipping=m.gradient_clipping)
        else:
            y, h = m._torch_forward(q=q, k=k, v=v, xf=f, h0=h,
                                    gradient_clipping=m.gradient_clipping, cu_seqlens=None, max_seqlen=None)
        if m.use_residual:
            y = y + v * m.D
        y = y.flatten(-2, -1)
        g = g.repeat_interleave(m.num_heads // m.num_g_heads, dim=-1)
        y = y * F.silu(g)
        y = m.g_norm(y)
        y = m.output_projection(y)
        return y, (conv_tail, h)

    def init_chunk_state(self, batch_size, device):
        m = self.mixer
        tail = torch.zeros(batch_size, m.kernel_size - 1, m.conv_dim, device=device,
                           dtype=m.input_projection.weight.dtype)
        return (tail, None)

    def forward_with_state(self, x, state, *, use_xma):
        xn = self.attn_norm(x)
        y, state = self._mixer_with_state(xn, state, use_xma=use_xma)
        x = x.float() + y.float()
        return x + self._ffn(self.ffn_norm(x)).float(), state


class M2RNNCanonLM(nn.Module):
    def __init__(self, vocab_size, d_model, layers, heads, head_dim, ffn_mult, *, kernel_size=4) -> None:
        super().__init__()
        M2RNN = _load_m2rnn_block()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        cfg = M2RNNCanonConfig(vocab_size=int(vocab_size), d_model=int(d_model), n_layers=int(layers),
                               heads=int(heads), head_dim=int(head_dim), ffn_mult=float(ffn_mult),
                               kernel_size=int(kernel_size))
        self.cfg = cfg
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.blocks = nn.ModuleList(M2RNNCanonBlock(cfg, M2RNN, i) for i in range(int(layers)))
        self.final_norm = RMSNorm(self.d_model)
        nn.init.normal_(self.embed.weight, std=self.d_model ** -0.5)

    def hidden(self, tokens, *, use_xma=False):
        del use_xma
        h = self.embed(tokens)
        for b in self.blocks:
            h = b(h, cache=None)
        return self.final_norm(h.float())

    def logits(self, tokens, *, use_xma=False):
        return self.hidden(tokens, use_xma=use_xma) @ self.embed.weight.float().T

    def chunked_logits(self, tokens, *, use_xma=False, chunk_size=0,
                       detach_boundaries=False, remat_chunks=False):
        """sqrt-exact BPTT on the twin: carry (conv ring-buffer, recurrent state h)
        per block across chunks, checkpoint each chunk. The xma kernel backprops into
        input_state (dht supported), so detach_boundaries False is exact."""
        T = int(tokens.shape[1])
        if int(chunk_size) <= 0 or int(chunk_size) >= T:
            return self.logits(tokens, use_xma=use_xma)
        nb = len(self.blocks)
        states = [b.init_chunk_state(int(tokens.shape[0]), tokens.device) for b in self.blocks]

        def flatten(sts):
            flat = []
            for tail, h in sts:
                flat.extend([tail, h])
            return flat

        def unflatten(flat):
            return [(flat[2 * i], flat[2 * i + 1]) for i in range(nb)]

        def run_chunk(chunk_tokens, *flat):
            # kernel ctx must be INSIDE: checkpoint recomputes this during backward.
            with _kernel_ctx(self.embed.weight):
                x = self.embed(chunk_tokens)
                nxt = []
                for b, st in zip(self.blocks, unflatten(flat)):
                    x, st = b.forward_with_state(x, st, use_xma=use_xma)
                    nxt.append(st)
                logits = self.final_norm(x.float()) @ self.embed.weight.float().T
            return (logits, *flatten(nxt))

        outs = []
        for start in range(0, T, int(chunk_size)):
            chunk = tokens[:, start:start + int(chunk_size)]
            flat = flatten(states)
            if remat_chunks and torch.is_grad_enabled():
                result = checkpoint(run_chunk, chunk, *flat, use_reentrant=False)
            else:
                result = run_chunk(chunk, *flat)
            outs.append(result[0])
            states = unflatten(list(result[1:]))
            if detach_boundaries:
                states = [(t.detach(), None if h is None else h.detach()) for t, h in states]
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def init_states(self, batch_size, device) -> ShimCache:
        del batch_size, device
        return ShimCache()

    @torch.no_grad()
    def step(self, token, states: ShimCache, update_mask):
        was_training = self.training
        self.eval()
        mask = update_mask.bool()
        old = states.snapshot()
        h = self.embed(token).unsqueeze(1)
        for b in self.blocks:
            h = b(h, cache=states)
        logits = self.final_norm(h[:, -1].float()) @ self.embed.weight.float().T
        states.blend(old, mask)
        if was_training:
            self.train()
        return torch.where(mask[:, None], logits, torch.zeros_like(logits)), states

    def num_params(self) -> int:
        seen, n = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p)); n += p.numel()
        return n


__all__ = ["M2RNNCanonLM", "ShimCache"]
