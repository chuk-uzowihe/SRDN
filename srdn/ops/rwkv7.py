"""RWKV-7 ("Goose") token mixer (fla.layers.RWKV7Attention).

Same cell class as GDN-2: a chunkable matrix-state linear attention whose state is an fla
Cache of recurrent_state + a token-shift conv_state, with a chunk kernel (training) and a
fused-recurrent decode kernel (rollout; fla auto-selects it at seq_len < 64 in eval). Owns
its pre-norm, returns the residual delta. CUDA-only (triton kernels).

v_first -- RWKV-7's cross-layer value residual: layer 0 *produces* v_first (it has no
v_lora), layers >0 *consume* it (lerp their value toward v_first via v_lora). fla builds
the right params from `layer_idx` and its forward takes/returns v_first, so we (a) pass the
true layer_idx per layer and (b) thread v_first through a shared VFirstBus across the stack's
RWKV layers: layer 0 writes, later layers read. Correct under the stack's sequential
execution -- v_first is a per-token residual across *depth* (not a recurrent state across
time), recomputed every forward / chunk / step, so no cross-call carry is needed.
build_rwkv7 wires one bus + the true layer_idx per layer.

M1: chunkable=False (full-seq forward + token-by-token decode -- enough for enwik8 / FRJT
and the step==full-seq gate). The chunkable sqrt-BPTT path (forward_with_state / flatten /
unflatten, like GDN2Mixer) is M2, for long-sequence graph-RL.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import fla.layers.rwkv7 as _fla_rwkv7
from fla.layers import RWKV7Attention
from fla.models.rwkv7.modeling_rwkv7 import RWKV7FeedForward as _RWKV7FeedForward
from fla.models.utils import Cache

from srdn.core import RMSNorm

# fla's RWKV7Attention.forward hardcodes the training chunk kernel at chunk_size=64, which
# needs ~128 KB of shared memory. Consumer Ampere/Ada (RTX 3070 Ti / 4090, sm_86/89) cap dynamic
# shared memory at ~100 KB, so chunk_size=64 OOMs there; datacenter GPUs (A100 164 KB, H100 227 KB)
# fit it. chunk_size is a pure tiling parameter (the chunked recurrence is mathematically identical
# for any chunk_size; verified fwd+bwd at 32), so on cards that can't fit 64 we shim the symbol
# fla's forward calls to force 32. On A100/H100 we leave the faster native 64. Decode
# (fused_mul_recurrent) is a different kernel and is untouched. Only the rwkv7 BASELINE uses this
# kernel; srdn's state-reading scan is pure-torch and unaffected either way.
def _chunk64_fits() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False  # no CUDA at import -> assume the safe (32) path
        return torch.cuda.get_device_properties(0).shared_memory_per_multiprocessor >= 128 * 1024
    except Exception:
        return False


if not _chunk64_fits() and not getattr(_fla_rwkv7, "_srdn_chunk32_patched", False):
    _orig_chunk_rwkv7 = _fla_rwkv7.chunk_rwkv7

    def _chunk_rwkv7_cs32(*args, **kwargs):
        kwargs["chunk_size"] = 32
        return _orig_chunk_rwkv7(*args, **kwargs)

    _fla_rwkv7.chunk_rwkv7 = _chunk_rwkv7_cs32
    _fla_rwkv7._srdn_chunk32_patched = True


class VFirstBus:
    """Shared one-slot carrier for RWKV-7's v_first across a stack's RWKV layers."""
    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = None


def _cache_at(state, layer_idx) -> Cache:
    """Wrap a single layer's (recurrent_state, conv_state) dict as a Cache with the state at
    slot `layer_idx`. fla indexes get/update_layer_cache by the mixer's layer_idx -- which we
    keep truthful (layer 0 produces v_first, layers >0 consume it via v_lora) -- so a layer>0
    mixer with a naive slot-0 Cache would miss its state and silently run stateless every step.
    Cache.update auto-extends (filling slots 0..layer_idx-1 with None), so seed via update at
    the true slot. offset=0: the forward's own update advances the token count."""
    cache = Cache()
    if state is not None:
        cache.update(layer_idx=int(layer_idx), offset=0,
                     recurrent_state=state.get("recurrent_state"),
                     conv_state=state.get("conv_state"))
    return cache


class RWKV7Mixer(nn.Module):
    chunkable = True  # sqrt-exact BPTT: state = the single-layer (recurrent_state, conv_state)

    def __init__(self, d_model, *, head_dim=64, layer_idx=0, num_layers=1, value_dim=None,
                 bus=None) -> None:
        super().__init__()
        self.norm = RMSNorm(int(d_model))
        self.layer_idx = int(layer_idx)
        self.bus = bus if bus is not None else VFirstBus()
        self.mixer = RWKV7Attention(
            mode="chunk", hidden_size=int(d_model), head_dim=int(head_dim),
            layer_idx=int(layer_idx), num_hidden_layers=int(num_layers),
            value_dim=None if value_dim is None else int(value_dim), fuse_norm=False,
        )

    def _run(self, x, *, past, use_cache):
        dtype = self.mixer.r_proj.weight.dtype
        v_first = None if self.layer_idx == 0 else self.bus.value
        o, _, past, v_first_out = self.mixer(
            self.norm(x).to(dtype), past_key_values=past, use_cache=use_cache, v_first=v_first,
        )
        if self.layer_idx == 0:
            self.bus.value = v_first_out
        return o.float(), past

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self._run(x, past=Cache(), use_cache=False)
        return o

    def init_state(self, B, device):
        # Seed a ZERO token-shift conv_state ([B, D]) rather than starting from an empty
        # Cache. fla's token_shift only stores the previous token into cache_out on its
        # "warm" path (cache is not None); a cold single-token first decode (cache=None)
        # silently drops token 0, so token 1's shift sees 0 instead of token 0 (fla assumes
        # the first call is a multi-token prefill -- but our rollout decodes one token from
        # t=0). Zeros == "previous token is 0", matching full-seq's ZeroPad at position 0.
        # recurrent_state stays None (the WKV kernel treats None as the zero state).
        dtype = self.mixer.r_proj.weight.dtype
        conv = torch.zeros(int(B), self.mixer.hidden_size, device=device, dtype=dtype)
        return {"recurrent_state": None, "conv_state": conv}

    # chunked sqrt-BPTT: carry (recurrent_state [WKV] + conv_state [token-shift]) across chunks.
    # conv_state MUST carry so the chunk-boundary token's token-shift sees the true previous
    # token (not 0). v_first is per-token across depth, recomputed each chunk via the bus.
    def forward_with_state(self, x, state):
        cache = _cache_at(state, self.layer_idx)
        o, cache = self._run(x, past=cache, use_cache=True)
        return o, cache[self.layer_idx]

    def flatten_state(self, state):
        if state is None:
            return [None, None]
        return [state.get("recurrent_state"), state.get("conv_state")]

    def unflatten_state(self, flat):
        rs, conv = flat
        if rs is None and conv is None:
            return None
        return {"recurrent_state": rs, "conv_state": conv}

    def step(self, x_t, state):
        cache = _cache_at(state, self.layer_idx)
        o, cache = self._run(x_t.unsqueeze(1), past=cache, use_cache=True)  # [B,1,D]
        return o[:, 0], cache[self.layer_idx]


class RWKVChannelMixer(nn.Module):
    """RWKV-7's native channel mix as the second sublayer -- the paper-faithful alternative to
    the shared SwiGLU ChannelMixer. RWKV-7's channel mix is NOT a normal FFN: it lerps the input
    with the previous position (a token-shift) and uses a squared-ReLU MLP (hidden_ratio 4, no
    gate). Same Block contract as ChannelMixer (pre-norm -> FFN -> residual, pop_router_logits).

    Full-sequence forward only (enwik8 / FRJT). The channel mix carries its OWN token-shift state
    (ffn_state), which the rollout `step` path does not thread -- so decode is an M2 task; the
    scaffold-matched (shared-SwiGLU) RWKV-7 remains the variant with verified decode.
    """

    breaks_chunking = True  # own token-shift state, not threaded through core's chunk loop

    def __init__(self, d_model, *, hidden_ratio=4, layer_idx=0, num_layers=1) -> None:
        super().__init__()
        self.norm = RMSNorm(int(d_model))
        # float ratio allowed: fla computes intermediate = int(hidden_size * ratio), so
        # fractional ratios give fine-grained width control for parameter equalization
        self.ffn = _RWKV7FeedForward(hidden_size=int(d_model), hidden_ratio=float(hidden_ratio),
                                     layer_idx=int(layer_idx), num_hidden_layers=int(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.ffn.key.weight.dtype
        out, _ = self.ffn(self.norm(x).to(dtype))
        return x + out.float()

    def pop_router_logits(self):
        return []


__all__ = ["RWKV7Mixer", "VFirstBus", "RWKVChannelMixer"]
