"""Token mixers -- the ONLY thing that varies across architectures.

Every mixer is an nn.Module driven by srdn.core.Block / SRDNLM. The uniform
interface:

  forward(x) -> y                              full-sequence teacher-forced mix (residual delta)
  chunkable: bool                              True -> exact state-carry BPTT chunking
  # recurrence-complete mixers (chunkable=True) ALSO implement:
  init_state(B, device) -> state
  forward_with_state(x, state) -> (y, state)   carry recurrent + conv state across chunks
  flatten_state(state) -> list[Tensor|None]    so core's checkpoint-remat loop is generic
  unflatten_state(flat) -> state
  # rollout (all mixers; "state" may be a library cache for parallelizable ones):
  init_state(B, device) -> state
  step(x_t, state) -> (y_t, state)             one token; x_t [B,d] -> y_t [B,d]

x is [B, T, d_model] (or [B, d_model] in step). Mixers map d_model -> d_model.

Imports are LAZY (via __getattr__): building SRDN must not require fla / refs / the
mamba-ssm kernels that only the baseline mixers need.
"""
from importlib import import_module

_MODULES = {
    "SRDNMixer": "ops.srdn",
    "AttentionMixer": "ops.attention",
    "Mamba3Mixer": "ops.mamba3",
    "M2RNNMixer": "ops.m2rnn",
    "GDN2Mixer": "ops.gdn2",
}


def __getattr__(name):
    if name in _MODULES:
        return getattr(import_module(_MODULES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_MODULES)
