"""NVIDIA GatedDeltaNet-2 token mixer (loaded from refs/GatedDeltaNet-2 at runtime).

GDN-2 is NVIDIA Source Code License-NC (non-commercial, NOT redistributable), so it
is referenced, never vendored: point refs/GatedDeltaNet-2 at a checkout (README has
the pinned commit). Owns its pre-norm, returns the residual delta.

chunkable=True: state = a single-layer FLA Cache (chunk_gdn2 recurrent_state +
ShortConvolution conv_state, both chunk-correct). Rollout forces the `chunk` kernel
(train mode) -- GDN-2's fused_recurrent kernel returns NaN at head_dim 28 / S=1.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from fla.models.utils import Cache

from srdn.norm import RMSNorm

_REFS = Path(__file__).resolve().parents[1] / "refs"


def import_gdn2_layer(gdn2_repo=None) -> type[nn.Module]:
    candidates = [Path(gdn2_repo).expanduser()] if gdn2_repo else []
    candidates += [_REFS / "GatedDeltaNet-2", Path.cwd() / "refs" / "GatedDeltaNet-2"]
    for repo in candidates:
        gdn2_path = repo / "lit_gpt" / "gdn2.py"
        if gdn2_path.exists():
            pkg_name = f"_srdn_gdn2_{abs(hash(str(repo.resolve())))}"
            if pkg_name not in sys.modules:
                pkg = types.ModuleType(pkg_name); pkg.__path__ = [str(repo / "lit_gpt")]
                sys.modules[pkg_name] = pkg
                ops = types.ModuleType(f"{pkg_name}.gdn2_ops"); ops.__path__ = [str(repo / "lit_gpt" / "gdn2_ops")]
                sys.modules[f"{pkg_name}.gdn2_ops"] = ops
            mod_name = f"{pkg_name}.gdn2"
            if mod_name not in sys.modules:
                spec = importlib.util.spec_from_file_location(mod_name, gdn2_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module
                spec.loader.exec_module(module)
            return sys.modules[mod_name].GatedDeltaNet2
    raise ImportError("GDN-2 not found: clone NVlabs/GatedDeltaNet-2 to refs/ (README has the pin).")


def _one_layer_cache(state) -> Cache:
    """Wrap a single layer's (recurrent_state, conv_state) dict as a layer-0 Cache."""
    if state is None:
        return Cache()
    return Cache.from_legacy_cache((state,))


class GDN2Mixer(nn.Module):
    chunkable = True

    def __init__(self, d_model, n_heads, head_dim, *, expand_v=1.0, use_short_conv=True,
                 allow_neg_eigval=True, gdn2_repo=None) -> None:
        super().__init__()
        layer_cls = import_gdn2_layer(gdn2_repo)
        self.norm = RMSNorm(int(d_model))
        self.mixer = layer_cls(hidden_size=int(d_model), expand_v=float(expand_v), head_dim=int(head_dim),
                               num_heads=int(n_heads), mode="chunk", use_short_conv=bool(use_short_conv),
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
        self.train()                                     # force chunk kernel (S=1 fused NaNs)
        x = self.norm(x_t).unsqueeze(1).to(dtype)
        y, _, cache = self.mixer(x, past_key_values=cache, use_cache=True)
        self.train(old_training)
        return y[:, 0].float(), cache[0]


__all__ = ["GDN2Mixer", "import_gdn2_layer"]
