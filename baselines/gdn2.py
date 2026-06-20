"""NVIDIA GatedDeltaNet-2 baseline.

GDN-2 is NVIDIA Source Code License-NC (non-commercial, NOT redistributable), so
it is NOT vendored: point `refs/GatedDeltaNet-2` at a local checkout (the README
records the pinned commit). Loaded at runtime. Wrapped with the same FFN + policy
surface (logits / chunked_logits / init_states / step) as SRDN.

Two things this wrapper adds on top of the stock layer:
  * the rollout NaN fix -- GDN-2 auto-switches to `fused_recurrent_gdn2` for
    q_len<=64 in eval, which returns NaN from finite inputs at head_dim 28 / S=1;
    we force the `chunk` kernel (train mode) in .step.
  * sqrt-exact-BPTT sequence checkpointing in chunked_logits -- carry the FLA
    Cache (chunk_gdn2 recurrent_state + ShortConvolution conv_state) across chunks.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from fla.models.utils import Cache

from .common import RMSNorm, REFS


def import_gdn2_layer(gdn2_repo: str | Path | None = None) -> type[nn.Module]:
    candidates = [Path(gdn2_repo).expanduser()] if gdn2_repo is not None else []
    candidates += [REFS / "GatedDeltaNet-2", Path.cwd() / "refs" / "GatedDeltaNet-2"]
    for repo in candidates:
        gdn2_path = repo / "lit_gpt" / "gdn2.py"
        if gdn2_path.exists():
            package_name = f"_srdn_external_gdn2_{abs(hash(str(repo.resolve())))}"
            if package_name not in sys.modules:
                pkg = types.ModuleType(package_name)
                pkg.__path__ = [str(repo / "lit_gpt")]
                sys.modules[package_name] = pkg
                ops_pkg = types.ModuleType(f"{package_name}.gdn2_ops")
                ops_pkg.__path__ = [str(repo / "lit_gpt" / "gdn2_ops")]
                sys.modules[f"{package_name}.gdn2_ops"] = ops_pkg
            module_name = f"{package_name}.gdn2"
            if module_name not in sys.modules:
                spec = importlib.util.spec_from_file_location(module_name, gdn2_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load GDN-2 module spec from {gdn2_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            return sys.modules[module_name].GatedDeltaNet2
    raise ImportError(
        "Could not import GDN-2. Clone https://github.com/NVlabs/GatedDeltaNet-2 "
        "under refs/GatedDeltaNet-2 (see README for the pinned commit) or pass gdn2_repo."
    )


def _where_rows(mask, new, old):
    if new is None and old is None:
        return None
    src = new if new is not None else torch.zeros_like(old)
    dst = old if old is not None else torch.zeros_like(src)
    view = mask
    while view.ndim < src.ndim:
        view = view.unsqueeze(-1)
    return torch.where(view, src, dst)


def _blend_state(update_mask, new, old):
    if isinstance(new, tuple) or isinstance(old, tuple):
        new_t = new if isinstance(new, tuple) else tuple(None for _ in old)
        old_t = old if isinstance(old, tuple) else tuple(None for _ in new)
        return tuple(_blend_state(update_mask, n, o) for n, o in zip(new_t, old_t))
    if torch.is_tensor(new) or torch.is_tensor(old):
        return _where_rows(update_mask, new, old)
    return new if new is not None else old


def _reset_state(reset_mask, state):
    if isinstance(state, tuple):
        return tuple(_reset_state(reset_mask, x) for x in state)
    if torch.is_tensor(state):
        view = reset_mask
        while view.ndim < state.ndim:
            view = view.unsqueeze(-1)
        return torch.where(view, torch.zeros_like(state), state)
    return state


def _reset_cache_rows(cache: Cache, reset_mask) -> Cache:
    if not bool(reset_mask.any()) or len(cache) == 0:
        return cache
    legacy = [{k: _reset_state(reset_mask, v) for k, v in ls.items()} for ls in cache]
    return Cache.from_legacy_cache(tuple(legacy))


def _blend_caches(old_cache: Cache, new_cache: Cache, update_mask) -> Cache:
    legacy = []
    for idx in range(max(len(old_cache), len(new_cache))):
        old_s = old_cache[idx] if idx < len(old_cache) else {}
        new_s = new_cache[idx] if idx < len(new_cache) else {}
        keys = set(old_s) | set(new_s)
        legacy.append({k: _blend_state(update_mask, new_s.get(k), old_s.get(k)) for k in keys})
    return Cache.from_legacy_cache(tuple(legacy))


class GDN2Block(nn.Module):
    def __init__(self, d_model, heads, head_dim, ffn_mult, *, expand_v, use_short_conv,
                 allow_neg_eigval, layer_idx, gdn2_layer_cls) -> None:
        super().__init__()
        self.norm = RMSNorm(int(d_model))
        self.mixer = gdn2_layer_cls(
            hidden_size=int(d_model), expand_v=float(expand_v), head_dim=int(head_dim),
            num_heads=int(heads), mode="chunk", use_short_conv=bool(use_short_conv),
            allow_neg_eigval=bool(allow_neg_eigval), layer_idx=int(layer_idx))
        self.use_ffn = float(ffn_mult) > 0.0
        if self.use_ffn:
            hidden = int(round(int(d_model) * float(ffn_mult)))
            self.ffn_norm = RMSNorm(int(d_model))
            self.ffn_in = nn.Linear(int(d_model), 2 * hidden, bias=False)
            self.ffn_out = nn.Linear(hidden, int(d_model), bias=False)

    def forward(self, x, *, past_key_values=None, use_cache=False, cu_seqlens=None):
        mixer_dtype = self.mixer.q_proj.weight.dtype
        y, _, past_key_values = self.mixer(self.norm(x).to(mixer_dtype), past_key_values=past_key_values,
                                           use_cache=use_cache, cu_seqlens=cu_seqlens)
        x = x.float() + y.float()
        if self.use_ffn:
            ff = self.ffn_norm(x)
            up, gate = self.ffn_in(ff.to(self.ffn_in.weight.dtype)).chunk(2, dim=-1)
            x = x + self.ffn_out(up * F.silu(gate)).float()
        return x, past_key_values


class GDN2LM(nn.Module):
    def __init__(self, vocab_size, d_model, layers, heads, head_dim, ffn_mult, *,
                 expand_v=1.0, use_short_conv=True, allow_neg_eigval=True, gdn2_repo=None) -> None:
        super().__init__()
        layer_cls = import_gdn2_layer(gdn2_repo)
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.layers = nn.ModuleList(
            GDN2Block(d_model, heads, head_dim, ffn_mult, expand_v=expand_v,
                      use_short_conv=use_short_conv, allow_neg_eigval=allow_neg_eigval,
                      layer_idx=i, gdn2_layer_cls=layer_cls) for i in range(int(layers)))
        self.final_norm = RMSNorm(self.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, tokens, *, cu_seqlens=None):
        x = self.embed(tokens)
        cache = Cache()
        for layer in self.layers:
            x, cache = layer(x, past_key_values=cache, use_cache=False, cu_seqlens=cu_seqlens)
        return self.final_norm(x.float())

    def logits(self, tokens, *, use_xma=False, cu_seqlens=None):
        del use_xma
        return self.forward(tokens, cu_seqlens=cu_seqlens) @ self.embed.weight.float().T

    def chunked_logits(self, tokens, *, use_xma=False, chunk_size=0,
                       detach_boundaries=False, remat_chunks=False):
        """sqrt-exact BPTT: carry the FLA Cache (chunk_gdn2 recurrent_state +
        ShortConvolution conv_state, both chunk-correct) across chunks, checkpoint
        each chunk. Forces train mode so the layer uses the `chunk` kernel."""
        del use_xma
        T = int(tokens.shape[1])
        if int(chunk_size) <= 0 or int(chunk_size) >= T:
            return self.logits(tokens)
        nlayers = len(self.layers)

        def build_cache(flat):
            if flat[0] is None:
                return Cache()
            legacy = []
            for li in range(nlayers):
                rs, cq, ck, cv = flat[4 * li:4 * li + 4]
                legacy.append({"recurrent_state": rs, "conv_state": None if cq is None else (cq, ck, cv)})
            return Cache.from_legacy_cache(tuple(legacy))

        def run_chunk(chunk_tokens, *flat):
            cache = build_cache(flat)
            x = self.embed(chunk_tokens)
            for layer in self.layers:
                x, cache = layer(x, past_key_values=cache, use_cache=True)
            logits = self.final_norm(x.float()) @ self.embed.weight.float().T
            newflat = []
            for li in range(nlayers):
                ls = cache[li]
                conv = ls.get("conv_state")
                cq, ck, cv = (None, None, None) if conv is None else conv
                newflat.extend([ls.get("recurrent_state"), cq, ck, cv])
            return (logits, *newflat)

        old_training = self.training
        self.train()
        flat = (None,) * (4 * nlayers)
        outs = []
        for start in range(0, T, int(chunk_size)):
            chunk = tokens[:, start:start + int(chunk_size)]
            if remat_chunks and torch.is_grad_enabled():
                result = checkpoint(run_chunk, chunk, *flat, use_reentrant=False)
            else:
                result = run_chunk(chunk, *flat)
            outs.append(result[0])
            flat = tuple(result[1:])
            if detach_boundaries:
                flat = tuple(None if t is None else t.detach() for t in flat)
        self.train(old_training)
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def init_states(self, batch_size, device) -> Cache:
        del batch_size, device
        return Cache()

    @torch.no_grad()
    def step(self, token, cache: Cache, update_mask, reset_mask=None):
        if token.ndim != 1:
            raise ValueError("token must have shape [batch]")
        update = update_mask.to(device=token.device, dtype=torch.bool)
        reset = torch.zeros_like(update) if reset_mask is None else reset_mask.to(device=token.device, dtype=torch.bool)
        old_training = self.training
        # force the chunk kernel even at S=1: fused_recurrent_gdn2 NaNs at head_dim 28.
        self.train()
        old_cache = _reset_cache_rows(cache, reset)
        new_cache = Cache.from_legacy_cache(old_cache.to_legacy_cache())
        x = self.embed(token).unsqueeze(1)
        for layer in self.layers:
            x, new_cache = layer(x, past_key_values=new_cache, use_cache=True)
        logits = self.final_norm(x[:, -1].float()) @ self.embed.weight.float().T
        blended = _blend_caches(old_cache, new_cache, update)
        self.train(old_training)
        return torch.where(update[:, None], logits, torch.zeros_like(logits)), blended

    def num_params(self) -> int:
        seen, n = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p)); n += p.numel()
        return n


__all__ = ["GDN2LM", "GDN2Block", "import_gdn2_layer"]
