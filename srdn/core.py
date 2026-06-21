"""Shared model scaffold: Block + LM. The ONLY thing that varies across
architectures is the token mixer (ops/*.py) plugged into Block; the channel mixer,
embedding, chunked-BPTT remat loop, rollout, and masking all live here, once.

Block:  x = x + mixer(x) ; x = channel(x)        (mixer/channel own their pre-norm)
LM:     embed -> blocks -> final_norm -> tied head; plus the policy surface
        (logits / chunked_logits / init_states / step) the tasks drive.

A mixer must implement: forward(x)->y, .chunkable (bool). chunkable mixers also
implement init_state/forward_with_state/flatten_state/unflatten_state. All mixers
implement step(x_t, state)->(y_t, state) for rollout (a parallelizable mixer's
"state" is its own KV/conv cache).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .channel import ChannelMixer
from .norm import RMSNorm


class Block(nn.Module):
    def __init__(self, mixer: nn.Module, channel: ChannelMixer) -> None:
        super().__init__()
        self.mixer = mixer
        self.channel = channel

    @property
    def chunkable(self) -> bool:
        return bool(getattr(self.mixer, "chunkable", False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() + self.mixer(x).float()
        return self.channel(x)

    # chunked: carry the mixer's recurrent state across the chunk boundary
    def forward_with_state(self, x, state):
        y, state = self.mixer.forward_with_state(x, state)
        x = x.float() + y.float()
        return self.channel(x), state

    def step(self, x_t, state):
        y, state = self.mixer.step(x_t, state)
        x = x_t.float() + y.float()
        return self.channel(x), state


class SRDNLM(nn.Module):
    """LM + RL-policy surface around a stack of `Block`s (any token mixer)."""

    def __init__(self, vocab_size: int, d_model: int, blocks: list[Block]) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(self.d_model)
        nn.init.normal_(self.embed.weight, std=self.d_model ** -0.5)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    # ---- full sequence ----
    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        for b in self.blocks:
            h = b(h)
        return self.final_norm(h)

    def forward(self, tokens):
        return self.hidden(tokens)

    def logits(self, tokens, *, use_xma: bool = False):
        del use_xma
        return self.hidden(tokens) @ self.embed.weight.float().T

    # ---- chunked: sqrt-exact BPTT (generic over the mixers' flatten/unflatten) ----
    def chunked_logits(self, tokens, *, use_xma: bool = False, chunk_size: int = 0,
                       detach_boundaries: bool = False, remat_chunks: bool = False):
        del use_xma
        T = int(tokens.shape[1])
        if int(chunk_size) <= 0 or int(chunk_size) >= T or not all(b.chunkable for b in self.blocks):
            return self.logits(tokens)
        B = int(tokens.shape[0])
        states = [b.mixer.init_state(B, tokens.device) for b in self.blocks]
        # per-block flat lengths, to slice the flat tuple back apart inside run_chunk
        lens = [len(b.mixer.flatten_state(s)) for b, s in zip(self.blocks, states)]

        def flatten(sts):
            flat = []
            for b, s in zip(self.blocks, sts):
                flat.extend(b.mixer.flatten_state(s))
            return flat

        def unflatten(flat):
            out, i = [], 0
            for b, n in zip(self.blocks, lens):
                out.append(b.mixer.unflatten_state(list(flat[i:i + n])))
                i += n
            return out

        def run_chunk(chunk_tokens, *flat):
            x = self.embed(chunk_tokens)
            nxt = []
            for b, st in zip(self.blocks, unflatten(flat)):
                x, st = b.forward_with_state(x, st)
                nxt.append(st)
            logits = self.final_norm(x) @ self.embed.weight.float().T
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
                states = [_detach_state(s) for s in states]
        self.pop_router_logits()
        return torch.cat(outs, dim=1)

    def pop_router_logits(self):
        out = []
        for b in self.blocks:
            out.extend(b.channel.pop_router_logits())
        return out

    # ---- rollout ----
    @torch.no_grad()
    def init_states(self, batch_size, device):
        return [b.mixer.init_state(int(batch_size), device) for b in self.blocks]

    @torch.no_grad()
    def step(self, token, states, update_mask=None):
        h = self.embed(token)
        new_states = []
        for b, st in zip(self.blocks, states):
            h, nst = b.step(h, st)
            if update_mask is not None:
                nst = _merge_state(update_mask, nst, st)
            new_states.append(nst)
        logits = self.final_norm(h) @ self.embed.weight.float().T
        self.pop_router_logits()
        if update_mask is not None:
            logits = torch.where(update_mask[:, None], logits, torch.zeros_like(logits))
        return logits, new_states

    def num_params(self) -> int:
        seen, n = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p)); n += p.numel()
        return n


def _detach_state(s):
    if isinstance(s, torch.Tensor):
        return s.detach()
    if isinstance(s, dict):
        return {k: _detach_state(v) for k, v in s.items()}
    if isinstance(s, (tuple, list)):
        return type(s)(_detach_state(v) for v in s)
    return s


def _merge_state(mask, new, old):
    """Keep `new` rows where mask is True, else `old`, through nested state. A None
    on either side means "no prior/new state at this slot" -> take whatever exists
    (e.g. first rollout step, where a library cache starts None and is fully populated)."""
    if old is None or new is None:
        return new if new is not None else old
    if isinstance(new, torch.Tensor):
        if new.dim() == 0 or new.shape[0] != mask.shape[0]:
            return new
        m = mask.view(-1, *([1] * (new.dim() - 1)))
        return torch.where(m, new, old)
    if isinstance(new, dict):
        return {k: _merge_state(mask, new[k], old.get(k)) for k in new}
    if isinstance(new, (tuple, list)):
        return type(new)(_merge_state(mask, n, o) for n, o in zip(new, old))
    return new


__all__ = ["Block", "SRDNLM"]
