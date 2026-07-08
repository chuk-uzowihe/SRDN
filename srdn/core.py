"""Shared model core: norm, the state-routed MoE, and the scaffold (Block + LM).

The ONLY thing that varies across architectures is the token mixer (srdn/ops/*.py)
plugged into Block; the channel mixer (srdn/channel.py), embedding, chunked-BPTT
remat loop, rollout, and masking all live here / next to here, once.

Block:  x = x + mixer(x) ; x = channel(x)        (mixer/channel own their pre-norm)
LM:     embed -> blocks -> final_norm -> tied head; plus the policy surface
        (logits / chunked_logits / init_states / step) the tasks drive.

A mixer implements: forward(x)->y, .chunkable (bool). chunkable mixers also implement
init_state / forward_with_state / flatten_state / unflatten_state. All mixers
implement init_state + step(x_t, state)->(y_t, state) for rollout (a parallelizable
mixer's "state" is its own KV/conv cache).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ----------------------------------------------------------------- norm
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        n = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (n * self.weight.float()).type_as(self.weight)


# ----------------------------------------------------------------- state-routed MoE
class StateRoutedMoE(nn.Module):
    """Top-k mixture of SwiGLU experts; expert/router inputs decoupled.

    The state enters via the ROUTER, not the additive write path: experts see
    `x_in`, the router sees `route_in`. Which expert fires is a function of what the
    recurrence has accumulated. forward(x_in, route_in) -> (out, router_logits).
    """

    def __init__(self, d_in: int, d_out: int, d_route: int,
                 n_experts: int, top_k: int, d_hidden: int = 0) -> None:
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.E, self.k = int(n_experts), int(top_k)
        self.d_hidden = int(d_hidden)
        self.router = nn.Linear(d_route, self.E, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        inner = self.d_hidden if self.d_hidden > 0 else d_out
        self.w_gate = nn.Parameter(torch.empty(self.E, d_in, inner))
        self.w_up = nn.Parameter(torch.empty(self.E, d_in, inner))
        bound = d_in ** -0.5
        nn.init.uniform_(self.w_gate, -bound, bound)
        nn.init.uniform_(self.w_up, -bound, bound)
        if self.d_hidden > 0:
            self.w_down = nn.Parameter(torch.empty(self.E, self.d_hidden, d_out))
            nn.init.uniform_(self.w_down, -(self.d_hidden ** -0.5), self.d_hidden ** -0.5)

    def forward(self, x_in: torch.Tensor, route_in: torch.Tensor):
        logits = self.router(route_in).float()
        topv, topi = logits.topk(self.k, dim=-1)
        topw = F.softmax(topv, dim=-1)
        weights = torch.zeros_like(logits).scatter(-1, topi, topw)
        g = torch.einsum("bd,edw->bew", x_in, self.w_gate.float())
        u = torch.einsum("bd,edw->bew", x_in, self.w_up.float())
        act = F.silu(g) * u
        if self.d_hidden > 0:
            act = torch.einsum("beh,ehd->bed", act, self.w_down.float())
        out = torch.einsum("be,bew->bw", weights, act)
        return out, logits


def moe_aux_loss(router_logits: list[torch.Tensor], top_k: int) -> torch.Tensor:
    """Switch-transformer load-balance loss: E * sum_e f_e*P_e (0 if list empty)."""
    if not router_logits:
        return torch.zeros(())
    total = 0.0
    for lg in router_logits:
        flat = lg.reshape(-1, lg.shape[-1])
        E = flat.shape[-1]
        P = F.softmax(flat, dim=-1).mean(0)
        topi = flat.topk(top_k, dim=-1).indices
        f = F.one_hot(topi, E).sum(1).clamp(max=1).float().mean(0)
        total = total + E * (f * P).sum()
    return total / len(router_logits)


# ----------------------------------------------------------------- scaffold
class Block(nn.Module):
    """x = x + mixer(x) ; x = channel(x). `mixer` and `channel` own their pre-norms."""

    def __init__(self, mixer: nn.Module, channel: nn.Module) -> None:
        super().__init__()
        self.mixer = mixer
        self.channel = channel

    @property
    def chunkable(self) -> bool:
        # a channel mixer with its own cross-token state that is NOT threaded through the chunk
        # loop (e.g. RWKVChannelMixer's token-shift) would silently reset at every chunk boundary
        return (bool(getattr(self.mixer, "chunkable", False))
                and not bool(getattr(self.channel, "breaks_chunking", False)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() + self.mixer(x).float()
        return self.channel(x)

    def forward_with_state(self, x, state):
        y, state = self.mixer.forward_with_state(x, state)
        x = x.float() + y.float()
        return self.channel(x), state

    def step(self, x_t, state):
        if getattr(self.channel, "breaks_chunking", False):
            raise RuntimeError(
                "this channel mixer carries its own cross-token state and has no step path "
                "(build with the scaffold SwiGLU channel for rollout)")
        y, state = self.mixer.step(x_t, state)
        x = x_t.float() + y.float()
        return self.channel(x), state


class SRDNLM(nn.Module):
    """LM + RL-policy surface around a stack of `Block`s (any token mixer)."""

    def __init__(self, vocab_size: int, d_model: int, blocks: list[Block],
                 pos_embed: torch.Tensor | None = None) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(self.d_model)
        nn.init.normal_(self.embed.weight, std=self.d_model ** -0.5)
        # NO blanket re-init here: each mixer/channel owns its init (library cells like fla's
        # RWKV-7 carry carefully scheduled inits -- zero o_proj, orthogonal r/k/v, LoRA bias
        # schedules -- that a tree-wide nn.Linear re-init would destroy)
        # optional absolute positional table (the transformer's sinusoidal PE), added ONCE at
        # the input embedding so it enters the residual stream -- the standard placement
        if pos_embed is not None:
            self.register_buffer("pos_embed", pos_embed, persistent=False)
        else:
            self.pos_embed = None

    # ---- full sequence ----
    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        if self.pos_embed is not None:
            # Vaswani et al.: embeddings are multiplied by sqrt(d_model) before adding PE --
            # with the d^-0.5 embed init this puts token content at unit scale, comparable to
            # the unit-amplitude sinusoids (unscaled, PE drowns the tokens ~8x at d=128)
            h = h * self.d_model ** 0.5 + self.pos_embed[: tokens.shape[1]].to(h.dtype)[None]
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
            if int(chunk_size) > 0 and int(chunk_size) < T and not getattr(self, "_warned_full_seq", False):
                self._warned_full_seq = True
                print("chunked_logits: model not chunkable -> full-sequence BPTT "
                      "(chunk_size/detach/remat flags ignored)", flush=True)
            return self.logits(tokens)
        B = int(tokens.shape[0])
        states = [b.mixer.init_state(B, tokens.device) for b in self.blocks]
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
        states = [b.mixer.init_state(int(batch_size), device) for b in self.blocks]
        if self.pos_embed is not None:
            # per-row absolute position for the embedding-level PE (advances only where updated)
            states.append(torch.zeros(int(batch_size), device=device, dtype=torch.long))
        return states

    @torch.no_grad()
    def step(self, token, states, update_mask=None):
        h = self.embed(token)
        pos = None
        if self.pos_embed is not None:
            states, pos = states[:-1], states[-1]
            h = h * self.d_model ** 0.5 + self.pos_embed[pos].to(h.dtype)
        new_states = []
        for b, st in zip(self.blocks, states):
            h, nst = b.step(h, st)
            if update_mask is not None:
                nst = _merge_state(update_mask, nst, st)
            new_states.append(nst)
        if pos is not None:
            new_states.append(pos + (update_mask.long() if update_mask is not None else 1))
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
    """Keep `new` rows where mask is True, else `old`, through nested state. A None on
    either side means take whatever exists (e.g. first rollout step, where a library
    cache starts None and is fully populated)."""
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


__all__ = ["RMSNorm", "StateRoutedMoE", "moe_aux_loss", "Block", "SRDNLM"]
