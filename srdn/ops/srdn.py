"""SRDN: the state-reading RWKV-7 token mixer.

Base cell = RWKV-7 (fla.layers.RWKV7Attention) -- a chunkable DPLR linear-attention whose
per-head state S in R^{dh x dh} evolves as (from the fla fused_recurrent kernel):

    S <- diag(exp w) . S  -  (a (.) kk) (kk^T S)      # decay + rank-1 removal  (DPLR transition)
    S <- S + k~ (x) v                                  # write (k~ = fused replacement key)
    o <- r^T S                                         # read after write

We make it RECURRENCE-COMPLETE by conditioning the projections on the *previous* state S_{t-1}
(SRDN's state-reading), which makes step-t input depend nonlinearly on step-(t-1) state and so
forfeits the chunk-parallel kernel -- the scan here is an explicit per-token loop.

  s0 = rmsnorm(diag S)        -> conditions r  (the read direction; RWKV-7's sole reader)
  sx = rmsnorm(q . S)         -> conditions {k, v, w(decay), a(removal)}  (the write/transition)

All conditioning is additive with ZERO-INIT adapter outputs (LoRA B=0 / full-rank W=0), so the
cell is EXACTLY RWKV-7 at init. The base projections and the entire output path (g_norm +
gate_output_correction + o_proj) are reused verbatim from the fla RWKV7Attention submodule, so
the init cell matches fla to kernel tolerance. The cross-layer v_first bus carries the layer-0
BASE v, exactly as in native RWKV-7 -- the bus is skeleton, not a conditioning target (see
_publish_v_first).

NOTE (why there is no ReZero gamma): gating the zero-init adapters with a ReZero scalar
gamma=0 would be a true fixed point -- gamma * adapter(z) with both factors zero has
identically zero gradient to gamma AND to the adapter -- so the state-reading path would never
train. Zero-init adapters alone give exact-RWKV-7-at-init with a live gradient (dL/dB ~ zA != 0).

Ablation switches:
  content_read_mode: "shared" (1 query: r does double duty) | "split" (2: a separate content
                     query feeds one shared sx) | "per_proj" (>2: a unique query per content
                     read; default -- split kept as the dose-response ablation)
  read_rank:         low-rank content-read queries (default head_dim, "lite"; 0 = full-rank)
  use_lora:          low-rank state-reading adapters (default True) vs full [dh x dh] per head
  neg_eigval:        a = (1+exp(_DECAY))*sigmoid -> eigenvalue range EXACTLY [-1,1] (vs native
                     a=sigmoid -> [-0.455, 1]); a linear rescale so the floor hits -1 w/o overshoot

CUDA-only (fla output ops are triton). Mirrors the RWKV7Mixer contract in ops/rwkv7.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from fla.layers import RWKV7Attention
from fla.ops.rwkv7.gate_output_correction import gate_output_correction

from srdn.core import RMSNorm
from .rwkv7 import VFirstBus  # shared v_first carrier across the stack's RWKV layers

_DECAY = -0.6065306597126334  # RWKV-7's fixed decay scale: w = _DECAY * sigmoid(w_lora(xw))

# The DPLR transition's key-direction eigenvalue is lambda = exp(w) - a (decay diagonal MINUS the
# rank-1 removal). exp(w) in [exp(_DECAY), 1] = [0.5453, 1], so:
#   native  a = sigmoid in [0,1]            -> lambda floor = exp(_DECAY) - 1   ~= -0.455  (range ~[-0.455, 1])
#   a = 2*sigmoid                           -> lambda floor = exp(_DECAY) - 2   ~= -1.455  (OVERSHOOTS [-1,1])
# To make the ACHIEVABLE eigenvalue spectrum EXACTLY [-1, 1], set a_max so the worst-case floor is -1:
#   exp(_DECAY) - a_max = -1  ->  a_max = 1 + exp(_DECAY) ~= 1.5453   (ceiling stays +1 at a=0, w=0).
NEG_A_SCALE = 1.0 + math.exp(_DECAY)  # ~= 1.54524; a = NEG_A_SCALE * sigmoid -> eigenvalues in [-1, 1]


def _l2norm(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _rmsnorm(x, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class StateAdapter(nn.Module):
    """Per-head map R^{dh} -> R^{dh} reading a state summary (s0 or sx). Full-rank [H,dh,dh]
    or low-rank [H,dh,r]@[H,r,dh] (RWKV-7-style LoRA). Output is exactly 0 at init (B=0 / W=0)
    so the cell starts as exact RWKV-7, while the gradient stays live (A is random)."""

    def __init__(self, H, dh, *, use_lora=True, rank=0):
        super().__init__()
        self.use_lora = bool(use_lora)
        if self.use_lora:
            r = int(rank) if rank else max(8, dh // 2)
            self.A = nn.Parameter(torch.randn(H, dh, r) * dh ** -0.5)
            self.B = nn.Parameter(torch.zeros(H, r, dh))
        else:
            self.W = nn.Parameter(torch.zeros(H, dh, dh))  # zero-init => exact RWKV-7

    def forward(self, z):  # z: [B,H,dh] -> [B,H,dh]
        if self.use_lora:
            return torch.einsum("bhr,hrd->bhd", torch.einsum("bhs,hsr->bhr", z, self.A), self.B)
        return torch.einsum("bhs,hsd->bhd", z, self.W)


class SRDNMixer(nn.Module):
    # state-carry BPTT: the scan is sequential either way, so chunking just bounds activation
    # memory -- state (S, prev raw token for token-shift) carries across chunk boundaries with
    # grad intact (sqrt-exact under core.chunked_logits remat), same contract as SRDNMixer.
    chunkable = True

    #: which projections sx conditions (the write/transition path)
    SX_TARGETS = ("k", "v", "w", "a")

    def __init__(self, d_model, *, head_dim=64, layer_idx=0, num_layers=1, value_dim=None,
                 bus=None, content_read_mode="per_proj", use_lora=True, lora_rank=0,
                 neg_eigval=False, fuse_scan=True, scan_block=16, read_rank=None):
        super().__init__()
        assert content_read_mode in ("shared", "split", "per_proj")
        d = int(d_model)
        self.norm = RMSNorm(d)
        self.layer_idx = int(layer_idx)
        self.bus = bus if bus is not None else VFirstBus()
        self.mode = content_read_mode
        self.a_scale = NEG_A_SCALE if neg_eigval else 1.0  # eigenvalue range [-1,1] vs native [-0.455,1]
        # Tier-1.5 fusion: torch.compile a BLOCK of scan_block unrolled steps (dynamic=False, so
        # the inner loop unrolls at compile time). The scan is irreducibly SERIAL (state-reading),
        # so this does NOT parallelize time -- it cuts dispatch cost from per-TOKEN to per-BLOCK.
        # _base/_output stay eager (triton). Distinct block lengths (the tail, rollout T=1)
        # compile separate specializations -- bounded in practice by chunking/padding. Do NOT
        # compile with cudagraphs ("reduce-overhead"): a fn replayed in a T/K-long Python loop
        # builds O(T) static buffers and OOMs.
        self.fuse_scan = bool(fuse_scan)
        self.scan_block = max(1, int(scan_block))
        self._sc = None  # lazily-compiled _run_block
        # reuse fla RWKV-7's projections, LoRAs, token-shift params, g_norm, o_proj, k_k/k_a/r_k
        self.rwkv = RWKV7Attention(
            mode="chunk", hidden_size=d, head_dim=int(head_dim), layer_idx=int(layer_idx),
            num_hidden_layers=int(num_layers), value_dim=None if value_dim is None else int(value_dim),
            fuse_norm=False,
        )
        H, dh = self.rwkv.num_heads, self.rwkv.head_dim
        self.H, self.dh = H, dh
        assert self.rwkv.head_v_dim == dh, "state-reading needs square heads (head_dim == head_v_dim)"
        # The adapters are per-head [dh x dh] maps, so LoRA only compresses (and only acts as a real
        # low-rank bottleneck) when rank << dh: a [dh,r]+[r,dh] factorization beats a full [dh,dh]
        # only for r < dh/2. Default to dh//4 so "LoRA on" is a genuine bottleneck and the smaller
        # option. (At head_dim=32 the adapters are tiny vs embed/FFN, so this ablation tests the
        # low-rank inductive bias, not param savings -- it has more teeth at head_dim >= 64.)
        rank = int(lora_rank) if lora_rank else max(4, dh // 4)

        # s0 -> r (the read direction)
        self.adapt_r = StateAdapter(H, dh, use_lora=use_lora, rank=rank)
        # sx -> {k,v,w,a} (the write/transition path)
        self.adapt = nn.ModuleDict({t: StateAdapter(H, dh, use_lora=use_lora, rank=rank)
                                    for t in self.SX_TARGETS})
        # content-read query projection(s): "shared" reuses r; "split" one query; "per_proj" one
        # per target. read_rank factorizes each query d -> rank -> inner (both factors
        # RANDOM-init: the adapters' zero-init already makes the cell exactly RWKV-7 at init, and
        # a zero factor here would kill the read gradient). With a FIXED rank the read overhead
        # grows ~linearly in d instead of quadratically. Default rank = head_dim ("lite" -- the
        # query needs no more capacity than the head it reads). 0 = full-rank.
        inner = H * dh
        read_rank = dh if read_rank is None else int(read_rank)

        def _q(rank):
            if int(rank) <= 0:
                lin = nn.Linear(d, inner, bias=False)
                nn.init.normal_(lin.weight, std=0.02)
                return lin
            down, up = nn.Linear(d, int(rank), bias=False), nn.Linear(int(rank), inner, bias=False)
            nn.init.normal_(down.weight, std=0.02)
            nn.init.normal_(up.weight, std=0.02)
            return nn.Sequential(down, up)

        if self.mode == "split":
            self.q_read = _q(read_rank)
        elif self.mode == "per_proj":
            self.q_read = nn.ModuleDict({t: _q(read_rank) for t in self.SX_TARGETS})

    # ---- base (state-independent) projections, computed once over the whole sequence ----
    def _base(self, x, prev=None):
        """Token-shift sees `prev` (the previous chunk's/step's last raw token) at position 0;
        prev=None marks sequence start (zeros, == fla's ZeroPad)."""
        m = self.rwkv
        dt = m.r_proj.weight.dtype
        xn = self.norm(x).to(dt)
        prev_n = self.norm(prev).to(dt) if prev is not None else torch.zeros_like(xn[:, :1])
        shifted = torch.cat([prev_n, xn[:, :-1]], dim=1)   # shift right by 1 along T
        delta = shifted - xn
        # token-shift lerp (== fla fused_addcmul): x_* selects per-proj mix of current/previous
        xr, xw, xk, xv, xa, xg = (xn + delta * p for p in
                                  (m.x_r, m.x_w, m.x_k, m.x_v, m.x_a, m.x_g))
        r = m.r_proj(xr).float()
        w_logit = m.w_lora(xw).float()            # w = _DECAY * sigmoid(w_logit + cond)
        k = m.k_proj(xk).float()
        v = m.v_proj(xv).float()
        if self.layer_idx > 0:
            # lerp toward layer 0's value stream (the CONDITIONED v, written post-scan)
            v = torch.lerp(v, self.bus.value.float(), m.v_lora(xv).sigmoid().float())
        a_logit = m.a_lora(xa).float()            # a = a_scale * sigmoid(a_logit + cond)
        g = m.g_lora(xg)                          # output gate (unconditioned)
        qx = None
        if self.mode == "split":
            qx = self.q_read(xn).float()
        elif self.mode == "per_proj":
            qx = {t: self.q_read[t](xn).float() for t in self.SX_TARGETS}
        return dict(r=r, w_logit=w_logit, k=k, v=v, a_logit=a_logit, g=g, qx=qx)

    def _read(self, q, S):  # q:[B,H,dh], S:[B,H,dh,dh]  ->  rmsnorm(q.S) : [B,H,dh]
        # rmsnorm makes the read scale-invariant in q, so q needs no normalization of its own
        return _rmsnorm(torch.einsum("bhk,bhkv->bhv", q, S))

    @staticmethod
    def _step_state(S, r, w, ktil, v, kk, a):
        """One DPLR recurrence step (matches fla fused_recurrent_rwkv7). S:[B,H,dh,dh]."""
        ew = torch.exp(w)                                          # [B,H,dh]
        proj = torch.einsum("bhk,bhkv->bhv", -kk, S)              # (-kk).S  : [B,H,dh]
        S = (ew[..., None] * S
             + (kk * a)[..., None] * proj[:, :, None, :]          # rank-1 removal
             + ktil[..., None] * v[:, :, None, :])                # write
        o = torch.einsum("bhk,bhkv->bhv", r, S)                   # read after write
        return o, S

    def _step(self, S, r_b, w_l, k_b, v_b, a_l, qx):
        """The fusible per-token core: pure tensors in/out (no Python-side mutation), so it
        torch.compiles cleanly. qx is None (shared), [B,H,dh] (split), or [B,H,4,dh] (per_proj,
        in SX_TARGETS order k,v,w,a). Returns (o, S, r, ktil, v) -- the last three feed the
        output path's gate_output_correction."""
        m = self.rwkv
        s0 = _rmsnorm(torch.diagonal(S, dim1=-2, dim2=-1))        # s0 -> r (read direction)
        r = r_b + self.adapt_r(s0)
        if self.mode == "shared":
            sx_k = sx_v = sx_w = sx_a = self._read(r, S)          # r does double duty
        elif self.mode == "split":
            sx = self._read(qx, S); sx_k = sx_v = sx_w = sx_a = sx
        else:  # per_proj: one read query per content target
            sx_k = self._read(qx[:, :, 0], S); sx_v = self._read(qx[:, :, 1], S)
            sx_w = self._read(qx[:, :, 2], S); sx_a = self._read(qx[:, :, 3], S)
        w = _DECAY * torch.sigmoid(w_l + self.adapt["w"](sx_w))
        a = self.a_scale * torch.sigmoid(a_l + self.adapt["a"](sx_a))
        k = k_b + self.adapt["k"](sx_k)
        v = v_b + self.adapt["v"](sx_v)
        kk = _l2norm(k * m.k_k.float().view(self.H, self.dh))     # normalized removal key
        ktil = k * (1.0 + (a - 1.0) * m.k_a.float().view(self.H, self.dh))  # fla fused_k_rwkv7
        o, S = self._step_state(S, r, w, ktil, v, kk, a)
        return o, S, r, ktil, v

    def _run_block(self, S, rh, wh, kh, vh, ah, qh):
        """K unrolled scan steps ([B,K,H,dh] inputs; qh also allows [B,K,H,4,dh] or None).
        Pure tensors in/out. Under torch.compile(dynamic=False) K is static, so the loop
        unrolls into ONE graph -- dispatch is paid per block, not per token."""
        os, rs, ks, vs = [], [], [], []
        for t in range(rh.shape[1]):
            qx_t = None if qh is None else qh[:, t]
            o, S, r, ktil, v = self._step(S, rh[:, t], wh[:, t], kh[:, t], vh[:, t], ah[:, t], qx_t)
            os.append(o); rs.append(r); ks.append(ktil); vs.append(v)
        stk = lambda xs: torch.stack(xs, dim=1)                  # [B,K,H,dh]
        return stk(os), stk(rs), stk(ks), stk(vs), S

    def _block_fn(self):
        if not self.fuse_scan:
            return self._run_block
        if self._sc is None:
            self._sc = torch.compile(self._run_block, dynamic=False)
        return self._sc

    def _scan(self, base, S):
        B, T = base["r"].shape[0], base["r"].shape[1]
        H, dh = self.H, self.dh
        def heads(x):
            return x.view(B, T, H, dh)
        rh, wh, kh, vh, ah = (heads(base[k]) for k in ("r", "w_logit", "k", "v", "a_logit"))
        if self.mode == "split":
            qh = heads(base["qx"])                               # [B,T,H,dh]
        elif self.mode == "per_proj":
            qh = torch.stack([heads(base["qx"][t]) for t in self.SX_TARGETS], dim=3)  # [B,T,H,4,dh]
        else:
            qh = None
        run = self._block_fn()
        K = self.scan_block if self.fuse_scan else T             # eager: one pass, same op order
        outs = []
        for s in range(0, T, K):
            e = min(s + K, T)
            q_blk = None if qh is None else qh[:, s:e]
            o, r, ktil, v, S = run(S, rh[:, s:e], wh[:, s:e], kh[:, s:e], vh[:, s:e], ah[:, s:e], q_blk)
            outs.append((o, r, ktil, v))
        cat = lambda i: torch.cat([b[i] for b in outs], dim=1)   # [B,T,H,dh]
        return cat(0), cat(1), cat(2), cat(3), S

    def _publish_v_first(self, v_base):
        """Layer 0 writes its BASE (unconditioned) v to the cross-layer bus: the v_first bus
        belongs to the RWKV-7 skeleton, which state reading leaves untouched -- all conditioning
        stays local to each cell's own projections. (Publishing the conditioned v instead was
        tried: it helps full-rank reads but destabilizes low-rank ones -- late-training
        collapses after mastery on FRJT.)"""
        if self.layer_idx == 0:
            self.bus.value = v_base

    def _output(self, o, r, k, v, g):
        """g_norm + gate_output_correction + o_proj (reused from fla, post-scan / state-free)."""
        m = self.rwkv
        B, T = o.shape[0], o.shape[1]
        on = m.g_norm(o.reshape(B * T, self.H * self.dh)).view(B, T, -1)
        on = gate_output_correction(on, r.to(on.dtype), k.to(on.dtype), m.r_k, v.to(on.dtype), g)
        return m.o_proj(on).float()

    def _init_S(self, B, device):
        return torch.zeros(B, self.H, self.dh, self.dh, device=device, dtype=torch.float32)

    # ---- interface ----
    def forward(self, x):
        base = self._base(x)
        self._publish_v_first(base["v"])
        o, r, k, v, _ = self._scan(base, self._init_S(x.shape[0], x.device))
        return self._output(o, r, k, v, base["g"])

    # rollout (single token); state = (S, prev_token) for token-shift
    def init_state(self, B, device):
        return (self._init_S(B, device),
                torch.zeros(B, 1, self.d_model_(), device=device, dtype=torch.float32))

    def d_model_(self):
        return self.rwkv.hidden_size

    def step(self, x_t, state):
        S, prev = state
        x = x_t.unsqueeze(1)                       # [B,1,D]
        base = self._base(x, prev)
        self._publish_v_first(base["v"])
        o, r, k, v, S = self._scan(base, S)
        out = self._output(o, r, k, v, base["g"])[:, 0]
        return out, (S, x_t.unsqueeze(1).float())

    # chunked sqrt-BPTT (M2): like step() but multi-token; token-shift sees the previous
    # chunk's last raw token, S carries with grad across the boundary.
    def forward_with_state(self, x, state):
        S, prev = state
        base = self._base(x, prev)
        self._publish_v_first(base["v"])
        o, r, k, v, S = self._scan(base, S)
        return self._output(o, r, k, v, base["g"]), (S, x[:, -1:].float())

    def flatten_state(self, state):
        return [state[0], state[1]]

    def unflatten_state(self, flat):
        S, prev = flat
        return (S, prev)


__all__ = ["SRDNMixer", "StateAdapter"]
