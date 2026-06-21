#!/usr/bin/env python3
"""Parity gate for the no-baggage strip: the clean SRDN *cell* (ops/srdn.py) must be
the SAME token mixer as the original configurable srdnladder.LadderLM at the srdn5
config (rung=moe, moe_read=sx, cond_q, cond_gates, out_mode=none, decay=mamba).

We isolate the CELL by disabling the channel mixer on both sides (ffn_mult=0,
moe_ffn=False) -- the refactor deliberately moved everyone to one shared channel
mixer, so whole-LM FFN parity is not the goal; cell parity is. We copy the original's
weights into the clean model by structural correspondence and assert logits +
parameter gradients match to fp32 roundoff.

Dev-only: requires the private srdnladder repo. Skips without SRDNLADDER_ROOT.
  SRDNLADDER_ROOT=/path/to/srdnladder uv run python tests/test_parity.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srdn import build_srdn

_root = os.environ.get("SRDNLADDER_ROOT")
if not _root or not (Path(_root) / "srdnladder" / "ladder.py").exists():
    print("test_parity: SKIP (set SRDNLADDER_ROOT to the srdnladder checkout to run)")
    raise SystemExit(0)
sys.path.insert(0, str(Path(_root)))
from srdnladder.ladder import LadderConfig, LadderLM   # noqa: E402

_COND = {"W_qs", "gamma_q", "W_kx", "W_vx", "gamma_kx", "gamma_vx",
         "W_ax", "W_bx", "gamma_ax", "gamma_bx"}


def _to_orig_name(clean: str) -> str:
    """clean `blocks.{i}.mixer.{tail}` -> orig `layers.{i}.{(rung.)?}{tail}`."""
    if not clean.startswith("blocks."):
        return clean                                 # embed.weight / final_norm.weight
    _, idx, _mixer, *rest = clean.split(".")
    tail = ".".join(rest)
    if rest[0] in _COND:                             # conditioning params live under .rung.
        return f"layers.{idx}.rung.{tail}"
    return f"layers.{idx}.{tail}"


def _copy_weights(clean, orig) -> None:
    osd = dict(orig.named_parameters())
    with torch.no_grad():
        for name, p in clean.named_parameters():
            src = osd[_to_orig_name(name)]
            assert src.shape == p.shape, f"shape mismatch {name}: {p.shape} vs {src.shape}"
            p.copy_(src)


def _check(short_conv: bool) -> None:
    torch.manual_seed(0)
    V, d, L, H, dh = 137, 64, 2, 4, 16
    clean = build_srdn(V, d, L, H, dh, ffn_mult=0.0, short_conv=short_conv)   # FFN OFF -> cell only
    ocfg = LadderConfig(vocab_size=V, d_model=d, n_layers=L, n_heads=H, d_head=dh,
                        ffn_mult=0.0, rung="moe", moe_read="sx", cond_q=True, cond_gates=True,
                        out_mode="none", moe_ffn=False, decay_param="mamba", short_conv=short_conv)
    orig = LadderLM(ocfg, tie_head=True)
    _copy_weights(clean, orig)
    nclean = sum(p.numel() for p in clean.parameters())
    norig = orig.num_params()

    toks = torch.randint(0, V, (3, 24))
    lc = clean.logits(toks); clean.pop_router_logits()
    lo = orig.logits(toks); orig.pop_router_logits()
    fwd = (lc - lo).abs().max().item()

    def scalar(z):
        return (z.float().tanh() * torch.linspace(0.1, 1.0, z.shape[-1])).sum()
    clean.zero_grad(); scalar(clean.logits(toks)).backward(); clean.pop_router_logits()
    orig.zero_grad(); scalar(orig.logits(toks)).backward(); orig.pop_router_logits()
    go = dict(orig.named_parameters())
    gmax = 0.0
    for n, p in clean.named_parameters():
        if p.grad is None:
            continue
        og = go[_to_orig_name(n)].grad
        gmax = max(gmax, (p.grad - og).abs().max().item())

    ok = fwd < 1e-5 and gmax < 1e-4 and nclean == norig
    print(f"cell conv={short_conv}: params clean={nclean} orig={norig} | "
          f"logit_diff={fwd:.2e} grad_diff={gmax:.2e} {'PASS' if ok else 'FAIL'}")


def main():
    for short_conv in (False, True):
        _check(short_conv)


if __name__ == "__main__":
    main()
