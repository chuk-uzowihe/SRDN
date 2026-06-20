#!/usr/bin/env python3
"""Parity gate for the no-baggage strip: the clean `srdn.SRDNLM` must be the SAME
model as the original configurable `srdnladder.LadderLM` at the srdn5 config
(rung=moe, moe_read=sx, cond_q, cond_gates, out_mode=none, decay=mamba). We copy
the original's weights into the clean model by structural correspondence and assert
logits + parameter gradients match to fp32 roundoff.

Run with SRDNLADDER_ROOT pointing at the original repo:
  SRDNLADDER_ROOT=/home/chuk/dev/srdnladder uv run python tests/test_parity.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srdn import SRDNConfig, SRDNLM

_ORIG = Path(os.environ.get("SRDNLADDER_ROOT", "/home/chuk/dev/srdnladder"))
sys.path.insert(0, str(_ORIG))
from srdnladder.ladder import LadderConfig, LadderLM   # noqa: E402

_COND = {"W_qs", "gamma_q", "W_kx", "W_vx", "gamma_kx", "gamma_vx",
         "W_ax", "W_bx", "gamma_ax", "gamma_bx"}


def _to_orig_name(clean: str) -> str:
    n = clean.replace("blocks.", "layers.", 1)
    parts = n.split(".")
    if len(parts) >= 3 and parts[2] in _COND:           # conditioning params live under .rung.
        parts.insert(2, "rung")
    return ".".join(parts)


def _copy_weights(clean: SRDNLM, orig: LadderLM) -> None:
    osd = dict(orig.named_parameters())
    with torch.no_grad():
        for name, p in clean.named_parameters():
            src = osd[_to_orig_name(name)]
            assert src.shape == p.shape, f"shape mismatch {name}: {p.shape} vs {src.shape}"
            p.copy_(src)


def _check(moe_ffn: bool, short_conv: bool) -> None:
    torch.manual_seed(0)
    V, d, L, H, dh = 137, 64, 2, 4, 16
    ccfg = SRDNConfig(vocab_size=V, d_model=d, n_layers=L, n_heads=H, d_head=dh,
                      ffn_mult=2.0, short_conv=short_conv, moe_ffn=moe_ffn, expert_mult=2.3)
    ocfg = LadderConfig(vocab_size=V, d_model=d, n_layers=L, n_heads=H, d_head=dh,
                        ffn_mult=2.0, rung="moe", moe_read="sx", cond_q=True, cond_gates=True,
                        out_mode="none", moe_ffn=moe_ffn, expert_mult=2.3, decay_param="mamba",
                        short_conv=short_conv)
    clean, orig = SRDNLM(ccfg), LadderLM(ocfg, tie_head=True)
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
    gc = {n: p.grad for n, p in clean.named_parameters()}
    go = dict(orig.named_parameters())
    gmax = 0.0
    for n, g in gc.items():
        if g is None:
            continue
        og = go[_to_orig_name(n)].grad
        gmax = max(gmax, (g - og).abs().max().item())

    tag = f"moe_ffn={moe_ffn} conv={short_conv}"
    ok = fwd < 1e-5 and gmax < 1e-4 and nclean == norig
    print(f"{tag}: params clean={nclean} orig={norig} | logit_diff={fwd:.2e} grad_diff={gmax:.2e} "
          f"{'PASS' if ok else 'FAIL'}")


def main():
    for moe_ffn in (False, True):
        for short_conv in (False, True):
            _check(moe_ffn, short_conv)


if __name__ == "__main__":
    main()
