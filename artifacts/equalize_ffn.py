#!/usr/bin/env python3
"""Verify/re-solve the parameter-equalized FFN knobs for this repo's archs (blog protocol).

ANCHOR = srdn per_proj at its typical FFN width (hardcoded targets below, recorded in
artifacts/param_equalization.json). Baselines are widened UP to the anchor by FFN width only
-- mixers untouched. Two families: 'swiglu' (everyone on the shared SwiGLU ChannelMixer) and
'faithful' (enwik8/FRJT: srdn/rwkv7 on the native token-shift+sqReLU channel mix via
--rwkv7-hidden-ratio; float ratios realize int(d*ratio)).

  python artifacts/equalize_ffn.py     # prints the solved table for THIS repo's builders
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import srdn  # noqa: E402

V, D, H = 256, 128, 4
# anchor = srdn per_proj @ typical width (see module docstring)
TARGETS = {("swiglu", 2): 589824, ("swiglu", 4): 1155072,
           ("faithful", 2): 655616, ("faithful", 4): 1286656}


def build(arch, L, *, ffn=2.0, hidden_ratio=4.0):
    if arch == "transformer":
        return srdn.build_transformer(V, D, L, H, ffn, max_seq_len=4096)
    if arch == "mamba3":
        return srdn.build_mamba3(V, D, L, ffn, state_size=128, head_dim=64)  # d128-final config
    if arch == "mamba2":
        return srdn.build_mamba2(V, D, L, ffn, state_size=128, head_dim=64)  # natural, matches mamba3
    # NATURAL head dims (dh32 = d/H); FFN width does the equalizing. dh32 makes matrix-state
    # size EXACTLY matched across gdn2/m2rnn/rwkv7/srdn: 4x32^2 = 4096/layer.
    if arch == "m2rnn":
        return srdn.build_m2rnn(V, D, L, H, 32, ffn)
    if arch == "gdn2":
        return srdn.build_gdn2(V, D, L, H, 32, ffn)
    if arch == "gdn1":
        return srdn.build_gdn1(V, D, L, H, 32, ffn)
    if arch == "rwkv7":
        return srdn.build_rwkv7(V, D, L, ffn, head_dim=32)
    if arch == "rwkv7_faithful":
        return srdn.build_rwkv7(V, D, L, ffn, head_dim=32, faithful_channel_mix=True,
                                hidden_ratio=hidden_ratio)
    raise ValueError(arch)


def solve_swiglu(arch, L, target):
    best = None
    for hidden in range(64, 4096 + 1, 64):  # ChannelMixer quantizes hidden to /64
        p = build(arch, L, ffn=hidden / D).num_params()
        if best is None or abs(p - target) < abs(best[1] - target):
            best = (hidden / D, p)
        if p > target + 60000:
            break
    return best


def solve_faithful_ratio(L, target):
    lo, hi = 0.5, 40.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if build("rwkv7_faithful", L, hidden_ratio=mid).num_params() < target:
            lo = mid
        else:
            hi = mid
    r = round((lo + hi) / 2 * D) / D
    return r, build("rwkv7_faithful", L, hidden_ratio=r).num_params()


def main():
    for family in ("swiglu", "faithful"):
        for L in (2, 4):
            target = TARGETS[(family, L)]
            print(f"\n== family={family} L={L} d{D} (anchor target {target:,}) ==")
            archs = ["transformer", "mamba3", "mamba2", "m2rnn", "gdn2", "gdn1"]
            archs.append("rwkv7_faithful" if family == "faithful" else "rwkv7")
            for arch in archs:
                if arch == "rwkv7_faithful":
                    r, p = solve_faithful_ratio(L, target)
                    knob = f"--rwkv7-hidden-ratio {r}"
                else:
                    m, p = solve_swiglu(arch, L, target)
                    knob = f"--ffn-mult {m}"
                print(f"  {arch:16s} {knob:28s} {p:>9,}  ({100*(p-target)/target:+.2f}%)")


if __name__ == "__main__":
    main()
