#!/usr/bin/env python3
"""At init the SRDN cell (state-reading RWKV-7) MUST equal fla's native RWKV-7 (same weights).

The cell reuses the fla RWKV7Attention submodule for all params + the output path, and runs its
own sequential DPLR scan. The zero-init adapters (LoRA B=0 / full-rank W=0) make the conditioning
inert at init, so forward(x) must match fla RWKV7Attention.forward(norm(x), v_first) to kernel
tolerance -- checked for BOTH layer 0 (v_first produced) and layer>0 (v_first consumed), against
BOTH fla kernel paths (eval short-seq -> fused_recurrent; train / long-seq -> the chunk kernel
the rwkv7 baseline actually trains with), and for the compiled Tier-1.5 scan (fuse_scan=True).
Also checks that the v_first bus carries exactly fla's v_first at init, step==forward, that the
neg_eigval rescale puts the DPLR transition eigenvalues in exactly [-1, 1], and that the
state-reading path RECEIVES GRADIENT at init (the zero-output init must be a live path, not a
fixed point -- a gamma x zero-LoRA product would have identically zero gradient).

  python tests/test_srdn_equiv.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from srdn.ops.srdn import SRDNMixer, NEG_A_SCALE, _DECAY

TOL = 1e-3  # vs the fused_recurrent kernel: true-fp32 both sides, so this is generous


def _kernel_noise_floor(dev):
    """fla's fused_recurrent and chunk kernels disagree with EACH OTHER (~6e-4 at T=48: the
    chunk kernel's triton matmuls use tf32 accumulation, our scan and fused_recurrent are true
    fp32). Comparisons against the chunk kernel are judged relative to this measured floor,
    not an absolute tolerance -- systematic drift still fails, kernel precision doesn't."""
    from fla.layers import RWKV7Attention
    torch.manual_seed(0)
    m = RWKV7Attention(mode="chunk", hidden_size=128, head_dim=32, layer_idx=0,
                       num_hidden_layers=3, fuse_norm=False).to(dev)
    torch.nn.init.normal_(m.o_proj.weight, std=0.5)
    x = torch.randn(2, 48, 128, device=dev)
    with torch.no_grad():
        m.eval();  a, _, _, _ = m(x, use_cache=False, v_first=None)   # T<=64 + eval -> fused
        m.train(); b, _, _, _ = m(x, use_cache=False, v_first=None)   # train -> chunk kernel
    return (a - b).abs().max().item() / (a.abs().max().item() + 1e-9)


def _equiv(dev, mode, lora, layer_idx, *, T=48, train=False, fuse_scan=False, label="", tol=TOL):
    torch.manual_seed(0)
    B, D, hd = 2, 128, 32
    m = SRDNMixer(D, head_dim=hd, layer_idx=layer_idx, num_layers=3,
                       content_read_mode=mode, use_lora=lora, fuse_scan=fuse_scan).to(dev)
    m.train(train)
    torch.nn.init.normal_(m.rwkv.o_proj.weight, std=0.5)  # RWKV-7 zero-inits o_proj; un-vacuum the test
    x = torch.randn(B, T, D, device=dev)
    vf = None if layer_idx == 0 else torch.randn(B, T, m.rwkv.value_dim, device=dev)
    if vf is not None:
        m.bus.value = vf                                  # layer>0 consumes v_first from the bus
    with torch.no_grad():
        mine = m(x)
        dt = m.rwkv.r_proj.weight.dtype
        m.rwkv.train(train)                               # train => fla dispatches the chunk kernel
        o_fla, _, _, vf_fla = m.rwkv(m.norm(x).to(dt), use_cache=False, v_first=vf)
    err = (mine - o_fla.float()).abs().max().item()
    scale = o_fla.float().abs().max().item() + 1e-9
    ok = err / scale < tol and scale > 1e-3
    if layer_idx == 0:
        # the bus must carry exactly fla's v_first at init (adapters zero => conditioned v == base v)
        vf_err = (m.bus.value - vf_fla.float()).abs().max().item()
        vf_scale = vf_fla.float().abs().max().item() + 1e-9
        ok &= vf_err / vf_scale < tol
    print(f"  layer={layer_idx} mode={mode:8s} lora={int(lora)} {label:14s} vs fla: "
          f"rel={err/scale:.2e} scale={scale:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def _eig_range(dev, a_scale, n=2000, dh=32):
    """ACHIEVABLE eigenvalue range (inf/sup) of M = diag(exp w) - (a (.) kk) kk^T.
    The extreme spectrum is the HOMOGENEOUS limit (all channels at the same w,a); per-channel
    random sampling only dilutes the rank-1 shift, so it under-shoots the true floor. Floor =
    most-decayed + max-removal (w=_DECAY, a=a_scale); ceiling = no-decay + no-removal (w=0,a=0)."""
    torch.manual_seed(1)
    kk = torch.randn(n, dh, device=dev); kk = kk / kk.norm(dim=-1, keepdim=True)
    # floor: w=_DECAY (exp=0.5453) and a=a_scale, both constant across channels
    Mf = math.exp(_DECAY) * torch.eye(dh, device=dev) - a_scale * kk.unsqueeze(-1) * kk.unsqueeze(-2)
    lo = torch.linalg.eigvalsh(Mf).min().item()
    # ceiling: w=0 (exp=1), a=0 -> identity; max eigenvalue is 1
    hi = 1.0
    return lo, hi


def _grad_flow(dev, lora):
    """The entry point of the state-reading path (LoRA B / full-rank W) must get a NONZERO grad
    at init. (A and q_read are legitimately zero-grad at step 1 -- they wake up once B moves,
    standard LoRA behavior.)"""
    torch.manual_seed(0)
    B, T, D, hd = 2, 24, 128, 32
    m = SRDNMixer(D, head_dim=hd, layer_idx=0, num_layers=2, content_read_mode="split",
                       use_lora=lora, fuse_scan=False).to(dev)
    torch.nn.init.normal_(m.rwkv.o_proj.weight, std=0.5)
    m(torch.randn(B, T, D, device=dev)).pow(2).mean().backward()
    entry = "B" if lora else "W"
    dead = [n for n, p in m.named_parameters()
            if ("adapt" in n and n.endswith("." + entry))
            and (p.grad is None or p.grad.abs().max().item() == 0.0)]
    ok = not dead
    print(f"grad-flow lora={int(lora)}: entry params ({entry}) all live: "
          f"{'PASS' if ok else 'FAIL ' + str(dead)}")
    return ok


def main():
    assert torch.cuda.is_available(), "CUDA required (fla triton)"
    dev = torch.device("cuda")
    allok = True
    print("init equivalence vs native fla RWKV-7 (eval short-seq -> fused_recurrent kernel):")
    for layer_idx in (0, 1):
        for mode in ("shared", "split", "per_proj"):
            for lora in (True, False):
                allok &= _equiv(dev, mode, lora, layer_idx, label="[eval/fused]")

    floor = _kernel_noise_floor(dev)
    tol_chunk = max(5 * floor, 1e-4)   # scale to the measured kernel-vs-kernel disagreement
    print(f"init equivalence vs the TRAINING chunk kernel (train mode, T=128; "
          f"fla fused-vs-chunk noise floor {floor:.2e} -> tol {tol_chunk:.2e}):")
    for layer_idx in (0, 1):
        allok &= _equiv(dev, "per_proj", True, layer_idx, T=128, train=True,
                        label="[train/chunk]", tol=tol_chunk)

    print("compiled Tier-1.5 scan (fuse_scan=True) vs the chunk kernel:")
    allok &= _equiv(dev, "per_proj", True, 0, T=128, train=True, fuse_scan=True,
                    label="[compiled]", tol=tol_chunk)

    for lora in (True, False):
        allok &= _grad_flow(dev, lora)

    # step == forward (rollout faithfulness), inert-adapter init
    torch.manual_seed(0)
    B, T, D, hd = 2, 48, 128, 32
    m = SRDNMixer(D, head_dim=hd, layer_idx=0, num_layers=2, content_read_mode="split",
                       fuse_scan=False).to(dev).eval()
    x = torch.randn(B, T, D, device=dev)
    with torch.no_grad():
        full = m(x)
        outs, st = [], m.init_state(B, dev)
        for t in range(T):
            o, st = m.step(x[:, t], st)
            outs.append(o)
        seq = torch.stack(outs, dim=1)
    err = (full - seq).abs().max().item() / (full.abs().max().item() + 1e-9)
    print(f"step==forward: rel={err:.2e}  {'PASS' if err < TOL else 'FAIL'}")
    allok &= err < TOL

    # eigenvalue range: native a=sigmoid -> ~[-0.455,1]; neg a=NEG_A_SCALE*sigmoid -> exactly [-1,1]
    nlo, nhi = _eig_range(dev, 1.0)
    glo, ghi = _eig_range(dev, NEG_A_SCALE)
    print(f"eigenvalue range  native(a=sigmoid): [{nlo:.3f}, {nhi:.3f}]   "
          f"(expect ~[{math.exp(_DECAY)-1:.3f}, 1.000])")
    print(f"eigenvalue range  neg(a={NEG_A_SCALE:.3f}*sigmoid): [{glo:.3f}, {ghi:.3f}]   (expect ~[-1.000, 1.000])")
    ok_eig = abs(glo + 1.0) < 0.05 and abs(ghi - 1.0) < 0.05 and abs(nlo - (math.exp(_DECAY) - 1)) < 0.05
    print(f"eigenvalue rescale: {'PASS' if ok_eig else 'FAIL'}")
    allok &= ok_eig
    print("ALL", "PASS" if allok else "FAIL")


if __name__ == "__main__":
    main()
