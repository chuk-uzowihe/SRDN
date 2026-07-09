#!/usr/bin/env python3
"""chunked_logits == logits (forward AND gradient) for the recurrence-complete archs.

This is the correctness gate for sequence checkpointing: with detach_boundaries=False
the chunked sqrt-exact-BPTT pass must reproduce the full-sequence forward + gradient
(to fp32 / triton-kernel tolerance). CUDA-only (fla/triton kernels).

  uv run python tests/test_chunk_equivalence.py --arch all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import srdn


def _grad_diff(model, tokens, *, chunk_size):
    def loss_of(logits):
        w = torch.linspace(0.1, 1.0, logits.shape[-1], device=logits.device)
        return (logits.float().tanh() * w).sum()

    model.zero_grad(set_to_none=True)
    loss_of(model.logits(tokens)).backward()
    g_full = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    full = model.logits(tokens).detach()

    model.zero_grad(set_to_none=True)
    chunked = model.chunked_logits(tokens, chunk_size=chunk_size, detach_boundaries=False, remat_chunks=True)
    loss_of(chunked).backward()

    logit_diff = (full - chunked.detach()).abs().max().item()
    rel = 0.0
    for n, p in model.named_parameters():
        if p.grad is None or n not in g_full:
            continue
        d = (p.grad - g_full[n]).abs().max().item()
        rel = max(rel, d / max(g_full[n].abs().max().item(), 1e-12))
    return logit_diff, rel


def _report(name, model, device, *, chunk_size, tol_logit, tol_rel):
    model = model.to(device)
    toks = torch.randint(0, model.vocab_size, (3, 40), device=device)
    ld, rel = _grad_diff(model, toks, chunk_size=chunk_size)
    ok = ld < tol_logit and rel < tol_rel
    print(f"{name:18s} logit_diff={ld:.2e} grad_rel={rel:.2e} {'PASS' if ok else 'FAIL'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["srdn", "gdn2", "gdn1", "rwkv7", "all"], default="all")
    args = p.parse_args()
    V = 261
    cuda = torch.cuda.is_available()
    if args.arch in ("gdn2", "all"):
        if not cuda:
            print("gdn2: SKIP (CUDA)")
        else:
            torch.manual_seed(0)
            m = srdn.build_gdn2(V, 64, 2, 4, 16, 2.0).cuda().train()  # chunk kernel path
            _report("gdn2", m, torch.device("cuda"), chunk_size=16, tol_logit=1e-3, tol_rel=1e-2)
    if args.arch in ("gdn1", "all"):
        if not cuda:
            print("gdn1: SKIP (CUDA)")
        else:
            # looser tols than gdn2: fla's gated_delta_rule kernels do TF32 tl.dot
            # (~1e-3 relative), so chunk-boundary regrouping shifts logits/grads by a flat
            # few-e-3 band (measured 7e-3 / 1.2e-2) that is precision, not state-carry error.
            torch.manual_seed(0)
            m = srdn.build_gdn1(V, 64, 2, 4, 16, 2.0).cuda().train()  # chunk kernel path
            _report("gdn1", m, torch.device("cuda"), chunk_size=16, tol_logit=2e-2, tol_rel=5e-2)
    if args.arch in ("rwkv7", "all"):
        if not cuda:
            print("rwkv7: SKIP (CUDA)")
        else:
            torch.manual_seed(0)
            m = srdn.build_rwkv7(V, 128, 2, 2.0, head_dim=32).cuda().train()  # chunk kernel path
            _report("rwkv7", m, torch.device("cuda"), chunk_size=16, tol_logit=1e-3, tol_rel=1e-2)
    if args.arch in ("srdn", "all"):
        if not cuda:
            print("srdn: SKIP (CUDA)")
        else:
            # SwiGLU channel (the chunkable configuration). The faithful channel mix breaks
            # chunking (own token-shift state) and must be rejected by Block.chunkable.
            torch.manual_seed(0)
            m = srdn.build_srdn(V, 128, 2, 2.0, head_dim=32, fuse_scan=False,
                                faithful_channel_mix=False).cuda().train()
            _report("srdn", m, torch.device("cuda"), chunk_size=16, tol_logit=1e-3, tol_rel=1e-2)
            mf = srdn.build_srdn(V, 128, 2, 2.0, head_dim=32, fuse_scan=False)  # faithful FFN
            guarded = not any(b.chunkable for b in mf.blocks)
            print(f"srdn faithful-FFN chunk guard: {'PASS' if guarded else 'FAIL'}")


if __name__ == "__main__":
    main()
