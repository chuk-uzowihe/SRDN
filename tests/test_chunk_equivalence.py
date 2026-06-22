#!/usr/bin/env python3
"""chunked_logits == full-sequence logits (forward AND parameter gradients), for the
recurrence-complete models. This is the correctness gate for sequence checkpointing:
detach_boundaries=False must give exactly the full-BPTT gradient (kernel tolerance),
so the chunked path is a pure memory lever, not an approximation.

SRDN runs on CPU (fp32, exact arithmetic). GDN-2 / M2RNN use triton/xma kernels and
are CUDA-only; their chunked == full holds to kernel fp32 backward tolerance
(relative). Transformer / Mamba-3 are parallelizable (not chunkable) -- skipped.

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

    model.zero_grad(set_to_none=True)
    loss_of(model.chunked_logits(tokens, chunk_size=chunk_size,
                                 detach_boundaries=False, remat_chunks=True)).backward()
    g_chunk = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    full = model.logits(tokens)
    chunk = model.chunked_logits(tokens, chunk_size=chunk_size, detach_boundaries=False, remat_chunks=True)
    ld = (full - chunk).abs().max().item()
    rel, gd, worst = 0.0, 0.0, ""
    for k in set(g_full) | set(g_chunk):
        d = (g_full[k] - g_chunk[k]).abs().max().item()
        scale = max(g_full[k].abs().max().item(), 1e-12)
        gd = max(gd, d)
        if d / scale > rel:
            rel, worst = d / scale, k
    print(f"  worst rel-grad param: {worst} rel={rel:.2e}")
    return ld, rel


def _run(name, model, *, device, chunk_size, tol_l, tol_r):
    model = model.to(device).train()
    torch.manual_seed(0)
    toks = torch.randint(0, model.vocab_size, (3, 40), device=device)
    ld, rel = _grad_diff(model, toks, chunk_size=chunk_size)
    ok = ld < tol_l and rel < tol_r
    print(f"{name}: logit_diff={ld:.2e} grad_rel={rel:.2e} {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["srdn", "m2rnn", "gdn2", "all"], default="all")
    args = p.parse_args()
    V = 261
    cuda = torch.cuda.is_available()

    if args.arch in ("srdn", "all"):
        for sc in (False, True):
            _run(f"srdn conv={sc}", srdn.build_srdn(V, 64, 2, 4, 16, 2.0, short_conv=sc),
                 device="cpu", chunk_size=13, tol_l=1e-4, tol_r=1e-3)   # exact fp32 (reordered)
    if args.arch in ("m2rnn", "all"):
        if not cuda:
            print("m2rnn: SKIP (CUDA required)")
        else:
            _run("m2rnn", srdn.build_m2rnn(V, 64, 2, 4, 16, 2.0, kernel_size=4),
                 device="cuda", chunk_size=16, tol_l=1e-3, tol_r=1e-2)   # xma kernel tolerance
    if args.arch in ("gdn2", "all"):
        if not cuda:
            print("gdn2: SKIP (CUDA required)")
        else:
            _run("gdn2", srdn.build_gdn2(V, 64, 2, 4, 16, 2.0, expand_v=1.0),
                 device="cuda", chunk_size=16, tol_l=1e-3, tol_r=1e-2)   # triton kernel tolerance


if __name__ == "__main__":
    main()
