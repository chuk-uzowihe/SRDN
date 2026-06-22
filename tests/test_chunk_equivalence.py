#!/usr/bin/env python3
"""chunked_logits == logits (forward AND gradient) for the recurrence-complete archs.

This is the correctness gate for sequence checkpointing: with detach_boundaries=False
the chunked sqrt-exact-BPTT pass must reproduce the full-sequence forward + gradient
(to fp32 / triton-kernel tolerance). SRDN runs on CPU; M2RNN/GDN-2 need CUDA.

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
    p.add_argument("--arch", choices=["srdn", "m2rnn", "gdn2", "all"], default="all")
    args = p.parse_args()
    V = 261
    if args.arch in ("srdn", "all"):
        torch.manual_seed(0)
        for sc in (False, True):
            _report(f"srdn conv={int(sc)}", srdn.build_srdn(V, 64, 2, 4, 16, 2.0, short_conv=sc),
                    torch.device("cpu"), chunk_size=13, tol_logit=1e-4, tol_rel=1e-3)
    cuda = torch.cuda.is_available()
    if args.arch in ("m2rnn", "all"):
        if not cuda:
            print("m2rnn: SKIP (CUDA)")
        else:
            torch.manual_seed(0)
            _report("m2rnn (twin)", srdn.build_m2rnn(V, 64, 2, 4, 16, 2.0, kernel_size=4),
                    torch.device("cuda"), chunk_size=16, tol_logit=1e-3, tol_rel=1e-2)
    if args.arch in ("gdn2", "all"):
        if not cuda:
            print("gdn2: SKIP (CUDA)")
        else:
            torch.manual_seed(0)
            m = srdn.build_gdn2(V, 64, 2, 4, 16, 2.0).cuda().train()  # chunk kernel path
            _report("gdn2", m, torch.device("cuda"), chunk_size=16, tol_logit=1e-3, tol_rel=1e-2)


if __name__ == "__main__":
    main()
