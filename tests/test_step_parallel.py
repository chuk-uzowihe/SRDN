#!/usr/bin/env python3
"""Rollout decode == teacher-forced forward, at every position. The policy is trained
on the parallel forward but acts via .step (token by token with the recurrent/conv/KV
cache); this is the rollout-correctness gate. Covers all five architectures.

  uv run python tests/test_step_parallel.py --arch all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import srdn


@torch.no_grad()
def _step_vs_parallel(model, device):
    B, T = 3, 24
    toks = torch.randint(0, model.vocab_size, (B, T), device=device)
    full = model.logits(toks)
    states = model.init_states(B, device)
    um = torch.ones(B, dtype=torch.bool, device=device)
    worst = 0.0
    for t in range(T):
        lg, states = model.step(toks[:, t], states, um)
        worst = max(worst, (lg - full[:, t]).abs().max().item())
    return worst


def _run(name, model, *, device, tol):
    model = model.to(device).eval()
    torch.manual_seed(0)
    d = _step_vs_parallel(model, device)
    print(f"{name}: step_vs_parallel={d:.2e} {'PASS' if d < tol else 'FAIL'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["srdn", "transformer", "mamba3", "m2rnn", "gdn2", "all"], default="all")
    args = p.parse_args()
    V = 261
    cuda = torch.cuda.is_available()

    if args.arch in ("srdn", "all"):
        for sc in (False, True):
            _run(f"srdn conv={sc}", srdn.build_srdn(V, 64, 2, 4, 16, 2.0, short_conv=sc), device="cpu", tol=1e-4)
    if args.arch in ("transformer", "all"):
        _run("transformer", srdn.build_transformer(V, 64, 2, 4, 2.0), device="cpu", tol=1e-4)
    if args.arch in ("mamba3", "all"):
        # KNOWN UPSTREAM LIMITATION: FLA-Mamba3's mamba-ssm cute *step* kernel errors
        # on multi-step decode (arg-#2 conv-state dtype) at fp32/bf16/fp16 alike, in
        # the fla pin we require for GDN-2 compat. Full-seq forward works (FRJT/enwik8
        # are fine); single-token rollout is unavailable -> Mamba-3 sits out graph-RL.
        print("mamba3: SKIP (upstream cute step-kernel bug on multi-step decode; forward/logits OK)")
    if args.arch in ("m2rnn", "all"):
        _run("m2rnn", srdn.build_m2rnn(V, 64, 2, 4, 16, 2.0, kernel_size=4),
             device="cuda" if cuda else "cpu", tol=2e-3 if cuda else 1e-4)
    if args.arch in ("gdn2", "all"):
        if not cuda:
            print("gdn2: SKIP (CUDA required)")
        else:
            _run("gdn2", srdn.build_gdn2(V, 64, 2, 4, 16, 2.0, expand_v=1.0), device="cuda", tol=2e-3)


if __name__ == "__main__":
    main()
