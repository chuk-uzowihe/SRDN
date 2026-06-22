#!/usr/bin/env python3
"""Rollout decode (.step, token by token) == teacher-forced full-seq forward, at every
position. The policy is trained on the parallel forward but acts via .step, so they
must agree. All five archs.

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
def _step_vs_parallel(model, tokens, device):
    B, T = tokens.shape
    full = model.logits(tokens)
    states = model.init_states(B, device)
    um = torch.ones(B, dtype=torch.bool, device=device)
    worst = 0.0
    for t in range(T):
        lg, states = model.step(tokens[:, t], states, um)
        worst = max(worst, (lg - full[:, t]).abs().max().item())
    return worst


def _report(name, model, device, tol):
    model = model.to(device).eval()
    toks = torch.randint(0, model.vocab_size, (3, 24), device=device)
    d = _step_vs_parallel(model, toks, device)
    print(f"{name:18s} step_vs_parallel={d:.2e} {'PASS' if d < tol else 'FAIL'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["srdn", "transformer", "mamba3", "m2rnn", "gdn2", "all"], default="all")
    args = p.parse_args()
    V = 261
    cpu, cuda = torch.device("cpu"), (torch.device("cuda") if torch.cuda.is_available() else None)

    if args.arch in ("srdn", "all"):
        torch.manual_seed(0)
        for sc in (False, True):
            _report(f"srdn conv={int(sc)}", srdn.build_srdn(V, 64, 2, 4, 16, 2.0, short_conv=sc), cpu, 1e-4)
    if args.arch in ("transformer", "all"):
        torch.manual_seed(0)
        _report("transformer", srdn.build_transformer(V, 64, 2, 4, 2.0), cpu, 1e-4)
    if args.arch in ("mamba3", "all"):
        # Mamba-3 has no working incremental decode at the pinned fla (mamba3_step_fn
        # kernel bug); it is a full-sequence-only baseline (FRJT, enwik8). No rollout
        # -> nothing to check here. See srdn/ops/mamba3.py.
        print("mamba3             SKIP (no incremental decode; full-sequence-only baseline)")
    if args.arch in ("m2rnn", "all"):
        dev = cuda or cpu
        torch.manual_seed(0)
        _report("m2rnn", srdn.build_m2rnn(V, 64, 2, 4, 16, 2.0, kernel_size=4), dev, 2e-3 if cuda else 1e-4)
    if args.arch in ("gdn2", "all"):
        if cuda is None:
            print("gdn2: SKIP (CUDA)")
        else:
            torch.manual_seed(0)
            _report("gdn2", srdn.build_gdn2(V, 64, 2, 4, 16, 2.0), cuda, 2e-3)


if __name__ == "__main__":
    main()
