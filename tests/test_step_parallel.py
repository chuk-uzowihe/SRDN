#!/usr/bin/env python3
"""Rollout decode (.step, token by token) == teacher-forced full-seq forward, at every
position, for every arch with a step path (m2rnn is full-sequence-forward only).

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
    p.add_argument("--arch", choices=["srdn", "transformer", "mamba3", "mamba2", "gdn2", "gdn1",
                                      "rwkv7", "all"], default="all")
    args = p.parse_args()
    V = 261
    cpu, cuda = torch.device("cpu"), (torch.device("cuda") if torch.cuda.is_available() else None)

    if args.arch in ("srdn", "all"):
        # SwiGLU channel: the only srdn configuration with a rollout step path (the faithful
        # channel mix is full-sequence-forward only). CUDA-only (fla triton output ops).
        if cuda is None:
            print("srdn: SKIP (CUDA)")
        else:
            torch.manual_seed(0)
            _report("srdn", srdn.build_srdn(V, 64, 2, 2.0, faithful_channel_mix=False), cuda, 2e-2)
    if args.arch in ("transformer", "all"):
        torch.manual_seed(0)
        _report("transformer", srdn.build_transformer(V, 64, 2, 4, 2.0), cpu, 1e-4)
    if args.arch in ("mamba3", "all"):
        # Mamba-3 decode runs the combined kernel at seq-len 1 with Input_States (fla's
        # own CUTE step kernel is broken at our pin). CUDA-only; loose tol (bf16 kernel).
        if cuda is None:
            print("mamba3             SKIP (CUDA)")
        else:
            torch.manual_seed(0)
            # tol: the SISO decode step and the chunked parallel op are DIFFERENT bf16
            # kernels, and inductor/autotune recompiles shift the worst-element error by
            # 1-3e-2 across equivalent builds (measured); 5e-2 is outside that noise band
            _report("mamba3", srdn.build_mamba3(V, 64, 2), cuda, 5e-2)
    if args.arch in ("mamba2", "all"):
        # Mamba-2 decode: mamba_ssm's Triton selective_state_update + conv-buffer roll vs the
        # full-seq mamba_chunk_scan_combined kernel. CUDA-only.
        if cuda is None:
            print("mamba2             SKIP (CUDA)")
        else:
            torch.manual_seed(0)
            _report("mamba2", srdn.build_mamba2(V, 64, 2), cuda, 2e-2)
    if args.arch in ("gdn2", "all"):
        if cuda is None:
            print("gdn2: SKIP (CUDA)")
        else:
            torch.manual_seed(0)
            _report("gdn2", srdn.build_gdn2(V, 64, 2, 4, 16, 2.0), cuda, 2e-3)
    if args.arch in ("gdn1", "all"):
        if cuda is None:
            print("gdn1: SKIP (CUDA)")
        else:
            # tol: fla's gated_delta_rule chunk kernel does TF32 tl.dot (~1e-3 relative);
            # regrouping the sequence into per-step calls rounds differently, giving a flat
            # (non-growing) few-e-3 band -- unlike gdn2's vendored fp32-dot kernel (2e-3).
            torch.manual_seed(0)
            _report("gdn1", srdn.build_gdn1(V, 64, 2, 4, 16, 2.0), cuda, 2e-2)
    if args.arch in ("rwkv7", "all"):
        # RWKV-7: decode (fused_recurrent) vs full-seq, both eval-mode (<64 -> same kernel).
        # CUDA-only; loose tol (bf16 LoRA path). 2-layer exercises v_first produce->consume.
        if cuda is None:
            print("rwkv7: SKIP (CUDA)")
        else:
            torch.manual_seed(0)
            _report("rwkv7", srdn.build_rwkv7(V, 64, 2), cuda, 2e-2)


if __name__ == "__main__":
    main()
