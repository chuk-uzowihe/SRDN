#!/usr/bin/env python3
"""enwik8 char-LM: train one architecture, report bits-per-byte (bpc = CE/ln2).

All archs use the identical dense SwiGLU channel mixer (the standard "swap only the
token mixer" comparison). Byte-level, vocab 256.

  uv run python experiments/train_lm.py --arch srdn --steps 1500 --out results/enwik8_srdn.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import srdn
from tasks.enwik8 import VOCAB_SIZE, load_enwik8, batch as lm_batch


def build(arch, args):
    d, L, H = args.d_model, args.n_layers, args.n_heads
    if arch == "srdn":
        return srdn.build_srdn(VOCAB_SIZE, d, L, H, d // H, args.ffn_mult, short_conv=bool(args.short_conv))
    if arch == "transformer":
        return srdn.build_transformer(VOCAB_SIZE, d, L, H, args.ffn_mult, max_seq_len=args.seq_len + 8)
    if arch == "mamba3":
        return srdn.build_mamba3(VOCAB_SIZE, d, L, args.ffn_mult, state_size=args.mamba_state,
                                 head_dim=args.mamba_head_dim)
    if arch == "m2rnn":
        return srdn.build_m2rnn(VOCAB_SIZE, d, L, H, args.m2rnn_head_dim, args.ffn_mult, kernel_size=4)
    if arch == "gdn2":
        return srdn.build_gdn2(VOCAB_SIZE, d, L, H, args.gdn2_head_dim, args.ffn_mult, gdn2_repo=args.gdn2_repo)
    raise ValueError(arch)


@torch.no_grad()
def eval_bpc(model, split, args, device, gen, n_batches=8):
    model.eval()
    tot = 0.0
    for _ in range(n_batches):
        toks = lm_batch(split, args.batch, args.seq_len, gen, device)
        logits = model.logits(toks[:, :-1])
        if hasattr(model, "pop_router_logits"):
            model.pop_router_logits()
        ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE).float(), toks[:, 1:].reshape(-1))
        tot += ce.item()
    model.train()
    return (tot / n_batches) / math.log(2.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=["srdn", "transformer", "mamba3", "m2rnn", "gdn2"])
    p.add_argument("--short-conv", dest="short_conv", action="store_true", default=True)
    p.add_argument("--no-short-conv", dest="short_conv", action="store_false")
    p.add_argument("--d-model", dest="d_model", type=int, default=128)
    p.add_argument("--n-layers", dest="n_layers", type=int, default=2)
    p.add_argument("--n-heads", dest="n_heads", type=int, default=4)
    p.add_argument("--ffn-mult", dest="ffn_mult", type=float, default=2.0)
    p.add_argument("--seq-len", dest="seq_len", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval-every", dest="eval_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mamba-state", dest="mamba_state", type=int, default=64)
    p.add_argument("--mamba-head-dim", dest="mamba_head_dim", type=int, default=48)
    p.add_argument("--m2rnn-head-dim", dest="m2rnn_head_dim", type=int, default=50)
    p.add_argument("--gdn2-head-dim", dest="gdn2_head_dim", type=int, default=28)
    p.add_argument("--gdn2-repo", dest="gdn2_repo", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    train, val, _ = load_enwik8()
    model = build(args.arch, args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    nparams = model.num_params()
    print(f"arch={args.arch} params={nparams/1e3:.1f}k d={args.d_model} L={args.n_layers}", flush=True)

    t0 = time.perf_counter()
    history = []
    for step in range(1, args.steps + 1):
        toks = lm_batch(train, args.batch, args.seq_len, gen, device)
        logits = model.logits(toks[:, :-1])
        if hasattr(model, "pop_router_logits"):
            model.pop_router_logits()
        ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE).float(), toks[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            vbpc = eval_bpc(model, val, args, device, gen)
            history.append({"step": step, "train_bpc": ce.item() / math.log(2.0), "val_bpc": vbpc})
            print(f"step {step} train_bpc={ce.item()/math.log(2):.4f} val_bpc={vbpc:.4f}", flush=True)

    result = {"task": "enwik8", "arch": args.arch, "args": vars(args), "params": nparams,
              "final_val_bpc": history[-1]["val_bpc"], "history": history,
              "wall_sec": round(time.perf_counter() - t0, 1)}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
