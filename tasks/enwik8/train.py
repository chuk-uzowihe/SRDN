#!/usr/bin/env python3
"""enwik8 char-LM: train one architecture, report bits-per-byte (bpc = CE/ln2).

All archs use the identical dense SwiGLU channel mixer (the standard "swap only the
token mixer" comparison). Byte-level, vocab 256.

  uv run python tasks/enwik8/train.py --arch srdn --steps 1500 --out results/enwik8_srdn.json
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import srdn  # noqa: E402
from tasks.enwik8 import VOCAB_SIZE, load_enwik8, batch as lm_batch  # noqa: E402
from tasks.seeding import set_seed, torch_generator  # noqa: E402


def build(arch, args):
    d, L, H = args.d_model, args.n_layers, args.n_heads
    if arch == "transformer":
        return srdn.build_transformer(VOCAB_SIZE, d, L, H, args.ffn_mult, max_seq_len=args.seq_len + 8)
    if arch == "mamba3":
        return srdn.build_mamba3(VOCAB_SIZE, d, L, args.ffn_mult, state_size=args.mamba_state,
                                 head_dim=args.mamba_head_dim)
    if arch == "mamba2":
        return srdn.build_mamba2(VOCAB_SIZE, d, L, args.ffn_mult, state_size=args.mamba_state,
                                 head_dim=args.mamba_head_dim)
    if arch == "m2rnn":
        return srdn.build_m2rnn(VOCAB_SIZE, d, L, H, args.m2rnn_head_dim, args.ffn_mult, kernel_size=4)
    if arch == "gdn2":
        return srdn.build_gdn2(VOCAB_SIZE, d, L, H, args.gdn2_head_dim, args.ffn_mult)
    if arch == "gdn1":
        return srdn.build_gdn1(VOCAB_SIZE, d, L, H, args.gdn1_head_dim, args.ffn_mult)
    if arch == "rwkv7":
        return srdn.build_rwkv7(VOCAB_SIZE, d, L, args.ffn_mult, head_dim=args.rwkv7_head_dim,
                                faithful_channel_mix=args.rwkv7_faithful,
                                hidden_ratio=args.rwkv7_hidden_ratio)
    if arch == "srdn":
        return srdn.build_srdn(VOCAB_SIZE, d, L, args.ffn_mult, head_dim=args.rwkv7_head_dim,
                                     content_read_mode=args.sr_mode, use_lora=not args.sr_no_lora,
                                     neg_eigval=args.sr_neg_eigval,
                                     faithful_channel_mix=not args.sr_swiglu_ffn,
                                     hidden_ratio=args.rwkv7_hidden_ratio,
                                     read_rank=args.sr_read_rank)
    raise ValueError(arch)


@torch.no_grad()
def eval_bpc_sequential(model, split, args, device):
    """Deterministic full-split bpc: contiguous non-overlapping seq_len windows covering the
    entire split, every byte predicted exactly once (except each window's first byte, which
    has no context -- the standard windowed protocol). Comparable across runs and to
    published enwik8 numbers, unlike the random-window quick eval below."""
    model.eval()
    L, B = int(args.seq_len), int(args.batch)
    n_windows = (len(split) - 1) // L
    tot_ce, tot_tok = 0.0, 0
    for start in range(0, n_windows * L, L * B):
        rows = [torch.as_tensor(split[s:s + L + 1], dtype=torch.long)
                for s in range(start, min(start + L * B, n_windows * L), L)]
        toks = torch.stack(rows).to(device)
        logits = model.logits(toks[:, :-1])
        if hasattr(model, "pop_router_logits"):
            model.pop_router_logits()
        ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE).float(), toks[:, 1:].reshape(-1),
                             reduction="sum")
        tot_ce += float(ce)
        tot_tok += int(toks[:, 1:].numel())
    model.train()
    return (tot_ce / tot_tok) / math.log(2.0)


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
    p.add_argument("--arch", required=True, choices=["srdn", "transformer", "mamba3", "mamba2",
                                                     "m2rnn", "gdn2", "gdn1", "rwkv7"])
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
    p.add_argument("--gdn1-head-dim", dest="gdn1_head_dim", type=int, default=32)
    p.add_argument("--rwkv7-head-dim", dest="rwkv7_head_dim", type=int, default=32)
    p.add_argument("--rwkv7-faithful", dest="rwkv7_faithful", action="store_true",
                   help="paper-faithful RWKV-7: native channel mix (token-shift + sqReLU) vs shared SwiGLU")
    p.add_argument("--rwkv7-hidden-ratio", dest="rwkv7_hidden_ratio", type=float, default=4.0,
                   help="faithful channel-mix width (float ok; fla realizes int(d*ratio)) -- "
                        "parameter equalization knob, see artifacts/param_equalization.json")
    # state-reading RWKV-7 ablation axes (srdn)
    p.add_argument("--sr-mode", dest="sr_mode", choices=["shared", "split", "per_proj"], default="per_proj",
                   help="content-read queries: shared=1 (r double-duty), split=2, per_proj=>2 (one per "
                        "k/v/w/a; default)")
    p.add_argument("--sr-no-lora", dest="sr_no_lora", action="store_true",
                   help="full-rank state-reading adapters instead of low-rank (default LoRA on)")
    p.add_argument("--sr-neg-eigval", dest="sr_neg_eigval", action="store_true",
                   help="a = 2*sigmoid (eigenvalue range ~[-1,1]) vs native sigmoid ([0,1])")
    p.add_argument("--sr-read-rank", dest="sr_read_rank", type=int, default=None,
                   help="srdn: low-rank content-read queries d->rank->inner (default None = "
                        "head_dim, 'lite'; 0 = full rank); fixed rank makes the read "
                        "overhead ~linear in d instead of quadratic")
    p.add_argument("--sr-swiglu-ffn", dest="sr_swiglu_ffn", action="store_true",
                   help="scaffold-matched shared SwiGLU FFN instead of the default paper-faithful "
                        "RWKV channel mix (token-shift + sqReLU)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    args = p.parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)
    gen = torch_generator(args.seed)

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

    seq_bpc = eval_bpc_sequential(model, val, args, device)
    print(f"final sequential full-val bpc={seq_bpc:.4f}", flush=True)
    result = {"task": "enwik8", "arch": args.arch, "args": vars(args), "params": nparams,
              "final_val_bpc": history[-1]["val_bpc"], "final_val_bpc_seq": seq_bpc,
              "history": history,
              "wall_sec": round(time.perf_counter() - t0, 1)}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
