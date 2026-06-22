#!/usr/bin/env python3
"""FRJT depth-generalization comparison across architectures.

Trains each architecture on FRJT programs (mixed train depths) and evaluates
halt-class accuracy at held-out, deeper jump-table depths. Recurrence-complete
models should hold accuracy as depth grows; chunk-parallel (TC0) models degrade.

  uv run python experiments/frjt_compare.py --archs srdn,transformer,mamba3,m2rnn,gdn2 \
      --seeds 0,1,2 --out results/frjt.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import srdn  # noqa: E402
from tasks.frjt import FRJTTaskConfig, generate_frjt_batch, frjt_vocab_size  # noqa: E402
from tasks.seeding import set_seed  # noqa: E402

IGNORE = -100


def build(arch: str, vocab: int, args) -> torch.nn.Module:
    d, L, H = args.d_model, args.layers, args.heads
    if arch == "srdn":
        return srdn.build_srdn(vocab, d, L, H, d // H, args.ffn_mult, short_conv=True)
    if arch == "transformer":
        return srdn.build_transformer(vocab, d, L, H, args.ffn_mult, max_seq_len=args.max_seq_len)
    if arch == "mamba3":
        return srdn.build_mamba3(vocab, d, L, args.ffn_mult, state_size=args.mamba_state,
                                 head_dim=args.mamba_head_dim)
    if arch == "m2rnn":
        return srdn.build_m2rnn(vocab, d, L, H, args.m2rnn_head_dim, args.ffn_mult, kernel_size=4)
    if arch == "gdn2":
        return srdn.build_gdn2(vocab, d, L, H, args.gdn2_head_dim, args.ffn_mult,
                               gdn2_repo=args.gdn2_repo)
    raise ValueError(arch)


def loss_and_acc(model, batch, device):
    x = batch.inputs.to(device)
    y = batch.targets.to(device)
    logits = model.logits(x)
    if hasattr(model, "pop_router_logits"):
        model.pop_router_logits()
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1), ignore_index=IGNORE)
    fi = batch.final_indices.to(device)
    ft = batch.final_targets.to(device)
    final_logits = logits[torch.arange(x.shape[0], device=device), fi]
    acc = (final_logits.argmax(-1) == ft).float().mean().item()
    return ce, acc


@torch.no_grad()
def evaluate(model, cfg, depths, args, device):
    model.eval()
    out = {}
    for D in depths:
        accs = []
        for j in range(args.eval_batches):
            b = generate_frjt_batch(cfg=cfg, batch_size=args.eval_batch, seed=10_000 + D * 100 + j,
                                    ignore_index=IGNORE, depth_override=D)
            _, acc = loss_and_acc(model, b, device)
            accs.append(acc)
        out[f"depth_{D}"] = sum(accs) / len(accs)
    model.train()
    return out


def run_one(arch, seed, args, device):
    set_seed(seed)
    vocab = frjt_vocab_size(args.max_jump)
    cfg = FRJTTaskConfig(depth_min=args.depth_min, depth_max=args.depth_max, max_jump=args.max_jump,
                         direct_halt_prob=args.direct_halt_prob, dense_supervision=bool(args.dense))
    model = build(arch, vocab, args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t0 = time.perf_counter()
    for step in range(args.steps):
        b = generate_frjt_batch(cfg=cfg, batch_size=args.batch, seed=seed * 10**6 + step, ignore_index=IGNORE)
        ce, _ = loss_and_acc(model, b, device)
        opt.zero_grad(set_to_none=True)
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    evald = evaluate(model, cfg, [int(d) for d in args.eval_depths.split(",")], args, device)
    return {"params": model.num_params(), "eval": evald, "wall_sec": round(time.perf_counter() - t0, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", default="srdn,transformer,mamba3,m2rnn,gdn2")
    p.add_argument("--seeds", default="0")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", dest="d_model", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ffn-mult", dest="ffn_mult", type=float, default=2.0)
    p.add_argument("--depth-min", dest="depth_min", type=int, default=8)
    p.add_argument("--depth-max", dest="depth_max", type=int, default=16)
    p.add_argument("--max-jump", dest="max_jump", type=int, default=4)
    p.add_argument("--direct-halt-prob", dest="direct_halt_prob", type=float, default=0.0)
    p.add_argument("--dense", action="store_true", help="dense per-block register supervision")
    p.add_argument("--eval-depths", dest="eval_depths", default="16,32,64,128")
    p.add_argument("--eval-batch", dest="eval_batch", type=int, default=256)
    p.add_argument("--eval-batches", dest="eval_batches", type=int, default=4)
    p.add_argument("--max-seq-len", dest="max_seq_len", type=int, default=4096)
    p.add_argument("--mamba-state", dest="mamba_state", type=int, default=64)
    p.add_argument("--mamba-head-dim", dest="mamba_head_dim", type=int, default=32)
    p.add_argument("--m2rnn-head-dim", dest="m2rnn_head_dim", type=int, default=16)
    p.add_argument("--gdn2-head-dim", dest="gdn2_head_dim", type=int, default=16)
    p.add_argument("--gdn2-repo", dest="gdn2_repo", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    args = p.parse_args()
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]

    results = {"task": "frjt", "args": vars(args), "runs": {}}
    for arch in args.archs.split(","):
        results["runs"][arch] = {}
        for seed in seeds:
            r = run_one(arch, seed, args, device)
            results["runs"][arch][f"seed_{seed}"] = r
            ev = " ".join(f"{k}={v:.3f}" for k, v in r["eval"].items())
            print(f"{arch:12s} seed={seed} params={r['params']/1e3:.0f}k {ev} ({r['wall_sec']}s)", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
