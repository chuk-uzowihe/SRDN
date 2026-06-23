# SRDN — recurrence-complete linear attention with state-conditioned projections

SRDN is a gated delta-rule linear-attention cell whose **projections are conditioned
on the recurrent state**, which makes it *recurrence-complete* (its step-`t` inputs
depend nonlinearly on the step-`(t-1)` state, so it cannot be flattened into a
chunk-parallel form — unlike GDN, Mamba, or attention). At matched parameters in an
otherwise-identical 2-layer stack it matches parallelizable models on language
modeling and **decisively beats them on a state-tracking RL task**.

The ladder of changes from Gated DeltaNet (GDN):
`GDN → q reads [x; ŝ₀]` (a diagonal state summary) `→ k,v and the decay/write gates
read [x; ŝₓ]` (the content read `q·S`), all additive ReZero paths (gamma=0 at init,
so SRDN is *exactly* GDN at init and the state coupling grows in). Decay is Mamba-2
multi-timescale; the read goes straight to the residual.

## Results

2-layer stacks, only the token mixer swapped, identical SwiGLU FFN, one 8 GB consumer GPU.
Params reported per row (matched `d_model`; the recurrent cells land ~0.43M, Mamba-3 is
heavier because it carries a state_size=128, attention is lighter).

**FRJT** (Flip-Register Jump Table, arXiv:2510.06828 — single-cell state tracking).
Halt-class accuracy, trained at depth 16, evaluated to depth 128; mean over 3 seeds:

| arch | params | d16 | d32 | d64 | d128 |
|---|---|---|---|---|---|
| **SRDN** | 206k | 0.991 | 0.995 | 0.992 | **0.990** |
| GDN-2 | 218k | 0.979 | 0.963 | 0.885 | 0.827 |
| M2RNN | 164k | 0.718 | 0.733 | 0.703 | 0.688 |
| Mamba-3 | 211k | 0.680 | 0.578 | 0.545 | 0.541 |
| Transformer | 165k | 0.606 | 0.572 | 0.590 | 0.588 |

SRDN holds ~0.99 flat as depth grows; parallelizable models (GDN-2, Mamba-3, Transformer)
degrade. M2RNN is recurrence-complete yet stuck ~0.70 — evidence that recurrence-completeness
is *necessary but not sufficient* (see Limitations; fairness diagnostic for the small
M2RNN/Mamba-3 FRJT configs in progress).

**enwik8** char-LM (bpc, dense SwiGLU FFN, seq 256, 1500 steps, 3 seeds, mean ± std):

| arch | params | val bpc |
|---|---|---|
| M2RNN | 433.5k | 2.149 ± .019 |
| **SRDN** | 438.9k | **2.159 ± .021** |
| GDN-2 | 432.1k | 2.180 ± .019 |
| Mamba-3 | 506.0k | 2.327 ± .043 |
| Transformer | 361.1k | 2.508 ± .041 |

The three delta-rule cells cluster within 0.03 bpc (overlapping within 1σ): SRDN ties the
strong recurrent baselines on plain LM — no quality tax for its state-tracking machinery.

**Graph traversal RL** (center-curriculum, 256 nodes — the recurrence-complete discriminator).
Eval reward/episode (sampled), 4k steps, seed 0:

| eval len | SRDN | M2RNN | GDN-2 | Transformer | Mamba-3 |
|---|---|---|---|---|---|
| 32   | **5.7**  | 0.4 | 1.1 | _tbd_ | _tbd_ |
| 512  | **10.3** | 3.2 | 2.4 | _tbd_ | _tbd_ |
| 2048 | **18.1** | 6.3 | 3.6 | _tbd_ | _tbd_ |
| curriculum reached | **80** | 32 | 64 | _tbd_ | _tbd_ |

> Status: FRJT and enwik8 complete (3 seeds). SRDN/M2RNN/GDN-2 graph-RL carried over from the
> validated runs; the Transformer (fresh) and Mamba-3 (pending the decode fix) graph-RL runs
> are still to come.

## Layout

The architecture is split so that **only the token mixer differs** between models:
the shared scaffold (block wiring, embedding, channel mixer, sqrt-exact-BPTT remat
loop, rollout) lives in `srdn/`, and each token mixer is one isolated file in `ops/`.
```
srdn/
  core.py      RMSNorm, the state-routed MoE, shared Block (x = x + mixer(x); x = channel(x))
               + SRDNLM scaffold (logits / chunked_logits / init_states / step), generic over any mixer
  channel.py   the SHARED channel mixer (SwiGLU FFN, optional MoE) -- identical for ALL archs
  builders.py  build_srdn / build_transformer / build_mamba3 / build_m2rnn / build_gdn2
  ops/         the ONLY thing that varies -- one token mixer per file, common interface:
    srdn.py · attention.py · mamba3.py · m2rnn.py · gdn2.py   (+ conv.py for the short conv)
tasks/         one subpackage per task -- each holds its data/generator AND its runner:
  seeding.py             set_seed + seeded generators (reproducibility, used by every runner)
  enwik8/   data.py · train.py
  frjt/     task.py · compare.py
  graph_traversal/  traversal.py · graph_rl.py · train.py
tests/         parity (clean cell == original), chunk-equivalence (chunked==full), step-vs-parallel.
refs/          gitignored symlinks to the external baseline checkouts (below).
```
The external baseline mixers (`srdn/ops/{mamba3,m2rnn,gdn2}.py`) load their cells
from `refs/` at runtime; `srdn/ops/__init__.py` imports lazily so building SRDN needs
neither fla nor the refs.

## Setup

```bash
uv sync                      # installs torch, fla@4b02d15d, mamba-ssm, ...
# mamba-ssm is a CUDA source build:
uv pip install --no-build-isolation mamba-ssm   # needs nvcc (CUDA_HOME)
```

External baselines are **not** vendored. Clone them at the pinned commits and symlink:

```bash
mkdir -p refs
git clone https://github.com/fla-org/lm-engine            refs/lm-engine    && git -C refs/lm-engine    checkout e94d13f   # Apache-2.0; canonical M2RNN
git clone https://github.com/fla-org/xma                  refs/xma          && git -C refs/xma          checkout a60444b   # Apache-2.0; M2RNN triton kernel
git clone https://github.com/NVlabs/GatedDeltaNet-2       refs/GatedDeltaNet-2 && git -C refs/GatedDeltaNet-2 checkout da7974d   # NVIDIA NC license (non-commercial); DO NOT redistribute
```

> NVIDIA GDN-2 is under the NVIDIA Source Code License-NC (non-commercial,
> non-redistributable), so it is referenced, never copied. M2RNN/xma are Apache-2.0
> but also referenced (kept canonical; the M2RNN twin in `srdn/ops/m2rnn.py` is
> parity-tested bit-exact against the lm-engine cell).

## Reproduce

```bash
# tests (correctness gates)
SRDNLADDER_ROOT=/path/to/srdnladder uv run python tests/test_parity.py        # clean SRDN == original srdn5 (bit-exact)
uv run python tests/test_chunk_equivalence.py --arch all                       # chunked_logits == full-seq (fwd+grad)
uv run python tests/test_step_parallel.py --arch all                           # rollout step == teacher-forced forward

# experiments (every run is fully determined by --seed)
uv run python tasks/frjt/compare.py          --archs srdn,transformer,mamba3,m2rnn,gdn2 --seeds 0,1,2
uv run python tasks/enwik8/train.py          --arch srdn --seed 0
uv run python tasks/graph_traversal/train.py --architecture srdn --seed 0 \
    --micro-batch-action-budget 4096 --train-sequence-chunk-size 512 --train-remat-chunks   # 8GB
```

## Sequence checkpointing
Training uses **sqrt-exact BPTT**: the sequence is split into chunks, each recurrent
model carries its state (recurrent + conv ring-buffer) across chunk boundaries with
`torch.utils.checkpoint` remat, and the gradient is exact (not truncated). Verified
chunked==full for SRDN (3e-6 rel grad), GDN-2 (4e-3, triton), and the M2RNN twin
(bit-exact forward). Transformer/Mamba-3 are parallelizable and run full-sequence.
