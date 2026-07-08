## Setup

Requirements: a CUDA GPU (8 GB is enough for everything in `results/`), NVIDIA driver
>= 550 (uv.lock pins torch 2.11+cu128), nvcc for the mamba-ssm source build, ~10 GB
disk for the venv. All tests are CUDA-only.

```bash
uv sync
mkdir -p refs
git clone https://github.com/open-lm-engine/lm-engine refs/lm-engine  && git -C refs/lm-engine checkout e94d13f
git clone https://github.com/NVlabs/GatedDeltaNet-2   refs/GatedDeltaNet-2 && git -C refs/GatedDeltaNet-2 checkout da7974d  # NVIDIA NC license: do not redistribute
# optional: the xma triton kernel accelerates the m2rnn baseline (results were produced
# with it); its pinned source is no longer publicly available -- without it m2rnn falls
# back to the pure-torch path (slower, same math).
```

## Scripts

```bash
# experiments
uv run python tasks/enwik8/train.py --arch srdn --rwkv7-hidden-ratio 5.0 --seed 0 --out /tmp/srdn_s0.json
uv run python tasks/frjt/compare.py --archs srdn,rwkv7 --seeds 0,1,2 --out /tmp/frjt.json

# full result sets
bash artifacts/run_srdn.sh              # srdn variants: enwik8 + FRJT + scaling ladder
bash artifacts/run_enwik8_d128.sh       # baseline archs, enwik8
bash artifacts/run_frjt_eq.sh           # baseline archs, FRJT
bash artifacts/run_rwkv7_faithful.sh    # rwkv7 (equalized), both tasks
bash artifacts/run_rwkv7_scaling.sh     # rwkv7 half of the scaling ladder
bash artifacts/run_transformer.sh       # transformer, both tasks
uv run python artifacts/equalize_ffn.py # re-solve the iso-param FFN knobs
```

The `artifacts/*.sh` scripts run `uv run python` by default (override with
`PYTHON=/path/to/python`), take hours end to end, and write into `results/` —
i.e. they overwrite the shipped result JSONs.

## License

Apache-2.0 (see `LICENSE`). The `refs/` checkouts keep their own upstream licenses —
NVIDIA's GatedDeltaNet-2 is non-commercially licensed and is never redistributed here.
