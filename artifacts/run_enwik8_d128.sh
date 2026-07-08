#!/usr/bin/env bash
# enwik8 for the baseline archs (blog protocol): d_model=128, 4 archs x 3 seeds, sequential.
# PARAM-EQUALIZED, 'faithful' family: natural head dims (dh32 -- matrix-state archs all carry
# 4x32^2=4096 state/layer; mamba3 gets its natural state_size=128 at head_dim=64); per-arch
# --ffn-mult widens each baseline to the anchor (srdn per_proj @ native channel mix r4 =
# 655,616 params; all within +-4%, see artifacts/param_equalization.json).
# rwkv7 and srdn run via run_rwkv7_faithful.sh / run_srdn.sh. Outputs -> results/enwik8_eq/.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-uv run python}"
export PYTHONPATH="$PWD"
LOG=artifacts/logs; mkdir -p "$LOG" results/enwik8_eq
stamp(){ printf '%s %s\n' "$(date +%H:%M:%S)" "$1" >> "$LOG/progress_enwik8_eq.txt"; }
for seed in 0 1 2; do
  for spec in \
    "transformer|--arch transformer --ffn-mult 5.0" \
    "mamba3|--arch mamba3 --mamba-state 128 --mamba-head-dim 64 --ffn-mult 3.5" \
    "m2rnn|--arch m2rnn --m2rnn-head-dim 32 --ffn-mult 5.0" \
    "gdn2|--arch gdn2 --gdn2-head-dim 32 --ffn-mult 4.0"; do
    name="${spec%%|*}"; args="${spec#*|}"
    stamp "$name seed=$seed START"
    $PY tasks/enwik8/train.py $args --d-model 128 --seed "$seed" \
      --out "results/enwik8_eq/${name}_s${seed}.json" > "$LOG/enwik8_${name}_s${seed}.log" 2>&1
    stamp "$name seed=$seed exit=$?"
  done
done
stamp "enwik8 d128 DONE"
