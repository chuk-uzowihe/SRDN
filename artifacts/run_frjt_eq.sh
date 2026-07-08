#!/usr/bin/env bash
# FRJT for the baseline archs at the unified blog scaffold: d128 L2 H4, PARAM-EQUALIZED
# per-arch FFN ('faithful' family anchor = srdn @ native channel mix = 655,616 params;
# knobs from artifacts/param_equalization.json / equalize_ffn.py). Paper-faithful protocol:
# mixed depths 1-16, 10k steps, max_jump 4, eval 16-128, seeds 0,1,2. Per-arch invocations
# because --ffn-mult is global in compare.py. rwkv7 runs via run_rwkv7_faithful.sh.
# Outputs -> results/frjt_eq/.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-uv run python}"; export PYTHONPATH="$PWD"
LOG=artifacts/logs; mkdir -p "$LOG" results/frjt_eq
stamp(){ printf '%s %s\n' "$(date '+%m-%d %H:%M:%S')" "$1" >> "$LOG/progress_frjt_eq.txt"; }
COMMON="--seeds 0,1,2 --steps 10000 --batch 64 --lr 1e-3 --d-model 128 --layers 2 --heads 4 \
--depth-min 1 --depth-max 16 --max-jump 4 --eval-depths 16,32,64,128"
for spec in \
  "transformer|--archs transformer --ffn-mult 5.0" \
  "mamba3|--archs mamba3 --mamba-state 128 --mamba-head-dim 64 --ffn-mult 3.5" \
  "m2rnn|--archs m2rnn --m2rnn-head-dim 32 --ffn-mult 5.0" \
  "gdn2|--archs gdn2 --gdn2-head-dim 32 --ffn-mult 4.0"; do
  name="${spec%%|*}"; args="${spec#*|}"
  stamp "frjt_eq $name START"
  $PY tasks/frjt/compare.py $args $COMMON \
    --out "results/frjt_eq/frjt_${name}.json" > "$LOG/frjt_eq_${name}.log" 2>&1
  stamp "frjt_eq $name exit=$?"
done
stamp "FRJT EQ DONE"
