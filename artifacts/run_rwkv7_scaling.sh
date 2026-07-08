#!/usr/bin/env bash
# rwkv7 half of the enwik8 scaling ladder (produces results/enwik8_scaling/rwkv7_d*_s*.json;
# the srdn half lives in run_srdn.sh). 5 sizes iso-param to the srdn lite-r4 anchor,
# rwkv7-faithful widened UP per size (params linear in hidden; ceil to 32 hidden units):
#   d64L2 r7.0 (+3.3%) | d96L2 r6.0 (+2.5%) | d128L2 r5.5 (+2.1%) | d192L3 r5.0 (+1.6%)
#   | d256L4 r4.75 (+1.3%). Tokens = 20N, batch 64 seq 256, lr = min(3e-3, 3e-3*128/d).
# Usage: run_rwkv7_scaling.sh [seed list], default "0 1 2".
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-uv run python}"
export PYTHONPATH="$PWD"
OUT=results/enwik8_scaling; LOG=artifacts/logs; mkdir -p "$OUT" "$LOG"
stamp(){ printf '%s %s\n' "$(date '+%m-%d %H:%M:%S')" "$1" | tee -a "$LOG/progress.txt"; }
one(){ # one <d> <L> <ratio> <steps> <eval_every> <lr> <seed>
  stamp "scaling rwkv7 d$1 s$7 START"
  $PY tasks/enwik8/train.py --arch rwkv7 --rwkv7-faithful --rwkv7-head-dim 32 \
    --rwkv7-hidden-ratio "$3" --d-model "$1" --n-layers "$2" \
    --steps "$4" --batch 64 --eval-every "$5" --lr "$6" --seed "$7" \
    --out "$OUT/rwkv7_d$1_s$7.json" > "$LOG/scaling_rwkv7_d$1_s$7.log" 2>&1
  stamp "scaling rwkv7 d$1 s$7 exit=$?"
}
SEEDS="${1:-0 1 2}"
for S in $SEEDS; do
  one  64 2 7.0    231  25 3e-3   "$S"
  one  96 2 6.0    451  25 3e-3   "$S"
  one 128 2 5.5    721  50 3e-3   "$S"
  one 192 3 5.0   2139 100 2e-3   "$S"
  one 256 4 4.75  4742 200 1.5e-3 "$S"
done
stamp "RWKV7 SCALING DONE (seeds: $SEEDS)"
