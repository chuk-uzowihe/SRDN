#!/usr/bin/env bash
# Paper-faithful RWKV-7 (native channel mix: token-shift lerp + squared-ReLU) on enwik8 + FRJT.
# Token mixer = fla RWKV7Attention (v_first verified identical to fla's reference stacking);
# chunk kernel uses chunk_size=32 (consumer-GPU shared-mem cap; see ops/rwkv7.py).
# PARAM-EQUALIZED, unified scaffold (blog protocol): --rwkv7-hidden-ratio 6.2578125
# -> 667,904 params (+1.9% vs the srdn anchor 655,616, artifacts/param_equalization.json).
# Outputs -> results/{enwik8_eq,frjt_eq}/ (the stored rwkv7 rows).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-uv run python}"; export PYTHONPATH="$PWD"
LOG=artifacts/logs; mkdir -p "$LOG" results/enwik8_eq results/frjt_eq
stamp(){ printf '%s %s\n' "$(date '+%m-%d %H:%M:%S')" "$1" >> "$LOG/progress_rwkv7faithful.txt"; }

for seed in 0 1 2; do
  stamp "enwik8 rwkv7faithful seed=$seed START"
  $PY tasks/enwik8/train.py --arch rwkv7 --rwkv7-faithful \
    --rwkv7-hidden-ratio 6.2578125 \
    --d-model 128 --seed "$seed" --out "results/enwik8_eq/rwkv7faithful_eq_s${seed}.json" \
    > "$LOG/enwik8_rwkv7faithful_s${seed}.log" 2>&1
  stamp "enwik8 rwkv7faithful seed=$seed exit=$?"
done

stamp "frjt rwkv7faithful ALL START"
$PY tasks/frjt/compare.py \
  --archs rwkv7 --rwkv7-faithful --rwkv7-hidden-ratio 6.2578125 --seeds 0,1,2 \
  --steps 10000 --batch 64 --lr 1e-3 --d-model 128 --layers 2 --heads 4 \
  --depth-min 1 --depth-max 16 --max-jump 4 --eval-depths 16,32,64,128 \
  --out results/frjt_eq/frjt_rwkv7faithful_eq.json > "$LOG/frjt_rwkv7faithful.log" 2>&1
stamp "frjt rwkv7faithful exit=$?"
stamp "ALL DONE"
