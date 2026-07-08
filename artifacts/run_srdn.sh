#!/usr/bin/env bash
# All srdn runs: per_proj-lite default, v_first bus carries the BASE v (native RWKV-7
# semantics; conditioning the bus was tried and rejected -- see ops/srdn._publish_v_first).
# Produces everything the cell touches: enwik8 + FRJT for lite / per_proj-full / split,
# and the scaling ladder's srdn half (rwkv7 half: run_rwkv7_scaling.sh).
# Blog protocol d128 L2 H4 hd32, seeds 0,1,2; knobs per artifacts/param_equalization.json.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-uv run python}"
export PYTHONPATH="$PWD"
export PYTORCH_ALLOC_CONF=expandable_segments:True
OUT_E=results/enwik8_eq; OUT_F=results/frjt_eq; OUT_S=results/enwik8_scaling
LOG=artifacts/logs; mkdir -p "$OUT_E" "$OUT_F" "$OUT_S" "$LOG"
stamp(){ printf '%s %s\n' "$(date '+%m-%d %H:%M:%S')" "$1" | tee -a "$LOG/progress.txt"; }

E8="tasks/enwik8/train.py --d-model 128 --n-layers 2 --rwkv7-head-dim 32"
FRJT="tasks/frjt/compare.py --d-model 128 --layers 2 --heads 4 --rwkv7-head-dim 32 \
--depth-min 1 --depth-max 16 --steps 10000 --batch 64 --lr 1e-3 --seeds 0,1,2 \
--max-jump 4 --eval-depths 16,32,64,128"

for S in 0 1 2; do
  stamp "enwik8 perproj_lite s$S START"
  $PY $E8 --arch srdn --rwkv7-hidden-ratio 5.0 --seed "$S" \
    --out "$OUT_E/perproj_lite_s${S}.json" > "$LOG/enwik8_lite_s${S}.log" 2>&1
  stamp "enwik8 perproj_lite s$S exit=$?"
  stamp "enwik8 perproj s$S START"
  $PY $E8 --arch srdn --sr-read-rank 0 --seed "$S" \
    --out "$OUT_E/perproj_s${S}.json" > "$LOG/enwik8_perproj_s${S}.log" 2>&1
  stamp "enwik8 perproj s$S exit=$?"
  stamp "enwik8 split s$S START"
  $PY $E8 --arch srdn --sr-mode split --sr-read-rank 0 --rwkv7-hidden-ratio 5.2578125 \
    --seed "$S" --out "$OUT_E/split_eq_s${S}.json" > "$LOG/enwik8_split_s${S}.log" 2>&1
  stamp "enwik8 split s$S exit=$?"
done
stamp "ENWIK8 DONE"

stamp "frjt perproj_lite START"
$PY $FRJT --archs srdn --rwkv7-hidden-ratio 5.0 \
  --out "$OUT_F/frjt_perproj_lite.json" > "$LOG/frjt_lite.log" 2>&1
stamp "frjt perproj_lite exit=$?"
stamp "frjt perproj START"
$PY $FRJT --archs srdn --sr-read-rank 0 \
  --out "$OUT_F/frjt_perproj.json" > "$LOG/frjt_perproj.log" 2>&1
stamp "frjt perproj exit=$?"
stamp "frjt split START"
$PY $FRJT --archs srdn --sr-mode split --sr-read-rank 0 --rwkv7-hidden-ratio 5.2578125 \
  --out "$OUT_F/frjt_split_eq.json" > "$LOG/frjt_split.log" 2>&1
stamp "frjt split exit=$? (FRJT DONE)"

scale_one(){ # scale_one <d> <L> <steps> <eval_every> <lr> <seed>
  stamp "scaling srdn d$1 s$6 START"
  $PY tasks/enwik8/train.py --arch srdn --rwkv7-head-dim 32 --d-model "$1" --n-layers "$2" \
    --steps "$3" --batch 64 --eval-every "$4" --lr "$5" --seed "$6" \
    --out "$OUT_S/srdn_d$1_s$6.json" > "$LOG/scaling_d$1_s$6.log" 2>&1
  stamp "scaling srdn d$1 s$6 exit=$?"
}
for S in 0 1 2; do
  scale_one  64 2   231  25 3e-3   "$S"
  scale_one  96 2   451  25 3e-3   "$S"
  scale_one 128 2   721  50 3e-3   "$S"
  scale_one 192 3  2139 100 2e-3   "$S"
  scale_one 256 4  4742 200 1.5e-3 "$S"
done
stamp "SRDN ALL DONE (enwik8 + frjt + scaling srdn half)"
