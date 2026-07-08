#!/usr/bin/env bash
# Transformer on enwik8 + FRJT (blog protocol, seeds 0,1,2). Uses the Vaswani embedding
# scale (h = embed * sqrt(d) + PE): without the sqrt(d), the unit-amplitude sinusoids drown
# the token content (embed init std d^-0.5) and destabilize training.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-uv run python}"
export PYTHONPATH="$PWD"
OUT_E=results/enwik8_eq; OUT_F=results/frjt_eq
LOG=artifacts/logs; mkdir -p "$OUT_E" "$OUT_F" "$LOG"
stamp(){ printf '%s %s\n' "$(date '+%m-%d %H:%M:%S')" "$1" | tee -a "$LOG/progress.txt"; }
for S in 0 1 2; do
  stamp "enwik8 transformer s$S START"
  $PY tasks/enwik8/train.py --arch transformer --ffn-mult 5.0 --d-model 128 --n-layers 2 \
    --seed "$S" --out "$OUT_E/transformer_s${S}.json" > "$LOG/enwik8_transformer_s${S}.log" 2>&1
  stamp "enwik8 transformer s$S exit=$?"
done
stamp "frjt transformer START"
$PY tasks/frjt/compare.py --archs transformer --ffn-mult 5.0 \
  --d-model 128 --layers 2 --heads 4 --depth-min 1 --depth-max 16 \
  --steps 10000 --batch 64 --lr 1e-3 --seeds 0,1,2 --max-jump 4 --eval-depths 16,32,64,128 \
  --out "$OUT_F/frjt_transformer.json" > "$LOG/frjt_transformer.log" 2>&1
stamp "frjt transformer exit=$? (TRANSFORMER DONE)"
