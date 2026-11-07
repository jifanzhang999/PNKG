#!/usr/bin/env bash
# WordNet: view-weight search, PNKG (random, single views, uniform and selected multi-view) and KGE baselines,
# 10 seeds of a 60/20/20 random edge split.
#
#   bash experiments/wordnet.sh                 # all stages
#   bash experiments/wordnet.sh search pnkg     # a subset of: search pnkg kge summary
set -euo pipefail
cd "$(dirname "$0")/.."

KG_CSV="${KG_CSV:-data/wordnet/wordnet_kg.csv}"
OUT="${OUT:-results/wordnet}"
SEEDS=(${SEEDS:-7 11 13 17 19 23 29 31 37 42})
STAGES=("${@:-search pnkg kge summary}")
STAGES=(${STAGES[*]})

VIEWS=(bert roberta electra deberta)
MODEL_IDS="bert-base-uncased;roberta-base;google/electra-base-discriminator;microsoft/deberta-v3-base"
COMMON=(--kg-csv "$KG_CSV" --train-ratio 0.7 --val-ratio 0.1 --test-ratio 0.2 --epochs 100 --batch-size 1024 --d 128)
PNKG=("${COMMON[@]}" --model-ids "$MODEL_IDS" --pooling mean --max-length 32 --emb-dir cache/wordnet_emb)

has() { [[ " ${STAGES[*]} " == *" $1 "* ]]; }
run() {  # name seed args...
  local dir="$OUT/$1_seed$2"; shift 2
  [ -f "$dir/metrics.json" ] || python run.py "$@" --out-dir "$dir"
}

if has search; then
  python scripts/weight_search.py --num-views 4 --step 0.25 --seeds 101 103 --out-dir "$OUT/weight_search" \
    -- "${PNKG[@]}"
fi
SELECTED="${SELECTED:-$(cat "$OUT/weight_search/best_weights.txt" 2>/dev/null || echo "0.25;0.5;0.25;0")}"

if has pnkg; then
  echo "selected weights: $SELECTED"
  for seed in "${SEEDS[@]}"; do
    run random "$seed" "${COMMON[@]}" --init-mode random --seed "$seed"
    for i in "${!VIEWS[@]}"; do
      w=(0 0 0 0); w[$i]=1
      run "${VIEWS[$i]}" "$seed" "${PNKG[@]}" --model-weights "$(IFS=';'; echo "${w[*]}")" --seed "$seed"
    done
    run uniform "$seed" "${PNKG[@]}" --model-weights "1;1;1;1" --seed "$seed"
    run selected "$seed" "${PNKG[@]}" --model-weights "$SELECTED" --seed "$seed"
  done
fi

if has kge; then
  for seed in "${SEEDS[@]}"; do
    for model in transe distmult complex rotate; do
      run "$model" "$seed" "${COMMON[@]}" --model-type "$model" --seed "$seed"
    done
  done
fi

if has summary; then
  python scripts/summarize.py "$OUT" --out "$OUT/summary.csv"
  python scripts/plot_results.py wordnet --summary "$OUT/summary.csv" --out figures/wordnet.pdf
fi
