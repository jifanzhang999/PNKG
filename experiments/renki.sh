#!/usr/bin/env bash
# RENKI-KG: seven encoder views, their multi-view and single-view (PCA) coordinates with the RENKI head
# (frozen coordinates), RENKI from random trainable embeddings, and KGE baselines; 10 seeds of a 70/10/20 edge split.
# Requires data/renki/cxr_all.csv and data/renki/cxr_str.csv (see README).
#
#   bash experiments/renki.sh
#   CONFIGS="mv7_d128 random" bash experiments/renki.sh
set -euo pipefail
cd "$(dirname "$0")/.."

EMB=data/renki/emb
OUT="${OUT:-results/renki}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
VIEWS=(bert roberta nerbert mentalbert scibert biobert clinicalbert)

python scripts/embed_renki.py --out-dir "$EMB" --model "${VIEWS[@]}"
[ -f "$EMB/cxr_mv7_d128.pq" ] || python scripts/build_renki_views.py --views "${VIEWS[@]}" --dim 128 --out "$EMB/cxr_mv7_d128.pq"
for v in "${VIEWS[@]}"; do
  [ -f "$EMB/cxr_pca128_$v.pq" ] || python scripts/build_renki_views.py --views "$v" --dim 128 --out "$EMB/cxr_pca128_$v.pq"
done

default=(mv7_d128); for v in "${VIEWS[@]}"; do default+=("pca128_$v"); done; default+=(random transe distmult complex rotate)
CONFIGS=(${CONFIGS:-${default[@]}})

for name in "${CONFIGS[@]}"; do
  case "$name" in
    random)                         args=(--dim 128) ;;
    transe|distmult|complex|rotate) args=(--model "$name" --dim 128) ;;
    *)                              args=(--embed-path "$EMB/cxr_$name.pq" --freeze-embed) ;;
  esac
  python scripts/train_renki.py "${args[@]}" --name "$name" --seeds $SEEDS --out-dir "$OUT"
done

python scripts/plot_results.py renki --results-dir "$OUT" --out figures/renki.pdf
