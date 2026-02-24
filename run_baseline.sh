#!/bin/bash
# Run 1 — Clean Baseline: 5 seeds, 50 steps each, batch 6, eval every 10
# Hypothesis: Establish baseline and confirm whether policy collapse recurs.

set -e

for seed in 1 2 3 4 5; do
    echo "========================================="
    echo "  Run 1 — Seed $seed / 5"
    echo "========================================="
    python src/train.py \
        --agent 006 \
        --steps 50 \
        --batch-size 6 \
        --eval-every 10 \
        --seed "$seed" \
        --run-label run1
    echo "Seed $seed complete."
    echo ""
done

echo "All 5 seeds complete."
