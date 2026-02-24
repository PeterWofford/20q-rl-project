#!/bin/bash
set -e
echo "=== Run 3 — Free-Form Questions (GRPO from base, single seed) ==="
python src/train.py \
    --agent run3 \
    --steps 50 \
    --batch-size 6 \
    --eval-every 10 \
    --seed 1 \
    --run-label run3
