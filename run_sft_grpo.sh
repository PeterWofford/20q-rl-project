#!/bin/bash
# Run 2 — SFT Warm-Start + GRPO
# Hypothesis: SFT warm-starting on oracle trajectories prevents policy collapse,
# and GRPO refines (rather than degrades) the SFT behavior. Target: 70-85% accuracy.
#
# Three phases:
#   1. Generate oracle SFT data (76 trajectories, deterministic)
#   2. SFT training (3 epochs, ~$5)
#   3. GRPO from SFT checkpoint (single seed first to validate before 5-seed run)

set -e

echo "========================================="
echo "  Run 2 — SFT Warm-Start + GRPO"
echo "========================================="

echo ""
echo "=== Phase 1: Generate oracle SFT data ==="
python src/generate_sft_data.py

echo ""
echo "=== Phase 2: SFT training ==="
python src/train_sft.py

echo ""
echo "=== Phase 3: GRPO from SFT checkpoint (single seed first) ==="
python src/train.py \
    --agent run2 \
    --steps 50 \
    --batch-size 6 \
    --eval-every 10 \
    --seed 1 \
    --run-label run2

echo ""
echo "========================================="
echo "  Run 2 complete (single seed)."
echo "  Review results before running 5 seeds."
echo "========================================="
