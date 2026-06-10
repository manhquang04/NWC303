#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

EPISODES="${EPISODES:-100}"
EVAL_EVERY="${EVAL_EVERY:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"

echo "======================================"
echo " NWC303 Real Dataset Experiment"
echo " Episodes=${EPISODES} | Device=${DEVICE}"
echo "======================================"

echo ""
echo "=== [1/5] Verify datasets ==="
python3 dataset/verify_datasets.py

echo ""
echo "=== [2/5] Train DRL on ARP dataset ==="
python3 train_real.py \
  --dataset arp \
  --episodes "${EPISODES}" \
  --reward v2 \
  --save-path runs/real_arp_v2 \
  --eval-every "${EVAL_EVERY}" \
  --num-workers "${NUM_WORKERS}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --seed "${SEED}"

echo ""
echo "=== [3/5] Evaluate on ARP dataset ==="
python3 evaluate_real.py \
  --dataset arp \
  --model runs/real_arp_v2/dqn_best.pt \
  --output runs/eval_real_arp.csv

echo ""
echo "=== [4/5] Train DRL on InSDN dataset ==="
python3 train_real.py \
  --dataset insdn \
  --episodes "${EPISODES}" \
  --reward v2 \
  --save-path runs/real_insdn_v2 \
  --eval-every "${EVAL_EVERY}" \
  --num-workers "${NUM_WORKERS}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --seed "${SEED}"

echo ""
echo "=== [5/5] Ablation study - reward configs ==="
python3 experiment_real.py \
  --dataset arp \
  --episodes 50 \
  --num-workers "${NUM_WORKERS}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --output runs/ablation_real_arp.csv

echo ""
echo "======================================"
echo " ALL DONE - Results in runs/"
echo "======================================"
ls -la runs/
