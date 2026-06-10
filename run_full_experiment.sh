#!/usr/bin/env bash
set -euo pipefail

EPISODES="${EPISODES:-100}"
MAX_STEPS="${MAX_STEPS:-100}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
ATTACK_RATIO="${ATTACK_RATIO:-0.3}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv311/bin/python" ]]; then
    PYTHON_BIN=".venv311/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

mkdir -p runs runs/checkpoints

echo "=== STEP 1: Train custom DQN ${EPISODES} episodes ==="
"${PYTHON_BIN}" train.py --episodes "${EPISODES}" --model custom_dqn --lr 0.001 --batch-size 64 --max-steps "${MAX_STEPS}" --attack-ratio "${ATTACK_RATIO}" --save-path runs/checkpoints

echo "=== STEP 2: Evaluate custom DQN ==="
"${PYTHON_BIN}" evaluate.py --model runs/checkpoints/dqn_final.pt --scenario all --episodes "${EVAL_EPISODES}" --max-steps "${MAX_STEPS}" --output runs/evaluation_results.csv

echo "=== STEP 3: Ablation Reward v1 (baseline) ==="
"${PYTHON_BIN}" experiment.py --reward config/reward_v1.yaml --episodes "${EPISODES}" --max-steps "${MAX_STEPS}" --attack-ratio "${ATTACK_RATIO}" --eval-episodes "${EVAL_EPISODES}" --output runs/exp_v1.csv

echo "=== STEP 4: Ablation Reward v2 (FN penalty) ==="
"${PYTHON_BIN}" experiment.py --reward config/reward_v2_fn_penalty.yaml --episodes "${EPISODES}" --max-steps "${MAX_STEPS}" --attack-ratio "${ATTACK_RATIO}" --eval-episodes "${EVAL_EPISODES}" --output runs/exp_v2.csv

echo "=== STEP 5: Ablation Reward v3 (isolate boost) ==="
"${PYTHON_BIN}" experiment.py --reward config/reward_v3_isolate_boost.yaml --episodes "${EPISODES}" --max-steps "${MAX_STEPS}" --attack-ratio "${ATTACK_RATIO}" --eval-episodes "${EVAL_EPISODES}" --output runs/exp_v3.csv

echo "=== STEP 6: Ablation Reward v4 (balanced) ==="
"${PYTHON_BIN}" experiment.py --reward config/reward_v4_balanced.yaml --episodes "${EPISODES}" --max-steps "${MAX_STEPS}" --attack-ratio "${ATTACK_RATIO}" --eval-episodes "${EVAL_EPISODES}" --output runs/exp_v4.csv

echo "=== STEP 7: Ablation Reward v5 (conservative) ==="
"${PYTHON_BIN}" experiment.py --reward config/reward_v5_conservative.yaml --episodes "${EPISODES}" --max-steps "${MAX_STEPS}" --attack-ratio "${ATTACK_RATIO}" --eval-episodes "${EVAL_EPISODES}" --output runs/exp_v5.csv

echo "=== STEP 8: Ablation Reward v6 (logic fixed) ==="
"${PYTHON_BIN}" experiment.py --reward config/reward_v6_logic_fixed.yaml --episodes "${EPISODES}" --max-steps "${MAX_STEPS}" --attack-ratio "${ATTACK_RATIO}" --eval-episodes "${EVAL_EPISODES}" --output runs/exp_v6.csv

echo "=== STEP 9: Aggregate results ==="
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import pandas as pd

inputs = [
    Path("runs/exp_v1.csv"),
    Path("runs/exp_v2.csv"),
    Path("runs/exp_v3.csv"),
    Path("runs/exp_v4.csv"),
    Path("runs/exp_v5.csv"),
    Path("runs/exp_v6.csv"),
]
missing = [str(p) for p in inputs if not p.exists()]
if missing:
    raise FileNotFoundError(f"Missing experiment output(s): {', '.join(missing)}")

out = Path("runs/final_research_results.csv")
out.parent.mkdir(parents=True, exist_ok=True)
pd.concat([pd.read_csv(p) for p in inputs], ignore_index=True).to_csv(out, index=False)
print(f"Aggregated {len(inputs)} files -> {out}")
PY

echo "=== DONE ==="
echo "Results saved to runs/final_research_results.csv"
echo "Evaluation saved to runs/evaluation_results.csv"
echo "Reward curve saved to runs/checkpoints/reward_curve.png"
