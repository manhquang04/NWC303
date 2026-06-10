"""Evaluate a trained UNSW-NB15 DQN and export research artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from agent.dqn_agent import DQNAgent
from config import ACTION_NAMES, CFG, NUM_ACTIONS
from dataset.unsw_nb15_loader import UNSWNB15Loader
from train_unsw_nb15 import action_distribution, binary_metrics, infer_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=CFG.dataset.path)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/unsw_nb15"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = args.run_dir / "dqn_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    split = UNSWNB15Loader(args.data_dir).load(persist=False)
    if split.X_test.shape[1] != checkpoint["state_dim"]:
        raise ValueError("Checkpoint feature dimension does not match the UNSW-NB15 loader.")
    agent = DQNAgent(checkpoint["state_dim"], checkpoint.get("num_actions", NUM_ACTIONS), "cpu")
    agent.load(checkpoint_path)
    confidence_threshold = float(checkpoint.get("confidence_threshold", 0.0))
    actions = infer_actions(agent, split.X_test, confidence_threshold)
    metrics = binary_metrics(split.y_test, actions)
    counts = Counter(ACTION_NAMES[int(action)] for action in actions)
    metrics["action_distribution"] = {
        name: int(counts.get(name, 0)) for name in ACTION_NAMES
    }
    metrics["normal_action_distribution"] = action_distribution(split.y_test, actions, 0)
    metrics["attack_action_distribution"] = action_distribution(split.y_test, actions, 1)
    metrics["confidence_threshold"] = confidence_threshold

    predictions = (actions != 0).astype(np.int64)
    matrix = confusion_matrix(split.y_test, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=["Normal", "Attack"])
    display.plot(cmap="Blues", values_format="d")
    plt.tight_layout()
    plt.savefig(args.run_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    (args.run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    flat = {
        key: value for key, value in metrics.items()
        if not isinstance(value, dict)
    }
    flat.update({f"action_{key}": value for key, value in metrics["action_distribution"].items()})
    pd.DataFrame([flat]).to_csv(args.run_dir / "metrics.csv", index=False)

    print("\n=== UNSW-NB15 DQN Test Metrics ===")
    for key in ("precision", "recall", "f1", "fpr", "tp", "fp", "tn", "fn"):
        print(f"{key}: {metrics[key]}")
    print(f"actions: {metrics['action_distribution']}")
    print(f"normal actions: {metrics['normal_action_distribution']}")
    print(f"attack actions: {metrics['attack_action_distribution']}")
    print(f"Saved results: {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
