"""Evaluate the best CTU-13 DQN checkpoint on the deterministic test split."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from agent.dqn_agent import DQNAgent
from config import ACTION_ALLOW, ACTION_NAMES, NUM_ACTIONS
from dataset.ctu13_loader import CTU13Loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("~/Tải về/CTU-13-Dataset/Dataset/"))
    parser.add_argument("--sample-frac", type=float, default=0.10)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/ctu13/dqn_best.pt"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser()
    if not data_dir.exists():
        data_dir = Path("dataset")
    X, y, _ = CTU13Loader().load(data_dir, args.sample_frac)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    X_test = ((X_test - checkpoint["scaler_mean"]) / checkpoint["scaler_scale"]).astype(np.float32)
    agent = DQNAgent(checkpoint["state_dim"], checkpoint.get("num_actions", NUM_ACTIONS), device="cpu")
    agent.load(args.checkpoint)

    action_parts = []
    agent.q_net.eval()
    with torch.no_grad():
        for start in range(0, len(X_test), 4096):
            batch = torch.from_numpy(X_test[start:start + 4096]).float()
            action_parts.append(agent.q_net(batch).argmax(dim=1).numpy())
    actions = np.concatenate(action_parts)
    y_pred = (actions != ACTION_ALLOW).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    distribution = Counter(ACTION_NAMES[int(action)] for action in actions)
    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / max(fp + tn, 1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        **{f"action_{name}": distribution.get(name, 0) for name in ACTION_NAMES},
    }
    print("\n=== CTU-13 DQN Evaluation ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"FPR:       {result['fpr']:.4f}")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    print("Action distribution:")
    for name in ACTION_NAMES:
        print(f"  {name}: {distribution.get(name, 0)}")
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame([result]).to_csv("results/eval_ctu13.csv", index=False)
    print("Saved: results/eval_ctu13.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
