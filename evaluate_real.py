"""Evaluate a trained DQN checkpoint on real CSV datasets without Ryu/Mininet."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from agent.dqn_agent import DQNAgent
from config import ACTION_ALLOW, ACTION_NAMES
from dataset.data_loader import ARPDataLoader, InSDNDataLoader


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Evaluate DRL agent on real datasets.")
    p.add_argument("--dataset", choices=["arp", "insdn", "both"], required=True)
    p.add_argument("--model", type=Path, default=Path("runs/checkpoints/dqn_final.pt"))
    p.add_argument("--dry-run", action="store_true", help="Run only a small sample.")
    p.add_argument("--limit", type=int, default=None, help="Optional max test rows.")
    p.add_argument("--output-dir", type=Path, default=Path("runs"))
    p.add_argument("--output", type=Path, default=None, help="Optional exact CSV output path.")
    return p.parse_args()


def checkpoint_state_dim(path: Path) -> int:
    """Infer DQN input dimension from the first Linear layer in a checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "input_dim" in ckpt:
        return int(ckpt["input_dim"])
    state = ckpt.get("q_net", ckpt.get("model_state_dict", ckpt))
    for key, value in state.items():
        if key.endswith("net.0.weight") or key == "net.0.weight":
            return int(value.shape[1])
    for value in state.values():
        if hasattr(value, "ndim") and value.ndim == 2:
            return int(value.shape[1])
    raise ValueError(f"Could not infer checkpoint input dimension from {path}")


def adapt_features(X: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or truncate feature vectors to match checkpoint input dimension."""
    if X.shape[1] == target_dim:
        return X.astype(np.float32)
    if X.shape[1] > target_dim:
        return X[:, :target_dim].astype(np.float32)
    pad = np.zeros((X.shape[0], target_dim - X.shape[1]), dtype=np.float32)
    return np.hstack([X.astype(np.float32), pad])


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Return normalized test features, test labels, and feature names."""
    if name == "arp":
        _, X_test, _, y_test, features = ARPDataLoader().load()
        return X_test, y_test, features
    if name == "insdn":
        _, X_test, _, y_test, features = InSDNDataLoader().load()
        return X_test, y_test, features
    raise ValueError(name)


def compute_metrics(y_true: np.ndarray, actions: np.ndarray) -> Dict[str, float | int | Dict[str, int]]:
    """Compute binary metrics where allow=normal and all other actions=attack."""
    y_pred = (actions != ACTION_ALLOW).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    action_dist = {name: int((actions == i).sum()) for i, name in enumerate(ACTION_NAMES)}
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "action_dist": action_dist,
    }


def evaluate_dataset(
    name: str,
    model_path: Path,
    output_dir: Path,
    limit: int | None = None,
    output_path: Path | None = None,
) -> Dict[str, object]:
    """Evaluate one dataset and export per-row predictions."""
    X_test, y_test, features = load_dataset(name)
    if limit is not None:
        X_test = X_test[:limit]
        y_test = y_test[:limit]

    state_dim = checkpoint_state_dim(model_path)
    X_eval = adapt_features(X_test, state_dim)
    agent = DQNAgent(state_dim=state_dim)
    agent.load(model_path)
    agent.epsilon = 0.0

    actions = np.array([agent.act(row, greedy=True) for row in X_eval], dtype=np.int64)
    metrics = compute_metrics(y_test, actions)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = output_path or (output_dir / f"eval_real_{name}_{ts}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "ground_truth", "action", "action_name", "predicted_attack", "correct"])
        for idx, (gt, action) in enumerate(zip(y_test, actions)):
            pred = int(action != ACTION_ALLOW)
            writer.writerow([idx, int(gt), int(action), ACTION_NAMES[int(action)], pred, int(pred == int(gt))])

    result = {
        "dataset": name,
        "rows": int(len(y_test)),
        "raw_features": int(len(features)),
        "model_state_dim": int(state_dim),
        "output": str(out),
        **metrics,
    }
    return result


def print_summary(results: Iterable[Dict[str, object]]) -> None:
    """Print a compact metrics table."""
    rows = list(results)
    headers = ["dataset", "rows", "accuracy", "precision", "recall", "f1", "fpr", "tp", "fp", "tn", "fn", "action_dist"]
    print(" | ".join(headers))
    print("-+-".join("-" * len(h) for h in headers))
    for r in rows:
        print(" | ".join(
            f"{r[h]:.4f}" if isinstance(r.get(h), float) else str(r.get(h))
            for h in headers
        ))
        print(f"Exported: {r['output']}")


def main() -> int:
    """Script entry point."""
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    datasets = ["arp", "insdn"] if args.dataset == "both" else [args.dataset]
    limit = args.limit
    if args.dry_run and limit is None:
        limit = 100
    results = [
        evaluate_dataset(
            name,
            args.model,
            args.output_dir,
            limit=limit,
            output_path=args.output if len(datasets) == 1 else None,
        )
        for name in datasets
    ]
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
