"""Evaluate trained InSDN reward-ablation checkpoints."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from agent.dqn_agent import DQNAgent
from config import NUM_ACTIONS
from dataset.data_loader import InSDNDataLoader
from evaluate_real import adapt_features, checkpoint_state_dim, compute_metrics
from experiment_real import reward_configs


def checkpoint_dirs(runs_dir: Path = Path("runs")) -> Dict[str, Path]:
    """Map reward config names to existing InSDN ablation checkpoints."""
    found: Dict[str, Path] = {}
    for cfg_path in reward_configs():
        cfg = cfg_path.stem
        candidates = [
            runs_dir / f"ablation_insdn_{cfg}" / "dqn_best.pt",
            runs_dir / f"real_insdn_{cfg}" / "dqn_best.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                found[cfg] = candidate
                break
    return found


def evaluate_checkpoint(ckpt_path: Path, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate one trained checkpoint using batched greedy inference."""
    state_dim = checkpoint_state_dim(ckpt_path)
    X_eval = adapt_features(X_test, state_dim)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    agent = DQNAgent(state_dim=state_dim, num_actions=NUM_ACTIONS, device="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt.get("q_net"))
    agent.q_net.load_state_dict(state_dict)
    agent.q_net.eval()

    X_tensor = torch.from_numpy(X_eval.astype(np.float32))
    actions_out = []
    with torch.no_grad():
        for start in range(0, len(X_tensor), 4096):
            actions_out.append(agent.q_net(X_tensor[start:start + 4096]).argmax(dim=1).numpy())
    actions = np.concatenate(actions_out).astype(np.int64)
    return compute_metrics(y_test, actions)


def run_ablation_insdn(output_dir: Path = Path("runs")) -> List[dict]:
    """Load/evaluate trained InSDN ablation checkpoints and export metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = checkpoint_dirs(output_dir)
    expected = [p.stem for p in reward_configs()]
    missing = [cfg for cfg in expected if cfg not in checkpoints]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            "Missing InSDN ablation checkpoints for: "
            f"{missing_list}. Run `python3 experiment_insdn.py` first."
        )

    _, X_test, _, y_test, _ = InSDNDataLoader().load()
    rows: List[dict] = []
    print(f"Loaded InSDN test split: {X_test.shape}, attack_ratio={y_test.mean():.4f}")

    for cfg in expected:
        ckpt_path = checkpoints[cfg]
        print(f"\nEvaluating {cfg}: {ckpt_path}")
        metrics = evaluate_checkpoint(ckpt_path, X_test, y_test)
        row = {
            "dataset": "InSDN",
            "reward_config": cfg,
            "checkpoint": str(ckpt_path),
            **metrics,
        }
        rows.append(row)
        print(
            f"  F1={metrics['f1']:.4f}  FPR={metrics['fpr']:.4f}  "
            f"Recall={metrics['recall']:.4f}  Precision={metrics['precision']:.4f}"
        )

    out_path = output_dir / "ablation_insdn_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nExported: {out_path}")
    return rows


def load_arp_ablation(path: Path = Path("runs/ablation_real_arp.csv")) -> list[dict] | None:
    """Load ARP ablation results if present."""
    if not path.exists():
        return None
    import pandas as pd

    return pd.read_csv(path).to_dict("records")


def print_ablation_table(insdn_rows: List[dict], arp_rows: list[dict] | None = None) -> None:
    """Print ARP-vs-InSDN reward ablation table."""
    print("\n" + "=" * 70)
    print("RQ2 ABLATION: Reward Config Comparison - ARP vs InSDN")
    print("=" * 70)
    print(f"\n  {'Config':<28} {'ARP F1':>8} {'ARP FPR':>9} {'InSDN F1':>10} {'InSDN FPR':>11}")
    print(f"  {'-' * 28} {'-' * 8} {'-' * 9} {'-' * 10} {'-' * 11}")

    arp_map = {}
    if arp_rows:
        for row in arp_rows:
            arp_map[row.get("reward_config", row.get("config", ""))] = row

    for row in insdn_rows:
        cfg = row["reward_config"]
        arp = arp_map.get(cfg, {})
        print(
            f"  {cfg:<28} "
            f"{float(arp.get('f1', float('nan'))):>8.4f} "
            f"{float(arp.get('fpr', float('nan'))):>9.4f} "
            f"{row['f1']:>10.4f} "
            f"{row['fpr']:>11.4f}"
        )


def main() -> int:
    """Script entry point."""
    rows = run_ablation_insdn()
    print_ablation_table(rows, load_arp_ablation())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
