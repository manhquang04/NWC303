"""Run one reward-config experiment for SDN DRL-IDS."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from config import CFG
from train import train
from evaluate import evaluate


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one reward ablation experiment."""
    parser = argparse.ArgumentParser(description="Run one reward ablation experiment.")
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--attack-ratio", type=float, default=CFG.attack.attack_ratio)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def experiment(reward_config_path: Path, episodes: int, max_steps: int,
               eval_episodes: int, output_path: Path, seed: int = 42,
               attack_ratio: float = CFG.attack.attack_ratio) -> Path:
    """Train and evaluate one reward config, then export scenario metrics."""
    import pandas as pd
    import yaml

    if not reward_config_path.exists():
        raise FileNotFoundError(reward_config_path)
    with open(reward_config_path, "r", encoding="utf-8") as f:
        reward_config = yaml.safe_load(f)
    if not isinstance(reward_config, dict):
        raise ValueError(f"Reward config must be a YAML mapping: {reward_config_path}")

    config_name = reward_config_path.stem
    save_path = Path("runs") / "checkpoints" / config_name
    eval_path = Path("runs") / f"{config_name}_evaluation.csv"

    train(
        episodes=episodes,
        model_type="custom_dqn",
        lr=0.001,
        batch_size=64,
        save_path=save_path,
        max_steps=max_steps,
        attack_ratio=attack_ratio,
        seed=seed,
        reward_config=reward_config_path,
    )
    evaluate(
        model_path=save_path / "dqn_final.pt",
        scenario="all",
        episodes=eval_episodes,
        max_steps=max_steps,
        output=eval_path,
        agent="custom",
    )

    rows = []
    for row in pd.read_csv(eval_path).to_dict(orient="records"):
        rows.append({
            "config": config_name,
            "scenario": row["scenario"],
            "avg_reward": row["cumulative_reward"] / max(float(row["episodes"]), 1.0),
            "tp": int(row["tp"]),
            "fp": int(row["fp"]),
            "tn": int(row["tn"]),
            "fn": int(row["fn"]),
            "recall": float(row["recall"]),
            "precision": float(row["precision"]),
            "f1": float(row["f1"]),
            "mttd_sec": float(row["mttd_sec"]),
            "mtti_sec": float(row["mtti_sec"]),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "config", "scenario", "avg_reward", "tp", "fp", "tn", "fn",
            "recall", "precision", "f1", "mttd_sec", "mtti_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[experiment.py] Results saved to {output_path}")
    return output_path


def main() -> int:
    """Script entry point."""
    args = parse_args()
    experiment(
        args.reward,
        args.episodes,
        args.max_steps,
        args.eval_episodes,
        args.output,
        args.seed,
        args.attack_ratio,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
