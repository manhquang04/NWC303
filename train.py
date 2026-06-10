"""Convenience training script for SDN DRL-IDS experiments."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for custom DQN training."""
    parser = argparse.ArgumentParser(description="Train DQN for SDN DRL-IDS.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--model", choices=["custom_dqn", "custom", "sb3"], default="custom_dqn")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-path", type=Path, default=Path("runs/checkpoints"))
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--attack-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-config", type=Path, default=None)
    return parser.parse_args()


def _copy_outputs(save_path: Path) -> None:
    """Copy canonical checkpoints and reward plot into the requested save path."""
    from config import CHECKPOINT_DIR, LOG_DIR
    from agent.train import plot_reward_curve

    save_path.mkdir(parents=True, exist_ok=True)
    final_ckpt = CHECKPOINT_DIR / "dqn_final.pt"
    if final_ckpt.exists():
        shutil.copy2(final_ckpt, save_path / "dqn_final.pt")
    for ckpt in CHECKPOINT_DIR.glob("dqn_ep*.pt"):
        shutil.copy2(ckpt, save_path / ckpt.name)
    reward_png = plot_reward_curve()
    if reward_png.exists():
        shutil.copy2(reward_png, save_path / "reward_curve.png")
    metrics_csv = LOG_DIR / "metrics.csv"
    if metrics_csv.exists():
        shutil.copy2(metrics_csv, save_path / "metrics.csv")


def train(episodes: int, model_type: str, lr: float, batch_size: int,
          save_path: Path, max_steps: int, attack_ratio: float, seed: int,
          reward_config: Path | None = None) -> None:
    """Run the project training loop and save checkpoints plus a reward curve."""
    if episodes <= 0:
        raise ValueError("--episodes must be positive")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if lr <= 0:
        raise ValueError("--lr must be positive")
    if reward_config is not None:
        if not reward_config.exists():
            raise FileNotFoundError(reward_config)

    algo = "sb3" if model_type == "sb3" else "custom"
    cmd = [
        sys.executable,
        "-m",
        "agent.train",
        "--algo", algo,
        "--episodes", str(episodes),
        "--max-steps", str(max_steps),
        "--attack-ratio", str(attack_ratio),
        "--seed", str(seed),
    ]
    env = os.environ.copy()
    if reward_config is not None:
        env["SDNIDS_REWARD_CONFIG"] = str(reward_config.resolve())
    print(f"[train.py] model={model_type} episodes={episodes} max_steps={max_steps} lr={lr} batch_size={batch_size}")
    subprocess.run(cmd, check=True, env=env)
    _copy_outputs(save_path)
    print(f"[train.py] Saved outputs to {save_path}")


def main() -> int:
    """Script entry point."""
    args = parse_args()
    train(
        episodes=args.episodes,
        model_type=args.model,
        lr=args.lr,
        batch_size=args.batch_size,
        save_path=args.save_path,
        max_steps=args.max_steps,
        attack_ratio=args.attack_ratio,
        seed=args.seed,
        reward_config=args.reward_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
