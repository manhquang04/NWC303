"""Convenience evaluation script for trained SDN DRL-IDS agents."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate DQN on SDN attack scenarios.")
    parser.add_argument("--model", type=Path, default=Path("runs/checkpoints/dqn_final.pt"))
    parser.add_argument("--scenario", choices=["normal", "arp", "rogue", "mixed", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation_results.csv"))
    parser.add_argument("--agent", choices=["custom", "baseline", "sb3", "all"], default="custom")
    parser.add_argument("--reward-config", type=Path, default=None)
    return parser.parse_args()


def _resolve_checkpoint(model_path: Path) -> Path:
    """Resolve checkpoint path and copy wrapper output into canonical location."""
    from config import CHECKPOINT_DIR

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    canonical = CHECKPOINT_DIR / "dqn_final.pt"
    if model_path.resolve() != canonical.resolve():
        canonical.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, canonical)
    return canonical


def evaluate(model_path: Path, scenario: str, episodes: int, max_steps: int,
             output: Path, agent: str, reward_config: Path | None = None) -> Path:
    """Evaluate a trained policy and export TP/FP/TN/FN, recall, and F1 to CSV."""
    if episodes <= 0:
        raise ValueError("--episodes must be positive")
    if reward_config is not None and not reward_config.exists():
        raise FileNotFoundError(reward_config)
    checkpoint = _resolve_checkpoint(model_path)

    if output.exists():
        output.unlink()
    cmd = [
        sys.executable,
        "-m",
        "evaluation.metrics",
        "--checkpoint", str(checkpoint),
        "--scenario", scenario,
        "--agent", agent,
        "--episodes", str(episodes),
        "--max-steps", str(max_steps),
        "--out-csv", str(output),
    ]
    env = os.environ.copy()
    if reward_config is not None:
        env["SDNIDS_REWARD_CONFIG"] = str(reward_config.resolve())
    print(f"[evaluate.py] scenario={scenario} agent={agent} episodes={episodes} max_steps={max_steps}")
    subprocess.run(cmd, check=True, env=env)
    print(f"[evaluate.py] Results saved to {output}")
    return output


def main() -> int:
    """Script entry point."""
    args = parse_args()
    evaluate(
        args.model,
        args.scenario,
        args.episodes,
        args.max_steps,
        args.output,
        args.agent,
        reward_config=args.reward_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
