"""Convenience evaluation script for trained SDN DRL-IDS agents."""

from __future__ import annotations

import argparse
import shutil
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
             output: Path, agent: str) -> Path:
    """Evaluate a trained policy and export TP/FP/TN/FN, recall, and F1 to CSV."""
    if episodes <= 0:
        raise ValueError("--episodes must be positive")
    checkpoint = _resolve_checkpoint(model_path)

    from evaluation.metrics import main as eval_main

    if output.exists():
        output.unlink()
    sys.argv = [
        "evaluate",
        "--checkpoint", str(checkpoint),
        "--scenario", scenario,
        "--agent", agent,
        "--episodes", str(episodes),
        "--max-steps", str(max_steps),
        "--out-csv", str(output),
    ]
    print(f"[evaluate.py] scenario={scenario} agent={agent} episodes={episodes} max_steps={max_steps}")
    rc = eval_main()
    if rc != 0:
        raise RuntimeError(f"Evaluation failed with exit code {rc}")
    print(f"[evaluate.py] Results saved to {output}")
    return output


def main() -> int:
    """Script entry point."""
    args = parse_args()
    evaluate(args.model, args.scenario, args.episodes, args.max_steps, args.output, args.agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
