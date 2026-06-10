"""Train all reward-ablation checkpoints on the InSDN dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_real import print_table, reward_configs, metrics_from_training_log, write_results
from train_real import train_one_dataset


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train InSDN reward ablation checkpoints.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("runs/ablation_insdn_train_results.csv"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Train one InSDN checkpoint per reward config."""
    args = parse_args()
    rows = []
    for cfg_path in reward_configs():
        save_path = args.output_dir / f"ablation_insdn_{cfg_path.stem}"
        print(f"\n[InSDN] training reward_config={cfg_path.name}")
        train_args = argparse.Namespace(
            dataset="insdn",
            episodes=args.episodes,
            reward=cfg_path.name,
            save_path=save_path,
            eval_every=max(args.episodes, 1),
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            max_train_samples=None,
            eval_limit=None,
            no_balanced_sampler=False,
            profile=False,
        )
        train_one_dataset("insdn", train_args)
        metrics = metrics_from_training_log(save_path / "training_log.csv")
        rows.append({
            "dataset": "insdn",
            "reward_config": cfg_path.stem,
            **metrics,
        })

    out = write_results(rows, "insdn", args.output_dir, args.output)
    print_table(rows)
    print(f"\nExported: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
