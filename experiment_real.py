"""Reward ablation on real tabular datasets using the vectorized trainer."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from train_real import train_one_dataset


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Run reward ablation on real datasets.")
    p.add_argument("--dataset", choices=["arp", "insdn", "both"], required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-train-samples", type=int, default=0,
                   help="Optional cap for train rows per dataset; 0 means all rows.")
    p.add_argument("--eval-limit", type=int, default=0,
                   help="Optional cap for test rows; 0 means all rows.")
    p.add_argument("--output-dir", type=Path, default=Path("runs"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def reward_configs() -> List[Path]:
    """Return all reward YAML configs."""
    return sorted(Path("config").glob("reward*.yaml"))


def metrics_from_training_log(path: Path) -> dict:
    """Use the best non-zero evaluation row from a training log."""
    df = pd.read_csv(path)
    eval_rows = df[(df["tp"] + df["fp"] + df["tn"] + df["fn"]) > 0].copy()
    if eval_rows.empty:
        eval_rows = df.copy()
    best_idx = eval_rows["f1"].idxmax()
    row = eval_rows.loc[best_idx]
    return {
        "accuracy": float(row.get("accuracy", 0.0)),
        "precision": float(row.get("precision", 0.0)),
        "recall": float(row.get("recall", 0.0)),
        "fpr": float(row.get("fpr", 0.0)),
        "f1": float(row.get("f1", 0.0)),
        "tp": int(row.get("tp", 0)),
        "fp": int(row.get("fp", 0)),
        "tn": int(row.get("tn", 0)),
        "fn": int(row.get("fn", 0)),
        "best_episode": int(row.get("episode", 0)),
    }


def run_one_dataset(name: str, args: argparse.Namespace) -> List[dict]:
    """Run all reward configs for one dataset."""
    rows: List[dict] = []
    for cfg_path in reward_configs():
        reward_name = cfg_path.name
        save_path = args.output_dir / f"ablation_{name}_{cfg_path.stem}"
        print(f"\n[{name}] reward_config={reward_name}")
        train_args = argparse.Namespace(
            dataset=name,
            episodes=args.episodes,
            reward=reward_name,
            save_path=save_path,
            eval_every=max(args.episodes, 1),
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            batch_size=args.batch_size or None,
            max_train_samples=args.max_train_samples or None,
            eval_limit=args.eval_limit or None,
            no_balanced_sampler=False,
            profile=False,
        )
        train_one_dataset(name, train_args)
        metrics = metrics_from_training_log(save_path / "training_log.csv")
        rows.append({
            "dataset": name,
            "reward_config": cfg_path.stem,
            "rows_train": "",
            "rows_test": "",
            "features": "",
            **metrics,
            "action_dist": "",
        })
    return rows


def write_results(rows: List[dict], dataset_name: str, output_dir: Path, output: Path | None) -> Path:
    """Export ablation results to CSV."""
    out = output or (output_dir / f"ablation_real_{dataset_name}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def print_table(rows: Iterable[dict]) -> None:
    """Print reward ablation comparison."""
    headers = ["dataset", "reward_config", "accuracy", "recall", "fpr", "f1", "best_episode"]
    print("\n" + " | ".join(headers))
    print("-+-".join("-" * len(h) for h in headers))
    for row in rows:
        print(" | ".join(
            f"{row[h]:.4f}" if isinstance(row.get(h), float) else str(row.get(h))
            for h in headers
        ))


def main() -> int:
    """Script entry point."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    datasets = ["arp", "insdn"] if args.dataset == "both" else [args.dataset]
    all_rows: List[dict] = []
    for name in datasets:
        all_rows.extend(run_one_dataset(name, args))
    out = write_results(all_rows, args.dataset, args.output_dir, args.output)
    print_table(all_rows)
    print(f"\nExported: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
