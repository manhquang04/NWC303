"""Run RQ2 reward ablation on the InSDN dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from experiment_real import metrics_from_training_log, reward_configs
from train_real import train_one_dataset


def run_ablation_insdn(
    output_dir: Path = Path("runs"),
    episodes: int = 50,
    seed: int = 42,
    device: str = "auto",
    num_workers: int = 8,
    batch_size: int = 512,
) -> List[dict]:
    """Train/evaluate all reward configs on InSDN and export a CSV summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []

    for cfg_path in reward_configs():
        reward_name = cfg_path.name
        save_path = output_dir / f"ablation_insdn_{cfg_path.stem}"
        print(f"\n[InSDN] reward_config={reward_name}")
        args = argparse.Namespace(
            dataset="insdn",
            episodes=episodes,
            reward=reward_name,
            save_path=save_path,
            eval_every=max(episodes, 1),
            seed=seed,
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            max_train_samples=None,
            eval_limit=None,
            no_balanced_sampler=False,
            profile=False,
        )
        train_one_dataset("insdn", args)
        metrics = metrics_from_training_log(save_path / "training_log.csv")
        row = {
            "dataset": "InSDN",
            "reward_config": cfg_path.stem,
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
