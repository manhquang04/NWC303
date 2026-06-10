"""Print RQ1/RQ2 result summaries for the paper."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Dict

import pandas as pd


ACTION_NAMES = ("allow", "flag", "block", "isolate")


def metrics_from_eval_csv(path: str | Path) -> Dict[str, object]:
    """Read evaluate_real output and return binary detection metrics."""
    df = pd.read_csv(path)
    if {"accuracy", "precision", "recall", "f1", "fpr"}.issubset(df.columns):
        row = df.iloc[-1]
        return {
            "accuracy": float(row["accuracy"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "fpr": float(row["fpr"]),
            "tp": int(row.get("tp", 0)),
            "fp": int(row.get("fp", 0)),
            "tn": int(row.get("tn", 0)),
            "fn": int(row.get("fn", 0)),
            "action_dist": row.get("action_dist", "{}"),
        }

    y_true = df["ground_truth"].astype(int)
    y_pred = df["predicted_attack"].astype(int)
    actions = df["action"].astype(int)
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
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "action_dist": action_dist,
    }


def print_rq1() -> None:
    """RQ1: DRL performance on real datasets."""
    print("\n" + "=" * 60)
    print("RQ1: DRL Performance on Real Datasets")
    print("=" * 60)

    for dataset, path in [
        ("ARP SDN (Mendeley)", "runs/eval_real_arp_final.csv"),
        ("InSDN (Kaggle)", "runs/eval_real_insdn_final.csv"),
    ]:
        if not os.path.exists(path):
            print(f"  {dataset}: file not found: {path}")
            continue
        metrics = metrics_from_eval_csv(path)
        print(f"\n  Dataset: {dataset}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  FPR:       {metrics['fpr']:.4f}")
        print(
            f"  TP={metrics['tp']} FP={metrics['fp']} "
            f"TN={metrics['tn']} FN={metrics['fn']}"
        )
        print(f"  Actions:   {metrics['action_dist']}")


def print_rq2() -> None:
    """RQ2: Reward function ablation on ARP dataset."""
    print("\n" + "=" * 60)
    print("RQ2: Reward Function Ablation Study (ARP dataset)")
    print("=" * 60)

    path = "runs/ablation_real_arp.csv"
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return

    df = pd.read_csv(path)
    if "f1" not in df.columns or "reward_config" not in df.columns:
        print("  Missing f1/reward_config columns.")
        return
    df = df.sort_values("f1", ascending=False).reset_index(drop=True)
    print(f"\n  {'Config':<28} {'Accuracy':>10} {'Recall':>10} {'F1':>10} {'FPR':>10}")
    print(f"  {'-' * 72}")
    for i, row in df.iterrows():
        marker = " <- BEST" if i == 0 else ""
        print(
            f"  {row['reward_config']:<28} "
            f"{row.get('accuracy', 0):>10.4f} "
            f"{row.get('recall', 0):>10.4f} "
            f"{row.get('f1', 0):>10.4f} "
            f"{row.get('fpr', 0):>10.4f}{marker}"
        )


def print_training_curves() -> None:
    """Print learning curve summaries."""
    print("\n" + "=" * 60)
    print("Training Curves")
    print("=" * 60)

    for dataset, path in [
        ("ARP", "runs/real_arp_v2/training_log.csv"),
        ("InSDN", "runs/real_insdn_v2/training_log.csv"),
    ]:
        if not os.path.exists(path):
            print(f"  {dataset}: {path} not found")
            continue
        df = pd.read_csv(path)
        if "f1" not in df.columns:
            continue
        if {"tp", "fp", "tn", "fn"}.issubset(df.columns):
            eval_df = df[(df["tp"] + df["fp"] + df["tn"] + df["fn"]) > 0].copy()
        else:
            eval_df = df.copy()
        if eval_df.empty:
            print(f"  {dataset}: no evaluated episodes found")
            continue
        best_idx = eval_df["f1"].idxmax()
        best_ep = int(eval_df.loc[best_idx, "episode"])
        print(f"\n  {dataset}:")
        print(f"  Best F1: {eval_df['f1'].max():.4f} at episode {best_ep}")
        print(f"  Final evaluated F1: {eval_df['f1'].iloc[-1]:.4f}")
        print("  F1 progression (evaluated episodes):")
        step_df = eval_df[["episode", "f1", "recall", "fpr"]]
        print(step_df.to_string(index=False))


def main() -> int:
    """Script entry point."""
    print_rq1()
    print_rq2()
    print_training_curves()
    print("\nResults ready for paper writing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
