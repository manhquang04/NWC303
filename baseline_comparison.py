"""Baseline comparison: DRL vs supervised classifiers on real datasets."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

from dataset.data_loader import ARPDataLoader, InSDNDataLoader
from results_summary import metrics_from_eval_csv


MODELS = {
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    ),
    "LogisticReg": lambda: LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    ),
    "LinearSVM": lambda: LinearSVC(
        max_iter=5000,
        random_state=42,
        class_weight="balanced",
        dual="auto",
    ),
}

DATASETS = {
    "ARP": ARPDataLoader,
    "InSDN": InSDNDataLoader,
}

DRL_EVAL_FILES = {
    "ARP": Path("runs/eval_real_arp_final.csv"),
    "InSDN": Path("runs/eval_real_insdn_final.csv"),
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | int]:
    """Compute binary classification metrics with attack as positive class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }


def current_drl_results() -> Dict[str, Dict[str, float]]:
    """Load verified DRL metrics from evaluate_real CSV outputs."""
    rows: Dict[str, Dict[str, float]] = {}
    for dataset, path in DRL_EVAL_FILES.items():
        if not path.exists():
            continue
        metrics = metrics_from_eval_csv(path)
        rows[dataset] = {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "fpr": float(metrics["fpr"]),
        }
    return rows


def run_baselines(output_dir: Path = Path("runs")) -> List[dict]:
    """Train and evaluate RF/LR/LinearSVM baselines on real dataset splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []

    for ds_name, loader_cls in DATASETS.items():
        print(f"\n{'=' * 50}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 50}")
        X_train, X_test, y_train, y_test, _ = loader_cls().load()
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"Test attack ratio: {y_test.mean():.4f}")

        for model_name, model_factory in MODELS.items():
            print(f"\n  Training {model_name}...", end=" ", flush=True)
            t0 = time.time()
            model = model_factory()
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            y_pred = np.asarray(model.predict(X_test), dtype=int)
            y_pred = np.clip(y_pred, 0, 1).astype(int)
            metrics = compute_metrics(y_test, y_pred)
            elapsed = time.time() - t0
            row = {
                "dataset": ds_name,
                "model": model_name,
                "train_time_s": round(train_time, 2),
                **metrics,
            }
            rows.append(row)
            print(f"done ({elapsed:.1f}s)")
            print(
                f"    F1={metrics['f1']:.4f}  FPR={metrics['fpr']:.4f}  "
                f"Recall={metrics['recall']:.4f}  Precision={metrics['precision']:.4f}"
            )

    out_path = output_dir / "baseline_results.csv"
    if rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nExported: {out_path}")
    return rows


def print_comparison_table(baseline_rows: List[dict], drl_results: Dict[str, Dict[str, float]]) -> None:
    """Print DRL-vs-baseline metrics for each dataset."""
    print("\n" + "=" * 70)
    print("COMPARISON TABLE: DRL vs Baselines")
    print("=" * 70)

    for ds in ["ARP", "InSDN"]:
        print(f"\n{ds} Dataset")
        print(f"  {'Model':<20} {'F1':>8} {'FPR':>8} {'Recall':>8} {'Precision':>10}")
        print(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")
        d = drl_results.get(ds, {})
        print(
            f"  {'DRL (ours)':<20} "
            f"{d.get('f1', float('nan')):>8.4f} "
            f"{d.get('fpr', float('nan')):>8.4f} "
            f"{d.get('recall', float('nan')):>8.4f} "
            f"{d.get('precision', float('nan')):>10.4f}  our method"
        )

        for row in baseline_rows:
            if row["dataset"] != ds:
                continue
            f1_note = "lower" if row["f1"] < d.get("f1", 0.0) else "same/higher"
            fpr_note = "higher" if row["fpr"] > d.get("fpr", 1.0) else "same/lower"
            print(
                f"  {row['model']:<20} "
                f"{row['f1']:>8.4f} "
                f"{row['fpr']:>8.4f} "
                f"{row['recall']:>8.4f} "
                f"{row['precision']:>10.4f}  F1 {f1_note}, FPR {fpr_note}"
            )


def main() -> int:
    """Script entry point."""
    rows = run_baselines()
    print_comparison_table(rows, current_drl_results())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
