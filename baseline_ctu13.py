"""Train supervised CTU-13 baselines and export their metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from dataset.ctu13_loader import CTU13Loader


DEFAULT_DATA_DIR = Path("~/Tải về/CTU-13-Dataset/Dataset/")


def metrics(y_true, y_pred) -> dict[str, float | int]:
    """Compute binary detection metrics including false-positive rate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / max(fp + tn, 1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--sample-frac", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser()
    if not data_dir.exists() and Path("dataset").exists():
        data_dir = Path("dataset")
        print(f"Dataset path not found; using local directory: {data_dir}")
    X, y, _ = CTU13Loader().load(data_dir, args.sample_frac)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
        ),
        "LinearSVC": make_pipeline(StandardScaler(), LinearSVC(max_iter=2000, random_state=42)),
    }
    rows = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        rows.append({"model": name, **metrics(y_test, model.predict(X_test))})

    results = pd.DataFrame(rows)
    print("\n=== CTU-13 Baseline Results ===")
    print(results[["model", "precision", "recall", "f1", "fpr"]].to_string(index=False))
    Path("results").mkdir(exist_ok=True)
    results.to_csv("results/baseline_ctu13.csv", index=False)
    print("\nSaved: results/baseline_ctu13.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

