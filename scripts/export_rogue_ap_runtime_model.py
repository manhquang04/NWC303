#!/usr/bin/env python3
"""Train and export a compact Rogue AP runtime scorer.

The exported bundle is intentionally small and deployment-oriented:
- train only on the training split
- tune the operating threshold on validation only
- exclude source/audit and label-derived columns from the feature matrix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline


AUDIT_COLUMNS = {
    "source_file",
    "file_id",
    "event_id",
    "window_kind",
    "row_start",
    "row_end",
}
LABEL_DERIVED_COLUMNS = {
    "label",
    "rogue_frames",
    "normal_frames",
    "rogue_frame_ratio",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("processed/rogue_ap_event_windows/tight_event"))
    p.add_argument("--out-dir", type=Path, default=Path("models/rogue_ap_runtime"))
    p.add_argument("--target-fpr", type=float, default=0.20)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--min-samples-leaf", type=int, default=2)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    files = sorted((data_dir / split).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_dir / split}")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def choose_features(df: pd.DataFrame) -> list[str]:
    blocked = AUDIT_COLUMNS | LABEL_DERIVED_COLUMNS
    features = []
    for col in df.columns:
        if col in blocked:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        features.append(col)
    return features


def fpr_at_threshold(y_true, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def metrics_at_threshold(y_true, score, threshold):
    pred = (score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, score)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / (fp + tn) if (fp + tn) else 0.0),
        "auroc": float(roc_auc_score(y_true, score)) if len(set(y_true)) == 2 else float("nan"),
        "pr_auc": float(auc(recall_curve, precision_curve)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def tune_threshold(y_true, score, target_fpr: float):
    candidates = np.unique(np.quantile(score, np.linspace(0, 1, 501)))
    best = None
    for threshold in candidates:
        metrics = metrics_at_threshold(y_true, score, threshold)
        if metrics["fpr"] <= target_fpr:
            if best is None or metrics["f1"] > best["f1"] or (metrics["f1"] == best["f1"] and metrics["recall"] > best["recall"]):
                best = metrics
    if best is not None:
        return best
    # If no threshold satisfies the target, choose the lowest-FPR operating point.
    return min((metrics_at_threshold(y_true, score, t) for t in candidates), key=lambda m: (m["fpr"], -m["f1"]))


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = load_split(args.data_dir, "train")
    val = load_split(args.data_dir, "val")
    test = load_split(args.data_dir, "test")
    features = choose_features(train)
    if "label" not in train.columns:
        raise ValueError("label column is required")

    x_train = train[features]
    y_train = train["label"].astype(int)
    x_val = val[features]
    y_val = val["label"].astype(int)
    x_test = test[features]
    y_test = test["label"].astype(int)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    min_samples_leaf=args.min_samples_leaf,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    val_score = model.predict_proba(x_val)[:, 1]
    threshold_metrics = tune_threshold(y_val.to_numpy(), val_score, args.target_fpr)
    threshold = threshold_metrics["threshold"]
    test_score = model.predict_proba(x_test)[:, 1]
    test_metrics = metrics_at_threshold(y_test.to_numpy(), test_score, threshold)

    rf = model.named_steps["rf"]
    importance = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(args.out_dir / "feature_importance.csv", index=False)

    bundle = {
        "model": model,
        "feature_columns": features,
        "threshold": threshold,
        "target_fpr": args.target_fpr,
        "audit_columns": sorted(AUDIT_COLUMNS),
        "label_derived_columns": sorted(LABEL_DERIVED_COLUMNS),
        "window_contract": {
            "recommended_window_frames": 100,
            "runtime_features": features,
            "source_file_used_as_feature": False,
        },
    }
    joblib.dump(bundle, args.out_dir / "rogue_ap_runtime_rf.joblib")
    report = {
        "data_dir": str(args.data_dir),
        "model_path": str(args.out_dir / "rogue_ap_runtime_rf.joblib"),
        "feature_count": len(features),
        "features": features,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_label_counts": y_train.value_counts().sort_index().to_dict(),
        "val_label_counts": y_val.value_counts().sort_index().to_dict(),
        "test_label_counts": y_test.value_counts().sort_index().to_dict(),
        "validation_threshold_metrics": threshold_metrics,
        "test_metrics": test_metrics,
        "top_features": importance.head(20).to_dict(orient="records"),
    }
    (args.out_dir / "runtime_model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "Rogue AP Runtime Model Export",
        "",
        f"Model: RandomForest balanced_subsample",
        f"Features: {len(features)}",
        f"Threshold tuned on validation only: {threshold:.6f}",
        f"Target FPR: {args.target_fpr:.2f}",
        "",
        "Test metrics:",
        f"- precision: {test_metrics['precision']:.4f}",
        f"- recall: {test_metrics['recall']:.4f}",
        f"- f1: {test_metrics['f1']:.4f}",
        f"- fpr: {test_metrics['fpr']:.4f}",
        f"- auroc: {test_metrics['auroc']:.4f}",
        f"- pr_auc: {test_metrics['pr_auc']:.4f}",
        "",
        "Safety:",
        "- source_file/file_id/event_id are excluded from features.",
        "- label-derived fields are excluded from features.",
    ]
    (args.out_dir / "runtime_model_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
