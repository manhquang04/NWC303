#!/usr/bin/env python3
"""
Train supervised baselines on the cleaned Rogue AP timing-dropped dataset.

The script uses only train/val/test parquet splits. It never fits preprocessing
or thresholds on test, and it ignores the inspect split.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


LEAK_RE = re.compile(
    r"(time|timestamp|mactime|tsf|tsft|relative|delta|bssid|hw_mac|mac_addr|"
    r"ip_src|ip_dst|proto_ipv4|payload|full_uri|http_host|json|ssh_cookie)",
    re.IGNORECASE,
)


@dataclass
class SplitCheck:
    split: str
    rows: int
    cols: int
    normal: int
    rogueap: int
    missing_values: int
    duplicate_rows: int
    duplicate_rate_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Rogue AP supervised baselines.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("processed/rogue_ap_timing_dropped"),
        help="Cleaned timing-dropped Rogue AP dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/rogue_ap_baseline_timing_dropped"),
        help="Directory for metrics and plots.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--mlp-negatives",
        type=int,
        default=100_000,
        help="Number of normal train rows sampled for the MLP; all positives are included.",
    )
    parser.add_argument(
        "--tree-negatives",
        type=int,
        default=150_000,
        help="Number of normal train rows sampled for Random Forest/XGBoost; all positives are included.",
    )
    parser.add_argument(
        "--fpr-limit",
        type=float,
        default=0.01,
        help="Validation FPR constraint for the secondary threshold policy.",
    )
    return parser.parse_args()


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    paths = sorted((data_dir / split).glob("part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet parts found for split={split} in {data_dir / split}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def check_split(split: str, df: pd.DataFrame) -> SplitCheck:
    if "label" not in df.columns:
        raise ValueError(f"{split}: missing label column")
    counts = df["label"].value_counts().to_dict()
    if sorted(counts) != [0, 1]:
        raise ValueError(f"{split}: expected labels 0/1, got {counts}")
    missing = int(df.isna().sum().sum())
    row_hashes = pd.util.hash_pandas_object(df, index=False)
    duplicates = int(row_hashes.duplicated().sum())
    return SplitCheck(
        split=split,
        rows=len(df),
        cols=df.shape[1],
        normal=int(counts.get(0, 0)),
        rogueap=int(counts.get(1, 0)),
        missing_values=missing,
        duplicate_rows=duplicates,
        duplicate_rate_pct=round(100.0 * duplicates / max(len(df), 1), 6),
    )


def evaluate_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fpr = fp / max(fp + tn, 1)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_candidates(y_val: np.ndarray, val_scores: np.ndarray, fpr_limit: float) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
    if len(thresholds) == 0:
        return {"default_0_5": 0.5, "val_best_f1": 0.5, "val_fpr_constrained": 0.5}
    pr_precision = precision[:-1]
    pr_recall = recall[:-1]
    f1 = (2 * pr_precision * pr_recall) / np.maximum(pr_precision + pr_recall, 1e-12)
    best_idx = int(np.nanargmax(f1))
    best_f1_threshold = float(thresholds[best_idx])

    fpr, tpr, roc_thresholds = roc_curve(y_val, val_scores)
    valid = np.flatnonzero(fpr <= fpr_limit)
    if len(valid):
        best_valid = valid[np.argmax(tpr[valid])]
        fpr_threshold = float(roc_thresholds[best_valid])
    else:
        fpr_threshold = best_f1_threshold
    return {
        "default_0_5": 0.5,
        "val_best_f1": best_f1_threshold,
        f"val_fpr_le_{fpr_limit:g}": fpr_threshold,
    }


def score_model(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError(f"Model {model} cannot produce scores")


def metrics_for_model(
    model_name: str,
    y_val: np.ndarray,
    val_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    fpr_limit: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    threshold_rows = []
    thresholds = threshold_candidates(y_val, val_scores, fpr_limit)
    aucs = {
        "val_auroc": float(roc_auc_score(y_val, val_scores)),
        "val_pr_auc": float(average_precision_score(y_val, val_scores)),
        "test_auroc": float(roc_auc_score(y_test, test_scores)),
        "test_pr_auc": float(average_precision_score(y_test, test_scores)),
    }
    for policy, threshold in thresholds.items():
        val_metrics = evaluate_at_threshold(y_val, val_scores, threshold)
        test_metrics = evaluate_at_threshold(y_test, test_scores, threshold)
        threshold_rows.append(
            {
                "model": model_name,
                "policy": policy,
                "selected_on": "validation",
                "threshold": threshold,
                **{f"val_{k}": v for k, v in val_metrics.items() if k != "threshold"},
                **{f"test_{k}": v for k, v in test_metrics.items() if k != "threshold"},
            }
        )
        for split, split_metrics in (("val", val_metrics), ("test", test_metrics)):
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "threshold_policy": policy,
                    **aucs,
                    **split_metrics,
                }
            )
    return rows, threshold_rows


def top_logistic_coefficients(model: LogisticRegression, feature_names: list[str], n: int = 30) -> pd.DataFrame:
    coefs = model.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1][:n]
    return pd.DataFrame(
        {
            "model": "logistic_regression",
            "feature": [feature_names[i] for i in order],
            "coefficient": coefs[order],
            "abs_coefficient": np.abs(coefs[order]),
            "direction": np.where(coefs[order] >= 0, "RogueAP", "Normal"),
        }
    )


def feature_importance_rows(model_name: str, model: Any, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    else:
        return pd.DataFrame(columns=["model", "feature", "importance"])
    order = np.argsort(values)[::-1][:30]
    return pd.DataFrame(
        {
            "model": model_name,
            "feature": [feature_names[i] for i in order],
            "importance": values[order],
        }
    )


def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0, 1], ["Pred Normal", "Pred RogueAP"])
    ax.set_yticks([0, 1], ["True Normal", "True RogueAP"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading train/val/test parquet splits", flush=True)
    train = read_split(args.data_dir, "train")
    val = read_split(args.data_dir, "val")
    test = read_split(args.data_dir, "test")

    checks = [check_split("train", train), check_split("val", val), check_split("test", test)]
    pd.DataFrame([c.__dict__ for c in checks]).to_csv(args.output_dir / "data_checks.csv", index=False)
    for item in checks:
        print(
            f"  {item.split:5s}: rows={item.rows:7d} cols={item.cols:3d} "
            f"Normal={item.normal:7d} RogueAP={item.rogueap:4d} "
            f"missing={item.missing_values} dup={item.duplicate_rows} ({item.duplicate_rate_pct}%)"
            ,
            flush=True,
        )

    feature_cols = [c for c in train.columns if c != "label"]
    suspicious = [c for c in feature_cols if LEAK_RE.search(c)]
    if suspicious:
        print("\nWARNING: suspicious feature names remain:", flush=True)
        for col in suspicious:
            print(f"  - {col}", flush=True)
    else:
        print("\nNo timing/source-id leak pattern found in exported feature names.", flush=True)

    X_train = train[feature_cols]
    y_train = train["label"].to_numpy(dtype=np.int8)
    X_val = val[feature_cols]
    y_val = val["label"].to_numpy(dtype=np.int8)
    X_test = test[feature_cols]
    y_test = test["label"].to_numpy(dtype=np.int8)

    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    print("\n[2/6] Training Logistic Regression", flush=True)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    models: dict[str, Any] = {}
    models["logistic_regression"] = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=args.random_state,
    ).fit(X_train, y_train)

    rng = np.random.default_rng(args.random_state)
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    n_tree_neg = min(len(neg_idx), args.tree_negatives)
    tree_idx = np.concatenate([pos_idx, rng.choice(neg_idx, size=n_tree_neg, replace=False)])
    rng.shuffle(tree_idx)
    X_tree = X_train.iloc[tree_idx]
    y_tree = y_train[tree_idx]

    print("[3/6] Training Random Forest", flush=True)
    models["random_forest"] = RandomForestClassifier(
        n_estimators=80,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.random_state,
    ).fit(X_tree, y_tree)

    print("[4/6] Training XGBoost with light validation tuning", flush=True)
    xgb_grid = [
        {"max_depth": 3, "learning_rate": 0.08, "n_estimators": 120},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 160},
    ]
    best_xgb = None
    best_xgb_score = -np.inf
    xgb_tuning_rows = []
    for params in xgb_grid:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=args.random_state,
            scale_pos_weight=scale_pos_weight,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            **params,
        )
        model.fit(X_tree, y_tree, verbose=False)
        val_scores = score_model(model, X_val)
        pr_auc = average_precision_score(y_val, val_scores)
        xgb_tuning_rows.append({"model": "xgboost", **params, "val_pr_auc": pr_auc})
        if pr_auc > best_xgb_score:
            best_xgb_score = pr_auc
            best_xgb = model
    assert best_xgb is not None
    models["xgboost"] = best_xgb
    pd.DataFrame(xgb_tuning_rows).to_csv(args.output_dir / "xgboost_tuning.csv", index=False)

    print("[5/6] Training small MLP on balanced train sample", flush=True)
    n_neg = min(len(neg_idx), args.mlp_negatives)
    sample_idx = np.concatenate([pos_idx, rng.choice(neg_idx, size=n_neg, replace=False)])
    rng.shuffle(sample_idx)
    X_mlp = X_train.iloc[sample_idx]
    y_mlp = y_train[sample_idx]
    mlp_weights = compute_sample_weight(class_weight="balanced", y=y_mlp)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        batch_size=1024,
        learning_rate_init=1e-3,
        max_iter=60,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=args.random_state,
    )
    try:
        mlp.fit(X_mlp, y_mlp, sample_weight=mlp_weights)
    except TypeError:
        mlp.fit(X_mlp, y_mlp)
    models["mlp_small"] = mlp

    print("[6/6] Evaluating models and writing outputs", flush=True)
    metric_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    importance_frames = []
    coefficient_frames = []
    model_scores: dict[str, dict[str, np.ndarray]] = {}

    for name, model in models.items():
        val_scores = score_model(model, X_val)
        test_scores = score_model(model, X_test)
        model_scores[name] = {"val": val_scores, "test": test_scores}
        rows, t_rows = metrics_for_model(
            name, y_val, val_scores, y_test, test_scores, args.fpr_limit
        )
        metric_rows.extend(rows)
        threshold_rows.extend(t_rows)
        if name == "logistic_regression":
            coefficient_frames.append(top_logistic_coefficients(model, feature_cols))
        if name in {"random_forest", "xgboost"}:
            importance_frames.append(feature_importance_rows(name, model, feature_cols))

    metrics_df = pd.DataFrame(metric_rows)
    thresholds_df = pd.DataFrame(threshold_rows)
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
    thresholds_df.to_csv(args.output_dir / "threshold_analysis.csv", index=False)
    pd.concat(importance_frames, ignore_index=True).to_csv(
        args.output_dir / "feature_importance.csv", index=False
    )
    pd.concat(coefficient_frames, ignore_index=True).to_csv(
        args.output_dir / "top_coefficients.csv", index=False
    )

    # Pick best model by validation F1 under validation-selected F1 threshold.
    selector = thresholds_df[thresholds_df["policy"] == "val_best_f1"].copy()
    selector = selector.sort_values(["val_f1", "test_f1"], ascending=False)
    best_model_name = selector.iloc[0]["model"]
    best_threshold = float(selector.iloc[0]["threshold"])
    best_test_scores = model_scores[best_model_name]["test"]
    best_pred = (best_test_scores >= best_threshold).astype(np.int8)
    cm = confusion_matrix(y_test, best_pred, labels=[0, 1])
    plot_confusion_matrix(
        cm,
        f"{best_model_name} test CM @ val-best-F1 threshold",
        args.output_dir / "confusion_matrix.png",
    )

    source_files = {}
    metadata_path = args.data_dir / "split_metadata.json"
    if metadata_path.exists():
        source_files = json.loads(metadata_path.read_text())

    summary_lines = [
        "Rogue AP timing-dropped supervised baseline summary",
        f"Data dir: {args.data_dir}",
        f"Output dir: {args.output_dir}",
        "",
        "Split checks:",
    ]
    for item in checks:
        summary_lines.append(
            f"- {item.split}: rows={item.rows}, Normal={item.normal}, RogueAP={item.rogueap}, "
            f"missing={item.missing_values}, duplicates_after_export={item.duplicate_rows} "
            f"({item.duplicate_rate_pct}%)"
        )
    summary_lines.extend(
        [
            "",
            f"Suspicious feature names remaining: {len(suspicious)}",
            *(f"- {col}" for col in suspicious[:30]),
            "",
            f"Best validation-F1 model: {best_model_name}",
            f"Selected threshold from validation: {best_threshold:.8f}",
        ]
    )
    best_rows = metrics_df[
        (metrics_df["model"] == best_model_name)
        & (metrics_df["threshold_policy"] == "val_best_f1")
    ]
    for _, row in best_rows.iterrows():
        summary_lines.append(
            f"- {row['split']}: precision={row['precision']:.6f}, recall={row['recall']:.6f}, "
            f"f1={row['f1']:.6f}, fpr={row['fpr']:.6f}, auroc={row[f'{row['split']}_auroc']:.6f}, "
            f"pr_auc={row[f'{row['split']}_pr_auc']:.6f}"
        )
    if source_files:
        summary_lines.extend(["", "Held-out source files:"])
        for split in ("train", "val", "test"):
            summary_lines.append(f"- {split}: {', '.join(source_files[split]['source_files'])}")
    summary_lines.extend(
        [
            "",
            "Interpretation notes:",
            "- This run uses timing-dropped data only; test is file-held-out.",
            "- Inspect split was not used.",
            "- Per-file metrics are not available because exported parquet rows do not keep source_file.",
            "- If val/test gap is large for a model, treat it as source/file bias risk.",
        ]
    )
    (args.output_dir / "short_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nDone. Best validation-F1 model: {best_model_name}", flush=True)
    print(f"Results written to: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
