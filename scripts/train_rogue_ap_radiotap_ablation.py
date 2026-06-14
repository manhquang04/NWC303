#!/usr/bin/env python3
"""Run strict-current vs strict+radiotap/PHY ablation for Rogue AP detection."""

from __future__ import annotations

import argparse
import json
import re
import warnings
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
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


# Base strict drops from the previous ablation.
STRICT_DROP_PATTERNS = (
    "wlan_duration",
    "wlan_fc_type",
    "wlan_radio_duration",
    "wlan_fc_protected",
    "frame_len",
)

# Additional drops for this run: radiotap / PHY / data-rate / channel / signal.
RADIOTAP_PHY_DROP_PATTERNS = (
    "radiotap_datarate",
    "wlan_radio_data_rate",
    "wlan_fc_subtype",
    "wlan_radio_signal_dbm",
    "radiotap_channel_freq",
    "radiotap_channel_flags",
    "radiotap_channel",
    "wlan_radio_phy",
    "wlan_radio_channel",
    "signal_dbm",
    "data_rate",
    "phy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rogue AP strict ablation training.")
    parser.add_argument("--data-dir", type=Path, default=Path("processed/rogue_ap_timing_dropped"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/rogue_ap_radiotap_phy_ablation"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tree-negatives", type=int, default=150_000)
    parser.add_argument("--mlp-negatives", type=int, default=100_000)
    parser.add_argument("--fpr-limit", type=float, default=0.01)
    parser.add_argument("--chunk-breakdown-size", type=int, default=25_000)
    return parser.parse_args()


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    paths = sorted((data_dir / split).glob("part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet parts found in {data_dir / split}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def split_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    counts = df["label"].value_counts().to_dict()
    hashes = pd.util.hash_pandas_object(df, index=False)
    duplicates = int(hashes.duplicated().sum())
    return {
        "split": name,
        "rows": len(df),
        "cols": df.shape[1],
        "normal": int(counts.get(0, 0)),
        "rogueap": int(counts.get(1, 0)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": duplicates,
        "duplicate_rate_pct": round(100 * duplicates / max(len(df), 1), 6),
    }


def matching_drop_columns(columns: list[str], patterns: tuple[str, ...]) -> list[str]:
    out = []
    for col in columns:
        low = col.lower()
        if any(pattern in low for pattern in patterns):
            out.append(col)
    return out


def eval_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / max(fp + tn, 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def thresholds_from_val(y: np.ndarray, scores: np.ndarray, fpr_limit: float) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if len(thresholds) == 0:
        return {"default_0_5": 0.5, "val_best_f1": 0.5, f"val_fpr_le_{fpr_limit:g}": 0.5}
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_f1 = float(thresholds[int(np.nanargmax(f1))])
    fpr, tpr, roc_thresholds = roc_curve(y, scores)
    valid = np.flatnonzero(fpr <= fpr_limit)
    if len(valid):
        best_valid = valid[np.argmax(tpr[valid])]
        constrained = float(roc_thresholds[best_valid])
    else:
        constrained = best_f1
    return {"default_0_5": 0.5, "val_best_f1": best_f1, f"val_fpr_le_{fpr_limit:g}": constrained}


def score(model: Any, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def train_models(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    random_state: int,
    tree_negatives: int,
    mlp_negatives: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rng = np.random.default_rng(random_state)
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    tree_idx = np.concatenate(
        [pos_idx, rng.choice(neg_idx, size=min(len(neg_idx), tree_negatives), replace=False)]
    )
    rng.shuffle(tree_idx)
    x_tree = x_train.iloc[tree_idx]
    y_tree = y_train[tree_idx]

    models: dict[str, Any] = {}
    models["logistic_regression"] = LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=1000, random_state=random_state
    ).fit(x_train, y_train)
    models["random_forest"] = RandomForestClassifier(
        n_estimators=80,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    ).fit(x_tree, y_tree)

    neg = int((y_train == 0).sum())
    pos = int(y_train.sum())
    scale_pos_weight = neg / max(pos, 1)
    xgb_rows = []
    best_xgb = None
    best_pr = -np.inf
    for params in [
        {"max_depth": 3, "learning_rate": 0.08, "n_estimators": 120},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 160},
    ]:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            **params,
        )
        model.fit(x_tree, y_tree, verbose=False)
        val_pr = average_precision_score(y_val, score(model, x_val))
        xgb_rows.append({"model": "xgboost", **params, "val_pr_auc": val_pr})
        if val_pr > best_pr:
            best_pr = val_pr
            best_xgb = model
    assert best_xgb is not None
    models["xgboost"] = best_xgb

    mlp_idx = np.concatenate(
        [pos_idx, rng.choice(neg_idx, size=min(len(neg_idx), mlp_negatives), replace=False)]
    )
    rng.shuffle(mlp_idx)
    x_mlp = x_train.iloc[mlp_idx]
    y_mlp = y_train[mlp_idx]
    mlp_weights = compute_sample_weight(class_weight="balanced", y=y_mlp)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=1e-4,
        batch_size=1024,
        learning_rate_init=1e-3,
        max_iter=60,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
    )
    try:
        mlp.fit(x_mlp, y_mlp, sample_weight=mlp_weights)
    except TypeError:
        mlp.fit(x_mlp, y_mlp)
    models["mlp_small"] = mlp
    return models, pd.DataFrame(xgb_rows)


def evaluate_models(
    feature_set: str,
    models: dict[str, Any],
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    fpr_limit: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    thresholds = []
    for model_name, model in models.items():
        val_scores = score(model, x_val)
        test_scores = score(model, x_test)
        val_auroc = roc_auc_score(y_val, val_scores)
        val_pr = average_precision_score(y_val, val_scores)
        test_auroc = roc_auc_score(y_test, test_scores)
        test_pr = average_precision_score(y_test, test_scores)
        for policy, threshold in thresholds_from_val(y_val, val_scores, fpr_limit).items():
            val_metrics = eval_at(y_val, val_scores, threshold)
            test_metrics = eval_at(y_test, test_scores, threshold)
            thresholds.append(
                {
                    "feature_set": feature_set,
                    "model": model_name,
                    "policy": policy,
                    "selected_on": "validation",
                    "threshold": threshold,
                    **{f"val_{k}": v for k, v in val_metrics.items() if k != "threshold"},
                    **{f"test_{k}": v for k, v in test_metrics.items() if k != "threshold"},
                }
            )
            for split, split_metrics in [("val", val_metrics), ("test", test_metrics)]:
                metrics.append(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "split": split,
                        "threshold_policy": policy,
                        "val_auroc": val_auroc,
                        "val_pr_auc": val_pr,
                        "test_auroc": test_auroc,
                        "test_pr_auc": test_pr,
                        **split_metrics,
                    }
                )
    return pd.DataFrame(metrics), pd.DataFrame(thresholds)


def importance_tables(feature_set: str, models: dict[str, Any], feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fi_rows = []
    coef_rows = []
    for model_name, model in models.items():
        if hasattr(model, "feature_importances_"):
            vals = np.asarray(model.feature_importances_)
            for idx in np.argsort(vals)[::-1][:30]:
                fi_rows.append(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "feature": feature_cols[idx],
                        "importance": float(vals[idx]),
                    }
                )
        if model_name == "logistic_regression":
            coefs = model.coef_[0]
            for idx in np.argsort(np.abs(coefs))[::-1][:30]:
                coef_rows.append(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "feature": feature_cols[idx],
                        "coefficient": float(coefs[idx]),
                        "abs_coefficient": float(abs(coefs[idx])),
                        "direction": "RogueAP" if coefs[idx] >= 0 else "Normal",
                    }
                )
    return pd.DataFrame(fi_rows), pd.DataFrame(coef_rows)


def chunk_breakdown(
    feature_set: str,
    model_name: str,
    model: Any,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    threshold: float,
    chunk_size: int,
) -> pd.DataFrame:
    rows = []
    scores = score(model, x_test)
    for start in range(0, len(y_test), chunk_size):
        end = min(start + chunk_size, len(y_test))
        y = y_test[start:end]
        if len(np.unique(y)) < 2:
            auroc = np.nan
            pr_auc = np.nan
        else:
            auroc = roc_auc_score(y, scores[start:end])
            pr_auc = average_precision_score(y, scores[start:end])
        m = eval_at(y, scores[start:end], threshold)
        rows.append(
            {
                "breakdown_type": "test_contiguous_row_chunk_no_source_file",
                "feature_set": feature_set,
                "model": model_name,
                "chunk_id": len(rows),
                "row_start": start,
                "row_end": end,
                "rows": end - start,
                "rogueap": int(y.sum()),
                "auroc": auroc,
                "pr_auc": pr_auc,
                **m,
            }
        )
    return pd.DataFrame(rows)


def source_file_breakdown(
    feature_set: str,
    model_name: str,
    metadata: pd.DataFrame,
    y_test: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    temp = metadata.copy()
    group_col = "source_file" if "source_file" in temp.columns else "file_id"
    temp["label"] = y_test
    temp["score"] = scores
    rows = []
    for key, group in temp.groupby(group_col, sort=True):
        y = group["label"].to_numpy(dtype=np.int8)
        s = group["score"].to_numpy()
        if len(np.unique(y)) < 2:
            auroc = np.nan
            pr_auc = np.nan
        else:
            auroc = roc_auc_score(y, s)
            pr_auc = average_precision_score(y, s)
        m = eval_at(y, s, threshold)
        rows.append(
            {
                "breakdown_type": "source_file",
                "feature_set": feature_set,
                "model": model_name,
                "source_file": key,
                "rows": int(len(group)),
                "normal": int((y == 0).sum()),
                "rogueap": int((y == 1).sum()),
                "auroc": auroc,
                "pr_auc": pr_auc,
                **m,
            }
        )
    return pd.DataFrame(rows)


def plot_best_cms(metrics: pd.DataFrame, y_test: np.ndarray, model_scores: dict[str, tuple[np.ndarray, float]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (feature_set, (scores, threshold)) in zip(axes, model_scores.items()):
        pred = (scores >= threshold).astype(np.int8)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(feature_set)
        ax.set_xticks([0, 1], ["Pred Normal", "Pred RogueAP"])
        ax.set_yticks([0, 1], ["True Normal", "True RogueAP"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[1/5] Loading data", flush=True)
    train = read_split(args.data_dir, "train")
    val = read_split(args.data_dir, "val")
    test = read_split(args.data_dir, "test")
    checks = pd.DataFrame([split_check("train", train), split_check("val", val), split_check("test", test)])
    checks.to_csv(args.output_dir / "data_checks_ablation.csv", index=False)
    print(checks.to_string(index=False), flush=True)

    metadata_cols = [c for c in train.columns if c.lower() in {"source_file", "file_id"}]
    has_source_file = bool(metadata_cols)
    feature_cols_all = [c for c in train.columns if c != "label" and c not in metadata_cols]
    base_strict_drops = matching_drop_columns(feature_cols_all, STRICT_DROP_PATTERNS)
    current_strict_cols = [c for c in feature_cols_all if c not in base_strict_drops]
    radiotap_phy_drops = matching_drop_columns(current_strict_cols, RADIOTAP_PHY_DROP_PATTERNS)
    variants = {
        "strict_ablation_current": current_strict_cols,
        "strict_ablation_plus_radiotap_phy": [c for c in current_strict_cols if c not in radiotap_phy_drops],
    }
    (args.output_dir / "ablation_drop_columns.json").write_text(
        json.dumps({
            "base_strict_drop_columns": base_strict_drops,
            "radiotap_phy_drop_columns": radiotap_phy_drops,
            "radiotap_phy_drop_patterns": RADIOTAP_PHY_DROP_PATTERNS,
            "has_source_file": has_source_file
        }, indent=2),
        encoding="utf-8",
    )

    y_train = train["label"].to_numpy(dtype=np.int8)
    y_val = val["label"].to_numpy(dtype=np.int8)
    y_test = test["label"].to_numpy(dtype=np.int8)

    all_metrics = []
    all_thresholds = []
    all_importance = []
    all_coefficients = []
    per_breakdowns = []
    best_cm_scores = {}
    xgb_tuning = []

    for feature_set, feature_cols in variants.items():
        print(f"[2/5] Training {feature_set} with {len(feature_cols)} features", flush=True)
        x_train = train[feature_cols]
        x_val = val[feature_cols]
        x_test = test[feature_cols]
        models, tuning = train_models(
            x_train,
            y_train,
            x_val,
            y_val,
            args.random_state,
            args.tree_negatives,
            args.mlp_negatives,
        )
        tuning.insert(0, "feature_set", feature_set)
        xgb_tuning.append(tuning)
        metrics, thresholds = evaluate_models(feature_set, models, x_val, y_val, x_test, y_test, args.fpr_limit)
        all_metrics.append(metrics)
        all_thresholds.append(thresholds)
        fi, coefs = importance_tables(feature_set, models, feature_cols)
        all_importance.append(fi)
        all_coefficients.append(coefs)

        selector = thresholds[thresholds["policy"] == "val_best_f1"].sort_values(
            ["val_f1", "test_f1"], ascending=False
        )
        best = selector.iloc[0]
        best_model = best["model"]
        best_threshold = float(best["threshold"])
        best_scores = score(models[best_model], x_test)
        best_cm_scores[feature_set] = (best_scores, best_threshold)
        if has_source_file:
            per_breakdowns.append(
                source_file_breakdown(
                    feature_set,
                    best_model,
                    test[metadata_cols],
                    y_test,
                    best_scores,
                    best_threshold,
                )
            )
        else:
            per_breakdowns.append(
                chunk_breakdown(
                    feature_set,
                    best_model,
                    models[best_model],
                    x_test,
                    y_test,
                    best_threshold,
                    args.chunk_breakdown_size,
                )
            )

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    thresholds_df = pd.concat(all_thresholds, ignore_index=True)
    importance_df = pd.concat(all_importance, ignore_index=True)
    coefficients_df = pd.concat(all_coefficients, ignore_index=True)
    breakdown_df = pd.concat(per_breakdowns, ignore_index=True)
    pd.concat(xgb_tuning, ignore_index=True).to_csv(args.output_dir / "xgboost_tuning_ablation.csv", index=False)
    metrics_df.to_csv(args.output_dir / "metrics_ablation.csv", index=False)
    thresholds_df.to_csv(args.output_dir / "threshold_analysis_ablation.csv", index=False)
    importance_df.to_csv(args.output_dir / "feature_importance_ablation.csv", index=False)
    coefficients_df.to_csv(args.output_dir / "top_coefficients_ablation.csv", index=False)
    breakdown_df.to_csv(args.output_dir / "per_file_breakdown.csv", index=False)
    plot_best_cms(metrics_df, y_test, best_cm_scores, args.output_dir / "confusion_matrix_ablation.png")

    summary = ["Rogue AP radiotap/PHY ablation summary", ""]
    summary.append("Per-file breakdown limitation:")
    if has_source_file:
        summary.append("- source_file/file_id is present.")
    else:
        summary.append("- source_file/file_id is NOT present in current Parquet export; per_file_breakdown.csv uses contiguous test row chunks as proxy.")
        summary.append("- Re-export with source_file metadata is required for true per-file metrics.")
    summary.extend(["", "Base strict dropped features:"])
    summary.extend(f"- {c}" for c in base_strict_drops)
    summary.extend(["", "Additional radiotap/PHY/data-rate/signal dropped features:"])
    summary.extend(f"- {c}" for c in radiotap_phy_drops)
    summary.extend(["", "Validation-selected F1 results:"])
    selected = metrics_df[metrics_df["threshold_policy"] == "val_best_f1"]
    for _, row in selected[selected["split"] == "test"].sort_values(["feature_set", "f1"], ascending=[True, False]).iterrows():
        summary.append(
            f"- {row['feature_set']} / {row['model']}: test precision={row['precision']:.6f}, "
            f"recall={row['recall']:.6f}, f1={row['f1']:.6f}, fpr={row['fpr']:.6f}, "
            f"auroc={row['test_auroc']:.6f}, pr_auc={row['test_pr_auc']:.6f}"
        )
    summary.extend(["", "Top tree features by feature set:"])
    for feature_set in variants:
        top = importance_df[importance_df["feature_set"] == feature_set].groupby("model").head(5)
        summary.append(f"- {feature_set}:")
        for _, row in top.iterrows():
            summary.append(f"  {row['model']}: {row['feature']} ({row['importance']:.6f})")
    summary.extend(
        [
            "",
            "Interpretation:",
            "- If strict ablation remains perfect or near-perfect, the dataset still contains very separable non-timing protocol/radiotap signatures.",
            "- Do not move to DQN until supervised hardening and true per-file metadata export are complete.",
        ]
    )
    (args.output_dir / "short_summary_ablation.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"[5/5] Wrote ablation outputs to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
