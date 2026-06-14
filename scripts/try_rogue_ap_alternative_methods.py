#!/usr/bin/env python3
"""Alternative clean Rogue AP detectors: anomaly and sequence/context baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


BASE_FEATURES = [
    "n_frames", "ratio_retry", "ratio_protected", "ratio_moredata", "ratio_pwrmgt", "ratio_order",
    "ratio_frag", "ratio_ds_0x00000000", "ratio_ds_0x00000001", "ratio_ds_0x00000002",
    "ratio_ds_0x00000003", "ratio_type_0", "ratio_type_1", "ratio_type_2",
    "ratio_llc_present", "ratio_ip_present", "ratio_tcp_present",
]
META = {"source_file", "file_id", "event_id", "window_kind", "row_start", "row_end", "rogue_frames", "normal_frames", "rogue_frame_ratio", "label"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("processed/rogue_ap_event_windows/tight_event"))
    p.add_argument("--results-dir", type=Path, default=Path("results/rogue_ap_alternative_methods"))
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def load_split(base, split):
    return pd.read_parquet(base / split / "part-00000.parquet").sort_values(["source_file", "row_start", "event_id"]).reset_index(drop=True)


def add_context(df):
    out = df.copy()
    for col in BASE_FEATURES:
        g = out.groupby("source_file")[col]
        out[f"{col}_ma3"] = g.transform(lambda s: s.rolling(3, min_periods=1).mean())
        out[f"{col}_ma5"] = g.transform(lambda s: s.rolling(5, min_periods=1).mean())
        out[f"{col}_delta1"] = g.diff().fillna(0)
    out["position_frac"] = out.groupby("source_file").cumcount() / out.groupby("source_file")["source_file"].transform("size").sub(1).clip(lower=1)
    return out


def features(df):
    return [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]


def safe_auc(y, s, kind):
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, s) if kind == "roc" else average_precision_score(y, s))


def eval_at(y, s, th):
    pred = (s >= th).astype(np.int8)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(th), "precision": float(pr), "recall": float(rc), "f1": float(f1), "fpr": float(fp / max(fp + tn, 1)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def choose_thresholds(y, s):
    p, r, th = precision_recall_curve(y, s)
    best = 0.5
    if len(th):
        f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
        best = float(th[int(np.nanargmax(f1))])
    fpr, tpr, rth = roc_curve(y, s)
    out = {"default_0_5": 0.5, "val_best_f1": best}
    for target in [0.10, 0.20, 0.30]:
        valid = np.flatnonzero(fpr <= target)
        out[f"val_fpr_le_{target:g}"] = float(rth[valid[np.argmax(tpr[valid])]]) if len(valid) else best
    return out


def score_pool(df, scores, mode):
    tmp = df[["source_file"]].copy()
    tmp["score"] = scores
    if mode == "raw":
        return scores
    if mode == "ma3":
        return tmp.groupby("source_file")["score"].transform(lambda s: s.rolling(3, min_periods=1).mean()).to_numpy()
    if mode == "ma5":
        return tmp.groupby("source_file")["score"].transform(lambda s: s.rolling(5, min_periods=1).mean()).to_numpy()
    if mode == "max3":
        return tmp.groupby("source_file")["score"].transform(lambda s: s.rolling(3, min_periods=1).max()).to_numpy()
    raise ValueError(mode)


def per_file(df, scores, th, model, policy):
    tmp = df[["source_file", "label", "rogue_frames"]].copy()
    tmp["score"] = scores
    rows = []
    for sf, g in tmp.groupby("source_file", sort=True):
        y, s = g.label.to_numpy(np.int8), g.score.to_numpy()
        row = {"model": model, "policy": policy, "source_file": sf, "windows": len(g), "positive_windows": int(y.sum()), "rogue_frames": int(g.rogue_frames.sum()), "auroc": safe_auc(y, s, "roc"), "pr_auc": safe_auc(y, s, "pr")}
        row.update(eval_at(y, s, th))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = [add_context(load_split(args.input_dir, s)) for s in ["train", "val", "test"]]
    feat = features(train)
    ytr, yv, yt = [df.label.to_numpy(np.int8) for df in [train, val, test]]
    spw = int((ytr == 0).sum()) / max(int(ytr.sum()), 1)
    models = {
        "rf_context": RandomForestClassifier(n_estimators=240, max_depth=10, min_samples_leaf=1, class_weight="balanced_subsample", random_state=args.random_state, n_jobs=-1),
        "xgb_context": XGBClassifier(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", max_depth=3, learning_rate=0.04, n_estimators=220, scale_pos_weight=spw, subsample=0.9, colsample_bytree=0.9, random_state=args.random_state),
        "mlp_context": Pipeline([("scaler", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=500, random_state=args.random_state, early_stopping=True))]),
        "logreg_context": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000, random_state=args.random_state))]),
    }
    metrics, per_files = [], []
    best = None
    for name, model in models.items():
        model.fit(train[feat], ytr)
        raw_val = model.predict_proba(val[feat])[:, 1]
        raw_test = model.predict_proba(test[feat])[:, 1]
        for pool in ["raw", "ma3", "ma5", "max3"]:
            vs, ts = score_pool(val, raw_val, pool), score_pool(test, raw_test, pool)
            aucs = {"val_auroc": safe_auc(yv, vs, "roc"), "val_pr_auc": safe_auc(yv, vs, "pr"), "test_auroc": safe_auc(yt, ts, "roc"), "test_pr_auc": safe_auc(yt, ts, "pr")}
            for pol, th in choose_thresholds(yv, vs).items():
                tm = eval_at(yt, ts, th)
                metrics.append({"model": name, "pooling": pool, "threshold_policy": pol, **aucs, **tm})
                if pol in {"val_best_f1", "val_fpr_le_0.2"}:
                    pf = per_file(test, ts, th, name, f"{pool}/{pol}")
                    per_files.append(pf)
                if best is None or tm["f1"] > best["f1"]:
                    best = {"model": name, "pooling": pool, "policy": pol, "scores": ts, **tm}

    # Normal-only anomaly detector.
    scaler = StandardScaler().fit(train.loc[train.label == 0, feat])
    iso = IsolationForest(n_estimators=300, contamination=0.33, random_state=args.random_state, n_jobs=-1)
    iso.fit(scaler.transform(train.loc[train.label == 0, feat]))
    vs = -iso.decision_function(scaler.transform(val[feat]))
    ts = -iso.decision_function(scaler.transform(test[feat]))
    for pol, th in choose_thresholds(yv, vs).items():
        tm = eval_at(yt, ts, th)
        metrics.append({"model": "isolation_forest_normal_only", "pooling": "raw", "threshold_policy": pol, "val_auroc": safe_auc(yv, vs, "roc"), "val_pr_auc": safe_auc(yv, vs, "pr"), "test_auroc": safe_auc(yt, ts, "roc"), "test_pr_auc": safe_auc(yt, ts, "pr"), **tm})
        if best is None or tm["f1"] > best["f1"]:
            best = {"model": "isolation_forest_normal_only", "pooling": "raw", "policy": pol, "scores": ts, **tm}

    mdf = pd.DataFrame(metrics).sort_values("f1", ascending=False)
    mdf.to_csv(args.results_dir / "alternative_metrics.csv", index=False)
    if per_files:
        pd.concat(per_files, ignore_index=True).to_csv(args.results_dir / "alternative_per_file_metrics.csv", index=False)
    if best:
        cm = confusion_matrix(yt, (best["scores"] >= best["threshold"]).astype(np.int8), labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{best['model']} {best['pooling']} {best['policy']}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(args.results_dir / "alternative_confusion_matrix.png", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    top = mdf.head(15).iloc[::-1]
    labels = top["model"] + "/" + top["pooling"] + "/" + top["threshold_policy"]
    ax.barh(labels, top["f1"], label="F1")
    ax.scatter(top["fpr"], labels, color="tab:red", label="FPR")
    ax.legend(); ax.set_title("Rogue AP alternative methods")
    fig.tight_layout()
    fig.savefig(args.results_dir / "alternative_comparison.png", dpi=180)
    plt.close(fig)
    lines = ["Rogue AP alternative methods summary", ""]
    for _, r in mdf.head(12).iterrows():
        lines.append(f"- {r.model}/{r.pooling}/{r.threshold_policy}: precision={r.precision:.4f}, recall={r.recall:.4f}, f1={r.f1:.4f}, fpr={r.fpr:.4f}, auroc={r.test_auroc:.4f}, pr_auc={r.test_pr_auc:.4f}")
    (args.results_dir / "short_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
