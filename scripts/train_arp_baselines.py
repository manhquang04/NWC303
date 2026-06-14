#!/usr/bin/env python3
"""Supervised baselines for clean ARP spoofing dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("processed/arp_spoofing_clean"))
    p.add_argument("--results-dir", type=Path, default=Path("results/arp_baselines"))
    p.add_argument("--train-sample", type=int, default=600_000)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def load_split(base: Path, split: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted((base / split).glob("part-*.parquet"))], ignore_index=True)


def safe_auc(y, s, kind):
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, s) if kind == "roc" else average_precision_score(y, s))


def eval_at(y, s, th):
    pred = (s >= th).astype(np.int8)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(th), "precision": float(pr), "recall": float(rc), "f1": float(f1), "fpr": float(fp / max(fp + tn, 1)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def thresholds(y, s):
    p, r, th = precision_recall_curve(y, s)
    if len(th) == 0:
        return {"default_0_5": 0.5, "val_best_f1": 0.5, "val_fpr_le_0.05": 0.5, "val_fpr_le_0.10": 0.5}
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    best = float(th[int(np.nanargmax(f1))])
    fpr, tpr, rth = roc_curve(y, s)
    out = {"default_0_5": 0.5, "val_best_f1": best}
    for lim in [0.05, 0.10]:
        valid = np.flatnonzero(fpr <= lim)
        out[f"val_fpr_le_{lim:g}"] = float(rth[valid[np.argmax(tpr[valid])]]) if len(valid) else best
    return out


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    parts = []
    per_class = max(1, n // 2)
    for label, g in df.groupby("label"):
        parts.append(g.sample(n=min(len(g), per_class), random_state=seed))
    out = pd.concat(parts)
    if len(out) < n:
        rest = df.drop(out.index)
        out = pd.concat([out, rest.sample(n=min(len(rest), n - len(out)), random_state=seed)])
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = [load_split(args.input_dir, s) for s in ["train", "val", "test"]]
    features = [c for c in train.columns if c not in {"label", "row_id"}]
    train_fit = stratified_sample(train, args.train_sample, args.random_state)
    xtr, ytr = train_fit[features], train_fit["label"].to_numpy(np.int8)
    xv, yv = val[features], val["label"].to_numpy(np.int8)
    xt, yt = test[features], test["label"].to_numpy(np.int8)
    spw = int((ytr == 0).sum()) / max(int(ytr.sum()), 1)
    models = {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(class_weight="balanced", solver="saga", max_iter=500, n_jobs=-1, random_state=args.random_state))]),
        "random_forest": RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=args.random_state),
        "xgboost": XGBClassifier(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", max_depth=4, learning_rate=0.06, n_estimators=160, subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw, random_state=args.random_state),
    }
    metrics, th_rows, importances = [], [], []
    best = None
    for name, model in models.items():
        model.fit(xtr, ytr)
        vs, ts = model.predict_proba(xv)[:, 1], model.predict_proba(xt)[:, 1]
        aucs = {"val_auroc": safe_auc(yv, vs, "roc"), "val_pr_auc": safe_auc(yv, vs, "pr"), "test_auroc": safe_auc(yt, ts, "roc"), "test_pr_auc": safe_auc(yt, ts, "pr")}
        for pol, th in thresholds(yv, vs).items():
            vm, tm = eval_at(yv, vs, th), eval_at(yt, ts, th)
            th_rows.append({"model": name, "policy": pol, "threshold": th, **{f"val_{k}": v for k, v in vm.items()}, **{f"test_{k}": v for k, v in tm.items()}})
            metrics.append({"model": name, "split": "val", "threshold_policy": pol, **aucs, **vm})
            metrics.append({"model": name, "split": "test", "threshold_policy": pol, **aucs, **tm})
            if pol == "val_best_f1" and (best is None or tm["f1"] > best["f1"]):
                best = {"model": name, **tm, "scores": ts}
        raw = model.named_steps["model"] if isinstance(model, Pipeline) else model
        vals = getattr(raw, "feature_importances_", None)
        if vals is None and hasattr(raw, "coef_"):
            vals = np.abs(raw.coef_[0])
        if vals is not None:
            for f, v in sorted(zip(features, vals), key=lambda x: abs(x[1]), reverse=True)[:30]:
                importances.append({"model": name, "feature": f, "importance": float(v)})

    pd.DataFrame(metrics).to_csv(args.results_dir / "metrics.csv", index=False)
    pd.DataFrame(th_rows).to_csv(args.results_dir / "threshold_analysis.csv", index=False)
    pd.DataFrame(importances).to_csv(args.results_dir / "feature_importance.csv", index=False)
    checks = []
    for split, df in [("train", train), ("val", val), ("test", test)]:
        vc = df.label.value_counts().to_dict()
        checks.append({"split": split, "rows": len(df), "cols": len(df.columns), "normal": int(vc.get(0, 0)), "arp_spoofing": int(vc.get(1, 0)), "positive_rate": float(df.label.mean()), "missing": int(df.isna().sum().sum()), "duplicates": int(df.duplicated(subset=features + ["label"]).sum())})
    pd.DataFrame(checks).to_csv(args.results_dir / "data_checks.csv", index=False)
    if best:
        cm = confusion_matrix(yt, (best["scores"] >= best["threshold"]).astype(np.int8), labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_title(best["model"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(args.results_dir / "confusion_matrix.png", dpi=180)
        plt.close(fig)
    lines = ["ARP spoofing supervised baseline summary", ""]
    met = pd.DataFrame(metrics)
    for _, r in met[(met.split == "test") & (met.threshold_policy == "val_best_f1")].sort_values("f1", ascending=False).iterrows():
        lines.append(f"- {r.model}: precision={r.precision:.4f}, recall={r.recall:.4f}, f1={r.f1:.4f}, fpr={r.fpr:.4f}, auroc={r.test_auroc:.4f}, pr_auc={r.test_pr_auc:.4f}")
    lines.append("")
    lines.append("Split is contiguous block-based, not random. Val/test are attack-phase dominant; report this limitation.")
    (args.results_dir / "short_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
