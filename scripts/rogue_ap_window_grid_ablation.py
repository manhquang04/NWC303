#!/usr/bin/env python3
"""Window-size and window-label-threshold ablation for Rogue AP aggregates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier


LABEL_MAP = {"Normal": 0, "RogueAP": 1}
TRAIN_IDS = set(range(0, 19)) | set(range(24, 32))
VAL_IDS = set(range(19, 23)) | set(range(32, 35))
TEST_IDS = set(range(35, 40))
INSPECT_IDS = {23}
WINDOW_SIZES = [100, 250, 500, 1000]
LABEL_THRESHOLDS = [0.01, 0.05, 0.10, 0.25, 0.50]
USECOLS = [
    "Label",
    "wlan.fc.retry",
    "wlan.fc.protected",
    "wlan.fc.moredata",
    "wlan.fc.pwrmgt",
    "wlan.fc.order",
    "wlan.fc.frag",
    "wlan.fc.ds",
]
CONSERVATIVE_FEATURES = [
    "n_frames",
    "ratio_retry",
    "ratio_protected",
    "ratio_moredata",
    "ratio_pwrmgt",
    "ratio_order",
    "ratio_frag",
    "ratio_ds_0x00000000",
    "ratio_ds_0x00000001",
    "ratio_ds_0x00000002",
    "ratio_ds_0x00000003",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("dataset/4.Rogue_AP"))
    p.add_argument("--processed-dir", type=Path, default=Path("processed/rogue_ap_window_grid"))
    p.add_argument("--results-dir", type=Path, default=Path("results/rogue_ap_window_grid_ablation"))
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--fpr-limit", type=float, default=0.01)
    return p.parse_args()


def file_id(path: Path) -> int:
    m = re.fullmatch(r"RogueAP_(\d+)\.csv", path.name)
    if not m:
        raise ValueError(path.name)
    return int(m.group(1))


def split_for(fid: int) -> str:
    if fid in TRAIN_IDS:
        return "train"
    if fid in VAL_IDS:
        return "val"
    if fid in TEST_IDS:
        return "test"
    if fid in INSPECT_IDS:
        return "inspect"
    return "unused"


def norm(v: Any) -> str:
    if pd.isna(v):
        return "MISSING"
    return str(v).strip()


def binary_ratio(df: pd.DataFrame, col: str) -> float:
    if col not in df:
        return 0.0
    s = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return float((s != 0).mean())


def add_ds_ratios(out: dict[str, Any], df: pd.DataFrame) -> None:
    vals = df["wlan.fc.ds"].map(norm) if "wlan.fc.ds" in df else pd.Series(["MISSING"] * len(df))
    counts = vals.value_counts(normalize=True)
    for k in ["0x00000000", "0x00000001", "0x00000002", "0x00000003"]:
        out[f"ratio_ds_{k}"] = float(counts.get(k, 0.0))


def aggregate_window(df: pd.DataFrame, source_file: str, fid: int, window_id: int) -> dict[str, Any]:
    labels = df["Label"].astype("string").str.strip().map(LABEL_MAP)
    rogue_frames = int(labels.sum())
    out = {
        "source_file": source_file,
        "file_id": fid,
        "window_id": window_id,
        "n_frames": int(len(df)),
        "rogue_frames": rogue_frames,
        "normal_frames": int((labels == 0).sum()),
        "rogue_frame_ratio": float(rogue_frames / max(len(df), 1)),
    }
    for raw, dst in [
        ("wlan.fc.retry", "ratio_retry"),
        ("wlan.fc.protected", "ratio_protected"),
        ("wlan.fc.moredata", "ratio_moredata"),
        ("wlan.fc.pwrmgt", "ratio_pwrmgt"),
        ("wlan.fc.order", "ratio_order"),
        ("wlan.fc.frag", "ratio_frag"),
    ]:
        out[dst] = binary_ratio(df, raw)
    add_ds_ratios(out, df)
    return out


def build_base_windows(input_dir: Path, processed_dir: Path) -> dict[int, dict[str, pd.DataFrame]]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[int, dict[str, pd.DataFrame]] = {}
    files = sorted(input_dir.glob("RogueAP_*.csv"), key=file_id)
    for ws in WINDOW_SIZES:
        split_rows = {s: [] for s in ["train", "val", "test", "inspect"]}
        for path in files:
            fid = file_id(path)
            split = split_for(fid)
            if split == "unused":
                continue
            header = pd.read_csv(path, nrows=0).columns.tolist()
            usecols = [c for c in USECOLS if c in header]
            df = pd.read_csv(path, usecols=usecols, low_memory=False)
            for window_id, start in enumerate(range(0, len(df), ws)):
                win = df.iloc[start : start + ws]
                if len(win):
                    split_rows[split].append(aggregate_window(win, path.name, fid, window_id))
        cache[ws] = {}
        for split, rows in split_rows.items():
            out = pd.DataFrame(rows).fillna(0)
            d = processed_dir / f"window_{ws}" / split
            d.mkdir(parents=True, exist_ok=True)
            out.to_parquet(d / "part-00000.parquet", index=False)
            cache[ws][split] = out
    return cache


def label_for_threshold(df: pd.DataFrame, threshold: float) -> np.ndarray:
    return (df["rogue_frame_ratio"].to_numpy() >= threshold).astype(np.int8)


def eval_at(y: np.ndarray, scores: np.ndarray, th: float) -> dict[str, Any]:
    pred = (scores >= th).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(th), "precision": float(precision), "recall": float(recall), "f1": float(f1), "fpr": float(fp / max(fp + tn, 1)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def choose_thresholds(y: np.ndarray, scores: np.ndarray, fpr_limit: float) -> dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"default_0_5": 0.5, "val_best_f1": 0.5, f"val_fpr_le_{fpr_limit:g}": 0.5}
    p, r, th = precision_recall_curve(y, scores)
    if len(th) == 0:
        return {"default_0_5": 0.5, "val_best_f1": 0.5, f"val_fpr_le_{fpr_limit:g}": 0.5}
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    best = float(th[int(np.nanargmax(f1))])
    fpr, tpr, rth = roc_curve(y, scores)
    valid = np.flatnonzero(fpr <= fpr_limit)
    constrained = float(rth[valid[np.argmax(tpr[valid])]]) if len(valid) else best
    return {"default_0_5": 0.5, "val_best_f1": best, f"val_fpr_le_{fpr_limit:g}": constrained}


def safe_auc(y: np.ndarray, scores: np.ndarray, kind: str) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, scores) if kind == "roc" else average_precision_score(y, scores))


def train_models(xtr, ytr, xval, yval, random_state: int):
    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000, random_state=random_state).fit(xtr, ytr),
        "random_forest": RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=1, class_weight="balanced_subsample", random_state=random_state, n_jobs=-1).fit(xtr, ytr),
    }
    spw = int((ytr == 0).sum()) / max(int(ytr.sum()), 1)
    best, best_pr = None, -1.0
    for params in [{"max_depth": 2, "learning_rate": 0.08, "n_estimators": 80}, {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 120}]:
        model = XGBClassifier(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", scale_pos_weight=spw, random_state=random_state, **params)
        model.fit(xtr, ytr, verbose=False)
        pr = safe_auc(yval, model.predict_proba(xval)[:, 1], "pr")
        if pr > best_pr:
            best, best_pr = model, pr
    models["xgboost"] = best
    return models


def split_stats(df: pd.DataFrame, labels: np.ndarray, split: str, ws: int, lt: float) -> dict[str, Any]:
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    pos_ratio = df.loc[labels == 1, "rogue_frame_ratio"].mean() if pos else 0.0
    return {
        "window_size": ws,
        "label_threshold": lt,
        "split": split,
        "windows": len(df),
        "positive_windows": pos,
        "negative_windows": neg,
        "positive_window_pct": 100 * pos / max(len(df), 1),
        "rogue_frames": int(df["rogue_frames"].sum()),
        "total_frames": int(df["n_frames"].sum()),
        "rogue_frame_ratio_split": float(df["rogue_frames"].sum() / max(df["n_frames"].sum(), 1)),
        "mean_rogue_frame_ratio_positive_windows": float(pos_ratio),
    }


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    cache = build_base_windows(args.input_dir, args.processed_dir)
    config_rows, metrics_rows, threshold_rows, per_file_rows, per_window_rows = [], [], [], [], []
    best_for_plot = None
    best_f1 = -1.0
    for ws in WINDOW_SIZES:
        data = cache[ws]
        x = {split: data[split][CONSERVATIVE_FEATURES] for split in ["train", "val", "test"]}
        for lt in LABEL_THRESHOLDS:
            y = {split: label_for_threshold(data[split], lt) for split in ["train", "val", "test", "inspect"]}
            for split in ["train", "val", "test", "inspect"]:
                config_rows.append(split_stats(data[split], y[split], split, ws, lt))
            if y["train"].sum() == 0 or y["val"].sum() == 0 or y["test"].sum() == 0:
                continue
            models = train_models(x["train"], y["train"], x["val"], y["val"], args.random_state)
            for model_name, model in models.items():
                vs = model.predict_proba(x["val"])[:, 1]
                ts = model.predict_proba(x["test"])[:, 1]
                aucs = {"val_auroc": safe_auc(y["val"], vs, "roc"), "val_pr_auc": safe_auc(y["val"], vs, "pr"), "test_auroc": safe_auc(y["test"], ts, "roc"), "test_pr_auc": safe_auc(y["test"], ts, "pr")}
                for policy, th in choose_thresholds(y["val"], vs, args.fpr_limit).items():
                    vm, tm = eval_at(y["val"], vs, th), eval_at(y["test"], ts, th)
                    threshold_rows.append({"window_size": ws, "label_threshold": lt, "model": model_name, "policy": policy, "threshold": th, **{f"val_{k}": v for k, v in vm.items() if k != "threshold"}, **{f"test_{k}": v for k, v in tm.items() if k != "threshold"}})
                    for split, met in [("val", vm), ("test", tm)]:
                        metrics_rows.append({"window_size": ws, "label_threshold": lt, "model": model_name, "split": split, "threshold_policy": policy, **aucs, **met})
                    if policy == "val_best_f1":
                        if tm["f1"] > best_f1:
                            best_f1 = tm["f1"]
                            best_for_plot = (ws, lt, model_name, ts, th, y["test"])
                        temp = data["test"][["source_file", "file_id", "window_id", "rogue_frames", "n_frames"]].copy()
                        temp["label"] = y["test"]
                        temp["score"] = ts
                        for sf, g in temp.groupby("source_file", sort=True):
                            gy, gs = g["label"].to_numpy(dtype=np.int8), g["score"].to_numpy()
                            per_file_rows.append({"window_size": ws, "label_threshold": lt, "model": model_name, "source_file": sf, "windows": len(g), "positive_windows": int(gy.sum()), "rogue_frames": int(g.rogue_frames.sum()), "auroc": safe_auc(gy, gs, "roc"), "pr_auc": safe_auc(gy, gs, "pr"), **eval_at(gy, gs, th)})
                        for _, g in temp.iterrows():
                            per_window_rows.append({"window_size": ws, "label_threshold": lt, "model": model_name, "source_file": g.source_file, "window_id": int(g.window_id), "label": int(g.label), "rogue_frames": int(g.rogue_frames), "n_frames": int(g.n_frames), "score": float(g.score), "pred": int(g.score >= th)})
    cfg = pd.DataFrame(config_rows)
    met = pd.DataFrame(metrics_rows)
    ths = pd.DataFrame(threshold_rows)
    pf = pd.DataFrame(per_file_rows)
    pw = pd.DataFrame(per_window_rows)
    cfg.to_csv(args.results_dir / "window_config_map.csv", index=False)
    met.to_csv(args.results_dir / "metrics_by_config.csv", index=False)
    ths.to_csv(args.results_dir / "threshold_analysis.csv", index=False)
    pf.to_csv(args.results_dir / "per_file_metrics.csv", index=False)
    pw.to_csv(args.results_dir / "per_window_metrics.csv", index=False)
    (args.results_dir / "window_size_ablation_report.txt").write_text(cfg.groupby(["window_size", "split"])[["windows", "positive_windows", "negative_windows", "rogue_frame_ratio_split"]].first().to_string(), encoding="utf-8")
    (args.results_dir / "label_threshold_ablation_report.txt").write_text(cfg.groupby(["label_threshold", "split"])[["windows", "positive_windows", "negative_windows", "mean_rogue_frame_ratio_positive_windows"]].mean().to_string(), encoding="utf-8")
    if best_for_plot:
        ws, lt, model_name, scores, th, ytest = best_for_plot
        cm = confusion_matrix(ytest, (scores >= th).astype(np.int8), labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"best: ws={ws}, lt={lt}, {model_name}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(args.results_dir / "confusion_matrix.png", dpi=180)
        plt.close(fig)
    # Pick configs with non-degenerate test balance and best F1.
    test_cfg = cfg[cfg.split == "test"].copy()
    fair = test_cfg[(test_cfg.positive_window_pct >= 20) & (test_cfg.positive_window_pct <= 80)]
    selected = met[(met.split == "test") & (met.threshold_policy == "val_best_f1")].sort_values("f1", ascending=False).head(10)
    lines = [
        "Window grid ablation summary",
        "",
        "Best val-selected-F1 test configs:",
        selected[["window_size", "label_threshold", "model", "precision", "recall", "f1", "fpr", "test_auroc", "test_pr_auc"]].to_string(index=False),
        "",
        "Fair-ish test balance configs (20%-80% positive windows):",
        fair[["window_size", "label_threshold", "windows", "positive_windows", "negative_windows", "positive_window_pct", "rogue_frame_ratio_split"]].to_string(index=False) if len(fair) else "None",
        "",
        "Interpretation: use configs with enough negative test windows; high F1 with very high positive-window share is misleading.",
    ]
    (args.results_dir / "short_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
