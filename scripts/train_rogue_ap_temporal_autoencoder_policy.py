#!/usr/bin/env python3
"""Train Rogue AP temporal models, normal-only autoencoder, and policy layer.

This experiment uses the event-centered tight_event dataset. Source/audit columns
and label-derived columns are never used as model features.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


AUDIT_COLS = {"source_file", "file_id", "event_id", "window_kind", "row_start", "row_end"}
LABEL_DERIVED = {"label", "rogue_frames", "normal_frames", "rogue_frame_ratio"}
TARGET_FPRS = [0.10, 0.15, 0.20, 0.30]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("processed/rogue_ap_event_windows/tight_event"))
    p.add_argument("--results-dir", type=Path, default=Path("results/rogue_ap_temporal_autoencoder_policy"))
    p.add_argument("--seq-len", type=int, default=5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def load_split(base: Path, split: str) -> pd.DataFrame:
    return (
        pd.read_parquet(base / split / "part-00000.parquet")
        .sort_values(["source_file", "row_start", "row_end", "event_id"])
        .reset_index(drop=True)
    )


def feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = AUDIT_COLS | LABEL_DERIVED
    return [c for c in df.columns if c not in blocked and pd.api.types.is_numeric_dtype(df[c])]


def safe_auc(y: np.ndarray, score: np.ndarray, kind: str) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score) if kind == "roc" else average_precision_score(y, score))


def metric_at(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (score >= threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / max(tn + fp, 1)),
        "auroc": safe_auc(y, score, "roc"),
        "pr_auc": safe_auc(y, score, "pr"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_thresholds(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    out = {"single_0_5": 0.5}
    p, r, th = precision_recall_curve(y, score)
    if len(th):
        f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
        out["single_best_f1"] = float(th[int(np.nanargmax(f1))])
    else:
        out["single_best_f1"] = 0.5
    fpr, tpr, rth = roc_curve(y, score)
    for target in TARGET_FPRS:
        valid = np.flatnonzero(fpr <= target)
        out[f"single_fpr_le_{target:.2f}"] = float(rth[valid[np.argmax(tpr[valid])]]) if len(valid) else 1.0
    return out


def smooth_by_file(df: pd.DataFrame, score: np.ndarray, mode: str) -> np.ndarray:
    tmp = df[["source_file"]].copy()
    tmp["score"] = score
    if mode == "raw":
        return score
    if mode == "ma3":
        return tmp.groupby("source_file", sort=False)["score"].transform(lambda s: s.rolling(3, min_periods=1).mean()).to_numpy()
    if mode == "ma5":
        return tmp.groupby("source_file", sort=False)["score"].transform(lambda s: s.rolling(5, min_periods=1).mean()).to_numpy()
    if mode == "max3":
        return tmp.groupby("source_file", sort=False)["score"].transform(lambda s: s.rolling(3, min_periods=1).max()).to_numpy()
    raise ValueError(mode)


def dual_threshold_predict(df: pd.DataFrame, score: np.ndarray, high: float, low: float) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.int8)
    tmp = df[["source_file"]].copy()
    tmp["score"] = score
    for _, idx in tmp.groupby("source_file", sort=False).groups.items():
        idx = np.asarray(list(idx))
        s = score[idx]
        ma3 = pd.Series(s).rolling(3, min_periods=1).mean().to_numpy()
        out[idx] = ((s >= high) | ((s >= low) & (ma3 >= high))).astype(np.int8)
    return out


def select_dual_threshold(y: np.ndarray, score: np.ndarray, target_fpr: float) -> tuple[float, float, dict[str, Any]]:
    candidates = np.quantile(score, np.linspace(0.05, 0.95, 46))
    best: tuple[float, float, dict[str, Any]] | None = None
    fake_df = pd.DataFrame({"source_file": ["validation"] * len(score)})
    for high in candidates:
        for low in candidates[candidates <= high]:
            pred = dual_threshold_predict(fake_df, score, float(high), float(low))
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            fpr = fp / max(fp + tn, 1)
            if fpr > target_fpr:
                continue
            precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
            row = {
                "threshold": float(high),
                "low_threshold": float(low),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fpr": float(fpr),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
            if best is None or (row["recall"], row["f1"]) > (best[2]["recall"], best[2]["f1"]):
                best = (float(high), float(low), row)
    if best is None:
        return 1.0, 1.0, metric_at(y, score, 1.0)
    return best


def make_sequences(df: pd.DataFrame, features: list[str], seq_len: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    xs, ys, meta = [], [], []
    for sf, g in df.sort_values(["source_file", "row_start", "event_id"]).groupby("source_file", sort=True):
        vals = g[features].to_numpy(float)
        labels = g.label.to_numpy(np.int8)
        records = g[["source_file", "event_id", "row_start", "row_end", "rogue_frames"]].reset_index(drop=True)
        for i in range(len(g)):
            start = max(0, i - seq_len + 1)
            seq = vals[start : i + 1]
            if len(seq) < seq_len:
                seq = np.vstack([np.repeat(seq[:1], seq_len - len(seq), axis=0), seq])
            xs.append(seq)
            ys.append(labels[i])
            meta.append(records.iloc[i].to_dict())
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), pd.DataFrame(meta)


class GRUDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


class TCNDetector(nn.Module):
    def __init__(self, input_dim: int, channels: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.LayerNorm(channels), nn.Linear(channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x.transpose(1, 2))).squeeze(-1)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        mid = max(8, input_dim * 2)
        bottleneck = max(4, input_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, mid),
            nn.ReLU(),
            nn.Linear(mid, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Scores:
    model: str
    val: np.ndarray
    test: np.ndarray


def train_temporal_model(
    kind: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    seq_len: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[Scores, pd.DataFrame]:
    torch.manual_seed(seed)
    xtr, ytr, _ = make_sequences(train, features, seq_len)
    xv, yv, _ = make_sequences(val, features, seq_len)
    xt, _yt, _ = make_sequences(test, features, seq_len)
    scaler = StandardScaler().fit(xtr.reshape(-1, xtr.shape[-1]))
    def scale(x: np.ndarray) -> np.ndarray:
        return scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
    xtr, xv, xt = scale(xtr), scale(xv), scale(xt)
    model = GRUDetector(len(features)) if kind == "gru" else TCNDetector(len(features))
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)), batch_size=batch_size, shuffle=True)
    pos_weight = torch.tensor([(len(ytr) - float(ytr.sum())) / max(float(ytr.sum()), 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    hist, best_state, best_pr, stale = [], None, -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            vs = torch.sigmoid(model(torch.from_numpy(xv))).numpy()
        pr_auc = safe_auc(yv.astype(np.int8), vs, "pr")
        hist.append({"model": kind, "epoch": epoch, "loss": float(np.mean(losses)), "val_pr_auc": pr_auc, "val_auroc": safe_auc(yv.astype(np.int8), vs, "roc")})
        if pr_auc > best_pr:
            best_pr = pr_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= 20 and stale >= 12:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return Scores(f"temporal_{kind}", torch.sigmoid(model(torch.from_numpy(xv))).numpy(), torch.sigmoid(model(torch.from_numpy(xt))).numpy()), pd.DataFrame(hist)


def train_autoencoder(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, features: list[str], epochs: int, batch_size: int, seed: int) -> tuple[Scores, pd.DataFrame]:
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(train.loc[train.label == 0, features])
    xtr = scaler.transform(train.loc[train.label == 0, features]).astype(np.float32)
    xv = scaler.transform(val[features]).astype(np.float32)
    xt = scaler.transform(test[features]).astype(np.float32)
    model = AutoEncoder(len(features))
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr)), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    hist = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for (xb,) in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_err = ((model(torch.from_numpy(xv)) - torch.from_numpy(xv)) ** 2).mean(dim=1).numpy()
            hist.append({"model": "autoencoder_normal_only", "epoch": epoch, "loss": float(np.mean(losses)), "val_pr_auc": safe_auc(val.label.to_numpy(np.int8), val_err, "pr"), "val_auroc": safe_auc(val.label.to_numpy(np.int8), val_err, "roc")})
    model.eval()
    with torch.no_grad():
        val_err = ((model(torch.from_numpy(xv)) - torch.from_numpy(xv)) ** 2).mean(dim=1).numpy()
        test_err = ((model(torch.from_numpy(xt)) - torch.from_numpy(xt)) ** 2).mean(dim=1).numpy()
    return Scores("autoencoder_normal_only", val_err, test_err), pd.DataFrame(hist)


def train_tabular_references(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int) -> list[Scores]:
    ytr = train.label.to_numpy(np.int8)
    spw = int((ytr == 0).sum()) / max(int(ytr.sum()), 1)
    models: dict[str, Any] = {
        "rf_reference": RandomForestClassifier(n_estimators=260, max_depth=10, min_samples_leaf=1, class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        "xgb_reference": XGBClassifier(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", max_depth=3, learning_rate=0.04, n_estimators=220, scale_pos_weight=spw, subsample=0.9, colsample_bytree=0.9, random_state=seed),
        "isolation_forest_normal_only": Pipeline([
            ("scaler", StandardScaler()),
            ("model", IsolationForest(n_estimators=320, contamination=0.33, random_state=seed, n_jobs=-1)),
        ]),
    }
    out: list[Scores] = []
    for name, model in models.items():
        if name == "isolation_forest_normal_only":
            model.fit(train.loc[train.label == 0, features])
            out.append(Scores(name, -model.decision_function(val[features]), -model.decision_function(test[features])))
        else:
            model.fit(train[features], ytr)
            out.append(Scores(name, model.predict_proba(val[features])[:, 1], model.predict_proba(test[features])[:, 1]))
    return out


def evaluate_scores(scores: list[Scores], val: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    yv = val.label.to_numpy(np.int8)
    yt = test.label.to_numpy(np.int8)
    metrics, thresholds, per_files = [], [], []
    for sc in scores:
        for smooth in ["raw", "ma3", "ma5", "max3"]:
            vs = smooth_by_file(val, sc.val, smooth)
            ts = smooth_by_file(test, sc.test, smooth)
            for policy, th in choose_thresholds(yv, vs).items():
                thresholds.append({"model": sc.model, "smoothing": smooth, "policy": policy, "split": "val", **metric_at(yv, vs, th)})
                row = {"model": sc.model, "smoothing": smooth, "policy": policy, "split": "test"}
                row.update(metric_at(yt, ts, th))
                metrics.append(row)
            for target in TARGET_FPRS:
                high, low, val_metric = select_dual_threshold(yv, vs, target)
                pred = dual_threshold_predict(test, ts, high, low)
                tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
                precision, recall, f1, _ = precision_recall_fscore_support(yt, pred, average="binary", zero_division=0)
                row = {
                    "model": sc.model,
                    "smoothing": smooth,
                    "policy": f"dual_threshold_target_fpr_{target:.2f}",
                    "split": "test",
                    "threshold": float(high),
                    "low_threshold": float(low),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "fpr": float(fp / max(tn + fp, 1)),
                    "auroc": safe_auc(yt, ts, "roc"),
                    "pr_auc": safe_auc(yt, ts, "pr"),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                    "val_precision": val_metric.get("precision"),
                    "val_recall": val_metric.get("recall"),
                    "val_f1": val_metric.get("f1"),
                    "val_fpr": val_metric.get("fpr"),
                }
                metrics.append(row)
                thresholds.append({"model": sc.model, "smoothing": smooth, "policy": row["policy"], "split": "val", **val_metric})

        # Per-file for the best-F1 single threshold and best constrained policy after all smoothing choices.
        model_rows = [r for r in metrics if r["model"] == sc.model and r["split"] == "test"]
        for best_row in sorted(model_rows, key=lambda r: r["f1"], reverse=True)[:2]:
            ts = smooth_by_file(test, sc.test, best_row["smoothing"])
            if str(best_row["policy"]).startswith("dual_threshold"):
                pred = dual_threshold_predict(test, ts, float(best_row["threshold"]), float(best_row.get("low_threshold", best_row["threshold"])))
                score_for_file = ts
            else:
                pred = (ts >= float(best_row["threshold"])).astype(np.int8)
                score_for_file = ts
            tmp = test[["source_file", "event_id", "label", "rogue_frames"]].copy()
            tmp["score"] = score_for_file
            tmp["pred"] = pred
            for sf, g in tmp.groupby("source_file", sort=True):
                y = g.label.to_numpy(np.int8)
                p = g.pred.to_numpy(np.int8)
                tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
                precision, recall, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
                per_files.append({
                    "model": sc.model,
                    "smoothing": best_row["smoothing"],
                    "policy": best_row["policy"],
                    "source_file": sf,
                    "windows": len(g),
                    "positive_windows": int(y.sum()),
                    "rogue_frames": int(g.rogue_frames.sum()),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "fpr": float(fp / max(tn + fp, 1)),
                    "auroc": safe_auc(y, g.score.to_numpy(float), "roc"),
                    "pr_auc": safe_auc(y, g.score.to_numpy(float), "pr"),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                })
    return (
        pd.DataFrame(metrics).sort_values(["f1", "pr_auc"], ascending=False),
        pd.DataFrame(thresholds),
        pd.DataFrame(per_files),
    )


def save_outputs(results_dir: Path, metrics: pd.DataFrame, thresholds: pd.DataFrame, per_file: pd.DataFrame, histories: list[pd.DataFrame], test: pd.DataFrame, scores: list[Scores]) -> None:
    metrics.to_csv(results_dir / "metrics_by_model_policy.csv", index=False)
    thresholds.to_csv(results_dir / "threshold_analysis.csv", index=False)
    per_file.to_csv(results_dir / "per_file_metrics.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(results_dir / "training_curves.csv", index=False)

    best = metrics.iloc[0]
    best_score = next(s.test for s in scores if s.model == best.model)
    best_score = smooth_by_file(test, best_score, best.smoothing)
    if str(best.policy).startswith("dual_threshold"):
        pred = dual_threshold_predict(test, best_score, float(best.threshold), float(best.low_threshold))
    else:
        pred = (best_score >= float(best.threshold)).astype(np.int8)
    cm = confusion_matrix(test.label.to_numpy(np.int8), pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["pred_normal", "pred_rogue"])
    ax.set_yticks([0, 1], ["true_normal", "true_rogue"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_title(f"{best.model}/{best.policy}")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    top = metrics.head(15).iloc[::-1]
    labels = top["model"] + "/" + top["smoothing"] + "/" + top["policy"]
    ax.barh(labels, top["f1"], label="F1")
    ax.scatter(top["fpr"], labels, color="tab:red", label="FPR")
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(results_dir / "model_policy_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    y = test.label.to_numpy(np.int8)
    ax.hist(best_score[y == 0], bins=30, alpha=0.65, label="Normal")
    ax.hist(best_score[y == 1], bins=30, alpha=0.65, label="RogueAP")
    ax.axvline(float(best.threshold), color="black", linestyle="--", label="high threshold")
    if "low_threshold" in best and not pd.isna(best.get("low_threshold", np.nan)):
        ax.axvline(float(best.low_threshold), color="gray", linestyle=":", label="low threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "score_distribution.png", dpi=180)
    plt.close(fig)

    summary = [
        "Rogue AP temporal model + autoencoder + policy layer summary",
        "",
        "Safety:",
        "- No random row split.",
        "- source_file/file_id/event_id/window_kind/row_start/row_end are audit-only.",
        "- rogue_frames/normal_frames/rogue_frame_ratio are label-derived and excluded.",
        "- Thresholds and policies are selected on validation only; test is held out.",
        "",
        "Top test results:",
    ]
    for _, r in metrics.head(12).iterrows():
        summary.append(
            f"- {r.model}/{r.smoothing}/{r.policy}: precision={r.precision:.4f}, "
            f"recall={r.recall:.4f}, f1={r.f1:.4f}, fpr={r.fpr:.4f}, "
            f"auroc={r.auroc:.4f}, pr_auc={r.pr_auc:.4f}"
        )
    summary += [
        "",
        "Interpretation guide:",
        "- Temporal GRU/TCN is useful only if it beats RF/XGBoost on held-out files.",
        "- Autoencoder is a normal-only anomaly baseline; use it for robustness/new-attack discussion if competitive.",
        "- Policy layer is production-friendly when it reduces FPR without collapsing recall.",
    ]
    (results_dir / "short_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    torch.set_num_threads(2)

    train, val, test = [load_split(args.input_dir, s) for s in ["train", "val", "test"]]
    features = feature_columns(train)
    data_audit = pd.DataFrame([
        {"split": "train", "rows": len(train), "positive": int(train.label.sum()), "negative": int((train.label == 0).sum()), "files": train.source_file.nunique()},
        {"split": "val", "rows": len(val), "positive": int(val.label.sum()), "negative": int((val.label == 0).sum()), "files": val.source_file.nunique()},
        {"split": "test", "rows": len(test), "positive": int(test.label.sum()), "negative": int((test.label == 0).sum()), "files": test.source_file.nunique()},
    ])
    data_audit.to_csv(args.results_dir / "data_audit.csv", index=False)
    (args.results_dir / "feature_columns.txt").write_text("\n".join(features), encoding="utf-8")

    scores = train_tabular_references(train, val, test, features, args.random_state)
    histories: list[pd.DataFrame] = []
    for kind in ["gru", "tcn"]:
        sc, hist = train_temporal_model(kind, train, val, test, features, args.seq_len, args.epochs, args.batch_size, args.random_state)
        scores.append(sc)
        histories.append(hist)
    ae_score, ae_hist = train_autoencoder(train, val, test, features, args.epochs, args.batch_size, args.random_state)
    scores.append(ae_score)
    histories.append(ae_hist)

    metrics, thresholds, per_file = evaluate_scores(scores, val, test)
    save_outputs(args.results_dir, metrics, thresholds, per_file, histories, test, scores)


if __name__ == "__main__":
    main()
