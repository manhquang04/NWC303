#!/usr/bin/env python3
"""Real-trace-calibrated synthetic Rogue AP experiment with sequence baselines.

Synthetic data is used only for training augmentation. Validation and test remain
the real held-out file splits from tight_event.
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


AUDIT_COLS = {"source_file", "file_id", "event_id", "window_kind", "row_start", "row_end"}
LABEL_DERIVED_COLS = {"label", "rogue_frames", "normal_frames", "rogue_frame_ratio"}
TARGET_FPRS = [0.10, 0.15, 0.20]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("processed/rogue_ap_event_windows/tight_event"))
    p.add_argument("--results-dir", type=Path, default=Path("results/rogue_ap_synthetic_sequence"))
    p.add_argument("--synthetic-multiplier", type=float, default=1.5)
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
    blocked = AUDIT_COLS | LABEL_DERIVED_COLS
    return [c for c in df.columns if c not in blocked and pd.api.types.is_numeric_dtype(df[c])]


def safe_auc(y: np.ndarray, score: np.ndarray, kind: str) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    if kind == "roc":
        return float(roc_auc_score(y, score))
    return float(average_precision_score(y, score))


def metrics_at(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (score >= threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / max(fp + tn, 1)),
        "auroc": safe_auc(y, score, "roc"),
        "pr_auc": safe_auc(y, score, "pr"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_thresholds(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    out = {"default_0_5": 0.5}
    p, r, th = precision_recall_curve(y, score)
    if len(th):
        f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
        out["val_best_f1"] = float(th[int(np.nanargmax(f1))])
    else:
        out["val_best_f1"] = 0.5
    fpr, tpr, rth = roc_curve(y, score)
    for target in TARGET_FPRS:
        valid = np.flatnonzero(fpr <= target)
        out[f"val_fpr_le_{target:.2f}"] = float(rth[valid[np.argmax(tpr[valid])]]) if len(valid) else 1.0
    return out


def clip_feature_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in features:
        if col == "n_frames":
            out[col] = out[col].clip(lower=1).round().astype(int)
        elif col.startswith("ratio_"):
            out[col] = out[col].clip(0, 1)
    return out


def synthetic_rows(train: pd.DataFrame, features: list[str], multiplier: float, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    by_label = {label: train[train.label == label].reset_index(drop=True) for label in [0, 1]}
    std = train[features].std(numeric_only=True).replace(0, 1e-6)

    def sample(label: int, n: int, noise_scale: float, kind: str) -> pd.DataFrame:
        base = by_label[label].sample(n=n, replace=True, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
        x = base[features].to_numpy(float)
        noise = rng.normal(0, std[features].to_numpy(float) * noise_scale, size=x.shape)
        part = pd.DataFrame(x + noise, columns=features)
        part["label"] = label
        part["synthetic_kind"] = kind
        return part

    real_normal = int((train.label == 0).sum())
    real_rogue = int((train.label == 1).sum())
    n_easy = max(1, int(real_normal * multiplier * 0.55))
    n_rogue = max(1, int(real_rogue * multiplier * 1.10))
    rows.append(sample(0, n_easy, 0.035, "easy_negative"))
    rows.append(sample(1, n_rogue, 0.04, "synthetic_positive"))
    counts["easy_negative"] = n_easy
    counts["synthetic_positive"] = n_rogue

    n_hard = max(1, int(real_normal * multiplier * 0.35))
    neg = by_label[0].sample(n=n_hard, replace=True, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    pos = by_label[1].sample(n=n_hard, replace=True, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    alpha = rng.uniform(0.62, 0.90, size=(n_hard, 1))
    x = alpha * neg[features].to_numpy(float) + (1 - alpha) * pos[features].to_numpy(float)
    x += rng.normal(0, std[features].to_numpy(float) * 0.025, size=x.shape)
    hard = pd.DataFrame(x, columns=features)
    hard["label"] = 0
    hard["synthetic_kind"] = "hard_negative"
    rows.append(hard)
    counts["hard_negative"] = n_hard

    n_border = max(1, int(real_normal * multiplier * 0.25))
    neg = by_label[0].sample(n=n_border, replace=True, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    pos = by_label[1].sample(n=n_border, replace=True, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    alpha = rng.uniform(0.50, 0.72, size=(n_border, 1))
    x = alpha * neg[features].to_numpy(float) + (1 - alpha) * pos[features].to_numpy(float)
    x += rng.normal(0, std[features].to_numpy(float) * 0.02, size=x.shape)
    border = pd.DataFrame(x, columns=features)
    border["label"] = 0
    border["synthetic_kind"] = "borderline_negative"
    rows.append(border)
    counts["borderline_negative"] = n_border

    syn = pd.concat(rows, ignore_index=True)
    syn = clip_feature_frame(syn, features)
    profile = {
        "seed": seed,
        "synthetic_multiplier": multiplier,
        "counts": counts,
        "real_train_counts": {str(k): int(v) for k, v in train.label.value_counts().sort_index().to_dict().items()},
        "feature_columns": features,
        "safety": "Synthetic rows are train-only; real val/test are untouched.",
    }
    return syn, profile


def train_tabular_models(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    synthetic: pd.DataFrame,
    features: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_aug = pd.concat([train[features + ["label"]], synthetic[features + ["label"]]], ignore_index=True)
    ytr = train.label.to_numpy(np.int8)
    yaug = train_aug.label.to_numpy(np.int8)
    spw = int((yaug == 0).sum()) / max(int((yaug == 1).sum()), 1)
    models: dict[str, Any] = {
        "logreg_real_only": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1500, random_state=seed)),
        ]),
        "rf_real_only": RandomForestClassifier(n_estimators=280, max_depth=10, min_samples_leaf=1, class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        "rf_real_plus_synthetic": RandomForestClassifier(n_estimators=320, max_depth=11, min_samples_leaf=1, class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        "xgb_real_plus_synthetic": XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            max_depth=3,
            learning_rate=0.035,
            n_estimators=260,
            scale_pos_weight=spw,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
        ),
    }
    fit_data = {
        "logreg_real_only": (train[features], ytr),
        "rf_real_only": (train[features], ytr),
        "rf_real_plus_synthetic": (train_aug[features], yaug),
        "xgb_real_plus_synthetic": (train_aug[features], yaug),
    }
    metrics, thresholds, per_file = [], [], []
    trained = {}
    for name, model in models.items():
        xfit, yfit = fit_data[name]
        model.fit(xfit, yfit)
        trained[name] = model
        val_score = model.predict_proba(val[features])[:, 1]
        test_score = model.predict_proba(test[features])[:, 1]
        for policy, th in choose_thresholds(val.label.to_numpy(np.int8), val_score).items():
            thresholds.append({"model": name, "policy": policy, "split": "val", **metrics_at(val.label.to_numpy(np.int8), val_score, th)})
            row = {"model": name, "policy": policy, "split": "test"}
            row.update(metrics_at(test.label.to_numpy(np.int8), test_score, th))
            metrics.append(row)
            if policy in {"val_best_f1", "val_fpr_le_0.20"}:
                tmp = test[["source_file", "event_id", "label", "rogue_frames"]].copy()
                tmp["score"] = test_score
                tmp["pred"] = (test_score >= th).astype(np.int8)
                for sf, g in tmp.groupby("source_file", sort=True):
                    prow = {
                        "model": name,
                        "policy": policy,
                        "source_file": sf,
                        "windows": len(g),
                        "positive_windows": int(g.label.sum()),
                        "rogue_frames": int(g.rogue_frames.sum()),
                    }
                    prow.update(metrics_at(g.label.to_numpy(np.int8), g.score.to_numpy(float), th))
                    per_file.append(prow)
    return pd.DataFrame(metrics), pd.DataFrame(thresholds), pd.DataFrame(per_file), trained


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
                pad = np.repeat(seq[:1], seq_len - len(seq), axis=0)
                seq = np.vstack([pad, seq])
            xs.append(seq)
            ys.append(labels[i])
            meta.append(records.iloc[i].to_dict())
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), pd.DataFrame(meta)


def synthetic_sequences(synthetic: pd.DataFrame, features: list[str], seq_len: int, n_seq: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 99)
    pos = synthetic[synthetic.label == 1].reset_index(drop=True)
    neg = synthetic[synthetic.label == 0].reset_index(drop=True)
    xs, ys = [], []
    for _ in range(n_seq):
        label = int(rng.random() < 0.45)
        steps = []
        if label == 1:
            burst = int(rng.integers(1, min(3, seq_len) + 1))
            for i in range(seq_len):
                src = pos if i >= seq_len - burst else (pos if rng.random() < 0.25 else neg)
                steps.append(src.iloc[int(rng.integers(0, len(src)))][features].to_numpy(float))
        else:
            for _i in range(seq_len):
                src = neg
                steps.append(src.iloc[int(rng.integers(0, len(src)))][features].to_numpy(float))
        xs.append(np.vstack(steps))
        ys.append(label)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


class GRUDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


@dataclass
class SequenceResult:
    metrics: pd.DataFrame
    thresholds: pd.DataFrame
    per_file: pd.DataFrame
    history: pd.DataFrame
    val_score: np.ndarray
    test_score: np.ndarray
    model_name: str


def train_gru(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    synthetic: pd.DataFrame,
    features: list[str],
    seq_len: int,
    epochs: int,
    batch_size: int,
    seed: int,
    use_synthetic: bool,
) -> SequenceResult:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xtr, ytr, _ = make_sequences(train, features, seq_len)
    xv, yv, _ = make_sequences(val, features, seq_len)
    xt, yt, meta_t = make_sequences(test, features, seq_len)
    model_name = "gru_real_plus_synthetic" if use_synthetic else "gru_real_only"
    if use_synthetic:
        xs, ys = synthetic_sequences(synthetic, features, seq_len, max(len(xtr), 1), seed)
        xtr = np.concatenate([xtr, xs], axis=0)
        ytr = np.concatenate([ytr, ys], axis=0)
        order = rng.permutation(len(xtr))
        xtr, ytr = xtr[order], ytr[order]

    scaler = StandardScaler().fit(xtr.reshape(-1, xtr.shape[-1]))
    def scale(x: np.ndarray) -> np.ndarray:
        return scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)

    xtr, xv, xt = scale(xtr), scale(xv), scale(xt)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )
    model = GRUDetector(len(features))
    pos_weight = torch.tensor([(len(ytr) - float(ytr.sum())) / max(float(ytr.sum()), 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state, best_pr, stale = None, -1.0, 0
    hist = []
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
        pr = safe_auc(yv.astype(np.int8), vs, "pr")
        hist.append({"model": model_name, "epoch": epoch, "loss": float(np.mean(losses)), "val_pr_auc": pr, "val_auroc": safe_auc(yv.astype(np.int8), vs, "roc")})
        if pr > best_pr:
            best_pr = pr
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= 20 and stale >= 12:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_score = torch.sigmoid(model(torch.from_numpy(xv))).numpy()
        test_score = torch.sigmoid(model(torch.from_numpy(xt))).numpy()

    metrics, thresholds, per_file = [], [], []
    for policy, th in choose_thresholds(yv.astype(np.int8), val_score).items():
        thresholds.append({"model": model_name, "policy": policy, "split": "val", **metrics_at(yv.astype(np.int8), val_score, th)})
        row = {"model": model_name, "policy": policy, "split": "test"}
        row.update(metrics_at(yt.astype(np.int8), test_score, th))
        metrics.append(row)
        if policy in {"val_best_f1", "val_fpr_le_0.20"}:
            tmp = meta_t.copy()
            tmp["label"] = yt.astype(np.int8)
            tmp["score"] = test_score
            for sf, g in tmp.groupby("source_file", sort=True):
                prow = {
                    "model": model_name,
                    "policy": policy,
                    "source_file": sf,
                    "windows": len(g),
                    "positive_windows": int(g.label.sum()),
                    "rogue_frames": int(g.rogue_frames.sum()),
                }
                prow.update(metrics_at(g.label.to_numpy(np.int8), g.score.to_numpy(float), th))
                per_file.append(prow)
    return SequenceResult(pd.DataFrame(metrics), pd.DataFrame(thresholds), pd.DataFrame(per_file), pd.DataFrame(hist), val_score, test_score, model_name)


def save_plots(results_dir: Path, metrics: pd.DataFrame, threshold_rows: pd.DataFrame, y_test: np.ndarray, best_score: np.ndarray, best_th: float) -> None:
    best_cm = confusion_matrix(y_test, (best_score >= best_th).astype(np.int8), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(best_cm, cmap="Blues")
    ax.set_xticks([0, 1], ["pred_normal", "pred_rogue"])
    ax.set_yticks([0, 1], ["true_normal", "true_rogue"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(best_cm[i, j]), ha="center", va="center")
    ax.set_title("Best synthetic/sequence test confusion matrix")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    top = metrics.sort_values("f1", ascending=False).head(12).iloc[::-1]
    labels = top["model"] + "/" + top["policy"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, top["f1"], label="F1")
    ax.scatter(top["fpr"], labels, color="tab:red", label="FPR")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.set_title("Synthetic + sequence method comparison")
    fig.tight_layout()
    fig.savefig(results_dir / "synthetic_sequence_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(best_score[y_test == 0], bins=30, alpha=0.65, label="Normal")
    ax.hist(best_score[y_test == 1], bins=30, alpha=0.65, label="RogueAP")
    ax.axvline(best_th, color="black", linestyle="--", label=f"threshold={best_th:.3f}")
    ax.legend()
    ax.set_title("Best model test score distribution")
    fig.tight_layout()
    fig.savefig(results_dir / "score_distribution.png", dpi=180)
    plt.close(fig)

    if not threshold_rows.empty:
        pivot = threshold_rows[threshold_rows.split == "val"].copy()
        fig, ax = plt.subplots(figsize=(9, 5))
        for metric in ["f1", "fpr", "recall"]:
            ax.plot(np.arange(len(pivot)), pivot[metric], marker="o", label=metric)
        ax.set_xticks(np.arange(len(pivot)), pivot["model"] + "/" + pivot["policy"], rotation=90)
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / "threshold_analysis.png", dpi=180)
        plt.close(fig)


def write_distribution_report(train: pd.DataFrame, synthetic: pd.DataFrame, features: list[str], out: Path) -> None:
    rows = []
    for label in [0, 1]:
        real = train[train.label == label]
        syn = synthetic[synthetic.label == label]
        for f in features:
            rows.append({
                "label": label,
                "feature": f,
                "real_mean": float(real[f].mean()),
                "synthetic_mean": float(syn[f].mean()),
                "mean_abs_diff": float(abs(real[f].mean() - syn[f].mean())),
                "real_std": float(real[f].std()),
                "synthetic_std": float(syn[f].std()),
            })
    pd.DataFrame(rows).sort_values(["label", "mean_abs_diff"], ascending=[True, False]).to_csv(out, index=False)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.random_state)
    torch.set_num_threads(2)

    train, val, test = [load_split(args.input_dir, split) for split in ["train", "val", "test"]]
    features = feature_columns(train)
    synthetic, profile = synthetic_rows(train, features, args.synthetic_multiplier, args.random_state)
    synthetic.to_parquet(args.results_dir / "synthetic_train_aug.parquet", index=False)
    (args.results_dir / "synthetic_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    write_distribution_report(train, synthetic, features, args.results_dir / "synthetic_distribution_report.csv")

    tab_metrics, tab_thresholds, tab_per_file, trained = train_tabular_models(train, val, test, synthetic, features, args.random_state)
    seq_real = train_gru(train, val, test, synthetic, features, args.seq_len, args.epochs, args.batch_size, args.random_state, False)
    seq_syn = train_gru(train, val, test, synthetic, features, args.seq_len, args.epochs, args.batch_size, args.random_state, True)

    metrics = pd.concat([tab_metrics, seq_real.metrics, seq_syn.metrics], ignore_index=True).sort_values("f1", ascending=False)
    thresholds = pd.concat([tab_thresholds, seq_real.thresholds, seq_syn.thresholds], ignore_index=True)
    per_file = pd.concat([tab_per_file, seq_real.per_file, seq_syn.per_file], ignore_index=True)
    history = pd.concat([seq_real.history, seq_syn.history], ignore_index=True)
    metrics.to_csv(args.results_dir / "metrics_by_model.csv", index=False)
    thresholds.to_csv(args.results_dir / "threshold_analysis.csv", index=False)
    per_file.to_csv(args.results_dir / "per_file_metrics.csv", index=False)
    history.to_csv(args.results_dir / "sequence_training_curves.csv", index=False)

    rows = []
    for name, model in trained.items():
        raw = model.named_steps["model"] if isinstance(model, Pipeline) else model
        vals = None
        if hasattr(raw, "feature_importances_"):
            vals = raw.feature_importances_
        elif hasattr(raw, "coef_"):
            vals = np.abs(raw.coef_[0])
        if vals is not None:
            for f, v in sorted(zip(features, vals), key=lambda x: abs(x[1]), reverse=True)[:30]:
                rows.append({"model": name, "feature": f, "importance": float(v)})
    pd.DataFrame(rows).to_csv(args.results_dir / "feature_importance.csv", index=False)

    best = metrics.iloc[0].to_dict()
    best_model, best_policy = best["model"], best["policy"]
    best_th = float(best["threshold"])
    y_test = test.label.to_numpy(np.int8)
    if best_model == seq_real.model_name:
        best_score = seq_real.test_score
    elif best_model == seq_syn.model_name:
        best_score = seq_syn.test_score
    else:
        best_score = trained[best_model].predict_proba(test[features])[:, 1]
    save_plots(args.results_dir, metrics, thresholds, y_test, best_score, best_th)

    summary = [
        "Rogue AP Digital Twin + Real-Trace-Calibrated Synthetic Sequence Experiment",
        "",
        "Safety design:",
        "- Synthetic samples are generated from train split only.",
        "- Validation/test remain real held-out files from tight_event.",
        "- source_file/file_id/event_id/window_kind/row_start/row_end are audit-only.",
        "- label-derived fields rogue_frames/normal_frames/rogue_frame_ratio are excluded from features.",
        "",
        f"Real split rows: train={len(train)}, val={len(val)}, test={len(test)}.",
        f"Synthetic train augmentation rows: {len(synthetic)} with counts {synthetic.label.value_counts().sort_index().to_dict()}.",
        "",
        "Top test configurations:",
    ]
    for _, r in metrics.head(8).iterrows():
        summary.append(
            f"- {r.model}/{r.policy}: precision={r.precision:.4f}, recall={r.recall:.4f}, "
            f"f1={r.f1:.4f}, fpr={r.fpr:.4f}, auroc={r.auroc:.4f}, pr_auc={r.pr_auc:.4f}"
        )
    summary += [
        "",
        "Interpretation:",
        "- If synthetic-augmented or GRU models do not beat the clean RF baseline on real test files, treat synthetic as robustness support, not as the main evidence.",
        "- The thesis-safe claim is digital-twin/synthetic augmentation was evaluated against held-out real traces, not that synthetic traffic is equivalent to real traffic.",
    ]
    (args.results_dir / "short_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
