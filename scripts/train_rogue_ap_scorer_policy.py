#!/usr/bin/env python3
"""Two-stage Rogue AP scorer + constrained decision policy.

Stage 1 trains supervised scorers on tight_event aggregate features.
Stage 2 chooses validation-only constrained policies that maximize recall under
target FPR budgets. Metadata is used only for audit grouping.
"""

from __future__ import annotations

import argparse
import json
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
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


FEATURES = [
    "n_frames", "ratio_retry", "ratio_protected", "ratio_moredata", "ratio_pwrmgt", "ratio_order",
    "ratio_frag", "ratio_ds_0x00000000", "ratio_ds_0x00000001", "ratio_ds_0x00000002",
    "ratio_ds_0x00000003", "ratio_type_0", "ratio_type_1", "ratio_type_2",
    "ratio_llc_present", "ratio_ip_present", "ratio_tcp_present",
]
AUDIT_COLS = ["source_file", "file_id", "event_id", "window_kind", "row_start", "row_end", "rogue_frames", "label"]
TARGET_FPRS = [0.10, 0.20]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("processed/rogue_ap_event_windows/tight_event"))
    p.add_argument("--event-results-dir", type=Path, default=Path("results/rogue_ap_event_windows"))
    p.add_argument("--old-rl-results-dir", type=Path, default=Path("results/rogue_ap_rl_dqn"))
    p.add_argument("--hardened-dqn-results-dir", type=Path, default=Path("results/rogue_ap_dqn_hard_negative"))
    p.add_argument("--results-dir", type=Path, default=Path("results/rogue_ap_scorer_policy"))
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def load_split(base: Path, split: str) -> pd.DataFrame:
    df = pd.read_parquet(base / split / "part-00000.parquet")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{split} missing features: {missing}")
    return df.sort_values(["source_file", "row_start", "row_end", "event_id"]).reset_index(drop=True)


def safe_auc(y: np.ndarray, s: np.ndarray, kind: str) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s) if kind == "roc" else average_precision_score(y, s))


def metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "fpr": float(fp / max(fp + tn, 1)), "auroc": safe_auc(y, score, "roc"),
        "pr_auc": safe_auc(y, score, "pr"), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def train_scorers(train: pd.DataFrame, val: pd.DataFrame, random_state: int) -> dict[str, Any]:
    xtr, ytr = train[FEATURES], train["label"].to_numpy(np.int8)
    xv, yv = val[FEATURES], val["label"].to_numpy(np.int8)
    spw = int((ytr == 0).sum()) / max(int(ytr.sum()), 1)
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000, random_state=random_state)),
        ]),
        "random_forest": RandomForestClassifier(n_estimators=180, max_depth=8, min_samples_leaf=1, class_weight="balanced_subsample", random_state=random_state, n_jobs=-1),
        "xgboost": XGBClassifier(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", max_depth=3, learning_rate=0.05, n_estimators=140, scale_pos_weight=spw, random_state=random_state),
    }
    out = {}
    for name, model in models.items():
        model.fit(xtr, ytr)
        score = model.predict_proba(xv)[:, 1]
        out[name] = {"model": model, "val_pr_auc": safe_auc(yv, score, "pr"), "val_auroc": safe_auc(yv, score, "roc")}
    return out


def policy_threshold(y: np.ndarray, score: np.ndarray, target_fpr: float) -> tuple[float, dict[str, Any]]:
    fpr, tpr, th = roc_curve(y, score)
    valid = np.flatnonzero(fpr <= target_fpr)
    if len(valid):
        idx = valid[np.argmax(tpr[valid])]
        threshold = float(th[idx])
    else:
        threshold = 1.0
    pred = (score >= threshold).astype(np.int8)
    return threshold, metrics(y, pred, score)


def balanced_policy(y: np.ndarray, score: np.ndarray, target_fpr: float) -> tuple[float, dict[str, Any]]:
    return policy_threshold(y, score, target_fpr)


def conservative_policy(y: np.ndarray, score: np.ndarray, target_fpr: float) -> tuple[float, dict[str, Any]]:
    return policy_threshold(y, score, target_fpr * 0.75)


def make_score_frame(df: pd.DataFrame, score: np.ndarray, model_name: str) -> pd.DataFrame:
    out = df[AUDIT_COLS].copy()
    out["scorer"] = model_name
    out["score"] = score
    out["score_ma3"] = out.groupby("source_file")["score"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    out["score_delta"] = out.groupby("source_file")["score"].diff().fillna(0)
    return out


def grouped(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, sort=True):
        kt = key if isinstance(key, tuple) else (key,)
        y, p, s = g.label.to_numpy(np.int8), g.pred.to_numpy(np.int8), g.score.to_numpy(float)
        row = dict(zip(group_cols, kt))
        row.update({"windows": len(g), "positive_windows": int(y.sum()), "rogue_frames": int(g.rogue_frames.sum())})
        row.update(metrics(y, p, s))
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_rows(args: argparse.Namespace, policy_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sup_path = args.event_results_dir / "metrics_by_variant.csv"
    if sup_path.exists():
        sup = pd.read_csv(sup_path)
        sup = sup[(sup.variant == "tight_event") & (sup.split == "test") & (sup.threshold_policy == "val_best_f1")]
        for _, r in sup.iterrows():
            rows.append({"system": "supervised_baseline", "model": r.model, "policy": "val_best_f1", "target_fpr": np.nan, "precision": r.precision, "recall": r.recall, "f1": r.f1, "fpr": r.fpr, "auroc": r.test_auroc, "pr_auc": r.test_pr_auc})
    for label, path, fname in [
        ("rl_baseline", args.old_rl_results_dir, "rl_metrics.csv"),
        ("dqn_hardened", args.hardened_dqn_results_dir, "dqn_metrics.csv"),
    ]:
        f = path / fname
        if f.exists():
            m = pd.read_csv(f)
            m = m[m.split == "test"]
            for _, r in m.iterrows():
                rows.append({"system": label, "model": r.get("agent", "dqn"), "policy": r.reward_variant, "target_fpr": np.nan, "precision": r.precision, "recall": r.recall, "f1": r.f1, "fpr": r.fpr, "auroc": r.auroc, "pr_auc": r.pr_auc})
    for _, r in policy_metrics[policy_metrics.split == "test"].iterrows():
        rows.append({"system": "scorer_policy", "model": r.scorer, "policy": r.policy, "target_fpr": r.target_fpr, "precision": r.precision, "recall": r.recall, "f1": r.f1, "fpr": r.fpr, "auroc": r.auroc, "pr_auc": r.pr_auc})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = [load_split(args.input_dir, s) for s in ["train", "val", "test"]]
    ytr, yv, yt = [d.label.to_numpy(np.int8) for d in [train, val, test]]

    scorers = train_scorers(train, val, args.random_state)
    score_frames = []
    model_metrics, threshold_rows, policy_rows, per_file_rows, per_event_rows = [], [], [], [], []
    best_cm = None
    best_f1 = -1.0

    for name, item in scorers.items():
        model = item["model"]
        scores = {
            "train": model.predict_proba(train[FEATURES])[:, 1],
            "val": model.predict_proba(val[FEATURES])[:, 1],
            "test": model.predict_proba(test[FEATURES])[:, 1],
        }
        for split, df, y in [("train", train, ytr), ("val", val, yv), ("test", test, yt)]:
            model_metrics.append({"scorer": name, "split": split, **metrics(y, (scores[split] >= 0.5).astype(np.int8), scores[split])})
            sf = make_score_frame(df, scores[split], name)
            sf["split"] = split
            score_frames.append(sf)

        # Validation-only constrained policies.
        for target in TARGET_FPRS:
            for policy_name, fn in [("balanced_policy", balanced_policy), ("conservative_policy", conservative_policy)]:
                th, val_m = fn(yv, scores["val"], target)
                threshold_rows.append({"scorer": name, "policy": policy_name, "target_fpr": target, "threshold": th, **{f"val_{k}": v for k, v in val_m.items()}})
                for split, df, y in [("val", val, yv), ("test", test, yt)]:
                    sc = scores[split]
                    pred = (sc >= th).astype(np.int8)
                    m = metrics(y, pred, sc)
                    policy_rows.append({"scorer": name, "policy": policy_name, "target_fpr": target, "split": split, "threshold": th, "utility": float((pred * y * 4 - pred * (1 - y) * 2 - (1 - pred) * y * 6 + (1 - pred) * (1 - y)).mean()), **m})
                    if split == "test":
                        pf = df[AUDIT_COLS].copy()
                        pf["scorer"] = name
                        pf["policy"] = policy_name
                        pf["target_fpr"] = target
                        pf["score"] = sc
                        pf["pred"] = pred
                        per_file_rows.append(grouped(pf, ["scorer", "policy", "target_fpr", "source_file"]))
                        pe = pf[["scorer", "policy", "target_fpr", "source_file", "event_id", "window_kind", "label", "rogue_frames", "score", "pred"]].copy()
                        per_event_rows.append(pe)
                        if m["f1"] > best_f1:
                            best_f1 = m["f1"]
                            best_cm = (name, policy_name, target, confusion_matrix(y, pred, labels=[0, 1]))

    pd.DataFrame(model_metrics).to_csv(args.results_dir / "metrics_by_model.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(args.results_dir / "policy_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(args.results_dir / "threshold_analysis.csv", index=False)
    pd.concat(per_file_rows, ignore_index=True).to_csv(args.results_dir / "per_file_metrics.csv", index=False)
    pd.concat(per_event_rows, ignore_index=True).to_csv(args.results_dir / "per_event_metrics.csv", index=False)
    pd.concat(score_frames, ignore_index=True).to_csv(args.results_dir / "scorer_scores.csv", index=False)
    cmp = comparison_rows(args, pd.DataFrame(policy_rows))
    cmp.to_csv(args.results_dir / "comparison_with_supervised.csv", index=False)

    best_scorer = max(scorers.items(), key=lambda kv: kv[1]["val_pr_auc"])
    (args.results_dir / "scorer_training_report.txt").write_text(
        "\n".join([
            "Supervised scorer training report",
            f"Features: {FEATURES}",
            "Preprocessing is fit on train only. Logistic Regression uses StandardScaler in-pipeline; tree models use raw aggregate features.",
            f"Validation PR-AUC by scorer: {json.dumps({k: {'val_pr_auc': v['val_pr_auc'], 'val_auroc': v['val_auroc']} for k, v in scorers.items()}, indent=2)}",
            f"Selected scorer by validation PR-AUC: {best_scorer[0]}",
        ]),
        encoding="utf-8",
    )
    (args.results_dir / "policy_design_report.txt").write_text(
        "Policy layer uses supervised score plus audit-only sequential context columns in exported score files. "
        "The actual decision is constrained thresholding: balanced_policy maximizes validation recall under target FPR; "
        "conservative_policy uses 75% of the target FPR budget. source_file/file_id/event_id are not scorer features.\n",
        encoding="utf-8",
    )
    (args.results_dir / "constrained_objective_report.txt").write_text(
        "Objective: choose threshold on validation only to maximize recall subject to FPR <= target. "
        "Targets tested: 0.10 and 0.20. Test is evaluated once with the chosen validation thresholds.\n",
        encoding="utf-8",
    )

    if best_cm is not None:
        name, pol, target, cm = best_cm
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name}/{pol}/FPR {target}")
        ax.set_xticks([0, 1], ["pred normal", "pred rogue"])
        ax.set_yticks([0, 1], ["true normal", "true rogue"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(args.results_dir / "confusion_matrix.png", dpi=180)
        plt.close(fig)

    best_policy = pd.DataFrame(policy_rows)
    best_test = best_policy[best_policy.split == "test"].sort_values("f1", ascending=False).iloc[0]
    best_cmp = cmp.sort_values("f1", ascending=False).head(8)
    lines = [
        "Scorer + constrained policy summary",
        "",
        f"Best policy test: {best_test.scorer}/{best_test.policy}/target_fpr={best_test.target_fpr}, precision={best_test.precision:.4f}, recall={best_test.recall:.4f}, f1={best_test.f1:.4f}, fpr={best_test.fpr:.4f}, auroc={best_test.auroc:.4f}, pr_auc={best_test.pr_auc:.4f}.",
        "",
        "Top comparison rows:",
        best_cmp.to_string(index=False),
        "",
        "This stage uses constrained supervised decisioning, not DQN, because the action can be represented as a validation-tuned decision threshold under an FPR budget.",
    ]
    (args.results_dir / "short_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
