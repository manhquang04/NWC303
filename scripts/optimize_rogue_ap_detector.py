#!/usr/bin/env python3
"""Optimize the Rogue AP tight_event detector with supervised scorers and policies."""

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
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


BASE_FEATURES = [
    "n_frames", "ratio_retry", "ratio_protected", "ratio_moredata", "ratio_pwrmgt", "ratio_order",
    "ratio_frag", "ratio_ds_0x00000000", "ratio_ds_0x00000001", "ratio_ds_0x00000002",
    "ratio_ds_0x00000003", "ratio_type_0", "ratio_type_1", "ratio_type_2",
    "ratio_llc_present", "ratio_ip_present", "ratio_tcp_present",
]
AUDIT_COLS = ["source_file", "file_id", "event_id", "window_kind", "row_start", "row_end", "rogue_frames", "label"]
TARGET_FPRS = [0.10, 0.15, 0.20]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("processed/rogue_ap_event_windows/tight_event"))
    p.add_argument("--event-results-dir", type=Path, default=Path("results/rogue_ap_event_windows"))
    p.add_argument("--scorer-policy-results-dir", type=Path, default=Path("results/rogue_ap_scorer_policy"))
    p.add_argument("--hardened-dqn-results-dir", type=Path, default=Path("results/rogue_ap_dqn_hard_negative"))
    p.add_argument("--results-dir", type=Path, default=Path("results/rogue_ap_detector_optimized"))
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def load_split(base: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(base / split / "part-00000.parquet").sort_values(["source_file", "row_start", "row_end", "event_id"]).reset_index(drop=True)


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in BASE_FEATURES:
        grp = out.groupby("source_file")[col]
        out[f"{col}_ma3"] = grp.transform(lambda s: s.rolling(3, min_periods=1).mean())
        out[f"{col}_delta1"] = grp.diff().fillna(0)
    out["position_frac"] = out.groupby("source_file").cumcount() / out.groupby("source_file")["source_file"].transform("size").sub(1).clip(lower=1)
    out["prev_score_placeholder"] = 0.0
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    label_derived = ["rogue_frame_ratio", "rogue_frames", "normal_frames"]
    forbidden = set(AUDIT_COLS + label_derived + ["source_file", "file_id", "event_id", "window_kind", "label"])
    return [c for c in df.columns if c not in forbidden and pd.api.types.is_numeric_dtype(df[c])]


def safe_auc(y: np.ndarray, s: np.ndarray, kind: str) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s) if kind == "roc" else average_precision_score(y, s))


def metric_row(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "fpr": float(fp / max(fp + tn, 1)), "auroc": safe_auc(y, score, "roc"),
        "pr_auc": safe_auc(y, score, "pr"), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def mine_weights(train: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    y = train.label.to_numpy(np.int8)
    weights = np.ones(len(train), dtype=float)
    scaler = StandardScaler().fit(train[features])
    pos = train[y == 1]
    neg = train[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return weights, {"warning": "single-class train"}
    sim = cosine_similarity(scaler.transform(neg[features]), scaler.transform(pos[features])).max(axis=1)
    distances = []
    for _, r in neg.iterrows():
        p = pos[pos.source_file == r.source_file]
        if len(p) == 0:
            distances.append(np.inf)
        else:
            distances.append(float(np.minimum(abs(p.row_start - r.row_end), abs(r.row_start - p.row_end)).clip(lower=0).min()))
    neg_idx = neg.index.to_numpy()
    finite = np.asarray([d for d in distances if np.isfinite(d)])
    near_cut = float(np.percentile(finite, 35)) if len(finite) else 0.0
    sim_hi = float(np.percentile(sim, 85))
    sim_mid = float(np.percentile(sim, 65))
    groups = np.full(len(neg), "easy_negative", dtype=object)
    distances_arr = np.asarray(distances)
    groups[distances_arr <= near_cut] = "hard_negative"
    groups[(sim >= sim_hi) | ((sim >= sim_mid) & (distances_arr <= near_cut * 2))] = "borderline_negative"
    # Recall-friendly: positives > hard negatives > borderline > easy, but no overbearing duplication.
    weights[y == 1] = 2.0
    for name, w in [("easy_negative", 1.0), ("hard_negative", 1.35), ("borderline_negative", 1.15)]:
        weights[neg_idx[groups == name]] = w
    return weights, {
        "groups": {k: int(v) for k, v in pd.Series(groups).value_counts().to_dict().items()},
        "positive_windows": int(y.sum()),
        "near_distance_cutoff_rows": near_cut,
        "similarity_borderline_cutoff": sim_hi,
        "weighting": {"positive": 2.0, "easy_negative": 1.0, "hard_negative": 1.35, "borderline_negative": 1.15},
    }


def train_models(train: pd.DataFrame, val: pd.DataFrame, features: list[str], sample_weight: np.ndarray, seed: int) -> dict[str, Any]:
    xtr, ytr = train[features], train.label.to_numpy(np.int8)
    xv, yv = val[features], val.label.to_numpy(np.int8)
    spw = int((ytr == 0).sum()) / max(int(ytr.sum()), 1)
    candidates: dict[str, Any] = {
        "logistic_regression_balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1500, random_state=seed)),
        ]),
        "random_forest_balanced": RandomForestClassifier(n_estimators=220, max_depth=8, min_samples_leaf=1, class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        "random_forest_weighted": RandomForestClassifier(n_estimators=260, max_depth=10, min_samples_leaf=1, random_state=seed, n_jobs=-1),
        "xgboost_balanced": XGBClassifier(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", max_depth=3, learning_rate=0.045, n_estimators=180, scale_pos_weight=spw, subsample=0.9, colsample_bytree=0.9, random_state=seed),
        "xgboost_recall_weighted": XGBClassifier(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", max_depth=2, learning_rate=0.04, n_estimators=220, scale_pos_weight=spw * 1.35, subsample=0.95, colsample_bytree=0.9, random_state=seed),
    }
    try:
        from lightgbm import LGBMClassifier
        candidates["lightgbm_balanced"] = LGBMClassifier(n_estimators=220, learning_rate=0.04, num_leaves=15, class_weight="balanced", random_state=seed, verbose=-1)
    except Exception:
        pass
    out = {}
    for name, model in candidates.items():
        if name == "logistic_regression_balanced":
            model.fit(xtr, ytr)
        elif name.endswith("weighted"):
            model.fit(xtr, ytr, sample_weight=sample_weight)
        else:
            model.fit(xtr, ytr)
        score = model.predict_proba(xv)[:, 1]
        out[name] = {"model": model, "val_pr_auc": safe_auc(yv, score, "pr"), "val_auroc": safe_auc(yv, score, "roc")}
    return out


def threshold_for_target(y: np.ndarray, score: np.ndarray, target: float, mode: str) -> tuple[float, dict[str, Any]]:
    budget = target if mode == "recall_friendly" else target * 0.75
    fpr, tpr, th = roc_curve(y, score)
    valid = np.flatnonzero(fpr <= budget)
    threshold = float(th[valid[np.argmax(tpr[valid])]]) if len(valid) else 1.0
    pred = (score >= threshold).astype(np.int8)
    return threshold, metric_row(y, pred, score)


def dual_threshold_policy(df: pd.DataFrame, score: np.ndarray, high: float, low: float) -> np.ndarray:
    tmp = df[["source_file"]].copy()
    tmp["score"] = score
    preds = np.zeros(len(tmp), dtype=np.int8)
    for _, idx in tmp.groupby("source_file", sort=False).groups.items():
        idx = np.asarray(list(idx))
        s = score[idx]
        ma = pd.Series(s).rolling(3, min_periods=1).mean().to_numpy()
        preds[idx] = ((s >= high) | ((s >= low) & (ma >= high))).astype(np.int8)
    return preds


def grouped(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(cols, sort=True):
        kt = key if isinstance(key, tuple) else (key,)
        y, p, s = g.label.to_numpy(np.int8), g.pred.to_numpy(np.int8), g.score.to_numpy(float)
        row = dict(zip(cols, kt))
        row.update({"windows": len(g), "positive_windows": int(y.sum()), "rogue_frames": int(g.rogue_frames.sum())})
        row.update(metric_row(y, p, s))
        rows.append(row)
    return pd.DataFrame(rows)


def feature_importance(models: dict[str, Any], features: list[str]) -> pd.DataFrame:
    rows = []
    for name, item in models.items():
        model = item["model"]
        raw_model = model.named_steps["model"] if isinstance(model, Pipeline) else model
        vals = None
        if hasattr(raw_model, "feature_importances_"):
            vals = raw_model.feature_importances_
        elif hasattr(raw_model, "coef_"):
            vals = np.abs(raw_model.coef_[0])
        if vals is not None:
            for f, v in sorted(zip(features, vals), key=lambda x: abs(x[1]), reverse=True)[:30]:
                rows.append({"model": name, "feature": f, "importance": float(v)})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train0, val0, test0 = [load_split(args.input_dir, s) for s in ["train", "val", "test"]]
    train, val, test = [add_context_features(d) for d in [train0, val0, test0]]
    features = feature_columns(train)
    sample_weight, hard_report = mine_weights(train, features)
    models = train_models(train, val, features, sample_weight, args.random_state)

    yv, yt = val.label.to_numpy(np.int8), test.label.to_numpy(np.int8)
    metrics_rows, policy_rows, threshold_rows, per_file_rows, per_event_rows = [], [], [], [], []
    best = None
    for name, item in models.items():
        model = item["model"]
        scores = {split: model.predict_proba(df[features])[:, 1] for split, df in [("train", train), ("val", val), ("test", test)]}
        for split, df in [("train", train), ("val", val), ("test", test)]:
            y = df.label.to_numpy(np.int8)
            metrics_rows.append({"model": name, "split": split, **metric_row(y, (scores[split] >= 0.5).astype(np.int8), scores[split])})
        for target in TARGET_FPRS:
            for mode in ["recall_friendly", "conservative"]:
                th, vm = threshold_for_target(yv, scores["val"], target, mode)
                threshold_rows.append({"model": name, "policy": mode, "target_fpr": target, "threshold": th, **{f"val_{k}": v for k, v in vm.items()}})
                for split, df, y in [("val", val, yv), ("test", test, yt)]:
                    pred = (scores[split] >= th).astype(np.int8)
                    row = {"model": name, "policy": mode, "target_fpr": target, "split": split, "threshold": th, **metric_row(y, pred, scores[split])}
                    policy_rows.append(row)
                    if split == "test":
                        audit = df[AUDIT_COLS].copy()
                        audit["model"], audit["policy"], audit["target_fpr"] = name, mode, target
                        audit["score"], audit["pred"] = scores[split], pred
                        per_file_rows.append(grouped(audit, ["model", "policy", "target_fpr", "source_file"]))
                        per_event_rows.append(audit[["model", "policy", "target_fpr", "source_file", "event_id", "window_kind", "label", "rogue_frames", "score", "pred"]])
                        if best is None or row["f1"] > best["f1"]:
                            best = {**row, "cm": confusion_matrix(y, pred, labels=[0, 1])}
            # Dual threshold: use recall-friendly high threshold and lower support threshold.
            high, _ = threshold_for_target(yv, scores["val"], target, "recall_friendly")
            low = max(0.0, high * 0.85)
            val_pred = dual_threshold_policy(val, scores["val"], high, low)
            test_pred = dual_threshold_policy(test, scores["test"], high, low)
            for split, df, y, pred in [("val", val, yv, val_pred), ("test", test, yt, test_pred)]:
                row = {"model": name, "policy": "dual_threshold_ma3", "target_fpr": target, "split": split, "threshold": high, "low_threshold": low, **metric_row(y, pred, scores[split])}
                policy_rows.append(row)
                if split == "test":
                    audit = df[AUDIT_COLS].copy()
                    audit["model"], audit["policy"], audit["target_fpr"] = name, "dual_threshold_ma3", target
                    audit["score"], audit["pred"] = scores[split], pred
                    per_file_rows.append(grouped(audit, ["model", "policy", "target_fpr", "source_file"]))
                    per_event_rows.append(audit[["model", "policy", "target_fpr", "source_file", "event_id", "window_kind", "label", "rogue_frames", "score", "pred"]])
                    if best is None or row["f1"] > best["f1"]:
                        best = {**row, "cm": confusion_matrix(y, pred, labels=[0, 1])}

    metrics_df = pd.DataFrame(metrics_rows)
    policy_df = pd.DataFrame(policy_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    metrics_df.to_csv(args.results_dir / "metrics_by_model.csv", index=False)
    policy_df.to_csv(args.results_dir / "policy_metrics.csv", index=False)
    threshold_df.to_csv(args.results_dir / "threshold_analysis.csv", index=False)
    pd.concat(per_file_rows, ignore_index=True).to_csv(args.results_dir / "per_file_metrics.csv", index=False)
    pd.concat(per_event_rows, ignore_index=True).to_csv(args.results_dir / "per_event_metrics.csv", index=False)
    feature_importance(models, features).to_csv(args.results_dir / "feature_importance.csv", index=False)

    comp = []
    sup_path = args.event_results_dir / "metrics_by_variant.csv"
    if sup_path.exists():
        sup = pd.read_csv(sup_path)
        sup = sup[(sup.variant == "tight_event") & (sup.split == "test") & (sup.threshold_policy == "val_best_f1")]
        for _, r in sup.iterrows():
            comp.append({"system": "supervised_baseline", "model": r.model, "policy": "val_best_f1", "target_fpr": np.nan, "precision": r.precision, "recall": r.recall, "f1": r.f1, "fpr": r.fpr, "auroc": r.test_auroc, "pr_auc": r.test_pr_auc})
    sp_path = args.scorer_policy_results_dir / "policy_metrics.csv"
    if sp_path.exists():
        sp = pd.read_csv(sp_path)
        sp = sp[(sp.split == "test") & (sp.scorer == "xgboost") & (sp.policy == "balanced_policy") & (sp.target_fpr == 0.2)]
        for _, r in sp.iterrows():
            comp.append({"system": "previous_scorer_policy", "model": r.scorer, "policy": r.policy, "target_fpr": r.target_fpr, "precision": r.precision, "recall": r.recall, "f1": r.f1, "fpr": r.fpr, "auroc": r.auroc, "pr_auc": r.pr_auc})
    dqn_path = args.hardened_dqn_results_dir / "dqn_metrics.csv"
    if dqn_path.exists():
        dqn = pd.read_csv(dqn_path)
        dqn = dqn[dqn.split == "test"]
        for _, r in dqn.iterrows():
            comp.append({"system": "hardened_dqn", "model": r.agent, "policy": r.reward_variant, "target_fpr": np.nan, "precision": r.precision, "recall": r.recall, "f1": r.f1, "fpr": r.fpr, "auroc": r.auroc, "pr_auc": r.pr_auc})
    for _, r in policy_df[policy_df.split == "test"].iterrows():
        comp.append({"system": "optimized_detector", "model": r.model, "policy": r.policy, "target_fpr": r.target_fpr, "precision": r.precision, "recall": r.recall, "f1": r.f1, "fpr": r.fpr, "auroc": r.auroc, "pr_auc": r.pr_auc})
    comp_df = pd.DataFrame(comp)
    comp_df.to_csv(args.results_dir / "comparison_with_supervised.csv", index=False)

    (args.results_dir / "scorer_tuning_report.txt").write_text(
        "Scorer tuning report\n\n"
        f"Feature count: {len(features)}. Added per-file rolling mean and delta features computed from row order only, without labels or source_file as feature.\n"
        f"Hard negative/sample weighting report: {json.dumps(hard_report, indent=2)}\n"
        f"Validation PR-AUC/AUROC: {json.dumps({k: {'val_pr_auc': v['val_pr_auc'], 'val_auroc': v['val_auroc']} for k, v in models.items()}, indent=2)}\n",
        encoding="utf-8",
    )
    (args.results_dir / "window_design_report.txt").write_text(
        "Window design update\n\nBase remains tight_event. No split or label changes. Added non-leaky context aggregates: rolling mean over last 3 windows and first difference per feature within source_file. These use row order only and do not use labels/test fitting.\n",
        encoding="utf-8",
    )
    (args.results_dir / "policy_design_report.txt").write_text(
        "Policy design\n\nrecall_friendly: maximize validation recall under target FPR. conservative: use 75% of target FPR budget. dual_threshold_ma3: allow a lower score only when local 3-window moving average supports it. Targets: 0.10, 0.15, 0.20. Thresholds are selected on validation only.\n",
        encoding="utf-8",
    )

    if best is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = best["cm"]
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{best['model']} / {best['policy']}")
        ax.set_xticks([0, 1], ["pred normal", "pred rogue"])
        ax.set_yticks([0, 1], ["true normal", "true rogue"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(args.results_dir / "confusion_matrix.png", dpi=180)
        plt.close(fig)

    top = comp_df.sort_values("f1", ascending=False).head(12)
    best_opt = policy_df[policy_df.split == "test"].sort_values("f1", ascending=False).iloc[0]
    lines = [
        "Optimized Rogue AP detector summary",
        "",
        f"Best optimized detector: {best_opt.model}/{best_opt.policy}/target_fpr={best_opt.target_fpr}, precision={best_opt.precision:.4f}, recall={best_opt.recall:.4f}, f1={best_opt.f1:.4f}, fpr={best_opt.fpr:.4f}, auroc={best_opt.auroc:.4f}, pr_auc={best_opt.pr_auc:.4f}.",
        "",
        "Top comparison rows:",
        top.to_string(index=False),
        "",
        "Conclusion: if the optimized detector does not beat the high-recall supervised baseline in F1, the bottleneck is scorer/window separability rather than the decision policy.",
    ]
    (args.results_dir / "short_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
