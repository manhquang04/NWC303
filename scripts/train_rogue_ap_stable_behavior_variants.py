#!/usr/bin/env python3
"""Train stable 802.11 behavior feature variants for Rogue AP detection."""

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
from xgboost import XGBClassifier


BASE_ARTIFACT_PATTERNS = (
    "frame_len",
    "frame_encap_type",
    "wlan_duration",
    "wlan_fc_type",
    "wlan_fc_subtype",
    "wlan_fc_protected",
    "wlan_radio_duration",
    "radiotap_channel",
    "radiotap_datarate",
    "wlan_radio_data_rate",
    "wlan_radio_signal_dbm",
    "wlan_radio_phy",
    "wlan_radio_channel",
    "wlan_radio_frequency",
    "signal_dbm",
    "data_rate",
    "phy",
    "time",
    "timestamp",
    "mactime",
    "tsf",
    "source_file",
    "file_id",
)

CONSERVATIVE_EXACT = {
    "num__wlan_fc_frag",
    "num__wlan_fc_order",
    "num__wlan_fc_moredata",
    "num__wlan_fc_pwrmgt",
    "num__wlan_fc_retry",
}

MODERATE_PREFIXES = (
    "cat_wlan_fc_ds_",
    "cat_llc_",
)

RICH_PREFIXES = (
    "cat_ip_proto_",
    "cat_ip_version_",
    "cat_tcp_flags_syn_",
    "cat_tcp_flags_ack_",
    "cat_tcp_flags_fin_",
    "cat_tcp_flags_push_",
    "cat_tcp_flags_reset_",
    "cat_tcp_checksum_status_",
)

DROP_VALUE_TOKENS = ("_MISSING", "_OTHER")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stable behavior Rogue AP variants.")
    parser.add_argument("--data-dir", type=Path, default=Path("processed/rogue_ap_with_source_metadata"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/rogue_ap_stable_behavior_variants"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tree-negatives", type=int, default=150_000)
    parser.add_argument("--fpr-limit", type=float, default=0.01)
    return parser.parse_args()


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(p) for p in sorted((data_dir / split).glob("part-*.parquet"))],
        ignore_index=True,
    )


def check_split(split: str, df: pd.DataFrame) -> dict[str, Any]:
    counts = df["label"].value_counts().to_dict()
    dup = int(pd.util.hash_pandas_object(df, index=False).duplicated().sum())
    return {
        "split": split,
        "rows": len(df),
        "cols": df.shape[1],
        "normal": int(counts.get(0, 0)),
        "rogueap": int(counts.get(1, 0)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": dup,
        "duplicate_rate_pct": round(100 * dup / max(len(df), 1), 6),
    }


def is_artifact(col: str) -> bool:
    low = col.lower()
    return any(p in low for p in BASE_ARTIFACT_PATTERNS)


def is_missing_artifact(col: str) -> bool:
    return any(tok in col for tok in DROP_VALUE_TOKENS)


def design_variants(columns: list[str]) -> tuple[dict[str, list[str]], pd.DataFrame, str]:
    feature_cols = [c for c in columns if c not in {"label", "source_file", "file_id"}]
    category_rows = []
    variants: dict[str, list[str]] = {}

    conservative = [c for c in feature_cols if c in CONSERVATIVE_EXACT]
    moderate = conservative + [
        c
        for c in feature_cols
        if c.startswith(MODERATE_PREFIXES)
        and not is_artifact(c)
        and not is_missing_artifact(c)
    ]
    rich = moderate + [
        c
        for c in feature_cols
        if c.startswith(RICH_PREFIXES)
        and not is_artifact(c)
        and not is_missing_artifact(c)
    ]
    # Keep raw sequence only in rich_behavior: it may capture behavior, but can also reflect ordering.
    if "num__wlan_seq" in feature_cols:
        rich.append("num__wlan_seq")

    variants["conservative"] = sorted(dict.fromkeys(conservative))
    variants["moderate"] = sorted(dict.fromkeys(moderate))
    variants["rich_behavior"] = sorted(dict.fromkeys(rich))

    keep_sets = {name: set(cols) for name, cols in variants.items()}
    for c in feature_cols:
        if c in keep_sets["conservative"]:
            group, decision, reason = "A_keep_sure", "keep_conservative", "Stable 802.11 FC behavior flag."
        elif c in keep_sets["moderate"]:
            group, decision, reason = "B_keep_ablate", "keep_moderate", "Protocol/DS semantic feature; useful but should be ablated."
        elif c in keep_sets["rich_behavior"]:
            group, decision, reason = "B_keep_ablate", "keep_rich_behavior", "Protocol flag/value or sequence behavior; may still encode traffic composition."
        elif is_artifact(c):
            group, decision, reason = "C_drop_artifact", "drop", "Timing/radiotap/PHY/channel/signal/length/type artifact pattern."
        elif is_missing_artifact(c):
            group, decision, reason = "C_drop_artifact", "drop", "Missing/OTHER indicator likely extraction/protocol-presence artifact."
        else:
            group, decision, reason = "D_uncertain", "drop_for_now", "Uncertain stability; inspect before using as baseline feature."
        category_rows.append(
            {
                "feature": c,
                "group": group,
                "decision": decision,
                "in_conservative": c in keep_sets["conservative"],
                "in_moderate": c in keep_sets["moderate"],
                "in_rich_behavior": c in keep_sets["rich_behavior"],
                "reason": reason,
            }
        )

    report = [
        "Stable 802.11 behavior feature design",
        "",
        "A. Keep surely: wlan FC behavior flags such as frag/order/moredata/pwrmgt/retry.",
        "B. Keep but ablate: DS/LLC/protocol flags and sequence behavior.",
        "C. Drop artifact/leak: timing, radiotap, PHY, channel, signal, data-rate, frame length/type/duration/protected, missing/OTHER indicators.",
        "D. Uncertain: remaining transformed fields not clearly stable.",
        "",
        "Variant conservative: only stable 802.11 FC behavior flags.",
        "Variant moderate: conservative + DS/LLC semantic categories without MISSING/OTHER.",
        "Variant rich_behavior: moderate + IP/TCP value flags and wlan_seq, still excluding radiotap/PHY/timing/source metadata and MISSING/OTHER.",
    ]
    return variants, pd.DataFrame(category_rows), "\n".join(report)


def eval_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
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


def thresholds(y: np.ndarray, scores: np.ndarray, fpr_limit: float) -> dict[str, float]:
    precision, recall, th = precision_recall_curve(y, scores)
    if len(th) == 0:
        return {"default_0_5": 0.5, "val_best_f1": 0.5, f"val_fpr_le_{fpr_limit:g}": 0.5}
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = float(th[int(np.nanargmax(f1))])
    fpr, tpr, roc_th = roc_curve(y, scores)
    valid = np.flatnonzero(fpr <= fpr_limit)
    constrained = float(roc_th[valid[np.argmax(tpr[valid])]]) if len(valid) else best
    return {"default_0_5": 0.5, "val_best_f1": best, f"val_fpr_le_{fpr_limit:g}": constrained}


def train_models(x_train, y_train, x_val, y_val, args):
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(args.random_state)
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    tree_idx = np.concatenate([pos_idx, rng.choice(neg_idx, size=min(args.tree_negatives, len(neg_idx)), replace=False)])
    rng.shuffle(tree_idx)
    x_tree, y_tree = x_train.iloc[tree_idx], y_train[tree_idx]
    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000, random_state=args.random_state).fit(x_train, y_train),
        "random_forest": RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=args.random_state).fit(x_tree, y_tree),
    }
    spw = int((y_train == 0).sum()) / max(int(y_train.sum()), 1)
    best_xgb, best_pr = None, -np.inf
    for params in [{"max_depth": 3, "learning_rate": 0.08, "n_estimators": 120}, {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 160}]:
        model = XGBClassifier(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=args.random_state, scale_pos_weight=spw, subsample=0.9, colsample_bytree=0.9, **params)
        model.fit(x_tree, y_tree, verbose=False)
        pr = average_precision_score(y_val, model.predict_proba(x_val)[:, 1])
        if pr > best_pr:
            best_pr, best_xgb = pr, model
    models["xgboost"] = best_xgb
    return models


def evaluate_variant(name, models, x_val, y_val, x_test, y_test, fpr_limit):
    metric_rows, threshold_rows = [], []
    for mname, model in models.items():
        vs = model.predict_proba(x_val)[:, 1]
        ts = model.predict_proba(x_test)[:, 1]
        aucs = {
            "val_auroc": roc_auc_score(y_val, vs),
            "val_pr_auc": average_precision_score(y_val, vs),
            "test_auroc": roc_auc_score(y_test, ts),
            "test_pr_auc": average_precision_score(y_test, ts),
        }
        for policy, th in thresholds(y_val, vs, fpr_limit).items():
            vm, tm = eval_at(y_val, vs, th), eval_at(y_test, ts, th)
            threshold_rows.append({"variant": name, "model": mname, "policy": policy, "threshold": th, **{f"val_{k}": v for k, v in vm.items() if k != "threshold"}, **{f"test_{k}": v for k, v in tm.items() if k != "threshold"}})
            for split, met in [("val", vm), ("test", tm)]:
                metric_rows.append({"variant": name, "model": mname, "split": split, "threshold_policy": policy, **aucs, **met})
    return pd.DataFrame(metric_rows), pd.DataFrame(threshold_rows)


def importance_tables(name, models, features):
    fi, coef = [], []
    for mname, model in models.items():
        if hasattr(model, "feature_importances_"):
            vals = np.asarray(model.feature_importances_)
            for i in np.argsort(vals)[::-1][:30]:
                fi.append({"variant": name, "model": mname, "feature": features[i], "importance": float(vals[i])})
        if mname == "logistic_regression":
            vals = model.coef_[0]
            for i in np.argsort(np.abs(vals))[::-1][:30]:
                coef.append({"variant": name, "model": mname, "feature": features[i], "coefficient": float(vals[i]), "abs_coefficient": float(abs(vals[i])), "direction": "RogueAP" if vals[i] >= 0 else "Normal"})
    return pd.DataFrame(fi), pd.DataFrame(coef)


def per_file(name, model_name, model, x_test, test_meta, y_test, threshold):
    scores = model.predict_proba(x_test)[:, 1]
    temp = test_meta.copy()
    temp["label"], temp["score"] = y_test, scores
    rows = []
    for sf, g in temp.groupby("source_file", sort=True):
        y, s = g["label"].to_numpy(dtype=np.int8), g["score"].to_numpy()
        met = eval_at(y, s, threshold)
        rows.append({"variant": name, "model": model_name, "source_file": sf, "rows": len(g), "normal": int((y == 0).sum()), "rogueap": int((y == 1).sum()), "auroc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s), **met})
    return pd.DataFrame(rows)


def plot_cms(best, y_test, out):
    fig, axes = plt.subplots(1, len(best), figsize=(5 * len(best), 4))
    if len(best) == 1:
        axes = [axes]
    for ax, (name, (scores, th)) in zip(axes, best.items()):
        cm = confusion_matrix(y_test, (scores >= th).astype(np.int8), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0, 1], ["Pred N", "Pred R"])
        ax.set_yticks([0, 1], ["True N", "True R"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = (read_split(args.data_dir, s) for s in ["train", "val", "test"])
    pd.DataFrame([check_split("train", train), check_split("val", val), check_split("test", test)]).to_csv(args.output_dir / "data_checks.csv", index=False)
    variants, fmap, report = design_variants(list(train.columns))
    fmap.to_csv(args.output_dir / "feature_variant_map.csv", index=False)
    (args.output_dir / "feature_set_design_report.txt").write_text(report + "\n\n" + json.dumps({k: v for k, v in variants.items()}, indent=2), encoding="utf-8")
    y_train, y_val, y_test = (df["label"].to_numpy(dtype=np.int8) for df in [train, val, test])
    all_m, all_t, all_fi, all_coef, all_pf = [], [], [], [], []
    best_plot = {}
    for name, cols in variants.items():
        print(f"Training {name}: {len(cols)} features", flush=True)
        models = train_models(train[cols], y_train, val[cols], y_val, args)
        met, ths = evaluate_variant(name, models, val[cols], y_val, test[cols], y_test, args.fpr_limit)
        all_m.append(met); all_t.append(ths)
        fi, coef = importance_tables(name, models, cols)
        all_fi.append(fi); all_coef.append(coef)
        selector = ths[ths.policy.eq("val_best_f1")].sort_values(["val_f1", "test_f1"], ascending=False).iloc[0]
        bm, bth = selector["model"], float(selector["threshold"])
        scores = models[bm].predict_proba(test[cols])[:, 1]
        best_plot[name] = (scores, bth)
        all_pf.append(per_file(name, bm, models[bm], test[cols], test[["source_file", "file_id"]], y_test, bth))
    metrics = pd.concat(all_m, ignore_index=True)
    thresholds_df = pd.concat(all_t, ignore_index=True)
    fi = pd.concat(all_fi, ignore_index=True)
    coef = pd.concat(all_coef, ignore_index=True)
    pf = pd.concat(all_pf, ignore_index=True)
    metrics.to_csv(args.output_dir / "metrics_by_variant.csv", index=False)
    thresholds_df.to_csv(args.output_dir / "threshold_analysis.csv", index=False)
    fi.to_csv(args.output_dir / "feature_importance.csv", index=False)
    coef.to_csv(args.output_dir / "top_coefficients.csv", index=False)
    pf.to_csv(args.output_dir / "per_file_metrics.csv", index=False)
    plot_cms(best_plot, y_test, args.output_dir / "confusion_matrix.png")
    lines = ["Stable behavior variants summary", "", "Variant sizes:"]
    lines += [f"- {k}: {len(v)} features" for k, v in variants.items()]
    lines += ["", "Validation-selected F1 test results:"]
    sel = metrics[metrics.threshold_policy.eq("val_best_f1") & metrics.split.eq("test")]
    for _, r in sel.sort_values(["variant", "f1"], ascending=[True, False]).iterrows():
        lines.append(f"- {r.variant} / {r.model}: precision={r.precision:.6f}, recall={r.recall:.6f}, f1={r.f1:.6f}, fpr={r.fpr:.6f}, auroc={r.test_auroc:.6f}, pr_auc={r.test_pr_auc:.6f}")
    lines += ["", "Interpretation:", "- Conservative/moderate/rich_behavior exclude source_file/file_id from features.", "- If performance is very low, prior high scores were largely driven by capture/PHY/protocol-presence artifacts.", "- Do not move to DQN until this supervised feature-set question is settled."]
    (args.output_dir / "short_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
