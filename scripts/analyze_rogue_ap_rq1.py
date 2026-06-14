#!/usr/bin/env python3
"""RQ1 analysis pack for the cleaned Rogue AP detector."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path("results/rogue_ap_detector_optimized")
EVENT_RESULTS = Path("results/rogue_ap_event_windows")
SCORER_POLICY_RESULTS = Path("results/rogue_ap_scorer_policy")
DQN_RESULTS = Path("results/rogue_ap_dqn_hard_negative")

BEST_MODEL = "random_forest_weighted"
BEST_POLICY = "dual_threshold_ma3"
BEST_TARGET = 0.20


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def metric_cols() -> list[str]:
    return ["precision", "recall", "f1", "fpr", "auroc", "pr_auc"]


def notes_for(row: pd.Series) -> str:
    model = str(row.get("model", ""))
    policy = str(row.get("policy", ""))
    target = row.get("target_fpr", np.nan)
    bits = []
    if "weighted" in model:
        bits.append("hard-negative/sample-weighted scorer")
    if "balanced" in model:
        bits.append("class-balanced scorer")
    if policy == "dual_threshold_ma3":
        bits.append("dual threshold with 3-window smoothing")
    elif policy == "recall_friendly":
        bits.append("single threshold, validation recall under FPR budget")
    elif policy == "conservative":
        bits.append("uses 75% of target FPR budget")
    if pd.notna(target):
        bits.append(f"target FPR={target:.2f}")
    return "; ".join(bits)


def build_ablation(policy: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    test = policy[policy["split"] == "test"].copy()
    rows = []
    for _, r in test.iterrows():
        rows.append({
            "ablation": f"{r.model} | {r.policy} | target={r.target_fpr}",
            "model": r.model,
            "policy": r.policy,
            "target_fpr": r.target_fpr,
            "weighted_rf": r.model == "random_forest_weighted",
            "plain_rf": r.model == "random_forest_balanced",
            "smoothing": r.policy == "dual_threshold_ma3",
            "hard_negative_weighting": "weighted" in str(r.model),
            **{c: r[c] for c in metric_cols()},
            "notes": notes_for(r),
        })
    # Include direct 0.5 operating points as no-policy ablations.
    for _, r in metrics[metrics["split"] == "test"].iterrows():
        rows.append({
            "ablation": f"{r.model} | default_0.5",
            "model": r.model,
            "policy": "default_0.5",
            "target_fpr": np.nan,
            "weighted_rf": r.model == "random_forest_weighted",
            "plain_rf": r.model == "random_forest_balanced",
            "smoothing": False,
            "hard_negative_weighting": "weighted" in str(r.model),
            **{c: r[c] for c in metric_cols()},
            "notes": "raw scorer at threshold 0.5",
        })
    out = pd.DataFrame(rows).sort_values(["f1", "pr_auc"], ascending=False)
    out.to_csv(RESULTS / "ablation_results.csv", index=False)
    best = out.iloc[0]
    lines = [
        "Ablation summary",
        "",
        f"Best ablation: {best.ablation}",
        f"precision={best.precision:.4f}, recall={best.recall:.4f}, f1={best.f1:.4f}, fpr={best.fpr:.4f}, auroc={best.auroc:.4f}, pr_auc={best.pr_auc:.4f}.",
        "",
        "Main pattern: dual-threshold smoothing improves recall/F1 at target FPR=0.20, while stricter 0.10/0.15 budgets reduce false alarms but lose recall.",
        "Weighted RF is the best cleaned operating point; it improves PR-AUC and reduces FPR versus the high-recall supervised baseline, but does not beat that baseline's F1.",
    ]
    (RESULTS / "ablation_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return out


def feature_explainability(fi: pd.DataFrame) -> pd.DataFrame:
    best = fi[fi["model"] == BEST_MODEL].copy()
    best = best.sort_values("importance", ascending=False).head(25)

    def group_feature(f: str) -> str:
        if f.endswith("_ma3"):
            return "rolling/window context"
        if f.endswith("_delta1"):
            return "change/stability"
        if "ratio_" in f:
            return "aggregate protocol ratio"
        return "window aggregate"

    best["feature_group"] = best["feature"].map(group_feature)
    best.to_csv(RESULTS / "feature_importance_topk.csv", index=False)
    groups = best.groupby("feature_group")["importance"].sum().sort_values(ascending=False)
    lines = [
        "Feature importance summary",
        "",
        f"Model: {BEST_MODEL}",
        "Top features are mostly protocol/frame-control aggregate ratios and short-context statistics, not source_file/file_id/label-derived fields.",
        "",
        "Top 10:",
    ]
    for _, r in best.head(10).iterrows():
        lines.append(f"- {r.feature}: {r.importance:.6f} ({r.feature_group})")
    lines += ["", "Group importance mass:", groups.to_string()]
    (RESULTS / "feature_importance_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return best


def error_analysis(per_event: pd.DataFrame, per_file: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    pe = per_event[
        (per_event["model"] == BEST_MODEL)
        & (per_event["policy"] == BEST_POLICY)
        & (np.isclose(per_event["target_fpr"], BEST_TARGET))
    ].copy()
    pe["error_type"] = np.select(
        [(pe.label == 0) & (pe.pred == 1), (pe.label == 1) & (pe.pred == 0), (pe.label == pe.pred)],
        ["false_positive", "false_negative", "correct"],
        default="unknown",
    )
    pe["score_margin"] = pe["score"] - threshold if threshold is not None else np.nan
    file_errors = pe.groupby(["source_file", "error_type"]).size().unstack(fill_value=0).reset_index()
    file_metrics = per_file[
        (per_file["model"] == BEST_MODEL)
        & (per_file["policy"] == BEST_POLICY)
        & (np.isclose(per_file["target_fpr"], BEST_TARGET))
    ].copy()
    out = file_metrics.merge(file_errors, on="source_file", how="left").fillna(0)
    for col in ["false_positive", "false_negative", "correct"]:
        if col not in out:
            out[col] = 0
    out["hardness_score"] = out["false_positive"] + out["false_negative"] + (1 - out["f1"]) * out["windows"]
    out = out.sort_values("hardness_score", ascending=False)
    out.to_csv(RESULTS / "error_analysis.csv", index=False)
    worst_recall = out.sort_values("recall").iloc[0]
    worst_fpr = out.sort_values("fpr", ascending=False).iloc[0]
    lines = [
        "Error analysis summary",
        "",
        f"Most difficult by combined errors: {out.iloc[0].source_file} (F1={out.iloc[0].f1:.4f}, FP={int(out.iloc[0].false_positive)}, FN={int(out.iloc[0].false_negative)}).",
        f"Lowest recall file: {worst_recall.source_file} (recall={worst_recall.recall:.4f}).",
        f"Highest FPR file: {worst_fpr.source_file} (FPR={worst_fpr.fpr:.4f}).",
        "",
        "False positives/false negatives remain concentrated in specific held-out files, indicating file/event distribution shift rather than a simple threshold-only problem.",
    ]
    (RESULTS / "error_analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return pe


def generalization(per_file: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = {
        "optimized_detector": per_file[
            (per_file["model"] == BEST_MODEL)
            & (per_file["policy"] == BEST_POLICY)
            & (np.isclose(per_file["target_fpr"], BEST_TARGET))
        ].copy()
    }
    baseline_pf = read_csv(EVENT_RESULTS / "per_file_metrics.csv")
    if not baseline_pf.empty:
        selected["supervised_rf_baseline"] = baseline_pf[
            (baseline_pf["variant"] == "tight_event") & (baseline_pf["model"] == "random_forest")
        ].copy()
    for name, df in selected.items():
        if df.empty:
            continue
        for m in ["f1", "recall", "fpr", "precision", "pr_auc"]:
            rows.append({
                "system": name,
                "metric": m,
                "mean": float(df[m].mean()),
                "std": float(df[m].std(ddof=0)),
                "min": float(df[m].min()),
                "max": float(df[m].max()),
                "best_file": str(df.sort_values(m, ascending=False).iloc[0]["source_file"]),
                "worst_file": str(df.sort_values(m, ascending=True).iloc[0]["source_file"]),
            })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "generalization_summary.csv", index=False)
    opt = out[(out.system == "optimized_detector") & (out.metric == "f1")].iloc[0]
    lines = [
        "Generalization summary",
        "",
        f"Optimized detector per-file F1 mean={opt['mean']:.4f}, std={opt['std']:.4f}, min={opt['min']:.4f}, max={opt['max']:.4f}.",
        f"Worst optimized file by F1: {opt['worst_file']}; best: {opt['best_file']}.",
        "Dispersion across held-out files is still visible, so RQ1 should report per-file metrics, not only aggregate metrics.",
    ]
    (RESULTS / "generalization_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return out


def final_comparison(ablation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cmp = read_csv(RESULTS / "comparison_with_supervised.csv")
    if not cmp.empty:
        wanted = [
            ("supervised_baseline", "random_forest"),
            ("previous_scorer_policy", "xgboost"),
            ("hardened_dqn", "double_dqn"),
        ]
        for system, model in wanted:
            sub = cmp[(cmp.system == system) & (cmp.model == model)].copy()
            if sub.empty:
                continue
            r = sub.sort_values("f1", ascending=False).iloc[0]
            rows.append({"system": system, "model_policy": f"{r.model}/{r.policy}", **{c: r[c] for c in metric_cols()}, "notes": "reference result"})
    best = ablation.head(5)
    for _, r in best.iterrows():
        rows.append({"system": "ablation_or_optimized", "model_policy": f"{r.model}/{r.policy}/target={r.target_fpr}", **{c: r[c] for c in metric_cols()}, "notes": r.notes})
    out = pd.DataFrame(rows).drop_duplicates("model_policy").sort_values("f1", ascending=False)
    out.to_csv(RESULTS / "final_comparison_table.csv", index=False)
    best_prod = out[out["model_policy"].str.contains(BEST_MODEL, na=False)].head(1)
    lines = [
        "Final RQ1 summary",
        "",
        "The optimized detector trades recall for a much lower false-positive rate. It does not surpass the high-recall RF baseline in F1, but it is more deployment-friendly under false-alarm constraints.",
        f"Candidate production config: {BEST_MODEL} + {BEST_POLICY} + target FPR={BEST_TARGET:.2f}.",
        "Main bottleneck: scorer/window separability under hard held-out files, not the policy layer.",
        "Recommended next robustness check: one more held-out capture/session or stricter negative sampling audit, rather than deeper RL.",
    ]
    if not best_prod.empty:
        r = best_prod.iloc[0]
        lines.append(f"Candidate metrics: precision={r.precision:.4f}, recall={r.recall:.4f}, F1={r.f1:.4f}, FPR={r.fpr:.4f}, PR-AUC={r.pr_auc:.4f}.")
    (RESULTS / "final_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return out


def charts(ablation: pd.DataFrame, top_features: pd.DataFrame, errors: pd.DataFrame, per_file: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    f = top_features.head(15).iloc[::-1]
    ax.barh(f["feature"], f["importance"])
    ax.set_title("Top feature importance")
    fig.tight_layout()
    fig.savefig(RESULTS / "feature_importance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    a = ablation.head(12).iloc[::-1]
    ax.barh(a["ablation"], a["f1"], label="F1")
    ax.scatter(a["fpr"], a["ablation"], label="FPR", color="tab:red")
    ax.set_title("Ablation comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "ablation_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    errors["error_type"].value_counts().reindex(["false_positive", "false_negative", "correct"]).fillna(0).plot(kind="bar", ax=ax)
    ax.set_title("Error distribution")
    fig.tight_layout()
    fig.savefig(RESULTS / "error_distribution.png", dpi=180)
    plt.close(fig)

    pf = per_file[
        (per_file["model"] == BEST_MODEL)
        & (per_file["policy"] == BEST_POLICY)
        & (np.isclose(per_file["target_fpr"], BEST_TARGET))
    ].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(pf))
    ax.plot(x, pf["f1"], marker="o", label="F1")
    ax.plot(x, pf["recall"], marker="o", label="Recall")
    ax.plot(x, pf["fpr"], marker="o", label="FPR")
    ax.set_xticks(x, pf["source_file"], rotation=30, ha="right")
    ax.set_title("Per-file performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "per_file_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    policy = read_csv(RESULTS / "policy_metrics.csv")
    metrics = read_csv(RESULTS / "metrics_by_model.csv")
    fi = read_csv(RESULTS / "feature_importance.csv")
    per_file = read_csv(RESULTS / "per_file_metrics.csv")
    per_event = read_csv(RESULTS / "per_event_metrics.csv")
    thresholds = read_csv(RESULTS / "threshold_analysis.csv")
    th = None
    if not thresholds.empty:
        sub = thresholds[
            (thresholds["model"] == BEST_MODEL)
            & (thresholds["policy"] == "recall_friendly")
            & (np.isclose(thresholds["target_fpr"], BEST_TARGET))
        ]
        if not sub.empty:
            th = float(sub.iloc[0]["threshold"])

    ablation = build_ablation(policy, metrics)
    top_features = feature_explainability(fi)
    errors = error_analysis(per_event, per_file, th)
    generalization(per_file)
    final_comparison(ablation)
    charts(ablation, top_features, errors, per_file)


if __name__ == "__main__":
    main()
