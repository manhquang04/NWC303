#!/usr/bin/env python3
"""RQ2 analysis pack: reward/policy effects for Rogue AP decision behavior."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path("results/rogue_ap_rq2")
RL = Path("results/rogue_ap_rl_dqn")
HARD_DQN = Path("results/rogue_ap_dqn_hard_negative")
SCORER_POLICY = Path("results/rogue_ap_scorer_policy")
OPT = Path("results/rogue_ap_detector_optimized")
EVENT = Path("results/rogue_ap_event_windows")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def metrics_cols() -> list[str]:
    return ["precision", "recall", "f1", "fpr", "auroc", "pr_auc"]


def collect_reward_ablation() -> pd.DataFrame:
    rows = []
    base = read_csv(RL / "rl_metrics.csv")
    if not base.empty:
        for _, r in base[base["split"] == "test"].iterrows():
            rows.append({
                "family": "baseline_dqn",
                "agent": r["agent"],
                "reward": r["reward_variant"],
                "policy": "argmax_action",
                "training": "original",
                "avg_reward": np.nan,
                "episode_return_mean": r.get("episode_return_mean", np.nan),
                "max_false_alarm_streak_mean": np.nan,
                "best_val_f1": r.get("best_val_f1", np.nan),
                **{c: r[c] for c in metrics_cols()},
            })
    hard = read_csv(HARD_DQN / "dqn_metrics.csv")
    if not hard.empty:
        for _, r in hard[hard["split"] == "test"].iterrows():
            rows.append({
                "family": "hardened_dqn",
                "agent": r.get("agent", "double_dqn"),
                "reward": r["reward_variant"],
                "policy": "argmax_action",
                "training": "hard_negative_augmented",
                "avg_reward": r.get("avg_reward", np.nan),
                "episode_return_mean": r.get("episode_return_mean", np.nan),
                "max_false_alarm_streak_mean": r.get("max_false_alarm_streak_mean", np.nan),
                "best_val_f1": r.get("best_val_f1", np.nan),
                **{c: r[c] for c in metrics_cols()},
            })
    out = pd.DataFrame(rows).sort_values(["f1", "pr_auc"], ascending=False)
    out.to_csv(OUT / "reward_ablation_results.csv", index=False)
    best_f1 = out.iloc[0]
    best_fpr = out.sort_values("fpr").iloc[0]
    lines = [
        "Reward ablation summary",
        "",
        f"Best F1 reward: {best_f1.family}/{best_f1.agent}/{best_f1.reward}, F1={best_f1.f1:.4f}, recall={best_f1.recall:.4f}, FPR={best_f1.fpr:.4f}.",
        f"Lowest FPR reward: {best_fpr.family}/{best_fpr.agent}/{best_fpr.reward}, F1={best_fpr.f1:.4f}, recall={best_fpr.recall:.4f}, FPR={best_fpr.fpr:.4f}.",
        "",
        "Aggressive rewards preserve recall and F1 better. FPR-constrained rewards strongly reduce false alarms but can collapse recall.",
        "This supports RQ2: reward shaping controls decision behavior, but too much FP penalty makes the agent overly conservative.",
    ]
    (OUT / "reward_ablation_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return out


def collect_policy_ablation() -> pd.DataFrame:
    rows = []
    opt = read_csv(OPT / "policy_metrics.csv")
    if not opt.empty:
        for _, r in opt[opt["split"] == "test"].iterrows():
            rows.append({
                "source": "optimized_detector",
                "model": r["model"],
                "policy": r["policy"],
                "target_fpr": r.get("target_fpr", np.nan),
                "threshold": r.get("threshold", np.nan),
                "notes": "clean optimized scorer/policy",
                **{c: r[c] for c in metrics_cols()},
            })
    sp = read_csv(SCORER_POLICY / "policy_metrics.csv")
    if not sp.empty:
        for _, r in sp[sp["split"] == "test"].iterrows():
            rows.append({
                "source": "previous_scorer_policy",
                "model": r["scorer"],
                "policy": r["policy"],
                "target_fpr": r.get("target_fpr", np.nan),
                "threshold": r.get("threshold", np.nan),
                "notes": "score-only constrained threshold policy",
                **{c: r[c] for c in metrics_cols()},
            })
    mm = read_csv(OPT / "metrics_by_model.csv")
    if not mm.empty:
        for _, r in mm[mm["split"] == "test"].iterrows():
            rows.append({
                "source": "optimized_detector",
                "model": r["model"],
                "policy": "plain_threshold_0.5",
                "target_fpr": np.nan,
                "threshold": 0.5,
                "notes": "plain scorer threshold",
                **{c: r[c] for c in metrics_cols()},
            })
    out = pd.DataFrame(rows).sort_values(["f1", "pr_auc"], ascending=False)
    out.to_csv(OUT / "policy_ablation_results.csv", index=False)
    th = read_csv(OPT / "threshold_analysis.csv")
    sp_th = read_csv(SCORER_POLICY / "threshold_analysis.csv")
    th_rows = []
    if not th.empty:
        th["source"] = "optimized_detector"
        th_rows.append(th)
    if not sp_th.empty:
        sp_th["source"] = "previous_scorer_policy"
        th_rows.append(sp_th)
    if th_rows:
        pd.concat(th_rows, ignore_index=True).to_csv(OUT / "policy_threshold_analysis.csv", index=False)
    best_policy = out.iloc[0]
    best_constrained = out[out["target_fpr"].notna()].iloc[0]
    lines = [
        "Policy ablation summary",
        "",
        f"Best overall policy row: {best_policy.model}/{best_policy.policy}, F1={best_policy.f1:.4f}, recall={best_policy.recall:.4f}, FPR={best_policy.fpr:.4f}.",
        f"Best constrained policy row: {best_constrained.model}/{best_constrained.policy}/target={best_constrained.target_fpr}, F1={best_constrained.f1:.4f}, recall={best_constrained.recall:.4f}, FPR={best_constrained.fpr:.4f}.",
        "",
        "Plain thresholds can maximize recall but often allow high FPR. Dual-threshold smoothing at target FPR=0.20 gives a clearer deployment operating point.",
    ]
    (OUT / "policy_ablation_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return out


def rl_vs_threshold(reward: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sup = read_csv(EVENT / "metrics_by_variant.csv")
    if not sup.empty:
        sup = sup[(sup["variant"] == "tight_event") & (sup["split"] == "test") & (sup["threshold_policy"] == "val_best_f1")]
        for _, r in sup.iterrows():
            rows.append({"system": "supervised_threshold_baseline", "model_policy": f"{r.model}/val_best_f1", "category": "supervised", **{c: r[c if c not in ['auroc', 'pr_auc'] else 'test_' + c] if c in ['auroc', 'pr_auc'] else r[c] for c in metrics_cols()}})
    if not reward.empty:
        for _, r in reward.head(8).iterrows():
            rows.append({"system": r.family, "model_policy": f"{r.agent}/{r.reward}", "category": "rl", **{c: r[c] for c in metrics_cols()}})
    if not policy.empty:
        for _, r in policy.head(8).iterrows():
            rows.append({"system": r.source, "model_policy": f"{r.model}/{r.policy}/target={r.target_fpr}", "category": "threshold_policy", **{c: r[c] for c in metrics_cols()}})
    out = pd.DataFrame(rows).sort_values("f1", ascending=False)
    out.to_csv(OUT / "rl_vs_threshold_comparison.csv", index=False)
    best_rl = out[out.category == "rl"].sort_values("f1", ascending=False).head(1)
    best_policy = out[out.category == "threshold_policy"].sort_values("f1", ascending=False).head(1)
    lines = [
        "RL vs threshold summary",
        "",
    ]
    if not best_rl.empty:
        r = best_rl.iloc[0]
        lines.append(f"Best RL: {r.model_policy}, F1={r.f1:.4f}, recall={r.recall:.4f}, FPR={r.fpr:.4f}, PR-AUC={r.pr_auc:.4f}.")
    if not best_policy.empty:
        r = best_policy.iloc[0]
        lines.append(f"Best threshold/policy: {r.model_policy}, F1={r.f1:.4f}, recall={r.recall:.4f}, FPR={r.fpr:.4f}, PR-AUC={r.pr_auc:.4f}.")
    lines += [
        "",
        "RL does not clearly dominate thresholding. Its useful contribution is showing reward-shaped operating behavior, while constrained supervised policy is simpler and more explainable.",
    ]
    (OUT / "rl_vs_threshold_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return out


def training_behavior() -> pd.DataFrame:
    rows = []
    curves = read_csv(RL / "rl_training_curves.csv")
    if not curves.empty:
        for keys, g in curves.groupby(["agent", "reward_variant"]):
            rows.append({
                "family": "baseline_dqn",
                "agent": keys[0],
                "reward": keys[1],
                "epochs": len(g),
                "final_val_f1": float(g["val_f1"].iloc[-1]),
                "best_val_f1": float(g["val_f1"].max()),
                "val_f1_std_last10": float(g["val_f1"].tail(10).std(ddof=0)),
                "final_val_fpr": float(g["val_fpr"].iloc[-1]),
                "final_train_return": float(g["train_return"].iloc[-1]),
                "final_loss": float(g["loss"].dropna().iloc[-1]) if g["loss"].notna().any() else np.nan,
            })
    hard_curves = read_csv(HARD_DQN / "dqn_training_curves.csv")
    if not hard_curves.empty:
        for reward, g in hard_curves.groupby("reward_variant"):
            rows.append({
                "family": "hardened_dqn",
                "agent": "double_dqn",
                "reward": reward,
                "epochs": len(g),
                "final_val_f1": float(g["val_f1"].iloc[-1]),
                "best_val_f1": float(g["val_f1"].max()),
                "val_f1_std_last10": float(g["val_f1"].tail(10).std(ddof=0)),
                "final_val_fpr": float(g["val_fpr"].iloc[-1]),
                "final_train_return": float(g["train_return"].iloc[-1]),
                "final_loss": float(g["loss"].dropna().iloc[-1]) if g["loss"].notna().any() else np.nan,
            })
    out = pd.DataFrame(rows).sort_values("best_val_f1", ascending=False)
    out.to_csv(OUT / "training_behavior_summary.csv", index=False)
    lines = [
        "Training behavior summary",
        "",
        "Validation F1 and last-10-epoch variance are used as lightweight stability indicators.",
    ]
    if not out.empty:
        stable = out.sort_values(["val_f1_std_last10", "best_val_f1"], ascending=[True, False]).iloc[0]
        best = out.iloc[0]
        lines.append(f"Best validation behavior: {best.family}/{best.agent}/{best.reward}, best_val_f1={best.best_val_f1:.4f}.")
        lines.append(f"Most stable last-10 curve: {stable.family}/{stable.agent}/{stable.reward}, std={stable.val_f1_std_last10:.4f}.")
    lines.append("Hardened FPR-constrained rewards are more conservative; aggressive rewards preserve detection behavior better but keep FPR higher.")
    (OUT / "training_behavior_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return out


def hard_file_summary() -> pd.DataFrame:
    hard_files = {"RogueAP_35.csv", "RogueAP_36.csv", "RogueAP_38.csv"}
    rows = []
    sources = [
        ("baseline_dqn", RL / "rl_per_file_metrics.csv", "reward_variant"),
        ("hardened_dqn", HARD_DQN / "dqn_per_file_metrics.csv", "reward_variant"),
        ("optimized_policy", OPT / "per_file_metrics.csv", "policy"),
    ]
    for family, path, policy_col in sources:
        df = read_csv(path)
        if df.empty or "source_file" not in df:
            continue
        df = df[df["source_file"].isin(hard_files)].copy()
        if df.empty:
            continue
        if family == "optimized_policy":
            df = df[(df["model"] == "random_forest_weighted") & (df["policy"] == "dual_threshold_ma3") & (np.isclose(df["target_fpr"], 0.20))]
            group_cols = ["model", "policy"]
        elif family == "baseline_dqn":
            group_cols = ["agent", "reward_variant"]
        else:
            group_cols = ["reward_variant"]
        for keys, g in df.groupby(group_cols):
            label = "/".join(keys) if isinstance(keys, tuple) else str(keys)
            rows.append({
                "family": family,
                "model_policy_reward": label,
                "hard_files": ",".join(sorted(hard_files)),
                "mean_f1": float(g["f1"].mean()),
                "min_f1": float(g["f1"].min()),
                "mean_recall": float(g["recall"].mean()),
                "min_recall": float(g["recall"].min()),
                "mean_fpr": float(g["fpr"].mean()),
                "max_fpr": float(g["fpr"].max()),
            })
    out = pd.DataFrame(rows).sort_values(["mean_f1", "min_recall"], ascending=False)
    out.to_csv(OUT / "hard_file_policy_summary.csv", index=False)
    lines = [
        "Hard-file policy summary",
        "",
        "Hard files inspected: RogueAP_35, RogueAP_36, RogueAP_38.",
    ]
    if not out.empty:
        best = out.iloc[0]
        lines.append(f"Best hard-file mean F1: {best.family}/{best.model_policy_reward}, mean_F1={best.mean_f1:.4f}, mean_recall={best.mean_recall:.4f}, mean_FPR={best.mean_fpr:.4f}.")
    lines.append("Files remain uneven; RQ2 should discuss reward/policy as operating-point control, not as a complete solution to domain shift.")
    (OUT / "hard_file_policy_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return out


def charts(reward: pd.DataFrame, policy: pd.DataFrame) -> None:
    curves = read_csv(RL / "rl_training_curves.csv")
    hard_curves = read_csv(HARD_DQN / "dqn_training_curves.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    if not curves.empty:
        for (agent, reward_name), g in curves.groupby(["agent", "reward_variant"]):
            ax.plot(g["epoch"], g["val_f1"], alpha=0.7, label=f"{agent}/{reward_name}")
    if not hard_curves.empty:
        for reward_name, g in hard_curves.groupby("reward_variant"):
            ax.plot(g["epoch"], g["val_f1"], linestyle="--", label=f"hard/{reward_name}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation F1")
    ax.set_title("Reward training curves")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "reward_training_curves.png", dpi=180)
    fig.savefig(OUT / "training_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    p = policy.head(14).iloc[::-1]
    labels = [
        f"{str(m)}/{str(pol)}/{str(t)}"
        for m, pol, t in zip(p["model"].tolist(), p["policy"].tolist(), p["target_fpr"].tolist())
    ]
    ax.barh(labels, p["f1"], label="F1")
    ax.scatter(p["fpr"], labels, color="tab:red", label="FPR")
    ax.set_title("Policy ablation: F1 vs FPR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "policy_ablation_chart.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reward = collect_reward_ablation()
    policy = collect_policy_ablation()
    rl_vs_threshold(reward, policy)
    training_behavior()
    hard_file_summary()
    charts(reward, policy)


if __name__ == "__main__":
    main()
