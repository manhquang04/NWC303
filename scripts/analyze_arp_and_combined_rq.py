#!/usr/bin/env python3
"""Summarize ARP results and combine with Rogue AP evidence for thesis RQs."""

from pathlib import Path
import pandas as pd


OUT = Path("results/combined_rq_arp_rogue")


def read(path):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    arp = read("results/arp_baselines/metrics.csv")
    if not arp.empty:
        for _, r in arp[(arp.split == "test") & (arp.threshold_policy == "val_best_f1")].iterrows():
            rows.append({"task": "ARP Spoofing", "level": "packet_feature_table", "system": "supervised", "model_policy_reward": r.model, "precision": r.precision, "recall": r.recall, "f1": r.f1, "fpr": r.fpr, "auroc": r.test_auroc, "pr_auc": r.test_pr_auc, "notes": "phase-block split; strong supervised baseline"})
    aw = read("results/arp_window_supervised/metrics.csv")
    if not aw.empty:
        for _, r in aw[aw.split == "test"].iterrows():
            rows.append({"task": "ARP Spoofing", "level": "window_aggregate", "system": "supervised", "model_policy_reward": r.model, "precision": r.precision, "recall": r.recall, "f1": r.f1, "fpr": r.fpr, "auroc": r.auroc, "pr_auc": r.pr_auc, "notes": "simple contiguous window aggregates"})
    ad = read("results/arp_window_dqn/arp_dqn_reward_ablation.csv")
    if not ad.empty:
        for _, r in ad[ad.split == "test"].iterrows():
            rows.append({"task": "ARP Spoofing", "level": "window_aggregate", "system": "DQN", "model_policy_reward": r.reward, "precision": r.precision, "recall": r.recall, "f1": r.f1, "fpr": r.fpr, "auroc": r.auroc, "pr_auc": r.pr_auc, "notes": "DQN reward ablation on ARP windows"})
    rogue = read("results/rogue_ap_detector_optimized/final_comparison_table.csv")
    if not rogue.empty:
        for _, r in rogue.head(8).iterrows():
            rows.append({"task": "Rogue AP", "level": "event_window", "system": r.system, "model_policy_reward": r.model_policy, "precision": r.precision, "recall": r.recall, "f1": r.f1, "fpr": r.fpr, "auroc": r.auroc, "pr_auc": r.pr_auc, "notes": r.notes})
    rq2 = read("results/rogue_ap_rq2/reward_ablation_results.csv")
    if not rq2.empty:
        for _, r in rq2.head(8).iterrows():
            rows.append({"task": "Rogue AP", "level": "event_window", "system": r.family, "model_policy_reward": f"{r.agent}/{r.reward}", "precision": r.precision, "recall": r.recall, "f1": r.f1, "fpr": r.fpr, "auroc": r.auroc, "pr_auc": r.pr_auc, "notes": "Rogue AP DQN reward ablation"})
    out = pd.DataFrame(rows).sort_values(["task", "f1"], ascending=[True, False])
    out.to_csv(OUT / "combined_results_table.csv", index=False)

    reward = out[out.system.str.contains("DQN|baseline_dqn|hardened", case=False, na=False)].copy()
    reward.to_csv(OUT / "combined_reward_ablation.csv", index=False)
    best_arp = out[out.task == "ARP Spoofing"].sort_values("f1", ascending=False).head(1)
    best_rogue = out[out.task == "Rogue AP"].sort_values("f1", ascending=False).head(1)
    lines = ["Combined ARP + Rogue AP RQ summary", ""]
    if not best_arp.empty:
        r = best_arp.iloc[0]
        lines.append(f"Best ARP result: {r.system}/{r.model_policy_reward} at {r.level}, F1={r.f1:.4f}, recall={r.recall:.4f}, FPR={r.fpr:.4f}.")
    if not best_rogue.empty:
        r = best_rogue.iloc[0]
        lines.append(f"Best Rogue AP result: {r.system}/{r.model_policy_reward} at {r.level}, F1={r.f1:.4f}, recall={r.recall:.4f}, FPR={r.fpr:.4f}.")
    lines += [
        "",
        "Interpretation for RQ1:",
        "ARP Spoofing is separable with the cleaned Kitsune feature table under phase-block split, especially with XGBoost. Rogue AP remains harder; optimized policy reduces FPR but does not beat the high-recall supervised baseline in F1.",
        "",
        "Interpretation for RQ2:",
        "Reward design affects behavior on both tasks. For Rogue AP, aggressive reward gives the best DQN F1 while FPR-constrained rewards reduce false alarms but collapse recall. For ARP windows, DQN does not beat packet-level supervised detection and still has high FPR; conservative reward is the best ARP-window DQN variant by F1.",
        "",
        "Thesis-safe claim:",
        "DRL/reward shaping is useful as a decision-policy experiment and for demonstrating recall-vs-FPR trade-offs, but the strongest production candidates are supervised scorers with constrained policies. Do not claim DRL universally outperforms supervised baselines.",
    ]
    (OUT / "combined_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
