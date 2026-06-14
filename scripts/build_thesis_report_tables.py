#!/usr/bin/env python3
"""Build thesis-ready RQ tables from existing experiment artifacts.

This script does not retrain models or rerun runtime experiments. It only reads
validated result files and writes a compact report package under
``reports/thesis``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "thesis"
TABLES = REPORT / "tables"
APPENDIX = REPORT / "appendix"

NUMERIC_COLUMNS = {
    "precision",
    "recall",
    "f1",
    "fpr",
    "auroc",
    "pr_auc",
    "recall_or_detection_rate",
    "fpr_or_false_alert_rate",
    "mean_detection_latency_sec",
    "mean_mitigation_latency_sec",
}


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def maybe_read_csv(path: str) -> pd.DataFrame:
    p = ROOT / path
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def fmt_value(value, column: str) -> str:
    if pd.isna(value):
        return ""
    if column in NUMERIC_COLUMNS:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = df.copy()
    for column in clean.columns:
        clean[column] = [fmt_value(value, column) for value in clean[column]]
    clean.to_csv(path, index=False)
    lines = [
        "| " + " | ".join(clean.columns) + " |",
        "| " + " | ".join(["---"] * len(clean.columns)) + " |",
    ]
    for _, row in clean.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in clean.columns) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_rq1() -> pd.DataFrame:
    rows = []
    for label, path in [
        ("ARP controlled SDN runtime", "results/sdn_runtime_arp_dqn_stress_150/benchmark_summary.json"),
        ("ARP hardcore SDN runtime", "results/sdn_runtime_arp_dqn_hardcore_120/benchmark_summary.json"),
    ]:
        p = ROOT / path
        if not p.exists():
            continue
        summary = read_json(path)
        rows.append(
            {
                "task": "ARP Spoofing",
                "experiment": label,
                "setting": f"{summary['episodes']} episodes ({summary['attack_episodes']} attack, {summary['normal_episodes']} normal)",
                "precision": summary["precision"],
                "recall": summary["recall"],
                "f1": summary["f1"],
                "fpr": summary["fpr"],
                "auroc": "",
                "pr_auc": "",
                "notes": "Controlled success" if "controlled" in label else "Hardcore robustness boundary",
            }
        )

    lofo = maybe_read_csv("results/rogue_ap_per_bssid_lofo/lofo_summary.csv")
    if not lofo.empty:
        selected = lofo[lofo["model"].eq("xgb_per_bssid") & lofo["policy"].eq("fpr_le_0.20")]
        for _, row in selected.iterrows():
            rows.append(
                {
                    "task": "Rogue AP",
                    "experiment": "Per-BSSID XGBoost LOFO",
                    "setting": f"{int(row['files'])} held-out positive files; {int(row['test_windows_total'])} test windows",
                    "precision": row["precision_mean"],
                    "recall": row["recall_mean"],
                    "f1": row["f1_mean"],
                    "fpr": row["fpr_mean"],
                    "auroc": row["auroc_mean"],
                    "pr_auc": row["pr_auc_mean"],
                    "notes": "Best offline Rogue AP result; source_file not used as feature",
                }
            )

    runtime = maybe_read_csv("results/rogue_ap_runtime_rq_summary/rogue_ap_runtime_scenario_summary.csv")
    if not runtime.empty:
        for _, row in runtime.iterrows():
            rows.append(
                {
                    "task": "Rogue AP",
                    "experiment": f"Live Wi-Fi runtime: {row['scenario']}",
                    "setting": f"{int(row['windows'])} monitor-mode windows",
                    "precision": "",
                    "recall": row["alert_rate"] if row["scenario"] != "normal_live_wifi" else "",
                    "f1": "",
                    "fpr": row["alert_rate"] if row["scenario"] == "normal_live_wifi" else "",
                    "auroc": "",
                    "pr_auc": "",
                    "notes": f"Hybrid monitor-mode policy; model_alerts=0, policy_alerts={int(row['policy_alerts'])}",
                }
            )
    return pd.DataFrame(rows)


def build_rq2() -> pd.DataFrame:
    source = ROOT / "results/hardcore_rq_suite_summary/rq2_reward_summary.csv"
    return pd.read_csv(source) if source.exists() else pd.DataFrame()


def build_rq3() -> pd.DataFrame:
    rows = []
    for label, path in [
        ("controlled_sdn_stress_150", "results/sdn_runtime_arp_dqn_stress_150/benchmark_summary.json"),
        ("hardcore_sdn_evasion_benign_churn_120", "results/sdn_runtime_arp_dqn_hardcore_120/benchmark_summary.json"),
    ]:
        p = ROOT / path
        if not p.exists():
            continue
        summary = read_json(path)
        rows.append(
            {
                "task": "ARP Spoofing",
                "runtime_setting": label,
                "episodes_or_windows": summary["episodes"],
                "precision": summary["precision"],
                "recall_or_detection_rate": summary["recall"],
                "f1": summary["f1"],
                "fpr_or_false_alert_rate": summary["fpr"],
                "mean_detection_latency_sec": summary["mean_detection_latency_sec"],
                "mean_mitigation_latency_sec": summary["mean_mitigation_latency_sec"],
                "robustness_note": "Controlled SDN testbed success"
                if label.startswith("controlled")
                else "Hardcore limitations under evasion and benign churn",
            }
        )

    runtime = maybe_read_csv("results/rogue_ap_runtime_rq_summary/rogue_ap_runtime_scenario_summary.csv")
    if not runtime.empty:
        for _, row in runtime.iterrows():
            rows.append(
                {
                    "task": "Rogue AP",
                    "runtime_setting": row["scenario"],
                    "episodes_or_windows": int(row["windows"]),
                    "precision": "",
                    "recall_or_detection_rate": row["alert_rate"] if row["scenario"] != "normal_live_wifi" else "",
                    "f1": "",
                    "fpr_or_false_alert_rate": row["alert_rate"] if row["scenario"] == "normal_live_wifi" else "",
                    "mean_detection_latency_sec": "",
                    "mean_mitigation_latency_sec": "",
                    "robustness_note": "Real USB Wi-Fi monitor-mode runtime; hybrid policy decision layer",
                }
            )
    return pd.DataFrame(rows)


def copy_appendix() -> None:
    APPENDIX.mkdir(parents=True, exist_ok=True)
    sources = {
        "results/sdn_runtime_arp_dqn_hardcore_120/variant_breakdown.csv": "arp_hardcore_variant_breakdown.csv",
        "results/sdn_runtime_arp_dqn_hardcore_120/scenario_breakdown.csv": "arp_hardcore_background_breakdown.csv",
        "results/rogue_ap_per_bssid_lofo/lofo_summary.csv": "rogue_ap_lofo_summary.csv",
        "results/rogue_ap_runtime_rq_summary/rogue_ap_runtime_scenario_summary.csv": "rogue_ap_runtime_scenarios.csv",
    }
    for src, dst in sources.items():
        p = ROOT / src
        if p.exists():
            pd.read_csv(p).to_csv(APPENDIX / dst, index=False)


def build_manifest() -> None:
    rows = [
        ("curated_report", "reports/thesis/README.md", "Thesis-ready report for RQ1-RQ3"),
        ("curated_table", "reports/thesis/tables/rq1_detection_results.csv", "Final detection results table"),
        ("curated_table", "reports/thesis/tables/rq2_reward_ablation.csv", "Reward ablation table"),
        ("curated_table", "reports/thesis/tables/rq3_runtime_robustness.csv", "Runtime robustness table"),
        ("primary_result", "results/sdn_runtime_arp_dqn_stress_150/", "Controlled ARP SDN runtime stress, 150 episodes"),
        ("primary_result", "results/sdn_runtime_arp_dqn_hardcore_120/", "Hardcore ARP SDN runtime, 120 episodes"),
        ("primary_result", "results/rogue_ap_per_bssid_lofo/", "Best Rogue AP offline LOFO evidence"),
        ("primary_result", "results/rogue_ap_runtime_rq_summary/", "Live Wi-Fi Rogue AP runtime scenario summary"),
        ("primary_model", "models/rogue_ap_runtime/", "Exported Rogue AP runtime model bundle"),
        ("primary_code", "sdn_runtime/", "Runtime controllers and sensors"),
        ("primary_code", "scripts/", "Dataset building, training, ablation, and analysis scripts"),
    ]
    pd.DataFrame(rows, columns=["category", "path", "description"]).to_csv(REPORT / "artifact_manifest.csv", index=False)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    write_table(build_rq1(), TABLES / "rq1_detection_results.csv")
    rq2 = build_rq2()
    if not rq2.empty:
        write_table(rq2, TABLES / "rq2_reward_ablation.csv")
    write_table(build_rq3(), TABLES / "rq3_runtime_robustness.csv")
    copy_appendix()
    build_manifest()
    print(f"Wrote thesis report tables to {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

