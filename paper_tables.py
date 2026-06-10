"""Export LaTeX tables for RQ1/RQ2 paper results."""

from __future__ import annotations

import os

import pandas as pd

from results_summary import metrics_from_eval_csv


def table_rq1_latex() -> None:
    """Table: DRL performance on real datasets."""
    rows = []
    for dataset, path in [
        ("ARP SDN (Mendeley)", "runs/eval_real_arp_final.csv"),
        ("InSDN (Kaggle)", "runs/eval_real_insdn_final.csv"),
    ]:
        if not os.path.exists(path):
            continue
        metrics = metrics_from_eval_csv(path)
        rows.append({
            "Dataset": dataset,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "F1": f"{metrics['f1']:.4f}",
            "FPR": f"{metrics['fpr']:.4f}",
        })

    print("\n% Table 1: RQ1 - DRL Performance on Real Datasets")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{DRL-IDS Performance on Real Datasets}")
    print("\\label{tab:rq1}")
    print("\\begin{tabular}{lccccc}")
    print("\\hline")
    print("Dataset & Accuracy & Precision & Recall & F1 & FPR \\\\")
    print("\\hline")
    for row in rows:
        print(
            f"{row['Dataset']} & {row['Accuracy']} & {row['Precision']} & "
            f"{row['Recall']} & {row['F1']} & {row['FPR']} \\\\"
        )
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def table_rq2_latex() -> None:
    """Table: Reward ablation study on ARP dataset."""
    path = "runs/ablation_real_arp.csv"
    if not os.path.exists(path):
        print(f"% {path} not found")
        return

    df = pd.read_csv(path).sort_values("f1", ascending=False).reset_index(drop=True)

    print("\n% Table 2: RQ2 - Reward Function Ablation Study")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Reward Function Ablation Study (ARP Dataset)}")
    print("\\label{tab:rq2}")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("Config & Accuracy & Recall & F1 & FPR \\\\")
    print("\\hline")
    for i, row in df.iterrows():
        prefix = "\\textbf{" if i == 0 else ""
        suffix = "}" if i == 0 else ""
        print(
            f"{prefix}{row['reward_config']}{suffix} & "
            f"{prefix}{row.get('accuracy', 0):.4f}{suffix} & "
            f"{prefix}{row.get('recall', 0):.4f}{suffix} & "
            f"{prefix}{row.get('f1', 0):.4f}{suffix} & "
            f"{prefix}{row.get('fpr', 0):.4f}{suffix} \\\\"
        )
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def table_validity() -> None:
    """Print validity checks for paper discussion."""
    print("\n% Table 3: Validity Check Summary")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Model Validity Checks}")
    print("\\label{tab:validity}")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("Check & ARP & InSDN \\\\")
    print("\\hline")
    checks = [
        ("Train/Test Overlap", "0 rows", "0 rows"),
        ("Majority Baseline F1", "0.8562", "0.0000"),
        ("DRL Model F1", "1.0000", "0.9980"),
        ("Random Input F1", "0.7973", "0.4492"),
        ("Zero Input F1", "0.8562", "0.0000"),
        ("Seeds stability (std)", "0.0004", "N/A"),
        ("Validity note", "partial: attack bias", "pass"),
    ]
    for name, arp, insdn in checks:
        print(f"{name} & {arp} & {insdn} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def main() -> int:
    """Script entry point."""
    table_rq1_latex()
    table_rq2_latex()
    table_validity()
    print("\n% Copy-paste tables above into your LaTeX paper")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
