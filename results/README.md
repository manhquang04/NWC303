# Results Directory

This directory now contains only the primary result artifacts needed by the root `README.md` and the thesis-ready package in `../reports/thesis/`.

Exploratory ablations and superseded result folders were removed during cleanup. The curated interpretation should be taken from `../reports/thesis/`; this directory preserves the underlying evidence used to build those tables.

## Current Result Folders

| Folder | Role | Used for |
|---|---|---|
| `sdn_runtime_arp_dqn_stress_150/` | Controlled ARP SDN runtime benchmark with 150 episodes. | RQ1, RQ3 |
| `sdn_runtime_arp_dqn_hardcore_120/` | Hardcore ARP runtime benchmark with evasive attacks and benign MAC/IP churn. | RQ1, RQ3, failure analysis |
| `rogue_ap_per_bssid_lofo/` | Best Rogue AP offline per-BSSID leave-one-file-out result. | RQ1 |
| `rogue_ap_runtime_rq_summary/` | Live Wi-Fi monitor-mode Rogue AP runtime summary. | RQ1, RQ3 |
| `hardcore_rq_suite_summary/` | Compact RQ-oriented source summaries used by the report builder. | RQ1, RQ2, RQ3 |

## Main Files to Inspect

| File | Meaning |
|---|---|
| `sdn_runtime_arp_dqn_stress_150/benchmark_summary.json` | Controlled ARP runtime summary. |
| `sdn_runtime_arp_dqn_hardcore_120/benchmark_summary.json` | Hardcore ARP runtime summary. |
| `sdn_runtime_arp_dqn_hardcore_120/variant_breakdown.csv` | ARP hard-case FN/FP breakdown by variant. |
| `rogue_ap_per_bssid_lofo/lofo_summary.csv` | Rogue AP LOFO model/policy summary. |
| `rogue_ap_runtime_rq_summary/rogue_ap_runtime_scenario_summary.csv` | Live Rogue AP runtime scenario summary. |
| `hardcore_rq_suite_summary/rq2_reward_summary.csv` | Reward ablation summary preserved for RQ2. |

## Important Caution

These artifacts should not be interpreted as independent claims without the framing in `../reports/thesis/README.md`. In particular:

- Controlled ARP `F1 = 1.0000` is a controlled-testbed result, not universal proof.
- Hardcore ARP results intentionally expose limitations.
- Rogue AP runtime alerts are hybrid policy decisions, not pure DQN/model-only alerts.

