# Thesis Result Report: Rogue AP and ARP Spoofing SDN DRL-IDS

This folder is the curated, thesis-ready result package for the three research questions. It is intentionally compact and should be used as the main source for writing the results chapter.

The full project contains raw datasets, processed datasets, runtime code, and selected primary result artifacts. Exploratory results have been cleaned from `../../results/`; the remaining result folders are the primary evidence supporting this report.

## Research Questions

| ID | Question | Final interpretation |
|---|---|---|
| RQ1 | Can the system distinguish normal traffic from Rogue AP and ARP Spoofing attacks in a dynamic SDN environment? | Yes, but not universally. ARP is highly effective in controlled SDN runtime and less robust under hardcore evasion. Rogue AP is strongest with per-BSSID/fingerprint-aware detection and a runtime policy layer. |
| RQ2 | Which reward function drives effective DRL policy learning? | Reward design changes behavior clearly. Aggressive rewards improve recall/F1; FPR-constrained rewards reduce false alarms but can sacrifice recall. No single reward is universally optimal. |
| RQ3 | Does performance remain stable when moving toward realistic runtime traffic? | Controlled ARP remains strong; hardcore ARP exposes robustness limits. Rogue AP has live monitor-mode validation, but it is hybrid policy evidence rather than pure DQN runtime transfer. |

## Tables

| File | Use |
|---|---|
| `tables/rq1_detection_results.csv` | Main RQ1 result table. |
| `tables/rq1_detection_results.md` | Markdown version of the RQ1 table. |
| `tables/rq2_reward_ablation.csv` | Main RQ2 reward comparison table. |
| `tables/rq2_reward_ablation.md` | Markdown version of the RQ2 table. |
| `tables/rq3_runtime_robustness.csv` | Main RQ3 runtime robustness table. |
| `tables/rq3_runtime_robustness.md` | Markdown version of the RQ3 table. |

## Appendix Tables

| File | Use |
|---|---|
| `appendix/arp_hardcore_variant_breakdown.csv` | Shows which ARP attack/benign variants cause FN/FP. |
| `appendix/arp_hardcore_background_breakdown.csv` | Shows performance under quiet/light/mixed/burst backgrounds. |
| `appendix/rogue_ap_lofo_summary.csv` | Rogue AP per-BSSID LOFO summary. |
| `appendix/rogue_ap_runtime_scenarios.csv` | Live Wi-Fi Rogue AP runtime scenarios. |

## Main Results

### RQ1

| task | experiment | setting | precision | recall | f1 | fpr | auroc | pr_auc | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARP Spoofing | ARP controlled SDN runtime | 150 episodes (75 attack, 75 normal) | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |  | Controlled success |
| ARP Spoofing | ARP hardcore SDN runtime | 120 episodes (56 attack, 64 normal) | 0.6471 | 0.7857 | 0.7097 | 0.3750 |  |  | Hardcore robustness boundary |
| Rogue AP | Per-BSSID XGBoost LOFO | 17 held-out positive files; 16487 test windows | 0.5532 | 0.9409 | 0.6745 | 0.0561 | 0.9786 | 0.7103 | Best offline Rogue AP result; source_file not used as feature |
| Rogue AP | Live Wi-Fi runtime: normal_live_wifi | 300 monitor-mode windows |  |  |  | 0.0100 |  |  | Hybrid monitor-mode policy; model_alerts=0, policy_alerts=3 |
| Rogue AP | Live Wi-Fi runtime: phone_hotspot_rogue | 300 monitor-mode windows |  | 1.0000 |  |  |  |  | Hybrid monitor-mode policy; model_alerts=0, policy_alerts=300 |
| Rogue AP | Live Wi-Fi runtime: evil_twin_same_ssid | 300 monitor-mode windows |  | 1.0000 |  |  |  |  | Hybrid monitor-mode policy; model_alerts=0, policy_alerts=300 |

### RQ2

The highest Rogue AP DQN F1 is from `aggressive_reward`, but it has high FPR. The lowest FPR comes from FPR-constrained reward, but recall drops substantially. For ARP window-level DQN, `conservative_reward` is strongest by F1, but still has high FPR.

See `tables/rq2_reward_ablation.csv`.

### RQ3

Controlled ARP runtime is strong, but hardcore runtime is the key robustness evidence. The detector fails on request-based poison variants and produces false positives on benign MAC/IP churn.

See `tables/rq3_runtime_robustness.csv` and `appendix/arp_hardcore_variant_breakdown.csv`.

## Rebuild This Report

From the project root:

```bash
python3 scripts/build_thesis_report_tables.py
```

The report builder reads only the retained primary artifacts under `../../results/` and writes the tables in this folder.

## Recommended Thesis Wording

The system performs strongly in controlled SDN runtime experiments, especially for ARP Spoofing, but hardcore testing reveals realistic robustness limits under evasive ARP variants and benign MAC/IP churn. For Rogue AP detection, the most reliable approach is a fingerprint-aware monitor-mode policy rather than a pure DQN detector. Reward shaping influences recall/FPR trade-offs, but no reward is universally optimal across both attack types.

