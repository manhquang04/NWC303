# SDN DRL-IDS for Rogue AP and ARP Spoofing

Research repository for a software-defined networking intrusion detection study on ARP Spoofing and Rogue Access Point detection. The project combines controlled SDN runtime experiments, 802.11 monitor-mode Rogue AP validation, supervised baselines, DRL/DQN reward studies, and robustness stress testing.

This README is the main entry point for the project. The thesis-ready result package is in [`reports/thesis/`](reports/thesis/).

## Abstract

The project investigates whether a DRL/SDN-based intrusion detection system can distinguish normal network behavior from ARP Spoofing and Rogue AP attacks, and whether reward shaping can improve decision policies under realistic runtime constraints. The final evaluation separates two attack domains instead of forcing incompatible feature spaces into one common model:

- **ARP Spoofing** is evaluated in a Mininet/Ryu SDN runtime with online mitigation.
- **Rogue AP** is evaluated using 802.11/radiotap datasets, per-BSSID windowing, leave-one-file-out generalization, and live USB Wi-Fi monitor-mode validation.

The strongest result is not a claim of universal perfect detection. Instead, the project provides controlled success cases, hard-case failure analysis, reward-policy trade-offs, and runtime robustness boundaries suitable for scientific reporting.

## Research Questions

| ID | Research question | Current answer |
|---|---|---|
| RQ1 | Can the proposed DRL/SDN-based detector distinguish normal behavior from Rogue AP and ARP Spoofing attacks in a dynamic SDN environment? | Yes, with qualifications. ARP Spoofing is detected very strongly in controlled SDN runtime. Rogue AP is strongest with a fingerprint-aware monitor-mode policy rather than pure DQN. Hardcore ARP testing exposes realistic failure modes. |
| RQ2 | Which reward function effectively encourages a DRL agent to learn useful policies for Rogue AP and ARP Spoofing handling? | Reward shaping clearly controls recall/FPR behavior. Aggressive rewards improve recall and F1 but increase false alarms; FPR-constrained rewards reduce false alarms but may collapse recall. No reward is universally optimal. |
| RQ3 | Does the system maintain stable performance and robustness when moving from simulation/testbed conditions to runtime traffic with noise and latency? | Partially. Controlled ARP SDN runtime is highly stable; hardcore ARP runtime shows robustness boundaries. Rogue AP live Wi-Fi tests validate a hybrid monitor-mode policy in real traffic, not a pure DRL transfer claim. |

## Project Structure

| Path | Purpose |
|---|---|
| [`reports/thesis/`](reports/thesis/) | Curated thesis-ready tables, appendix files, artifact manifest, and final RQ interpretation. Start here for writing. |
| [`results/`](results/) | Full experiment outputs, including runtime logs, ablations, training curves, summaries, and figures. |
| [`processed/`](processed/) | Cleaned and transformed datasets, including Rogue AP windows and ARP aggregates. |
| [`dataset/`](dataset/) | Raw/source datasets. Large files are ignored by git and should be backed up externally. |
| [`scripts/`](scripts/) | Offline audit, preprocessing, model training, ablation, and report-building scripts. |
| [`sdn_runtime/`](sdn_runtime/) | Ryu/Mininet runtime code, ARP guard controller, Rogue AP live sensor, and digital-twin helpers. |
| [`models/`](models/) | Exported runtime model bundles. |

## Methodology Overview

### 1. Dataset Governance

The project intentionally avoids merging ARP MITM, Rogue AP, and UNSW-NB15 style data into one flat table because they represent different capture layers and feature spaces.

| Data source | Layer / feature space | Final use |
|---|---|---|
| [Kitsune Network Attack Dataset](https://archive.ics.uci.edu/dataset/516/kitsune+network+attack+dataset) / ARP MITM / ARP runtime | ARP/flow/window/runtime SDN events | ARP Spoofing detection and mitigation experiments. |
| [AWID3 Dataset CSV](https://www.kaggle.com/datasets/suumia/awid3-dataset?select=CSV) / Rogue AP CSV / 802.11 radiotap | Wireless frame/window/per-BSSID features | Rogue AP offline and live monitor-mode experiments. |
| UNSW-NB15 style normal/benign | IP flow level | Not merged into the final Rogue AP or ARP pipelines. |

Important leakage controls:

- `source_file`, `file_id`, `event_id`, row ranges, and labels are audit metadata only.
- Random row splitting is avoided for Rogue AP where file/session leakage is a concern.
- Timing and source-identity features were ablated or dropped where they created leakage risk.
- Rogue AP live runtime uses policy audit fields for decision logging, not as supervised training features.

### 2. ARP Spoofing Runtime Pipeline

The ARP runtime experiment uses Mininet and Ryu:

1. A Mininet topology runs victim, gateway/target, and attacker hosts.
2. `arp_guard_controller.py` observes ARP sender IP-to-MAC behavior.
3. A DQN-style policy chooses allow, flag, or block.
4. For block decisions, the controller installs OpenFlow drop rules.
5. Runtime events are written to CSV/JSON for episode-level evaluation.

Main files:

- [`sdn_runtime/arp_guard_controller.py`](sdn_runtime/arp_guard_controller.py)
- [`sdn_runtime/run_arp_sdn_runtime.py`](sdn_runtime/run_arp_sdn_runtime.py)
- [`sdn_runtime/run_arp_sdn_runtime_benchmark.py`](sdn_runtime/run_arp_sdn_runtime_benchmark.py)

### 3. Rogue AP Pipeline

Rogue AP evaluation is treated as a wireless monitor-mode detection problem:

1. Frame-level CSVs are audited and cleaned.
2. Source metadata is preserved for per-file metrics but excluded from feature matrices.
3. Window/event/per-BSSID features are built.
4. Supervised and DQN/DRL variants are evaluated.
5. Live runtime is validated with a USB Wi-Fi adapter in monitor mode.

The most reliable Rogue AP result currently comes from per-BSSID/per-transmitter feature rebuild plus leave-one-file-out evaluation. Runtime validation uses hybrid monitor-mode policy decisions, especially BSSID/fingerprint allowlist logic.

Main files:

- [`scripts/rogue_ap_per_bssid_lofo.py`](scripts/rogue_ap_per_bssid_lofo.py)
- [`scripts/train_rogue_ap_dqn.py`](scripts/train_rogue_ap_dqn.py)
- [`scripts/train_rogue_ap_dqn_hard_negatives.py`](scripts/train_rogue_ap_dqn_hard_negatives.py)
- [`sdn_runtime/rogue_ap_live_sensor.py`](sdn_runtime/rogue_ap_live_sensor.py)

## Methods Evaluated

The project evaluated multiple families of methods before selecting the final thesis framing. Not every method is used as the final detector; several are retained as ablations or evidence for RQ2/RQ3.

| Category | Methods evaluated | Purpose | Final role |
|---|---|---|---|
| Supervised baselines | Logistic Regression, Random Forest, XGBoost, MLP | Establish non-DRL reference performance and identify whether the task is separable without RL. | Used as core baselines; XGBoost/RF remain stronger and easier to explain than pure DQN for Rogue AP. |
| Rogue AP frame-level models | Timing-kept, timing-dropped, strict feature ablation, radiotap/PHY ablation | Detect leakage/source artifacts and check whether near-perfect scores came from frame-level shortcuts. | Frame-level results are treated cautiously; they motivated stricter feature design. |
| Rogue AP window/event models | Fixed contiguous windows, event-centered windows, conservative/moderate/rich aggregates | Reduce dependence on single-frame artifacts and evaluate session/window-level behavior. | Useful intermediate stage; `tight_event` was used as the practical RL starting point. |
| Rogue AP per-BSSID/per-transmitter models | Per-BSSID aggregation, leave-one-file-out evaluation, XGBoost/RF | Test generalization by transmitter/file and reduce source leakage. | Best offline Rogue AP evidence. Current strongest result: XGBoost LOFO with low FPR and high recall. |
| Alternative Rogue AP learning | Temporal sequence model, Autoencoder/anomaly detection, synthetic sequence augmentation | Explore whether temporal or unsupervised methods improve over tree-based baselines. | Useful robustness checks, but not selected as the main detector because performance was stable but not stronger. |
| DRL/DQN for Rogue AP | DQN, Double DQN, Dueling DQN, hard-negative mining | Study reward-shaped decision policies and false-alarm trade-offs. | Used mainly for RQ2 analysis. DQN did not replace supervised/fingerprint-aware policy. |
| ARP supervised/offline models | Logistic Regression, Random Forest, XGBoost, ARP window aggregates | Compare offline feature-table and window-level detection against runtime SDN policy. | Provides supporting baselines; offline models are not the strongest final ARP evidence. |
| ARP DQN/reward ablation | Conservative, aggressive, recall-prioritized, FPR-constrained rewards | Evaluate how reward design changes ARP decision behavior. | Supports RQ2; conservative reward is best by F1 in ARP window DQN but FPR remains high. |
| SDN runtime policy | Ryu ARP guard, DQN-style policy, OpenFlow drop-rule mitigation | Validate online detection and mitigation under dynamic SDN traffic. | Main ARP runtime evidence for RQ1/RQ3. |
| Hardcore runtime stress | Evasive ARP attacks, benign MAC/IP churn, burst background traffic | Test whether perfect controlled results survive harder, more realistic cases. | Key robustness evidence; exposes realistic limitations and prevents overclaiming. |
| Rogue AP live runtime | USB Wi-Fi monitor mode, `tshark` frame extraction, BSSID/fingerprint policy | Validate Rogue AP detection with real Wi-Fi traffic, phone hotspot, and same-SSID evil twin. | Main Rogue AP runtime evidence; framed as hybrid monitor-mode policy, not pure DQN. |
| Decision policy layer | Single threshold, dual threshold, moving-average smoothing, FPR-constrained operating points | Select deployment operating points without tuning on test data. | Retained as the practical decision layer for stable supervised/hybrid detection. |

Overall, the final thesis framing uses:

- **ARP Spoofing:** SDN runtime detection and mitigation as the primary result, with hardcore stress testing to expose limits.
- **Rogue AP:** per-BSSID/fingerprint-aware supervised detection and live monitor-mode policy validation as the primary result.
- **DRL/DQN:** reward and decision-policy analysis, not a universal replacement for supervised or policy-based detectors.

## Thesis-Ready Results

The curated report package is:

```text
reports/thesis/
├── README.md
├── artifact_manifest.csv
├── tables/
│   ├── rq1_detection_results.csv
│   ├── rq1_detection_results.md
│   ├── rq2_reward_ablation.csv
│   ├── rq2_reward_ablation.md
│   ├── rq3_runtime_robustness.csv
│   └── rq3_runtime_robustness.md
└── appendix/
    ├── arp_hardcore_background_breakdown.csv
    ├── arp_hardcore_variant_breakdown.csv
    ├── rogue_ap_lofo_summary.csv
    └── rogue_ap_runtime_scenarios.csv
```

To rebuild the curated tables from existing result artifacts:

```bash
python3 scripts/build_thesis_report_tables.py
```

This command does not retrain models or rerun runtime experiments. It only reads existing validated artifacts and regenerates the compact thesis tables.

## Main Results

### RQ1: Detection Results

| Task | Experiment | Setting | Precision | Recall | F1 | FPR | AUROC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ARP Spoofing | Controlled SDN runtime | 150 episodes | 1.0000 | 1.0000 | 1.0000 | 0.0000 | - | - |
| ARP Spoofing | Hardcore SDN runtime | 120 episodes | 0.6471 | 0.7857 | 0.7097 | 0.3750 | - | - |
| Rogue AP | Per-BSSID XGBoost LOFO | 17 held-out files | 0.5532 | 0.9409 | 0.6745 | 0.0561 | 0.9786 | 0.7103 |
| Rogue AP | Live normal Wi-Fi | 300 windows | - | - | - | 0.0100 alert rate | - | - |
| Rogue AP | Live phone hotspot Rogue AP | 300 windows | - | 1.0000 detection | - | - | - | - |
| Rogue AP | Live evil twin same SSID | 300 windows | - | 1.0000 detection | - | - | - | - |

Interpretation:

- Controlled ARP runtime demonstrates strong SDN mitigation under expected attacks.
- Hardcore ARP runtime is more scientifically useful for robustness because it reveals failure modes.
- Rogue AP detection is most reliable when treated as fingerprint-aware wireless monitoring rather than pure DQN classification.

### RQ2: Reward Ablation

| Task | Reward / agent | Precision | Recall | F1 | FPR | Interpretation |
|---|---|---:|---:|---:|---:|---|
| Rogue AP | DQN `aggressive_reward` | 0.4868 | 0.8237 | 0.6119 | 0.4342 | Best DQN F1/recall, high false alarm rate. |
| Rogue AP | Hardened DQN `fpr_constrained_reward` | 0.6217 | 0.3192 | 0.4218 | 0.0971 | Lowest FPR, recall collapses. |
| Rogue AP | Hardened DQN `recall_prioritized_reward` | 0.5621 | 0.4040 | 0.4701 | 0.1574 | More balanced than strict FPR reward. |
| ARP | DQN window `conservative_reward` | 0.6829 | 0.9551 | 0.7964 | 0.6675 | Best ARP offline DQN F1, still high FPR. |

Interpretation:

- Reward functions successfully shape policy behavior.
- High recall and low FPR are in tension.
- DRL is valuable for studying decision policies, but the deployment candidate should remain a constrained supervised/policy hybrid unless further robustness work improves DRL stability.

### RQ3: Runtime Robustness

| Task | Runtime setting | Episodes/windows | Detection / recall | FPR / false alert rate | Latency |
|---|---|---:|---:|---:|---:|
| ARP Spoofing | Controlled SDN runtime | 150 episodes | 1.0000 | 0.0000 | 0.3623s detection |
| ARP Spoofing | Hardcore SDN runtime | 120 episodes | 0.7857 | 0.3750 | 0.3725s detection |
| Rogue AP | Normal live Wi-Fi | 300 windows | - | 0.0100 alert rate | - |
| Rogue AP | Phone hotspot Rogue AP | 300 windows | 1.0000 | - | - |
| Rogue AP | Evil twin same SSID | 300 windows | 1.0000 | - | - |

Hardcore ARP failure modes:

| Error type | Variant | Observation |
|---|---|---|
| False negative | `request_poison` | 8/8 missed across backgrounds. |
| False negative | `unicast_reply` | 4/8 missed, mainly in burst/mixed backgrounds. |
| False positive | `gateway_failover` | Legitimate failover-like MAC change flagged as spoofing. |
| False positive | `ip_reassignment` | Benign IP/MAC reassignment flagged as spoofing. |
| False positive | `arp_storm` | ARP-heavy benign behavior flagged as attack. |

Interpretation:

- The runtime system is effective in controlled SDN experiments.
- Robustness is not universal; context-aware ARP state tracking is needed for benign churn.
- Rogue AP live runtime evidence is real Wi-Fi monitor-mode evidence, but it should be reported as hybrid policy validation rather than pure DRL transfer.

## How to Run

This project has two execution modes:

- **Local analysis mode:** rebuilds report tables from existing artifacts and runs offline scripts.
- **Runtime experiment mode:** runs Mininet/Ryu or live Wi-Fi monitor-mode experiments, usually inside the paired Ubuntu VM.

### 1. Local quick check

From the project root:

```bash
python3 scripts/build_thesis_report_tables.py
python3 -m py_compile scripts/build_thesis_report_tables.py \
  sdn_runtime/run_arp_sdn_runtime.py \
  sdn_runtime/run_arp_sdn_runtime_benchmark.py
```

Expected output:

```text
Wrote thesis report tables to reports/thesis
```

This verifies that the curated report package can be regenerated without rerunning expensive experiments.

### 2. Rebuild thesis-ready tables

```bash
python3 scripts/build_thesis_report_tables.py
```

Outputs:

```text
reports/thesis/tables/rq1_detection_results.csv
reports/thesis/tables/rq2_reward_ablation.csv
reports/thesis/tables/rq3_runtime_robustness.csv
reports/thesis/appendix/
reports/thesis/artifact_manifest.csv
```

### 3. Run ARP controlled SDN benchmark

This should be run in the Ubuntu VM with Mininet and Ryu installed.

VM access pattern used during the experiments:

```bash
ssh -p 22022 manhquang@localhost
cd /home/manhquang/Downloads/NWC303_LOCAL_RUNTIME
```

Run the controlled stress benchmark:

```bash
sudo python3 sdn_runtime/run_arp_sdn_runtime_benchmark.py \
  --attack-episodes 75 \
  --normal-episodes 75 \
  --duration 14 \
  --stress-grid \
  --controller-cmd "/home/manhquang/Downloads/NWC303/.venv311/bin/ryu-manager sdn_runtime/arp_guard_controller.py" \
  --out-dir results/sdn_runtime_arp_dqn_stress_150
```

Expected outputs:

```text
results/sdn_runtime_arp_dqn_stress_150/benchmark_summary.json
results/sdn_runtime_arp_dqn_stress_150/benchmark_metrics.csv
results/sdn_runtime_arp_dqn_stress_150/scenario_breakdown.csv
results/sdn_runtime_arp_dqn_stress_150/variant_breakdown.csv
```

### 4. Run ARP hardcore robustness benchmark

This benchmark is intentionally harder and should not be expected to produce perfect scores.

```bash
sudo python3 sdn_runtime/run_arp_sdn_runtime_benchmark.py \
  --attack-episodes 56 \
  --normal-episodes 64 \
  --duration 12 \
  --hardcore-grid \
  --controller-cmd "/home/manhquang/Downloads/NWC303/.venv311/bin/ryu-manager sdn_runtime/arp_guard_controller.py" \
  --out-dir results/sdn_runtime_arp_dqn_hardcore_120
```

Hardcore cases include:

- Evasive attacks: `request_poison`, `unicast_reply`, `burst_then_sleep`.
- Benign churn: `gateway_failover`, `ip_reassignment`, `arp_storm`.
- Background modes: `quiet`, `light`, `mixed`, `burst`.

Expected outputs:

```text
results/sdn_runtime_arp_dqn_hardcore_120/benchmark_summary.json
results/sdn_runtime_arp_dqn_hardcore_120/benchmark_metrics.csv
results/sdn_runtime_arp_dqn_hardcore_120/scenario_breakdown.csv
results/sdn_runtime_arp_dqn_hardcore_120/variant_breakdown.csv
```

If results are produced on the VM, copy them back to the local project before writing the final report.

### 5. Run Rogue AP live monitor-mode sensor

Rogue AP runtime validation requires:

- A Wi-Fi adapter that supports monitor mode.
- `tshark`.
- The exported runtime model in `models/rogue_ap_runtime/`.
- A known authorized BSSID list for policy-based unknown-BSSID detection.

Example command pattern:

```bash
sudo python3 sdn_runtime/rogue_ap_live_sensor.py \
  --interface wlx60313bd892bc \
  --model models/rogue_ap_runtime/rogue_ap_runtime_rf.joblib \
  --window-frames 100 \
  --stride-frames 10 \
  --max-windows 300 \
  --authorized-bssid 4c:12:e8:fc:47:50 \
  --flag-unknown-bssid \
  --out-dir results/rogue_ap_runtime_new_test \
  --dry-run
```

Expected outputs:

```text
results/rogue_ap_runtime_new_test/rogue_ap_runtime_summary.json
results/rogue_ap_runtime_new_test/rogue_ap_runtime_windows.csv
results/rogue_ap_runtime_new_test/rogue_ap_runtime_alerts.jsonl
```

Important interpretation:

- `model_flag` indicates the supervised runtime model fired.
- `policy_flag` indicates the BSSID/fingerprint policy fired.
- In the current live Rogue AP evidence, alerts are policy-driven, not model-driven.

### 6. Run offline Rogue AP LOFO evaluation

The strongest offline Rogue AP result is already stored in:

```text
results/rogue_ap_per_bssid_lofo/
```

To rerun the evaluation, use:

```bash
python3 scripts/rogue_ap_per_bssid_lofo.py
```

Expected outputs:

```text
results/rogue_ap_per_bssid_lofo/lofo_summary.csv
results/rogue_ap_per_bssid_lofo/lofo_metrics_by_file.csv
results/rogue_ap_per_bssid_lofo/lofo_ablation_summary.csv
```

### 7. What not to do

For this research design, avoid:

- Random row splits for Rogue AP.
- Training with `source_file`, `file_id`, `event_id`, labels, row ranges, or other audit-only metadata.
- Directly merging ARP MITM, Rogue AP, and UNSW-NB15 into one common feature table.
- Reporting controlled ARP `F1 = 1.0000` as universal real-world proof.
- Tuning thresholds or rewards on the test set.

### 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ryu-manager` not found | VM virtual environment not active or wrong path | Use the full `.venv311/bin/ryu-manager` path in `--controller-cmd`. |
| Mininet errors or stale switches | Previous Mininet state remains | Run `sudo mn -c` before rerunning runtime benchmarks. |
| Rogue AP sensor sees no frames | Adapter not in monitor mode or wrong interface | Check `iw dev`, set monitor mode, and verify with `tcpdump`/`tshark`. |
| Rogue AP live alerts are all policy-based | Expected in current runtime setup | Report as hybrid monitor-mode policy, not pure ML detection. |
| ARP controlled score is perfect | Controlled scenario is too clean | Use `--hardcore-grid` for robustness evidence. |
| `source_file` appears in model features | Leakage risk | Remove audit metadata from the feature matrix before training. |

## Reproducibility

### Rebuild thesis tables

```bash
python3 scripts/build_thesis_report_tables.py
```

### Controlled ARP benchmark

The primary controlled benchmark artifact is already present:

```text
results/sdn_runtime_arp_dqn_stress_150/
```

The VM runtime command used this pattern:

```bash
sudo python3 sdn_runtime/run_arp_sdn_runtime_benchmark.py \
  --attack-episodes 75 \
  --normal-episodes 75 \
  --duration 14 \
  --stress-grid \
  --controller-cmd "/home/manhquang/Downloads/NWC303/.venv311/bin/ryu-manager sdn_runtime/arp_guard_controller.py" \
  --out-dir results/sdn_runtime_arp_dqn_stress_150
```

### Hardcore ARP benchmark

The hardcore benchmark artifact is already present:

```text
results/sdn_runtime_arp_dqn_hardcore_120/
```

The benchmark includes:

- Attack variants: `standard`, `slow`, `intermittent`, `random_mac`, `request_poison`, `unicast_reply`, `burst_then_sleep`.
- Benign variants: `standard`, `gratuitous`, `dhcp_like`, `arp_scan`, `noisy_gratuitous`, `gateway_failover`, `ip_reassignment`, `arp_storm`.
- Background modes: `quiet`, `light`, `mixed`, `burst`.

Command pattern:

```bash
sudo python3 sdn_runtime/run_arp_sdn_runtime_benchmark.py \
  --attack-episodes 56 \
  --normal-episodes 64 \
  --duration 12 \
  --hardcore-grid \
  --controller-cmd "/home/manhquang/Downloads/NWC303/.venv311/bin/ryu-manager sdn_runtime/arp_guard_controller.py" \
  --out-dir results/sdn_runtime_arp_dqn_hardcore_120
```

### Rogue AP runtime

Rogue AP live validation uses a USB Wi-Fi adapter in monitor mode and `tshark` frame extraction. The runtime model/policy bundle is in:

```text
models/rogue_ap_runtime/
```

The runtime scenario summary is:

```text
results/rogue_ap_runtime_rq_summary/rogue_ap_runtime_scenario_summary.csv
```

## Primary Artifacts

| Artifact | Purpose |
|---|---|
| `reports/thesis/` | Final curated reporting package. |
| `results/sdn_runtime_arp_dqn_stress_150/` | Controlled ARP SDN runtime evidence. |
| `results/sdn_runtime_arp_dqn_hardcore_120/` | Hardcore ARP robustness evidence. |
| `results/rogue_ap_per_bssid_lofo/` | Best Rogue AP offline LOFO evidence. |
| `results/rogue_ap_runtime_rq_summary/` | Rogue AP live Wi-Fi runtime summary. |
| `results/hardcore_rq_suite_summary/` | Compact RQ-oriented summary. |
| `models/rogue_ap_runtime/` | Runtime Rogue AP model bundle. |

## Limitations

This repository does not support a universal claim that a DRL model perfectly detects all Rogue AP and ARP Spoofing attacks. Current limitations are:

- ARP F1 = 1.0000 is valid only for controlled SDN runtime settings.
- Hardcore ARP testing exposes false negatives for request-based poisoning and false positives under benign MAC/IP churn.
- Rogue AP runtime decisions are currently policy-driven; model scores alone did not trigger the live alerts.
- Rogue AP results are strongest under per-BSSID/fingerprint-aware framing, not purely behavior-only generalization.
- More real deployments, additional AP hardware, and longer multi-day traffic traces would be needed for stronger external validity.

## Recommended Thesis Claim

The proposed system achieves strong detection and mitigation in controlled SDN runtime experiments, especially for ARP Spoofing. However, hardcore robustness testing reveals realistic limitations under evasive ARP variants and benign MAC/IP churn. For Rogue AP detection, the most reliable evidence comes from fingerprint-aware monitor-mode detection with a policy layer rather than pure DQN classification. Reward shaping is effective for controlling recall/FPR trade-offs, but no single reward function is universally optimal across both attack types.

## Maintenance Notes

- Keep raw datasets in `dataset/` backed up externally.
- Treat `reports/thesis/` as the clean reporting interface.
- Do not delete primary result folders unless they are archived elsewhere.
- Use `scripts/build_thesis_report_tables.py` after new experiments to refresh curated RQ tables.
- Avoid random row splits for Rogue AP data; preserve file/session-level evaluation.
