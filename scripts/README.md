# Scripts

This directory contains the offline experiment pipeline.

## Main Script Groups

| Script group | Purpose |
|---|---|
| `build_*`, `audit_*` | Dataset audit and export. |
| `rogue_ap_*windows*.py` | Rogue AP window construction and aggregation. |
| `train_rogue_ap_*.py` | Rogue AP supervised, DQN, hard-negative, and alternative model experiments. |
| `train_arp_*.py`, `arp_window_aggregates.py` | ARP supervised and DQN experiments. |
| `analyze_*` | RQ-oriented summaries and comparison tables. |

## Reproducibility

Most scripts write into `processed/` or `results/`. Prefer creating a new output directory for a new experiment rather than overwriting a prior result.

