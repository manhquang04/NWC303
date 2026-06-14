# Processed Datasets

This directory contains cleaned or transformed datasets produced from the raw ARP and Rogue AP sources. The `dataset/` folder is left intact; this directory holds derived artifacts used by scripts and retained experiments.

## Required for Current README/Report

| Folder | Use |
|---|---|
| `arp_spoofing_clean/` | Cleaned ARP Spoofing data retained for reproducibility and possible rebuilds. |
| `arp_window_aggregates/` | ARP window-level aggregates used by retained reward/RQ summaries. |
| `rogue_ap_per_bssid_windows/` | Per-BSSID/per-transmitter Rogue AP windows used for the strongest LOFO evaluation. |

## Additional Processed Artifacts

The following may still exist as intermediate datasets from earlier stages:

| Folder/file | Use |
|---|---|
| `rogue_ap_event_windows/` | Event-centered Rogue AP windows used during DQN exploration. |
| `rogue_ap_with_source_metadata/` | Frame-level Rogue AP data with source metadata for audit only. |
| `rogue_ap_timing_dropped/` | Frame-level Rogue AP dataset with suspicious timing fields removed. |
| `rogue_ap_timing_kept/` | Timing-kept ablation dataset. Kept only if additional ablation is needed. |
| `rogue_ap_window_aggregates/` | Aggregate window features from earlier Rogue AP experiments. |
| `rogue_ap_window_grid/` | Window-size/threshold grid artifacts from earlier ablation. |

## Leakage Rule

Columns such as `source_file`, `file_id`, event IDs, row ranges, labels, or other audit-only fields must not be used as model features. They are allowed only for grouping, audit, per-file metrics, or runtime policy logging.

