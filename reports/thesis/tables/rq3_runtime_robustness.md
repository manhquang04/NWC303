| task | runtime_setting | episodes_or_windows | precision | recall_or_detection_rate | f1 | fpr_or_false_alert_rate | mean_detection_latency_sec | mean_mitigation_latency_sec | robustness_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARP Spoofing | controlled_sdn_stress_150 | 150 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.3623 | 0.3625 | Controlled SDN testbed success |
| ARP Spoofing | hardcore_sdn_evasion_benign_churn_120 | 120 | 0.6471 | 0.7857 | 0.7097 | 0.3750 | 0.3725 | 0.3727 | Hardcore limitations under evasion and benign churn |
| Rogue AP | normal_live_wifi | 300 |  |  |  | 0.0100 |  |  | Real USB Wi-Fi monitor-mode runtime; hybrid policy decision layer |
| Rogue AP | phone_hotspot_rogue | 300 |  | 1.0000 |  |  |  |  | Real USB Wi-Fi monitor-mode runtime; hybrid policy decision layer |
| Rogue AP | evil_twin_same_ssid | 300 |  | 1.0000 |  |  |  |  | Real USB Wi-Fi monitor-mode runtime; hybrid policy decision layer |
