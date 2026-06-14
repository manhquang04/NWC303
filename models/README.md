# Models

This directory stores exported model bundles used by runtime or report experiments.

| Folder | Purpose |
|---|---|
| `rogue_ap_runtime/` | Random Forest based Rogue AP runtime scorer and feature metadata used by `sdn_runtime/rogue_ap_live_sensor.py`. |

Runtime audit fields such as `source_file`, `file_id`, BSSID group IDs, or event IDs must not be included in model feature matrices unless explicitly part of a policy-only audit layer.

