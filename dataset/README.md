# Raw Dataset Area

This directory stores raw or source datasets. It is intentionally excluded from cleanup operations and should be backed up externally because some files are large.

## Current Sources

| Source | External reference | Notes |
|---|---|---|
| `ARP MitM_dataset-002.csv` | [Kitsune Network Attack Dataset](https://archive.ics.uci.edu/dataset/516/kitsune+network+attack+dataset) | Large ARP MITM source table used for ARP Spoofing experiments after auditing/cleaning. |
| `ARP MitM_labels.csv` | [Kitsune Network Attack Dataset](https://archive.ics.uci.edu/dataset/516/kitsune+network+attack+dataset) | Label source for the ARP MITM data. |
| `4.Rogue_AP/` | [AWID3 Dataset CSV](https://www.kaggle.com/datasets/suumia/awid3-dataset?select=CSV) | Rogue AP / 802.11 CSV source folder used to build frame, window, event, and per-BSSID datasets. |

## Dataset Rules

- Do not merge ARP MITM, Rogue AP, and UNSW-NB15 directly into one common feature table.
- Treat ARP and Rogue AP as separate tasks because they come from different capture layers and feature spaces.
- Preserve raw files when possible; rebuild cleaned artifacts into `processed/`.
- Keep raw datasets out of git if they are large or license-restricted.

