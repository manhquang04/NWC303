#!/usr/bin/env python3
"""Build a clean ARP MitM dataset with contiguous block splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DROP_COLS = [f"f{i:03d}" for i in range(66, 80)]  # timestamp/large identifier-like from audit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, default=Path("dataset/ARP MitM_dataset-002.csv"))
    p.add_argument("--labels", type=Path, default=Path("/Users/manhquang/Downloads/kitsune+network+attack+dataset/arp_mitm/ARP MitM_labels.csv.gz"))
    p.add_argument("--out-dir", type=Path, default=Path("processed/arp_spoofing_clean"))
    p.add_argument("--chunksize", type=int, default=100_000)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def split_for_row(row_id: int) -> str:
    # Phase-block split, no random row split.
    # Benign background phase is split into contiguous train/val/test blocks.
    # Attack phase is also split into contiguous train/val/test blocks.
    # This avoids val/test becoming all-positive while still preserving time/block structure.
    if row_id < 900_000:
        return "train"
    if row_id < 1_100_000:
        return "val"
    if row_id < 1_300_000:
        return "test"
    if row_id < 1_902_560:
        return "train"
    if row_id < 2_203_414:
        return "val"
    return "test"


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test", "inspect"]:
        (args.out_dir / split).mkdir(parents=True, exist_ok=True)

    n_cols = pd.read_csv(args.features, header=None, nrows=1).shape[1]
    names = [f"f{i:03d}" for i in range(n_cols)]
    keep = [c for c in names if c not in DROP_COLS]
    labels = pd.read_csv(args.labels, usecols=["x"])["x"].astype("int8")

    buffers = {s: [] for s in ["train", "val", "test"]}
    part = {s: 0 for s in ["train", "val", "test"]}
    counts = {s: {0: 0, 1: 0, "rows": 0, "parts": 0} for s in ["train", "val", "test"]}
    row_start = 0
    for chunk in pd.read_csv(args.features, header=None, names=names, chunksize=args.chunksize, low_memory=False):
        n = len(chunk)
        chunk = chunk.apply(pd.to_numeric, errors="coerce").astype("float32")
        chunk["label"] = labels.iloc[row_start: row_start + n].to_numpy(dtype=np.int8)
        chunk["row_id"] = np.arange(row_start, row_start + n, dtype=np.int64)
        chunk["split"] = [split_for_row(int(i)) for i in chunk["row_id"]]
        for split, g in chunk.groupby("split", sort=False):
            out = g[keep + ["label", "row_id"]].reset_index(drop=True)
            path = args.out_dir / split / f"part-{part[split]:05d}.parquet"
            out.to_parquet(path, index=False)
            part[split] += 1
            vc = out["label"].value_counts().to_dict()
            counts[split][0] += int(vc.get(0, 0))
            counts[split][1] += int(vc.get(1, 0))
            counts[split]["rows"] += len(out)
            counts[split]["parts"] += 1
        row_start += n

    feature_schema = {
        "source": str(args.features),
        "labels": str(args.labels),
        "n_original_features": n_cols,
        "drop_cols": DROP_COLS,
        "kept_features": keep,
        "label_col": "label",
        "metadata_cols": ["row_id"],
        "split_strategy": "phase-block split: contiguous benign blocks and contiguous attack-phase blocks assigned to train/val/test; no random row split",
        "counts": counts,
    }
    (args.out_dir / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
    for split, meta in counts.items():
        (args.out_dir / split / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(feature_schema, indent=2))


if __name__ == "__main__":
    main()
