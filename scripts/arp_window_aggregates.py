#!/usr/bin/env python3
"""Build ARP contiguous window aggregates for DRL/supervised sequence experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("processed/arp_spoofing_clean"))
    p.add_argument("--out-dir", type=Path, default=Path("processed/arp_window_aggregates"))
    p.add_argument("--window-size", type=int, default=500)
    p.add_argument("--label-threshold", type=float, default=0.01)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_split(base: Path, split: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted((base / split).glob("part-*.parquet"))], ignore_index=True)


def aggregate_split(df: pd.DataFrame, split: str, window_size: int, threshold: float) -> pd.DataFrame:
    features = [c for c in df.columns if c not in {"label", "row_id"}]
    rows = []
    for wid, start in enumerate(range(0, len(df), window_size)):
        g = df.iloc[start:start + window_size]
        if len(g) == 0:
            continue
        y = g["label"].to_numpy()
        ratio = float(y.mean())
        row = {
            "split": split,
            "window_id": wid,
            "row_start": int(g["row_id"].iloc[0]),
            "row_end": int(g["row_id"].iloc[-1]),
            "n_packets": int(len(g)),
            "attack_packets": int(y.sum()),
            "attack_ratio": ratio,
            "label": int(ratio >= threshold),
        }
        means = g[features].mean()
        stds = g[features].std(ddof=0).fillna(0)
        for f in features:
            row[f"{f}_mean"] = float(means[f])
            row[f"{f}_std"] = float(stds[f])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"window_size": args.window_size, "label_threshold": args.label_threshold, "splits": {}}
    for split in ["train", "val", "test"]:
        df = load_split(args.input_dir, split)
        out = aggregate_split(df, split, args.window_size, args.label_threshold)
        d = args.out_dir / split
        d.mkdir(parents=True, exist_ok=True)
        out.to_parquet(d / "part-00000.parquet", index=False)
        vc = out.label.value_counts().to_dict()
        meta["splits"][split] = {
            "windows": int(len(out)),
            "normal_windows": int(vc.get(0, 0)),
            "attack_windows": int(vc.get(1, 0)),
            "attack_packet_ratio": float(df.label.mean()),
            "mean_attack_ratio_positive_windows": float(out.loc[out.label == 1, "attack_ratio"].mean()) if (out.label == 1).any() else 0.0,
        }
    (args.out_dir / "window_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
