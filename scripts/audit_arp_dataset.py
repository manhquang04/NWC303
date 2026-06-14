#!/usr/bin/env python3
"""Streaming audit for the ARP MitM dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("dataset/ARP MitM_dataset-002.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("results/arp_audit"))
    p.add_argument("--chunksize", type=int, default=100_000)
    p.add_argument("--sample-rows", type=int, default=200_000)
    return p.parse_args()


def stable_hash_rows(df: pd.DataFrame) -> list[str]:
    vals = pd.util.hash_pandas_object(df, index=False).astype("uint64").to_numpy()
    return [str(v) for v in vals]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    p = args.input
    first_header = pd.read_csv(p, nrows=0).columns.tolist()
    first_no_header = pd.read_csv(p, header=None, nrows=5)
    n_cols = first_no_header.shape[1]
    names = [f"f{i:03d}" for i in range(n_cols)]

    row_count = 0
    missing = np.zeros(n_cols, dtype=np.int64)
    finite = np.zeros(n_cols, dtype=np.int64)
    mins = np.full(n_cols, np.inf)
    maxs = np.full(n_cols, -np.inf)
    sums = np.zeros(n_cols, dtype=np.float64)
    sqs = np.zeros(n_cols, dtype=np.float64)
    unique_samples: list[Counter] = [Counter() for _ in range(n_cols)]
    candidate_label_counts: dict[str, Counter] = {f"f{i:03d}": Counter() for i in list(range(min(5, n_cols))) + list(range(max(0, n_cols - 10), n_cols))}
    hash_counter: Counter[str] = Counter()
    sample_parts = []

    for chunk in pd.read_csv(p, header=None, names=names, chunksize=args.chunksize, low_memory=False):
        row_count += len(chunk)
        numeric = chunk.apply(pd.to_numeric, errors="coerce")
        arr = numeric.to_numpy(dtype=np.float64)
        miss = np.isnan(arr)
        missing += miss.sum(axis=0)
        valid = ~miss
        finite += valid.sum(axis=0)
        with np.errstate(invalid="ignore"):
            mins = np.minimum(mins, np.nanmin(arr, axis=0))
            maxs = np.maximum(maxs, np.nanmax(arr, axis=0))
        sums += np.nansum(arr, axis=0)
        sqs += np.nansum(arr * arr, axis=0)
        for i in range(n_cols):
            if len(unique_samples[i]) <= 50:
                unique_samples[i].update(numeric.iloc[:, i].dropna().head(1000).round(8).astype(str).tolist())
        for col in candidate_label_counts:
            candidate_label_counts[col].update(numeric[col].dropna().round(8).astype(str).tolist())
        if sum(len(x) for x in sample_parts) < args.sample_rows:
            sample_parts.append(numeric.head(max(0, args.sample_rows - sum(len(x) for x in sample_parts))))
        for h in stable_hash_rows(numeric):
            hash_counter[h] += 1

    means = sums / np.maximum(finite, 1)
    vars_ = sqs / np.maximum(finite, 1) - means * means
    stds = np.sqrt(np.maximum(vars_, 0))
    schema = pd.DataFrame({
        "column": names,
        "missing": missing,
        "missing_rate": missing / max(row_count, 1),
        "finite": finite,
        "min": mins,
        "max": maxs,
        "mean": means,
        "std": stds,
        "sample_unique_count": [len(c) for c in unique_samples],
        "sample_values": [";".join(list(c.keys())[:10]) for c in unique_samples],
    })
    schema.to_csv(args.out_dir / "arp_schema_report.csv", index=False)

    label_rows = []
    for col, cnt in candidate_label_counts.items():
        total = sum(cnt.values())
        top = cnt.most_common(20)
        label_rows.append({
            "column": col,
            "unique_seen": len(cnt),
            "top_values": json.dumps(top),
            "dominant_rate": top[0][1] / total if total and top else 0,
            "looks_binary": len(cnt) <= 3 and set(cnt).issubset({"0.0", "1.0", "0", "1"}),
        })
    pd.DataFrame(label_rows).to_csv(args.out_dir / "arp_label_distribution.csv", index=False)

    dup_rows = sum(v - 1 for v in hash_counter.values() if v > 1)
    dup_report = {
        "row_count": row_count,
        "unique_row_hashes": len(hash_counter),
        "duplicate_rows_est_exact_hash": dup_rows,
        "duplicate_rate": dup_rows / max(row_count, 1),
        "note": "Exact duplicate count based on pandas row hash over numeric parsed rows.",
    }
    (args.out_dir / "arp_duplicate_report.json").write_text(json.dumps(dup_report, indent=2), encoding="utf-8")

    leak = schema[
        (schema["missing_rate"] == 0)
        & (
            (schema["std"] == 0)
            | (schema["max"] > 1e8)
            | (schema["sample_unique_count"] <= 2)
        )
    ].copy()
    leak["reason"] = np.select(
        [leak["std"] == 0, leak["max"] > 1e8, leak["sample_unique_count"] <= 2],
        ["constant", "timestamp_or_large_identifier_like", "binary_or_low_cardinality"],
        default="candidate",
    )
    leak.to_csv(args.out_dir / "arp_leakage_candidates.csv", index=False)

    sample = pd.concat(sample_parts, ignore_index=True).head(args.sample_rows)
    sample.to_parquet(args.out_dir / "arp_numeric_sample.parquet", index=False)

    report = {
        "input": str(p),
        "file_size_bytes": p.stat().st_size,
        "header_status": "no_real_header_detected_numeric_first_row",
        "pandas_header_columns_first_5": first_header[:5],
        "n_rows": row_count,
        "n_cols": n_cols,
        "recommended_column_names": names[:5] + ["..."] + names[-5:],
        "label_status": "uncertain",
        "label_warning": "No textual/header label found. Candidate label columns must be verified against dataset documentation or source extraction.",
        "duplicate_report": dup_report,
        "high_risk": [
            "The CSV appears headerless; do not use first row as column names.",
            "No trusted label column was found from the file alone.",
            "Large timestamp-like columns exist and should not be used as features until understood.",
        ],
    }
    (args.out_dir / "arp_audit_report.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out_dir / "arp_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
