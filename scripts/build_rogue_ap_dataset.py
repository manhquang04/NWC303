#!/usr/bin/env python3
"""
Build a clean Rogue AP dataset from AWID-like 802.11/radiotap CSV files.

This script intentionally does not merge ARP MITM, UNSW-NB15, or any other
source. It audits RogueAP_*.csv files, drops leakage-prone columns, creates a
file-based split, fits preprocessing on train only, and exports Parquet chunks.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


LABEL_COL = "Label"
LABEL_MAP = {"Normal": 0, "RogueAP": 1}

TRAIN_IDS = set(range(0, 19)) | set(range(24, 32))
VAL_IDS = set(range(19, 23)) | set(range(32, 35))
TEST_IDS = set(range(35, 40))
INSPECT_IDS = {23}

EXACT_LEAKAGE_COLUMNS = {
    "frame.number",
    "frame.time",
    "frame.time_delta_displayed",
    "frame.time_epoch",
    "frame.time_relative",
    "radiotap.mactime",
    "radiotap.timestamp.ts",
    "wlan.fixed.timestamp",
    "wlan_radio.timestamp",
    "wlan.bssid",
    "wlan.sa",
    "wlan.ta",
    "wlan.ra",
    "wlan.da",
    "dhcp.hw.mac_addr",
    "ip.src",
    "ip.dst",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "data.data",
    "tcp.payload",
    "udp.payload",
    "http.host",
    "http.file_data",
    "http.request.full_uri",
    "http.request.line",
    "http.request.uri.path",
    "http.request.uri.query",
    "http.request.uri.query.parameter",
    "http.referer",
    "json.value.string",
    "json.key",
    "ssh.cookie",
    "ssh.padding_string",
}

RAW_TEXT_PREFIXES = (
    "http.request.",
    "http.response.",
    "json.",
)

TIMING_LEAKAGE_RE = re.compile(
    r"(^frame\.time_delta$|^frame\.time_delta_displayed$|^frame\.time_relative$|"
    r"^tcp\.time_delta$|^tcp\.time_relative$|^udp\.time_delta$|^udp\.time_relative$|"
    r"time_epoch|timestamp|mactime|tsft|start_tsf|end_tsf)",
    re.IGNORECASE,
)

SUSPECT_LEAK_RE = re.compile(
    r"(frame\.number|frame\.time|time_epoch|time_relative|timestamp|mactime|tsft|"
    r"bssid|wlan\.(sa|ta|ra|da)$|hw_mac|mac_addr|ip\.src|ip\.dst|"
    r"proto_ipv4|payload|file_data|full_uri|uri\.|http\.host|json\.|ssh\.cookie)",
    re.IGNORECASE,
)


@dataclass
class FileAudit:
    file: str
    split: str
    rows: int
    cols: int
    normal: int
    rogueap: int
    rogueap_pct: float
    missing_values: int
    missing_pct: float
    missing_ge_95_cols: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean, split, and export Rogue AP CSV files to Parquet."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset/4.Rogue_AP"),
        help="Directory containing RogueAP_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed/rogue_ap"),
        help="Output directory for parquet splits and metadata.",
    )
    parser.add_argument(
        "--read-chunk-size",
        type=int,
        default=50_000,
        help="Rows per CSV read chunk.",
    )
    parser.add_argument(
        "--export-chunk-size",
        type=int,
        default=200_000,
        help="Rows per exported parquet part.",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.95,
        help="Drop columns with missing rate >= this threshold.",
    )
    parser.add_argument(
        "--numeric-threshold",
        type=float,
        default=0.98,
        help="Treat a column as numeric if this share of non-null train values parses as numeric.",
    )
    parser.add_argument(
        "--max-categorical-cardinality",
        type=int,
        default=30,
        help="One-hot encode categorical columns up to this train cardinality; drop higher cardinality.",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Do not standardize numeric features. Imputation still fits on train only.",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Drop exact duplicate processed rows within each split.",
    )
    parser.add_argument(
        "--drop-timing",
        action="store_true",
        help=(
            "Drop remaining timing/order columns such as frame.time_delta, "
            "tcp.time_relative, tcp.time_delta, and TSF/timestamp-like fields."
        ),
    )
    parser.add_argument(
        "--include-source-metadata",
        action="store_true",
        help="Append source_file and file_id columns after feature preprocessing for audit only.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only scan and write audit/manifest metadata; do not fit preprocessing or export parquet.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before writing.",
    )
    return parser.parse_args()


def require_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception:
        try:
            import fastparquet  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Parquet export requires pyarrow or fastparquet. Install one, for example: "
                "python3 -m pip install pyarrow"
            ) from exc


def file_id(path: Path) -> int:
    match = re.fullmatch(r"RogueAP_(\d+)\.csv", path.name)
    if not match:
        raise ValueError(f"Unexpected Rogue AP file name: {path.name}")
    return int(match.group(1))


def split_for_file(path: Path) -> str:
    idx = file_id(path)
    if idx in TRAIN_IDS:
        return "train"
    if idx in VAL_IDS:
        return "val"
    if idx in TEST_IDS:
        return "test"
    if idx in INSPECT_IDS:
        return "inspect"
    return "unused"


def list_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("RogueAP_*.csv"), key=file_id)
    if not files:
        raise FileNotFoundError(f"No RogueAP_*.csv files found in {input_dir}")
    return files


def validate_labels(labels: pd.Series, file_name: str) -> pd.Series:
    normalized = labels.astype("string").str.strip()
    unknown = sorted(set(normalized.dropna().unique()) - set(LABEL_MAP))
    if labels.isna().any():
        unknown.append("<NA>")
    if unknown:
        raise ValueError(f"{file_name}: unexpected label values: {unknown}")
    return normalized


def scan_files(
    files: list[Path],
    read_chunk_size: int,
    missing_threshold: float,
) -> tuple[list[FileAudit], pd.Series, list[str], dict[str, list[str]]]:
    audits: list[FileAudit] = []
    global_missing: pd.Series | None = None
    global_rows = 0
    reference_cols: list[str] | None = None
    manifest: dict[str, list[str]] = defaultdict(list)

    print("\n[1/7] Scanning files and auditing labels/missing values")
    for path in files:
        split = split_for_file(path)
        manifest[split].append(path.name)
        rows = 0
        cols = 0
        label_counts: Counter[str] = Counter()
        missing_values = 0
        file_missing: pd.Series | None = None

        for chunk in pd.read_csv(path, chunksize=read_chunk_size, low_memory=False):
            if LABEL_COL not in chunk.columns:
                raise ValueError(f"{path.name}: missing required label column '{LABEL_COL}'")
            if reference_cols is None:
                reference_cols = list(chunk.columns)
                global_missing = pd.Series(0, index=reference_cols, dtype="int64")
            elif list(chunk.columns) != reference_cols:
                raise ValueError(f"{path.name}: schema mismatch against first file")

            labels = validate_labels(chunk[LABEL_COL], path.name)
            label_counts.update(labels.value_counts(dropna=False).to_dict())

            if file_missing is None:
                cols = len(chunk.columns)
                file_missing = pd.Series(0, index=chunk.columns, dtype="int64")
            rows += len(chunk)
            global_rows += len(chunk)
            chunk_missing = chunk.isna().sum()
            file_missing += chunk_missing
            assert global_missing is not None
            global_missing += chunk_missing
            missing_values += int(chunk_missing.sum())

        if file_missing is None:
            raise ValueError(f"{path.name}: empty or unreadable CSV")

        missing_rates = file_missing / max(rows, 1)
        audit = FileAudit(
            file=path.name,
            split=split,
            rows=rows,
            cols=cols,
            normal=int(label_counts.get("Normal", 0)),
            rogueap=int(label_counts.get("RogueAP", 0)),
            rogueap_pct=round(100.0 * label_counts.get("RogueAP", 0) / max(rows, 1), 6),
            missing_values=missing_values,
            missing_pct=round(100.0 * missing_values / max(rows * cols, 1), 6),
            missing_ge_95_cols=int((missing_rates >= missing_threshold).sum()),
        )
        audits.append(audit)
        role = "normal/background" if audit.rogueap == 0 else "contains RogueAP"
        print(
            f"  {audit.file:15s} split={split:7s} rows={audit.rows:7d} "
            f"Normal={audit.normal:7d} RogueAP={audit.rogueap:4d} "
            f"RogueAP%={audit.rogueap_pct:8.5f} missing={audit.missing_pct:7.3f}% "
            f"({role})"
        )

    assert reference_cols is not None and global_missing is not None
    missing_rates = global_missing / max(global_rows, 1)
    sparse_cols = sorted(missing_rates[missing_rates >= missing_threshold].index.tolist())
    return audits, missing_rates, reference_cols, dict(manifest)


def is_leakage_column(col: str) -> bool:
    if col in EXACT_LEAKAGE_COLUMNS:
        return True
    if col.startswith("dhcp.ip."):
        return True
    if col.startswith("arp.") and col.endswith("hw_mac"):
        return True
    if col.startswith(RAW_TEXT_PREFIXES):
        return True
    if "payload" in col.lower():
        return True
    if "full_uri" in col.lower() or "uri.query" in col.lower():
        return True
    return False


def build_drop_plan(
    columns: list[str],
    missing_rates: pd.Series,
    missing_threshold: float,
    drop_timing: bool,
) -> tuple[dict[str, str], list[str], list[str]]:
    drop_reasons: dict[str, str] = {}
    for col in columns:
        if col == LABEL_COL:
            continue
        if is_leakage_column(col):
            drop_reasons[col] = "explicit_leakage_or_raw_identifier"
        if drop_timing and TIMING_LEAKAGE_RE.search(col):
            drop_reasons[col] = "drop_timing_flag"
    for col, rate in missing_rates.items():
        if col == LABEL_COL:
            continue
        if rate >= missing_threshold and col not in drop_reasons:
            drop_reasons[col] = f"sparse_missing_rate_ge_{missing_threshold}"

    kept = [c for c in columns if c != LABEL_COL and c not in drop_reasons]
    suspicious_kept = [c for c in kept if SUSPECT_LEAK_RE.search(c)]
    return drop_reasons, kept, suspicious_kept


def update_unique_set(values: set[str], series: pd.Series, max_cardinality: int) -> bool:
    non_null = series.dropna().astype("string")
    for value in non_null.unique():
        values.add(str(value))
        if len(values) > max_cardinality:
            return False
    return True


def infer_feature_types(
    train_files: list[Path],
    feature_cols: list[str],
    read_chunk_size: int,
    numeric_threshold: float,
    max_cardinality: int,
) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    non_null = Counter()
    numeric_ok = Counter()
    unique_values: dict[str, set[str]] = {col: set() for col in feature_cols}
    high_cardinality: set[str] = set()

    print("\n[2/7] Inferring numeric/categorical columns from train files only")
    for path in train_files:
        for chunk in pd.read_csv(path, chunksize=read_chunk_size, usecols=feature_cols + [LABEL_COL], low_memory=False):
            validate_labels(chunk[LABEL_COL], path.name)
            for col in feature_cols:
                s = chunk[col]
                present = s.notna()
                count_present = int(present.sum())
                non_null[col] += count_present
                if count_present == 0:
                    continue
                converted = pd.to_numeric(s, errors="coerce")
                numeric_ok[col] += int(converted[present].notna().sum())
                if col not in high_cardinality:
                    still_low = update_unique_set(unique_values[col], s[present], max_cardinality)
                    if not still_low:
                        high_cardinality.add(col)

    numeric_cols: list[str] = []
    categorical_vocabs: dict[str, list[str]] = {}
    dropped: dict[str, str] = {}

    for col in feature_cols:
        if non_null[col] == 0:
            dropped[col] = "all_missing_in_train_after_initial_drop"
            continue
        numeric_rate = numeric_ok[col] / max(non_null[col], 1)
        if numeric_rate >= numeric_threshold:
            numeric_cols.append(col)
        elif col not in high_cardinality and len(unique_values[col]) <= max_cardinality:
            vocab = sorted(unique_values[col])
            if "__MISSING__" not in vocab:
                vocab.append("__MISSING__")
            if "__OTHER__" not in vocab:
                vocab.append("__OTHER__")
            categorical_vocabs[col] = vocab
        else:
            dropped[col] = f"high_cardinality_or_non_numeric_train_values_gt_{max_cardinality}"

    print(f"  Numeric features: {len(numeric_cols)}")
    print(f"  Categorical features one-hot encoded: {len(categorical_vocabs)}")
    print(f"  Additional dropped after type inference: {len(dropped)}")
    return numeric_cols, categorical_vocabs, dropped


def safe_name(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:120] or "empty"


def fit_numeric_medians(
    train_files: list[Path],
    numeric_cols: list[str],
    read_chunk_size: int,
) -> dict[str, float]:
    print("\n[3/7] Fitting numeric medians on train only")
    if not numeric_cols:
        return {}

    values: dict[str, list[np.ndarray]] = {col: [] for col in numeric_cols}
    for path in train_files:
        for chunk in pd.read_csv(path, chunksize=read_chunk_size, usecols=numeric_cols + [LABEL_COL], low_memory=False):
            validate_labels(chunk[LABEL_COL], path.name)
            for col in numeric_cols:
                arr = pd.to_numeric(chunk[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if not arr.empty:
                    values[col].append(arr.to_numpy(dtype=np.float32, copy=False))

    medians: dict[str, float] = {}
    for col, chunks in values.items():
        if chunks:
            medians[col] = float(np.nanmedian(np.concatenate(chunks)))
        else:
            medians[col] = 0.0
    return medians


def make_numeric_matrix(
    chunk: pd.DataFrame,
    numeric_cols: list[str],
    medians: dict[str, float],
) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame(index=chunk.index)
    data = {}
    for col in numeric_cols:
        s = pd.to_numeric(chunk[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        data[f"num__{safe_name(col)}"] = s.fillna(medians.get(col, 0.0)).astype("float32")
    return pd.DataFrame(data, index=chunk.index)


def make_categorical_matrix(
    chunk: pd.DataFrame,
    categorical_vocabs: dict[str, list[str]],
) -> pd.DataFrame:
    if not categorical_vocabs:
        return pd.DataFrame(index=chunk.index)
    frames = []
    for col, vocab in categorical_vocabs.items():
        base = chunk[col].astype("string").fillna("__MISSING__")
        base = base.where(base.isin(vocab), "__OTHER__")
        cat = pd.Categorical(base, categories=vocab)
        dummies = pd.get_dummies(cat, prefix=f"cat__{safe_name(col)}", dtype=np.uint8)
        dummies.columns = [safe_name(c) for c in dummies.columns]
        frames.append(dummies)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=chunk.index)


def fit_scaler(
    train_files: list[Path],
    numeric_cols: list[str],
    medians: dict[str, float],
    read_chunk_size: int,
) -> StandardScaler | None:
    if not numeric_cols:
        return None
    print("\n[4/7] Fitting StandardScaler on train numeric features only")
    scaler = StandardScaler()
    usecols = numeric_cols + [LABEL_COL]
    for path in train_files:
        for chunk in pd.read_csv(path, chunksize=read_chunk_size, usecols=usecols, low_memory=False):
            validate_labels(chunk[LABEL_COL], path.name)
            numeric = make_numeric_matrix(chunk, numeric_cols, medians)
            scaler.partial_fit(numeric.to_numpy(dtype=np.float32))
    return scaler


def transform_chunk(
    chunk: pd.DataFrame,
    numeric_cols: list[str],
    categorical_vocabs: dict[str, list[str]],
    medians: dict[str, float],
    scaler: StandardScaler | None,
) -> pd.DataFrame:
    labels = validate_labels(chunk[LABEL_COL], "<chunk>").map(LABEL_MAP).astype("int8")
    numeric = make_numeric_matrix(chunk, numeric_cols, medians)
    if scaler is not None and not numeric.empty:
        numeric.loc[:, :] = scaler.transform(numeric.to_numpy(dtype=np.float32)).astype("float32")
    categorical = make_categorical_matrix(chunk, categorical_vocabs)
    out = pd.concat([numeric, categorical], axis=1)
    out["label"] = labels.to_numpy()
    return out


def write_part(split_dir: Path, part_idx: int, frame: pd.DataFrame) -> Path:
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / f"part-{part_idx:05d}.parquet"
    frame.to_parquet(out_path, index=False)
    return out_path


def export_split(
    split: str,
    files: list[Path],
    output_dir: Path,
    read_chunk_size: int,
    export_chunk_size: int,
    numeric_cols: list[str],
    categorical_vocabs: dict[str, list[str]],
    medians: dict[str, float],
    scaler: StandardScaler | None,
    dedup: bool,
    include_source_metadata: bool,
) -> dict[str, Any]:
    print(f"\n[6/7] Exporting split={split}")
    split_dir = output_dir / split
    part_idx = 0
    buffer: list[pd.DataFrame] = []
    buffered_rows = 0
    seen_hashes: set[int] = set()
    duplicate_rows = 0
    dropped_duplicate_rows = 0
    rows_before = 0
    rows_after = 0
    class_counts = Counter()
    part_files: list[str] = []

    usecols = list(dict.fromkeys(numeric_cols + list(categorical_vocabs) + [LABEL_COL]))
    for path in files:
        for chunk in pd.read_csv(path, chunksize=read_chunk_size, usecols=usecols, low_memory=False):
            transformed = transform_chunk(chunk, numeric_cols, categorical_vocabs, medians, scaler)
            if include_source_metadata:
                transformed["source_file"] = path.name
                transformed["file_id"] = file_id(path)
            rows_before += len(transformed)
            hashes = pd.util.hash_pandas_object(transformed, index=False).astype("uint64")
            duplicate_mask = hashes.isin(seen_hashes) | hashes.duplicated(keep="first")
            duplicate_rows += int(duplicate_mask.sum())
            seen_hashes.update(int(h) for h in hashes[~duplicate_mask].to_numpy())
            if dedup:
                transformed = transformed.loc[~duplicate_mask].copy()
                dropped_duplicate_rows += int(duplicate_mask.sum())
            class_counts.update(transformed["label"].value_counts().to_dict())
            rows_after += len(transformed)

            if not transformed.empty:
                buffer.append(transformed)
                buffered_rows += len(transformed)

            while buffered_rows >= export_chunk_size:
                combined = pd.concat(buffer, ignore_index=True)
                to_write = combined.iloc[:export_chunk_size].copy()
                rest = combined.iloc[export_chunk_size:].copy()
                part_path = write_part(split_dir, part_idx, to_write)
                part_files.append(part_path.name)
                print(f"  wrote {split}/{part_path.name} rows={len(to_write)}")
                part_idx += 1
                buffer = [rest] if not rest.empty else []
                buffered_rows = len(rest)

    if buffer:
        combined = pd.concat(buffer, ignore_index=True)
        part_path = write_part(split_dir, part_idx, combined)
        part_files.append(part_path.name)
        print(f"  wrote {split}/{part_path.name} rows={len(combined)}")

    metadata = {
        "split": split,
        "source_files": [p.name for p in files],
        "rows_before_dedup": rows_before,
        "rows_after_dedup": rows_after,
        "cols": None,
        "class_counts": {"Normal": int(class_counts.get(0, 0)), "RogueAP": int(class_counts.get(1, 0))},
        "duplicate_rows": duplicate_rows,
        "duplicate_rate_pct": round(100.0 * duplicate_rows / max(rows_before, 1), 6),
        "dropped_duplicate_rows": dropped_duplicate_rows,
        "part_files": part_files,
        "include_source_metadata": include_source_metadata,
    }
    if part_files:
        first_part = pd.read_parquet(split_dir / part_files[0])
        metadata["cols"] = int(first_part.shape[1])
    (split_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def prepare_output_dir(output_dir: Path, overwrite: bool, audit_only: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not audit_only:
        existing_parts = list(output_dir.glob("**/part-*.parquet"))
        if existing_parts and not overwrite:
            raise FileExistsError(
                f"{output_dir} already contains parquet parts. Use --overwrite to replace them."
            )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safety_checks(
    audits: list[FileAudit],
    manifest: dict[str, list[str]],
    suspicious_kept: list[str],
) -> None:
    by_split = defaultdict(Counter)
    for item in audits:
        by_split[item.split]["Normal"] += item.normal
        by_split[item.split]["RogueAP"] += item.rogueap

    if by_split["test"]["RogueAP"] <= 0:
        raise ValueError("Safety check failed: test split has no RogueAP rows")
    if by_split["train"]["Normal"] <= 0:
        raise ValueError("Safety check failed: train split has no Normal background rows")

    all_split_files = []
    for split in ("train", "val", "test"):
        all_split_files.extend(manifest.get(split, []))
    if len(all_split_files) != len(set(all_split_files)):
        raise ValueError("Safety check failed: file overlap between train/val/test")

    if suspicious_kept:
        print("\nWARNING: suspicious columns remain after drop plan. Review before trusting results:")
        for col in suspicious_kept:
            print(f"  - {col}")

    test_pos = by_split["test"]["RogueAP"]
    if test_pos < 50:
        print(f"\nWARNING: test split has only {test_pos} RogueAP rows; metrics may be unstable.")


def main() -> int:
    args = parse_args()
    files = list_input_files(args.input_dir)
    prepare_output_dir(args.output_dir, args.overwrite, args.audit_only)
    if not args.audit_only:
        require_parquet_engine()

    audits, missing_rates, columns, manifest = scan_files(
        files, args.read_chunk_size, args.missing_threshold
    )
    drop_reasons, initially_kept, suspicious_kept = build_drop_plan(
        columns, missing_rates, args.missing_threshold, args.drop_timing
    )

    safety_checks(audits, manifest, suspicious_kept)

    audit_rows = [asdict(item) for item in audits]
    pd.DataFrame(audit_rows).to_csv(args.output_dir / "audit.csv", index=False)
    write_json(args.output_dir / "audit.json", audit_rows)
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(
        args.output_dir / "drop_plan.json",
        {
            "drop_reasons": drop_reasons,
            "drop_timing": args.drop_timing,
            "initially_kept_columns": initially_kept,
            "suspicious_kept_columns": suspicious_kept,
            "global_missing_rates": {k: float(v) for k, v in missing_rates.items()},
        },
    )

    print("\n[Drop plan]")
    print(f"  Total input columns: {len(columns)}")
    print(f"  Dropped columns before type inference: {len(drop_reasons)}")
    print(f"  Kept candidate feature columns: {len(initially_kept)}")
    print("  Kept candidate columns:")
    for col in initially_kept:
        print(f"    - {col}")

    if args.audit_only:
        print("\nAudit-only mode complete. No parquet files were written.")
        return 0

    path_by_name = {p.name: p for p in files}
    train_files = [path_by_name[name] for name in manifest.get("train", [])]
    val_files = [path_by_name[name] for name in manifest.get("val", [])]
    test_files = [path_by_name[name] for name in manifest.get("test", [])]
    inspect_files = [path_by_name[name] for name in manifest.get("inspect", [])]

    numeric_cols, categorical_vocabs, type_dropped = infer_feature_types(
        train_files,
        initially_kept,
        args.read_chunk_size,
        args.numeric_threshold,
        args.max_categorical_cardinality,
    )
    final_feature_cols = numeric_cols + list(categorical_vocabs)
    medians = fit_numeric_medians(train_files, numeric_cols, args.read_chunk_size)
    scaler = None if args.no_scale else fit_scaler(train_files, numeric_cols, medians, args.read_chunk_size)

    feature_schema = {
        "label_mapping": LABEL_MAP,
        "drop_timing": args.drop_timing,
        "include_source_metadata": args.include_source_metadata,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_vocabs,
        "dropped_columns": {**drop_reasons, **type_dropped},
        "final_original_feature_columns": final_feature_cols,
        "numeric_medians_train_only": medians,
        "scaler": None
        if scaler is None
        else {
            "type": "StandardScaler",
            "fit_on": "train_only",
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "numeric_output_columns": [f"num__{safe_name(c)}" for c in numeric_cols],
        },
    }
    write_json(args.output_dir / "feature_schema.json", feature_schema)

    print("\n[5/7] Split manifest")
    for split, split_files in {
        "train": train_files,
        "val": val_files,
        "test": test_files,
        "inspect": inspect_files,
    }.items():
        print(f"  {split:7s}: {', '.join(p.name for p in split_files) or '(none)'}")

    split_metadata = {}
    for split, split_files in (
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
        ("inspect", inspect_files),
    ):
        if not split_files:
            continue
        split_metadata[split] = export_split(
            split,
            split_files,
            args.output_dir,
            args.read_chunk_size,
            args.export_chunk_size,
            numeric_cols,
            categorical_vocabs,
            medians,
            scaler,
            args.dedup,
            args.include_source_metadata,
        )

    write_json(args.output_dir / "split_metadata.json", split_metadata)

    total_rows_before = sum(item.rows for item in audits)
    total_rows_after = sum(meta["rows_after_dedup"] for meta in split_metadata.values())
    final_report = {
        "input_files_processed": len(files),
        "total_rows_before_cleaning": total_rows_before,
        "total_rows_after_export": total_rows_after,
        "splits": split_metadata,
        "columns_dropped_total": len(feature_schema["dropped_columns"]),
        "columns_dropped": feature_schema["dropped_columns"],
        "final_original_feature_columns": final_feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": list(categorical_vocabs),
        "suspicious_kept_columns": suspicious_kept,
        "warnings": [],
    }
    for split in ("train", "val", "test"):
        meta = split_metadata.get(split)
        if not meta:
            continue
        rogue = meta["class_counts"]["RogueAP"]
        total = max(meta["rows_after_dedup"], 1)
        rogue_pct = 100.0 * rogue / total
        if split == "test" and rogue == 0:
            final_report["warnings"].append("test split has no RogueAP rows")
        if rogue_pct < 0.05:
            final_report["warnings"].append(
                f"{split} split is extremely imbalanced: RogueAP={rogue} ({rogue_pct:.6f}%)"
            )
    write_json(args.output_dir / "final_report.json", final_report)

    print("\n[7/7] Final report")
    print(f"  Files processed: {len(files)}")
    print(f"  Rows before cleaning: {total_rows_before}")
    print(f"  Rows after export: {total_rows_after}")
    for split in ("train", "val", "test", "inspect"):
        meta = split_metadata.get(split)
        if not meta:
            continue
        counts = meta["class_counts"]
        print(
            f"  {split:7s}: rows={meta['rows_after_dedup']:8d} "
            f"Normal={counts['Normal']:8d} RogueAP={counts['RogueAP']:5d} "
            f"dups={meta['duplicate_rows']} ({meta['duplicate_rate_pct']}%)"
        )
    print(f"  Dropped columns: {len(feature_schema['dropped_columns'])}")
    print(f"  Final original feature columns: {len(final_feature_cols)}")
    if final_report["warnings"]:
        print("\nWarnings:")
        for warning in final_report["warnings"]:
            print(f"  - {warning}")
    print(f"\nDone. Metadata and parquet outputs are under: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
