"""Verify real datasets can be loaded on macOS without Ryu/Mininet."""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

try:
    from dataset.data_loader import ARPDataLoader, DatasetInfo, InSDNDataLoader
except ModuleNotFoundError:  # Allow: python dataset/verify_datasets.py
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataset.data_loader import ARPDataLoader, DatasetInfo, InSDNDataLoader


def print_csv_preview() -> None:
    """Print column names, first rows, and candidate label distributions."""
    print("=== dataset/ files ===")
    for path in sorted(Path("dataset").iterdir()):
        print(path)

    for f in glob.glob("dataset/*.csv"):
        df = pd.read_csv(f, nrows=5, low_memory=False)
        print(f"\n=== {f} ===")
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        print(df.head(2))

        label_cols = [
            c for c in df.columns
            if c.lower() in {"label", "class", "attack", "category"}
            or "label" in c.lower()
            or "class" in c.lower()
            or "attack" in c.lower()
        ]
        full_cols = pd.read_csv(f, nrows=0, low_memory=False).columns.tolist()
        for c in full_cols:
            if c not in label_cols and (
                c.lower() in {"label", "class", "attack", "category"}
                or "label" in c.lower()
                or "class" in c.lower()
                or "attack" in c.lower()
            ):
                label_cols.append(c)
        for c in label_cols:
            print(f"\nLabel distribution: {c}")
            print(pd.read_csv(f, usecols=[c], low_memory=False)[c].value_counts(dropna=False).to_string())


def main() -> int:
    """Run dataset verification."""
    print_csv_preview()
    print("\n=== Loader verification ===")
    ARPDataLoader().load()
    InSDNDataLoader().load()
    print("\n=== Dataset comparison ===")
    DatasetInfo().compare()
    print("\nBoth datasets OK - ready for DRL training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
