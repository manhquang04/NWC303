"""Aggregate reward ablation CSV files into one research table."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for CSV aggregation."""
    parser = argparse.ArgumentParser(description="Aggregate experiment CSV files.")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aggregate(input_paths: list[Path], output_path: Path) -> Path:
    """Combine many experiment CSV files into one output CSV."""
    import pandas as pd

    if not input_paths:
        raise ValueError("At least one input CSV is required.")
    frames = []
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if "config" not in df.columns:
            df["config"] = path.stem
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"[aggregate_results.py] Exported {len(combined)} rows to {output_path}")
    return output_path


def main() -> int:
    """Script entry point."""
    args = parse_args()
    aggregate(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
