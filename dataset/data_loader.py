"""Dataset loaders for macOS-only real-data evaluation/training.

The loaders deliberately avoid Mininet/Ryu imports. They produce normalized
tabular state vectors and binary labels: 0 = normal, 1 = attack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ArraySplit = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]


def _find_label_column(df: pd.DataFrame) -> str:
    """Find the label column used by ARP/InSDN-style CSVs."""
    preferred = ["Label", "label", "Class", "class", "Attack", "attack", "Category", "category"]
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        low = col.lower()
        if "label" in low or "class" in low or "attack" in low:
            return col
    raise ValueError("Could not infer label column. Expected a Label/class/attack column.")


def _numeric_features(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, List[str]]:
    """Keep numeric feature columns and drop string identifiers such as IP/MAC."""
    features = df.drop(columns=[label_col], errors="ignore")
    numeric = features.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
    return numeric, numeric.columns.tolist()


def _split_scale(df: pd.DataFrame, y: pd.Series, test_size: float, random_state: int) -> ArraySplit:
    """Create a stratified train/test split and standardize features."""
    X, feature_names = _numeric_features(df, _find_label_column(df))
    y_arr = y.astype(int).to_numpy()
    stratify = y_arr if len(np.unique(y_arr)) > 1 and min(np.bincount(y_arr)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(dtype=np.float32),
        y_arr,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train.astype(np.int64), y_test.astype(np.int64), feature_names


def _print_summary(name: str, rows: int, y: Sequence[int], feature_names: Sequence[str]) -> None:
    """Print a compact dataset summary."""
    y_arr = np.asarray(y, dtype=int)
    attack_ratio = float((y_arr == 1).mean()) if len(y_arr) else 0.0
    print(f"[{name}] Loaded: {rows} rows, {len(feature_names)} features, {attack_ratio * 100:.1f}% attack")
    print(f"[{name}] Features: {list(feature_names)}")


def _read_csv(path: Path) -> pd.DataFrame:
    """Read CSV with low-memory disabled for stable dtype inference."""
    return pd.read_csv(path, low_memory=False)


@dataclass
class ARPDataLoader:
    """Load ARP Poisoning & Flood in SDN dataset (Mendeley)."""

    test_size: float = 0.2
    random_state: int = 42

    def _resolve_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.exists():
            return path
        candidates = [
            *Path("dataset").glob("*arp*.csv"),
            *Path("dataset").glob("*ARP*.csv"),
            *Path("dataset").glob("arp_stats*.csv"),
        ]
        if candidates:
            return sorted(candidates)[0]
        raise FileNotFoundError(path)

    def load(self, path: str | Path = "dataset/arp_stats.csv") -> ArraySplit:
        """Load ARP CSV and merge poison/flood labels into one attack class."""
        csv_path = self._resolve_path(path)
        df = _read_csv(csv_path)
        label_col = _find_label_column(df)
        labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
        y = (labels != 0).astype(int)
        X_train, X_test, y_train, y_test, feature_names = _split_scale(
            df, y, self.test_size, self.random_state
        )
        _print_summary("ARP", len(df), y, feature_names)
        return X_train, X_test, y_train, y_test, feature_names


@dataclass
class InSDNDataLoader:
    """Load InSDN Dataset 2020 CSV files (Kaggle)."""

    test_size: float = 0.2
    random_state: int = 42

    def _csv_files(self, path_dir: str | Path) -> List[Path]:
        root = Path(path_dir)
        files = sorted(root.glob("*.csv"))
        files = [p for p in files if "arp" not in p.name.lower()]
        if not files:
            raise FileNotFoundError(f"No InSDN CSV files found in {root}")
        return files

    def load(self, path_dir: str | Path = "dataset/") -> ArraySplit:
        """Load and merge InSDN CSVs, mapping Normal to 0 and attacks to 1."""
        files = self._csv_files(path_dir)
        frames = [_read_csv(p) for p in files]
        df = pd.concat(frames, ignore_index=True)
        label_col = _find_label_column(df)
        labels = df[label_col].astype(str).str.strip().str.lower()
        y = (labels != "normal").astype(int)
        X_train, X_test, y_train, y_test, feature_names = _split_scale(
            df, y, self.test_size, self.random_state
        )
        _print_summary("InSDN", len(df), y, feature_names)
        return X_train, X_test, y_train, y_test, feature_names


class DatasetInfo:
    """Print statistics for ARP and InSDN datasets."""

    def _safe_stats(self, name: str, loader) -> dict:
        try:
            X_train, X_test, y_train, y_test, feature_names = loader.load()
        except Exception as exc:
            return {
                "Dataset": name,
                "Total rows": "N/A",
                "Normal": "N/A",
                "Attack": "N/A",
                "Attack%": f"ERROR: {exc}",
                "Features": "N/A",
            }
        y = np.concatenate([y_train, y_test])
        normal = int((y == 0).sum())
        attack = int((y == 1).sum())
        total = int(len(y))
        return {
            "Dataset": name,
            "Total rows": total,
            "Normal": normal,
            "Attack": attack,
            "Attack%": f"{(attack / total * 100.0) if total else 0.0:.1f}",
            "Features": len(feature_names),
        }

    def compare(self) -> None:
        """Print a side-by-side summary for the two real datasets."""
        rows = [
            self._safe_stats("ARP", ARPDataLoader()),
            self._safe_stats("InSDN", InSDNDataLoader()),
        ]
        headers = ["Dataset", "Total rows", "Normal", "Attack", "Attack%", "Features"]
        widths = {
            h: max(len(h), *(len(str(row[h])) for row in rows))
            for h in headers
        }
        print(" | ".join(h.ljust(widths[h]) for h in headers))
        print("-+-".join("-" * widths[h] for h in headers))
        for row in rows:
            print(" | ".join(str(row[h]).ljust(widths[h]) for h in headers))
