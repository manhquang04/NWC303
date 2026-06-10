"""Loader for the 13 CTU-13 bidirectional flow scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class CTU13Loader:
    """Load, label, sample, and extract numeric CTU-13 flow features."""

    COLUMNS = [
        "StartTime", "Dur", "Proto", "SrcAddr", "Sport", "Dir",
        "DstAddr", "Dport", "State", "sTos", "dTos", "TotPkts",
        "TotBytes", "SrcBytes", "Label",
    ]
    FEATURE_NAMES = [
        "Dur", "TotPkts", "TotBytes", "SrcBytes", "Proto", "Sport", "Dport",
    ]
    PROTOCOL_MAP = {"tcp": 0, "udp": 1, "icmp": 2}

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        """Read required CTU-13 columns, falling back to latin-1."""
        try:
            return pd.read_csv(path, sep=",", usecols=CTU13Loader.COLUMNS)
        except UnicodeDecodeError:
            return pd.read_csv(
                path,
                sep=",",
                usecols=CTU13Loader.COLUMNS,
                encoding="latin-1",
            )

    @staticmethod
    def _label_rows(labels: pd.Series) -> pd.Series:
        """Map Normal/Botnet labels and leave all other rows unmapped."""
        text = labels.astype(str)
        mapped = pd.Series(np.nan, index=labels.index, dtype="float64")
        mapped[text.str.contains("Normal", case=False, na=False)] = 0
        mapped[text.str.contains("Botnet", case=False, na=False)] = 1
        mapped[text.str.contains("Background", case=False, na=False)] = np.nan
        return mapped

    @staticmethod
    def _stratified_sample(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
        """Sample the same fraction independently from each available class."""
        if fraction >= 1.0:
            return frame
        parts = []
        for label, group in frame.groupby("target", sort=True):
            count = max(1, int(round(len(group) * fraction)))
            parts.append(group.sample(n=min(count, len(group)), random_state=seed + int(label)))
        return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0]

    def _extract(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Convert raw CTU-13 fields into the seven numeric model features."""
        output = pd.DataFrame(index=frame.index)
        output["Dur"] = pd.to_numeric(frame["Dur"], errors="coerce").fillna(0)
        for column in ("TotPkts", "TotBytes", "SrcBytes"):
            output[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        output["Proto"] = (
            frame["Proto"].astype(str).str.lower().map(self.PROTOCOL_MAP).fillna(3)
        )
        output["Sport"] = pd.to_numeric(frame["Sport"], errors="coerce").fillna(-1)
        output["Dport"] = pd.to_numeric(frame["Dport"], errors="coerce").fillna(-1)
        output["target"] = frame["target"].astype(int)
        return output.replace([np.inf, -np.inf], np.nan).dropna()

    def load(
        self,
        data_dir: str | Path,
        sample_frac: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load 13 scenarios and return float32 features, integer labels, and names."""
        if not 0 < sample_frac <= 1:
            raise ValueError("sample_frac must be in the interval (0, 1].")
        directory = Path(data_dir).expanduser()
        files = sorted(directory.glob("*.binetflow"))
        if len(files) != 13:
            raise FileNotFoundError(
                f"Expected 13 .binetflow files in {directory}, found {len(files)}."
            )

        sampled_frames: list[pd.DataFrame] = []
        for index, path in enumerate(files, start=1):
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"[{index}/13] Loading: {path.name} ({size_mb:.1f} MB)")
            raw = self._read_csv(path)
            raw["target"] = self._label_rows(raw["Label"])
            raw = raw.dropna(subset=["target"])
            features = self._extract(raw)
            sampled_frames.append(
                self._stratified_sample(features, sample_frac, seed=42 + index)
            )

        combined = pd.concat(sampled_frames, ignore_index=True)
        combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
        y = combined.pop("target").to_numpy(dtype=np.int64)
        X = combined[self.FEATURE_NAMES].to_numpy(dtype=np.float32)

        normal = int((y == 0).sum())
        botnet = int((y == 1).sum())
        total = len(y)
        print("\n=== CTU-13 Load Summary ===")
        print(f"Total flows (after sampling): {total}")
        print(f"Normal: {normal} ({normal / max(total, 1) * 100:.2f}%)")
        print(f"Botnet: {botnet} ({botnet / max(total, 1) * 100:.2f}%)")
        print(f"Scenarios loaded: {len(files)}")
        return X, y, list(self.FEATURE_NAMES)

