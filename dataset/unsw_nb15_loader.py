"""UNSW-NB15 loader using the official train/test flow split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CFG


@dataclass
class UNSWNB15Split:
    """Preprocessed official train, validation, and test arrays."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer


class UNSWNB15Loader:
    """Load UNSW-NB15 flow features with safe categorical encoding."""

    CATEGORICAL_COLUMNS = ["proto", "service", "state"]
    DROP_COLUMNS = ["id", "attack_cat", "label"]

    def __init__(self, data_dir: str | Path = CFG.dataset.path) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.split_dir = self.data_dir / "official_split"

    def _paths(self) -> tuple[Path, Path]:
        train_path = self.split_dir / "UNSW_NB15_training-set.csv"
        test_path = self.split_dir / "UNSW_NB15_testing-set.csv"
        missing = [str(path) for path in (train_path, test_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing official UNSW-NB15 split files: {missing}")
        return train_path, test_path

    @staticmethod
    def _clean(frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize labels, missing values, and infinite numeric values."""
        frame = frame.copy()
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        if "label" not in frame:
            raise ValueError("UNSW-NB15 split must contain a binary label column.")
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(np.int64)
        for column in UNSWNB15Loader.CATEGORICAL_COLUMNS:
            frame[column] = frame[column].fillna("unknown").astype(str).str.strip().str.lower()
        numeric = frame.select_dtypes(include=[np.number]).columns
        frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan).fillna(0)
        return frame

    @staticmethod
    def _preprocessor(numeric_columns: list[str]) -> ColumnTransformer:
        """Build a train-fitted transformer for numeric and categorical flows."""
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)
        return ColumnTransformer(
            [
                ("numeric", StandardScaler(), numeric_columns),
                ("categorical", encoder, UNSWNB15Loader.CATEGORICAL_COLUMNS),
            ],
            remainder="drop",
        )

    def load(
        self,
        validation_size: float = CFG.dataset.validation_size,
        random_state: int = CFG.dataset.seed,
        persist: bool = True,
    ) -> UNSWNB15Split:
        """Return preprocessed arrays using the official test set unchanged."""
        train_path, test_path = self._paths()
        print(f"Loading official train split: {train_path}")
        train_frame = self._clean(pd.read_csv(train_path, low_memory=False))
        print(f"Loading official test split:  {test_path}")
        test_frame = self._clean(pd.read_csv(test_path, low_memory=False))

        y_full = train_frame["label"].to_numpy(dtype=np.int64)
        y_test = test_frame["label"].to_numpy(dtype=np.int64)
        X_full = train_frame.drop(columns=self.DROP_COLUMNS, errors="ignore")
        X_test_frame = test_frame.drop(columns=self.DROP_COLUMNS, errors="ignore")
        X_train_frame, X_val_frame, y_train, y_val = train_test_split(
            X_full,
            y_full,
            test_size=validation_size,
            stratify=y_full,
            random_state=random_state,
        )
        numeric_columns = [
            column for column in X_train_frame.columns
            if column not in self.CATEGORICAL_COLUMNS
        ]
        preprocessor = self._preprocessor(numeric_columns)
        X_train = preprocessor.fit_transform(X_train_frame).astype(np.float32)
        X_val = preprocessor.transform(X_val_frame).astype(np.float32)
        X_test = preprocessor.transform(X_test_frame).astype(np.float32)
        feature_names = preprocessor.get_feature_names_out().tolist()

        print("\n=== UNSW-NB15 Summary ===")
        print(f"Train: {len(y_train)} ({y_train.mean() * 100:.2f}% attack)")
        print(f"Validation: {len(y_val)} ({y_val.mean() * 100:.2f}% attack)")
        print(f"Test: {len(y_test)} ({y_test.mean() * 100:.2f}% attack)")
        print(f"Encoded flow features: {len(feature_names)}")

        split = UNSWNB15Split(
            X_train, X_val, X_test, y_train, y_val, y_test,
            feature_names, preprocessor,
        )
        if persist:
            output = CFG.dataset.processed_path
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output / "official_split.npz",
                X_train=X_train, X_val=X_val, X_test=X_test,
                y_train=y_train, y_val=y_val, y_test=y_test,
            )
            joblib.dump(preprocessor, output / "preprocessor.joblib")
            (output / "feature_names.txt").write_text(
                "\n".join(feature_names), encoding="utf-8"
            )
        return split
