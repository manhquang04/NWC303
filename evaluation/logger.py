"""Logging setup and TensorBoard/CSV metric logger."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

from config import CFG

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment,misc]
    _TB_AVAILABLE = False


def setup_logging(level: Optional[str] = None) -> None:
    """Configure global logging — call once at entry point."""
    lvl = level or CFG.logging_cfg.log_level
    logging.basicConfig(
        level=getattr(logging, lvl.upper(), logging.INFO),
        format=CFG.logging_cfg.log_format,
    )


class MetricsLogger:
    """Log episode metrics to TensorBoard and CSV."""

    def __init__(
        self,
        tb_dir: Path = CFG.logging_cfg.tensorboard_dir,
        csv_path: Path = CFG.logging_cfg.csv_metrics_path,
    ) -> None:
        self.tb_dir = Path(tb_dir)
        self.csv_path = Path(csv_path)
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self.tb_dir)) if _TB_AVAILABLE else None

        write_header = (not self.csv_path.exists()) or self.csv_path.stat().st_size == 0
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_file)
        if write_header:
            self._csv.writerow(["episode", "reward", "loss", "epsilon"])
            self._csv_file.flush()

    def log_episode(self, ep: int, reward: float, loss: float, epsilon: float) -> None:
        if self.writer is not None:
            self.writer.add_scalar("episode/reward", reward, ep)
            self.writer.add_scalar("episode/loss", loss, ep)
            self.writer.add_scalar("episode/epsilon", epsilon, ep)
        self._csv.writerow([ep, f"{reward:.4f}", f"{loss:.6f}", f"{epsilon:.4f}"])
        self._csv_file.flush()

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        self._csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
