"""Plotting utilities: reward curve, confusion matrix, action distribution."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import CFG  # noqa: E402

log = logging.getLogger(__name__)


def plot_reward_curve(
    csv_path: Path = CFG.logging_cfg.csv_metrics_path,
    out_path: Path = CFG.logging_cfg.reward_curve_path,
    smooth_window: int = 20,
) -> Path:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    eps, rewards = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps.append(int(row["episode"]))
            rewards.append(float(row["reward"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eps, rewards, alpha=0.4, label="reward (raw)")
    if len(rewards) >= smooth_window:
        smoothed = np.convolve(rewards, np.ones(smooth_window) / smooth_window, mode="valid")
        ax.plot(eps[smooth_window - 1:], smoothed,
                linewidth=2.0, label=f"reward (MA{smooth_window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("DRL training reward curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Reward curve → %s", out_path)
    return Path(out_path)


def plot_confusion_matrix(
    tp: int, fp: int, tn: int, fn: int,
    out_path: Path = CFG.logging_cfg.confusion_matrix_path,
) -> Path:
    cm = np.array([[tp, fn], [fp, tn]], dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Detected", "Allowed"])
    ax.set_yticklabels(["Attack (GT)", "Normal (GT)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Confusion matrix → %s", out_path)
    return Path(out_path)


def plot_action_distribution(
    actions: Sequence[int],
    out_path: Path = CFG.LOG_DIR / "action_dist.png",
) -> Path:
    from config import ACTION_NAMES
    counts = [int((np.array(actions) == i).sum()) for i in range(len(ACTION_NAMES))]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ACTION_NAMES, counts, color=["#4caf50", "#ffc107", "#ff5722", "#9c27b0"])
    ax.set_ylabel("Count")
    ax.set_title("Action distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)
