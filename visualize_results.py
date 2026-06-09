"""Visualize SDN DRL-IDS experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


RESULTS_DIR = Path("results")
RUNS_DIR = Path("runs")


def _require_file(path: Path) -> Path:
    """Return an existing file path or raise a clear error."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required results file: {path}")
    return path


def plot_reward_ablation(ablation_df: pd.DataFrame) -> Path:
    """Plot average reward for every reward config and scenario."""
    scenario_order = ["normal", "arp", "rogue", "mixed"]
    config_order = [
        "reward_v1",
        "reward_v2_fn_penalty",
        "reward_v3_isolate_boost",
        "reward_v4_balanced",
        "reward_v5_conservative",
    ]
    labels = ["v1 baseline", "v2 FN", "v3 isolate", "v4 balanced", "v5 conservative"]
    colors = ["#607d8b", "#1976d2", "#d32f2f", "#7b1fa2"]

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.18
    x_positions = list(range(len(config_order)))
    for idx, scenario in enumerate(scenario_order):
        rewards = []
        for config in config_order:
            row = ablation_df[
                (ablation_df["config"] == config)
                & (ablation_df["scenario"].str.lower() == scenario)
            ]
            rewards.append(float(row["avg_reward"].iloc[0]) if not row.empty else 0.0)
        offsets = [x + (idx - 1.5) * width for x in x_positions]
        ax.bar(offsets, rewards, width=width, label=scenario.upper(), color=colors[idx])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average reward")
    ax.set_title("Reward Ablation Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / "reward_ablation_bar.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_detection_recall(eval_df: pd.DataFrame) -> Path:
    """Plot custom DQN recall/F1 for attack scenarios."""
    scenario_order = ["arp", "rogue", "mixed"]
    recalls = []
    f1_scores = []
    for scenario in scenario_order:
        row = eval_df[eval_df["scenario"].str.lower() == scenario]
        recalls.append(float(row["recall"].iloc[-1]) if not row.empty else 0.0)
        f1_scores.append(float(row["f1"].iloc[-1]) if not row.empty else 0.0)

    x_positions = list(range(len(scenario_order)))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar([x - 0.18 for x in x_positions], recalls, width=0.36, label="Recall", color="#1976d2")
    ax.bar([x + 0.18 for x in x_positions], f1_scores, width=0.36, label="F1", color="#ef6c00")
    ax.axhline(0.05, linestyle="--", color="#455a64", linewidth=1.2, label="Baseline recall 0.05")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([s.upper() for s in scenario_order])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Detection Performance: Baseline vs Custom DQN")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / "detection_recall.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> int:
    """Load CSV results and write visualization PNGs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_df = pd.read_csv(_require_file(RUNS_DIR / "evaluation_results.csv"))
    ablation_df = pd.read_csv(_require_file(RUNS_DIR / "final_research_results.csv"))
    outputs = [
        plot_reward_ablation(ablation_df),
        plot_detection_recall(eval_df),
    ]
    print("Visualizations saved:")
    for path in outputs:
        print(f" - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
