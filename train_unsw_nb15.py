"""Train the four-action DQN policy on UNSW-NB15 flow records."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from agent.dqn_agent import DQNAgent
from config import (
    ACTION_ALLOW, ACTION_BLOCK, ACTION_FLAG, ACTION_ISOLATE,
    ACTION_NAMES, CFG, NUM_ACTIONS,
)
from dataset.unsw_nb15_loader import UNSWNB15Loader


REWARD_ALLOW_NORMAL = 1.5
REWARD_FLAG_ATTACK = 2.0
REWARD_BLOCK_ATTACK = 5.0
REWARD_ISOLATE_ATTACK = 6.0
PENALTY_MISS_ATTACK = 6.0
FP_PENALTY = 12.0
TARGET_VALIDATION_FPR = 0.045
NORMAL_FLAG_PENALTY = 13.0
NORMAL_BLOCK_PENALTY = 24.0
NORMAL_ISOLATE_PENALTY = 28.0


def reward_for(
    action: int,
    label: int,
    correct_allow: float = REWARD_ALLOW_NORMAL,
    attack_flag_reward: float = REWARD_FLAG_ATTACK,
    attack_block_reward: float = REWARD_BLOCK_ATTACK,
    attack_isolate_reward: float = REWARD_ISOLATE_ATTACK,
    missed_attack_penalty: float = PENALTY_MISS_ATTACK,
    normal_flag_penalty: float = NORMAL_FLAG_PENALTY,
    normal_block_penalty: float = NORMAL_BLOCK_PENALTY,
    normal_isolate_penalty: float = NORMAL_ISOLATE_PENALTY,
) -> float:
    """Apply balanced flow-level rewards with an explicit false-positive cost."""
    if label == 0:
        penalty = {
            ACTION_ALLOW: correct_allow,
            ACTION_FLAG: -normal_flag_penalty,
            ACTION_BLOCK: -normal_block_penalty,
            ACTION_ISOLATE: -normal_isolate_penalty,
        }[action]
        return penalty
    return {
        ACTION_ALLOW: -missed_attack_penalty,
        ACTION_FLAG: attack_flag_reward,
        ACTION_BLOCK: attack_block_reward,
        ACTION_ISOLATE: attack_isolate_reward,
    }[action]


def infer_policy(agent: DQNAgent, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return greedy actions and non-allow softmax confidence."""
    import torch

    action_chunks = []
    confidence_chunks = []
    agent.q_net.eval()
    with torch.no_grad():
        for start in range(0, len(features), 4096):
            batch = torch.from_numpy(features[start:start + 4096]).to(agent.device)
            q_values = agent.q_net(batch)
            probabilities = torch.softmax(q_values, dim=1)
            action_chunks.append(q_values.argmax(dim=1).cpu().numpy())
            confidence_chunks.append(probabilities[:, 1:].max(dim=1).values.cpu().numpy())
    agent.q_net.train()
    return np.concatenate(action_chunks), np.concatenate(confidence_chunks)


def infer_actions(
    agent: DQNAgent, features: np.ndarray, confidence_threshold: float = 0.0,
) -> np.ndarray:
    """Perform greedy inference and gate low-confidence interventions to allow."""
    actions, confidence = infer_policy(agent, features)
    return np.where(
        (actions != ACTION_ALLOW) & (confidence < confidence_threshold),
        ACTION_ALLOW,
        actions,
    )


def select_confidence_threshold(
    labels: np.ndarray,
    proposed_actions: np.ndarray,
    confidence: np.ndarray,
    target_fpr: float = TARGET_VALIDATION_FPR,
) -> tuple[float, np.ndarray, dict[str, float | int]]:
    """Select the highest-F1 validation policy satisfying the FPR constraint."""
    candidates = np.linspace(0.0, 0.99, 100)
    feasible = []
    fallback = []
    for threshold in candidates:
        actions = np.where(
            (proposed_actions != ACTION_ALLOW) & (confidence < threshold),
            ACTION_ALLOW,
            proposed_actions,
        )
        metrics = binary_metrics(labels, actions)
        item = (float(metrics["f1"]), -float(metrics["fpr"]), float(threshold), actions, metrics)
        fallback.append(item)
        if metrics["fpr"] <= target_fpr:
            feasible.append(item)
    _, _, threshold, actions, metrics = max(feasible or fallback, key=lambda item: item[:3])
    return threshold, actions, metrics


def binary_metrics(labels: np.ndarray, actions: np.ndarray) -> dict[str, float | int]:
    """Convert non-allow actions into attack detections and score them."""
    predictions = (actions != ACTION_ALLOW).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "fpr": float(fp / max(fp + tn, 1)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def action_distribution(labels: np.ndarray, actions: np.ndarray, label: int) -> dict[str, int]:
    """Count each executed action for one ground-truth class."""
    selected = actions[labels == label]
    return {
        name: int((selected == action).sum())
        for action, name in enumerate(ACTION_NAMES)
    }


def balanced_episode_indices(
    labels: np.ndarray, steps: int, rng: np.random.Generator,
) -> np.ndarray:
    """Sample equal normal and attack transitions for each training episode."""
    normal = np.flatnonzero(labels == 0)
    attack = np.flatnonzero(labels == 1)
    normal_count = steps // 2
    attack_count = steps - normal_count
    indices = np.concatenate([
        rng.choice(normal, normal_count, replace=len(normal) < normal_count),
        rng.choice(attack, attack_count, replace=len(attack) < attack_count),
    ])
    rng.shuffle(indices)
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=CFG.dataset.path)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/unsw_nb15"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=CFG.dataset.seed)
    parser.add_argument("--correct-allow", type=float, default=REWARD_ALLOW_NORMAL)
    parser.add_argument("--attack-flag-reward", type=float, default=REWARD_FLAG_ATTACK)
    parser.add_argument("--attack-block-reward", type=float, default=REWARD_BLOCK_ATTACK)
    parser.add_argument("--attack-isolate-reward", type=float, default=REWARD_ISOLATE_ATTACK)
    parser.add_argument("--missed-attack-penalty", type=float, default=PENALTY_MISS_ATTACK)
    parser.add_argument("--normal-flag-penalty", type=float, default=NORMAL_FLAG_PENALTY)
    parser.add_argument("--normal-block-penalty", type=float, default=NORMAL_BLOCK_PENALTY)
    parser.add_argument("--normal-isolate-penalty", type=float, default=NORMAL_ISOLATE_PENALTY)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--fixed-confidence-threshold", type=float, default=None)
    parser.add_argument("--early-stop-fpr", type=float, default=0.40)
    parser.add_argument("--early-stop-after", type=int, default=20)
    parser.add_argument("--success-stop-fpr", type=float, default=None)
    parser.add_argument("--success-stop-f1", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    split = UNSWNB15Loader(args.data_dir).load(random_state=args.seed)
    agent = DQNAgent(split.X_train.shape[1], NUM_ACTIONS)
    if args.resume_checkpoint is not None:
        agent.load(args.resume_checkpoint)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    best_score = -float("inf")
    best_f1 = -1.0
    best_episode = -1
    rows = []
    checkpoint_path = args.run_dir / "dqn_best.pt"

    config_used = {
        "dataset": CFG.dataset.name,
        "dataset_path": str(Path(args.data_dir).resolve()),
        "label_mode": CFG.dataset.label_mode,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "correct_allow": args.correct_allow,
        "attack_flag_reward": args.attack_flag_reward,
        "attack_block_reward": args.attack_block_reward,
        "attack_isolate_reward": args.attack_isolate_reward,
        "missed_attack_penalty": args.missed_attack_penalty,
        "normal_flag_penalty": args.normal_flag_penalty,
        "normal_block_penalty": args.normal_block_penalty,
        "normal_isolate_penalty": args.normal_isolate_penalty,
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "start_episode": args.start_episode,
        "fixed_confidence_threshold": args.fixed_confidence_threshold,
        "early_stop_fpr": args.early_stop_fpr,
        "target_validation_fpr": TARGET_VALIDATION_FPR,
        "state_dim": split.X_train.shape[1],
        "actions": list(ACTION_NAMES),
        "reward": {
            "allow_normal": args.correct_allow,
            "flag_normal": -args.normal_flag_penalty,
            "block_normal": -args.normal_block_penalty,
            "isolate_normal": -args.normal_isolate_penalty,
            "allow_attack": -args.missed_attack_penalty,
            "flag_attack": args.attack_flag_reward,
            "block_attack": args.attack_block_reward,
            "isolate_attack": args.attack_isolate_reward,
        },
        "dqn": vars(CFG.dqn),
    }
    (args.run_dir / "config_used.json").write_text(
        json.dumps(config_used, indent=2, default=str), encoding="utf-8"
    )

    for local_episode in range(args.episodes):
        episode = args.start_episode + local_episode
        indices = balanced_episode_indices(split.y_train, args.max_steps, rng)
        rewards, losses = [], []
        for step, index in enumerate(indices):
            state = split.X_train[index]
            action = agent.act(state)
            reward = reward_for(
                action,
                int(split.y_train[index]),
                args.correct_allow,
                args.attack_flag_reward,
                args.attack_block_reward,
                args.attack_isolate_reward,
                args.missed_attack_penalty,
                args.normal_flag_penalty,
                args.normal_block_penalty,
                args.normal_isolate_penalty,
            )
            next_index = indices[(step + 1) % len(indices)]
            # Each dataset row is an independent flow decision, not a temporal
            # successor of the next sampled row. Terminal transitions prevent
            # unrelated-flow Q-values from contaminating the reward target.
            agent.remember(state, action, reward, split.X_train[next_index], True)
            loss = agent.learn()
            agent.decay_epsilon()
            rewards.append(reward)
            if loss:
                losses.append(loss)

        metrics = {
            "precision": np.nan, "recall": np.nan, "f1": np.nan, "fpr": np.nan,
            "composite_score": np.nan, "normal_allow_pct": np.nan,
            "normal_flag_pct": np.nan, "normal_block_pct": np.nan,
            "normal_isolate_pct": np.nan,
        }
        if local_episode % args.eval_every == 0 or local_episode == args.episodes - 1:
            proposed_actions, confidence = infer_policy(agent, split.X_val)
            if args.fixed_confidence_threshold is None:
                threshold, val_actions, val_metrics = select_confidence_threshold(
                    split.y_val, proposed_actions, confidence
                )
            else:
                threshold = args.fixed_confidence_threshold
                val_actions = np.where(
                    (proposed_actions != ACTION_ALLOW) & (confidence < threshold),
                    ACTION_ALLOW,
                    proposed_actions,
                )
                val_metrics = binary_metrics(split.y_val, val_actions)
            metrics.update(val_metrics)
            metrics["confidence_threshold"] = threshold
            normal_dist = action_distribution(split.y_val, val_actions, label=0)
            normal_total = max(sum(normal_dist.values()), 1)
            metrics.update({
                "composite_score": 0.7 * metrics["f1"] - 0.3 * metrics["fpr"],
                **{
                    f"normal_{name}_pct": count / normal_total
                    for name, count in normal_dist.items()
                },
            })
            print(f"  validation normal actions: {normal_dist}")
            if metrics["normal_isolate_pct"] > 0.30:
                print("  SANITY WARNING: isolate(normal) exceeds 30%; reward remains unbalanced.")
            if metrics["composite_score"] > best_score:
                best_score = float(metrics["composite_score"])
                best_f1 = float(metrics["f1"])
                best_episode = episode
                agent.save(checkpoint_path, metadata={
                    "state_dim": split.X_train.shape[1],
                    "num_actions": NUM_ACTIONS,
                    "feature_names": split.feature_names,
                    "best_val_f1": best_f1,
                    "best_composite_score": best_score,
                    "best_val_fpr": float(metrics["fpr"]),
                    "confidence_threshold": threshold,
                    "episode": episode,
                })
        row = {
            "episode": episode, "reward": float(np.sum(rewards)),
            "avg_reward": float(np.mean(rewards)),
            "epsilon": float(agent.epsilon),
            "loss": float(np.mean(losses)) if losses else 0.0,
            **metrics,
        }
        rows.append(row)
        print(
            f"Episode {episode:3d} | reward={row['reward']:.2f} | "
            f"epsilon={row['epsilon']:.4f} | loss={row['loss']:.4f} | "
            f"val_f1={row['f1']:.4f} | val_fpr={row['fpr']:.4f} | "
            f"score={row['composite_score']:.4f}"
        )
        if (
            local_episode >= args.early_stop_after
            and local_episode % args.eval_every == 0
            and metrics["fpr"] > args.early_stop_fpr
        ):
            print(
                f"Early stop: validation FPR {metrics['fpr']:.4f} exceeds "
                f"{args.early_stop_fpr:.4f} after episode {episode}."
            )
            break
        if (
            args.success_stop_fpr is not None
            and args.success_stop_f1 is not None
            and local_episode % args.eval_every == 0
            and metrics["fpr"] < args.success_stop_fpr
            and metrics["f1"] > args.success_stop_f1
        ):
            print(
                f"Early success stop: validation FPR {metrics['fpr']:.4f} < "
                f"{args.success_stop_fpr:.4f} and F1 {metrics['f1']:.4f} > "
                f"{args.success_stop_f1:.4f}."
            )
            break

    pd.DataFrame(rows).to_csv(args.run_dir / "training_log.csv", index=False)
    print(
        f"Best checkpoint: {checkpoint_path} "
        f"(episode={best_episode}, validation F1={best_f1:.4f}, score={best_score:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
