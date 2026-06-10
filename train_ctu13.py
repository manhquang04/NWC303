"""Train the repository DQN on sampled CTU-13 flow records."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from agent.dqn_agent import DQNAgent
from config import ACTION_ALLOW, ACTION_BLOCK, ACTION_FLAG, ACTION_ISOLATE, CFG, NUM_ACTIONS
from dataset.ctu13_loader import CTU13Loader


def reward_for(action: int, label: int) -> float:
    """Return the configured reward for an executed action and binary label."""
    reward = CFG.reward
    if label == 0:
        return {
            ACTION_ALLOW: reward.r_normal_allowed,
            ACTION_FLAG: reward.r_normal_flagged,
            ACTION_BLOCK: reward.r_normal_blocked,
            ACTION_ISOLATE: reward.r_isolate_wrong,
        }[action]
    return {
        ACTION_ALLOW: reward.r_attack_ignored,
        ACTION_FLAG: reward.r_attack_flagged,
        ACTION_BLOCK: reward.r_attack_blocked,
        ACTION_ISOLATE: reward.r_isolate_correct,
    }[action]


def greedy_actions(agent: DQNAgent, X: np.ndarray) -> np.ndarray:
    """Run batched greedy inference with the DQN policy."""
    import torch

    output = []
    agent.q_net.eval()
    with torch.no_grad():
        for start in range(0, len(X), 4096):
            batch = torch.from_numpy(X[start:start + 4096]).float().to(agent.device)
            output.append(agent.q_net(batch).argmax(dim=1).cpu().numpy())
    agent.q_net.train()
    return np.concatenate(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("~/Tải về/CTU-13-Dataset/Dataset/"))
    parser.add_argument("--sample-frac", type=float, default=0.10)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser()
    if not data_dir.exists():
        data_dir = Path("dataset")
    X, y, feature_names = CTU13Loader().load(data_dir, args.sample_frac)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    agent = DQNAgent(state_dim=X_train.shape[1], num_actions=NUM_ACTIONS)
    rng = np.random.default_rng(CFG.dqn.seed)
    random.seed(CFG.dqn.seed)
    rows = []
    best_f1 = -1.0
    checkpoint = Path("runs/ctu13/dqn_best.pt")

    for episode in range(args.episodes):
        indices = rng.choice(len(X_train), size=args.max_steps, replace=len(X_train) < args.max_steps)
        rewards = []
        losses = []
        for step, index in enumerate(indices):
            state = X_train[index]
            action = agent.act(state)
            reward = reward_for(action, int(y_train[index]))
            next_index = indices[(step + 1) % len(indices)]
            done = step == len(indices) - 1
            agent.remember(state, action, reward, X_train[next_index], done)
            loss = agent.learn()
            agent.decay_epsilon()
            rewards.append(reward)
            if loss:
                losses.append(loss)

        actions = greedy_actions(agent, X_test)
        y_pred = (actions != ACTION_ALLOW).astype(np.int64)
        eval_f1 = f1_score(y_test, y_pred, zero_division=0)
        row = {
            "episode": episode,
            "reward": float(np.sum(rewards)),
            "epsilon": agent.epsilon,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "eval_f1": eval_f1,
        }
        rows.append(row)
        if eval_f1 > best_f1:
            best_f1 = eval_f1
            agent.save(checkpoint, metadata={
                "state_dim": X_train.shape[1],
                "num_actions": NUM_ACTIONS,
                "feature_names": feature_names,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "best_f1": best_f1,
                "episode": episode,
            })
        if episode % 10 == 0 or episode == args.episodes - 1:
            print(
                f"Episode {episode:3d} | reward={row['reward']:.2f} | "
                f"epsilon={row['epsilon']:.4f} | loss={row['loss']:.4f} | F1={eval_f1:.4f}"
            )

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("results/train_ctu13.csv", index=False)
    print(f"Best checkpoint: {checkpoint} (F1={best_f1:.4f})")
    print("Saved: results/train_ctu13.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

