#!/usr/bin/env python3
"""DQN experiments for Rogue AP event-centered windows.

The environment is intentionally small and auditable: each episode is one
source file, each step is one event-centered window ordered by row_start, and
source metadata is kept only for grouping metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn


FEATURES = [
    "n_frames",
    "ratio_retry",
    "ratio_protected",
    "ratio_moredata",
    "ratio_pwrmgt",
    "ratio_order",
    "ratio_frag",
    "ratio_ds_0x00000000",
    "ratio_ds_0x00000001",
    "ratio_ds_0x00000002",
    "ratio_ds_0x00000003",
    "ratio_type_0",
    "ratio_type_1",
    "ratio_type_2",
    "ratio_llc_present",
    "ratio_ip_present",
    "ratio_tcp_present",
]

ACTION_NAMES = {0: "allow_ignore", 1: "flag_rogue_ap", 2: "isolate_escalate"}

REWARDS = {
    "conservative_reward": {
        "tp_flag": 4.0,
        "tp_isolate": 3.0,
        "tn": 1.0,
        "fp_flag": -2.0,
        "fp_isolate": -5.0,
        "fn": -6.0,
        "delay_penalty": -0.02,
    },
    "aggressive_reward": {
        "tp_flag": 5.0,
        "tp_isolate": 7.0,
        "tn": 0.5,
        "fp_flag": -1.5,
        "fp_isolate": -3.0,
        "fn": -10.0,
        "delay_penalty": -0.01,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("processed/rogue_ap_event_windows/tight_event"))
    p.add_argument("--supervised-results-dir", type=Path, default=Path("results/rogue_ap_event_windows"))
    p.add_argument("--results-dir", type=Path, default=Path("results/rogue_ap_rl_dqn"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target-sync", type=int, default=100)
    p.add_argument("--buffer-size", type=int, default=10000)
    p.add_argument("--train-every", type=int, default=8)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(base: Path, split: str) -> pd.DataFrame:
    df = pd.read_parquet(base / split / "part-00000.parquet")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in {split}: {missing}")
    return df.sort_values(["source_file", "row_start", "row_end", "event_id"]).reset_index(drop=True)


def safe_auc(y: np.ndarray, score: np.ndarray, kind: str) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score) if kind == "roc" else average_precision_score(y, score))


def eval_binary(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / max(fp + tn, 1)),
        "auroc": safe_auc(y, score, "roc"),
        "pr_auc": safe_auc(y, score, "pr"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def reward_for(action: int, label: int, position: int, length: int, cfg: dict[str, float]) -> float:
    delay = position / max(length - 1, 1)
    if label == 1:
        if action == 0:
            return cfg["fn"]
        if action == 1:
            return cfg["tp_flag"] + cfg["delay_penalty"] * delay
        return cfg["tp_isolate"] + cfg["delay_penalty"] * delay
    if action == 0:
        return cfg["tn"]
    if action == 1:
        return cfg["fp_flag"]
    return cfg["fp_isolate"]


@dataclass
class ReplayItem:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf: deque[ReplayItem] = deque(maxlen=capacity)

    def add(self, item: ReplayItem) -> None:
        self.buf.append(item)

    def sample(self, batch_size: int) -> list[ReplayItem]:
        return random.sample(self.buf, min(batch_size, len(self.buf)))

    def __len__(self) -> int:
        return len(self.buf)


class QNet(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, dueling: bool = False) -> None:
        super().__init__()
        self.dueling = dueling
        self.body = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        if dueling:
            self.value = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
            self.adv = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_actions))
        else:
            self.head = nn.Linear(64, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x)
        if not self.dueling:
            return self.head(z)
        value = self.value(z)
        adv = self.adv(z)
        return value + adv - adv.mean(dim=1, keepdim=True)


def make_episodes(df: pd.DataFrame, x: np.ndarray) -> list[dict[str, Any]]:
    tmp = df[["source_file", "file_id", "event_id", "window_kind", "row_start", "row_end", "rogue_frames", "label"]].copy()
    tmp["_idx"] = np.arange(len(tmp))
    episodes = []
    for source_file, g in tmp.groupby("source_file", sort=True):
        idx = g["_idx"].to_numpy()
        episodes.append({"source_file": source_file, "meta": g.reset_index(drop=True), "x": x[idx], "y": g["label"].to_numpy(dtype=np.int8)})
    return episodes


def train_agent(
    train_eps: list[dict[str, Any]],
    val_eps: list[dict[str, Any]],
    input_dim: int,
    agent_name: str,
    reward_name: str,
    reward_cfg: dict[str, float],
    args: argparse.Namespace,
) -> tuple[QNet, pd.DataFrame, dict[str, float]]:
    n_actions = 3
    double = agent_name in {"double_dqn", "dueling_dqn"}
    dueling = agent_name == "dueling_dqn"
    q = QNet(input_dim, n_actions, dueling=dueling)
    target = QNet(input_dim, n_actions, dueling=dueling)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()
    replay = ReplayBuffer(args.buffer_size)
    steps = 0
    curves = []
    best_state = None
    best_val_f1 = -1.0

    for epoch in range(args.epochs):
        eps = max(0.05, 1.0 - epoch / max(args.epochs * 0.75, 1))
        random.shuffle(train_eps)
        epoch_return = 0.0
        epoch_loss = []
        for ep in train_eps:
            x, y = ep["x"], ep["y"]
            ep_return = 0.0
            for t in range(len(x)):
                state = x[t].astype(np.float32)
                done = t == len(x) - 1
                next_state = x[t + 1].astype(np.float32) if not done else np.zeros_like(state)
                if random.random() < eps:
                    action = random.randrange(n_actions)
                else:
                    with torch.no_grad():
                        action = int(q(torch.tensor(state[None, :], dtype=torch.float32)).argmax(dim=1).item())
                r = reward_for(action, int(y[t]), t, len(x), reward_cfg)
                replay.add(ReplayItem(state, action, r, next_state, done))
                ep_return += r
                steps += 1
                if len(replay) >= args.batch_size and steps % args.train_every == 0:
                    batch = replay.sample(args.batch_size)
                    states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32)
                    actions = torch.tensor([b.action for b in batch], dtype=torch.int64)
                    rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32)
                    next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32)
                    dones = torch.tensor([b.done for b in batch], dtype=torch.float32)
                    q_sa = q(states).gather(1, actions[:, None]).squeeze(1)
                    with torch.no_grad():
                        if double:
                            next_actions = q(next_states).argmax(dim=1)
                            next_q = target(next_states).gather(1, next_actions[:, None]).squeeze(1)
                        else:
                            next_q = target(next_states).max(dim=1).values
                        target_q = rewards + args.gamma * (1.0 - dones) * next_q
                    loss = loss_fn(q_sa, target_q)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                    opt.step()
                    epoch_loss.append(float(loss.item()))
                if steps % args.target_sync == 0:
                    target.load_state_dict(q.state_dict())
            epoch_return += ep_return
        val = evaluate_agent(q, val_eps, reward_cfg)
        curves.append({
            "agent": agent_name,
            "reward_variant": reward_name,
            "epoch": epoch + 1,
            "epsilon": eps,
            "train_return": epoch_return / max(len(train_eps), 1),
            "loss": float(np.mean(epoch_loss)) if epoch_loss else float("nan"),
            "val_f1": val["f1"],
            "val_fpr": val["fpr"],
            "val_return": val["episode_return_mean"],
        })
        if val["f1"] > best_val_f1:
            best_val_f1 = val["f1"]
            best_state = {k: v.detach().clone() for k, v in q.state_dict().items()}
    if best_state is not None:
        q.load_state_dict(best_state)
    return q, pd.DataFrame(curves), {"best_val_f1": best_val_f1}


def evaluate_agent(model: QNet, episodes: list[dict[str, Any]], reward_cfg: dict[str, float]) -> dict[str, Any]:
    y_all, pred_all, score_all, returns, latencies = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for ep in episodes:
            x, y = ep["x"], ep["y"]
            qv = model(torch.tensor(x, dtype=torch.float32)).numpy()
            actions = qv.argmax(axis=1)
            # Positive score is the best non-allow Q value relative to all actions.
            exp = np.exp(qv - qv.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
            scores = probs[:, 1:].sum(axis=1)
            rewards = [reward_for(int(a), int(lbl), t, len(x), reward_cfg) for t, (a, lbl) in enumerate(zip(actions, y))]
            returns.append(float(np.sum(rewards)))
            positives = np.flatnonzero(y == 1)
            detections = np.flatnonzero((y == 1) & (actions > 0))
            if len(positives) and len(detections):
                latencies.append(float(max(0, detections[0] - positives[0])))
            y_all.extend(y.tolist())
            pred_all.extend((actions > 0).astype(np.int8).tolist())
            score_all.extend(scores.tolist())
    y_arr = np.asarray(y_all, dtype=np.int8)
    pred_arr = np.asarray(pred_all, dtype=np.int8)
    score_arr = np.asarray(score_all, dtype=float)
    out = eval_binary(y_arr, pred_arr, score_arr)
    out.update({
        "episode_return_mean": float(np.mean(returns)) if returns else 0.0,
        "episode_return_sum": float(np.sum(returns)) if returns else 0.0,
        "avg_decision_latency": float(np.mean(latencies)) if latencies else float("nan"),
    })
    return out


def prediction_frame(model: QNet, episodes: list[dict[str, Any]], reward_cfg: dict[str, float]) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for ep in episodes:
            qv = model(torch.tensor(ep["x"], dtype=torch.float32)).numpy()
            actions = qv.argmax(axis=1)
            exp = np.exp(qv - qv.max(axis=1, keepdims=True))
            scores = (exp / exp.sum(axis=1, keepdims=True))[:, 1:].sum(axis=1)
            meta = ep["meta"].copy()
            meta["action"] = actions
            meta["action_name"] = [ACTION_NAMES[int(a)] for a in actions]
            meta["pred"] = (actions > 0).astype(np.int8)
            meta["score"] = scores
            meta["reward"] = [reward_for(int(a), int(lbl), t, len(meta), reward_cfg) for t, (a, lbl) in enumerate(zip(actions, meta["label"]))]
            rows.append(meta)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def grouped_metrics(preds: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in preds.groupby(group_cols, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        y = g["label"].to_numpy(dtype=np.int8)
        p = g["pred"].to_numpy(dtype=np.int8)
        s = g["score"].to_numpy(dtype=float)
        row = dict(zip(group_cols, key_tuple))
        row.update({
            "windows": len(g),
            "positive_windows": int(y.sum()),
            "rogue_frames": int(g["rogue_frames"].sum()),
            "episode_return": float(g["reward"].sum()),
        })
        row.update(eval_binary(y, p, s))
        rows.append(row)
    return pd.DataFrame(rows)


def write_reports(args: argparse.Namespace, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    env_lines = [
        "RL environment for Rogue AP tight_event windows",
        "",
        "Episode: one source_file, ordered by row_start within the fixed file split.",
        "State: 17 aggregate behavior features only; source_file, file_id, event_id, row positions, window_kind, and labels are excluded from the feature matrix.",
        "Actions: 0=allow/ignore, 1=flag Rogue AP, 2=isolate/escalate.",
        "Transition: next event/window in the same source_file episode.",
        "Terminal: end of source_file episode.",
        "Evaluation: train on train files, select best checkpoint by validation F1, report test once.",
        "",
        f"Train rows={len(train)}, val rows={len(val)}, test rows={len(test)}.",
        f"Train label counts={train.label.value_counts().to_dict()}, val={val.label.value_counts().to_dict()}, test={test.label.value_counts().to_dict()}.",
    ]
    (args.results_dir / "rl_environment_design_report.txt").write_text("\n".join(env_lines), encoding="utf-8")

    reward_lines = ["Reward variants", ""]
    for name, cfg in REWARDS.items():
        reward_lines.append(f"{name}: {json.dumps(cfg, sort_keys=True)}")
        if name == "conservative_reward":
            reward_lines.append("Trade-off: punishes isolate false positives heavily, so it should prefer precision/lower FPR.")
        else:
            reward_lines.append("Trade-off: punishes false negatives most heavily and rewards isolate on true positives, so it should prefer recall.")
        reward_lines.append("")
    (args.results_dir / "reward_design_report.txt").write_text("\n".join(reward_lines), encoding="utf-8")


def supervised_comparison(supervised_dir: Path, rl_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    path = supervised_dir / "metrics_by_variant.csv"
    if path.exists():
        sup = pd.read_csv(path)
        sup = sup[(sup["variant"] == "tight_event") & (sup["split"] == "test") & (sup["threshold_policy"] == "val_best_f1")]
        for _, r in sup.iterrows():
            rows.append({
                "system": "supervised",
                "model": r["model"],
                "reward_variant": "",
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "fpr": r["fpr"],
                "auroc": r["test_auroc"],
                "pr_auc": r["test_pr_auc"],
                "avg_decision_latency": np.nan,
            })
    best_rl = rl_metrics[rl_metrics["split"] == "test"].copy()
    for _, r in best_rl.iterrows():
        rows.append({
            "system": "rl",
            "model": r["agent"],
            "reward_variant": r["reward_variant"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "fpr": r["fpr"],
            "auroc": r["auroc"],
            "pr_auc": r["pr_auc"],
            "avg_decision_latency": r["avg_decision_latency"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(args.input_dir, "train")
    val = load_split(args.input_dir, "val")
    test = load_split(args.input_dir, "test")
    write_reports(args, train, val, test)

    scaler = StandardScaler().fit(train[FEATURES])
    xtr = scaler.transform(train[FEATURES]).astype(np.float32)
    xv = scaler.transform(val[FEATURES]).astype(np.float32)
    xt = scaler.transform(test[FEATURES]).astype(np.float32)
    train_eps = make_episodes(train, xtr)
    val_eps = make_episodes(val, xv)
    test_eps = make_episodes(test, xt)

    all_metrics, all_curves, all_per_file, all_per_event = [], [], [], []
    best_key, best_f1, best_preds = None, -1.0, None
    agents = ["dqn", "double_dqn", "dueling_dqn"]
    for reward_name, reward_cfg in REWARDS.items():
        for agent in agents:
            model, curves, info = train_agent(train_eps, val_eps, len(FEATURES), agent, reward_name, reward_cfg, args)
            all_curves.append(curves)
            for split_name, eps in [("val", val_eps), ("test", test_eps)]:
                met = evaluate_agent(model, eps, reward_cfg)
                all_metrics.append({"agent": agent, "reward_variant": reward_name, "split": split_name, **met, **info})
            preds = prediction_frame(model, test_eps, reward_cfg)
            pf = grouped_metrics(preds, ["source_file"])
            pf.insert(0, "reward_variant", reward_name)
            pf.insert(0, "agent", agent)
            all_per_file.append(pf)
            pe = preds[["source_file", "event_id", "window_kind", "label", "rogue_frames", "action", "action_name", "pred", "score", "reward"]].copy()
            pe.insert(0, "reward_variant", reward_name)
            pe.insert(0, "agent", agent)
            all_per_event.append(pe)
            test_f1 = all_metrics[-1]["f1"]
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_key = (agent, reward_name)
                best_preds = preds

    metrics = pd.DataFrame(all_metrics)
    curves_df = pd.concat(all_curves, ignore_index=True)
    per_file = pd.concat(all_per_file, ignore_index=True)
    per_event = pd.concat(all_per_event, ignore_index=True)
    metrics.to_csv(args.results_dir / "rl_metrics.csv", index=False)
    per_file.to_csv(args.results_dir / "rl_per_file_metrics.csv", index=False)
    per_event.to_csv(args.results_dir / "rl_per_event_metrics.csv", index=False)
    curves_df.to_csv(args.results_dir / "rl_training_curves.csv", index=False)

    comparison = supervised_comparison(args.supervised_results_dir, metrics)
    comparison.to_csv(args.results_dir / "comparison_with_supervised.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (agent, reward), g in curves_df.groupby(["agent", "reward_variant"]):
        ax.plot(g["epoch"], g["val_f1"], label=f"{agent}/{reward.replace('_reward','')}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation F1")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.results_dir / "rl_training_curves.png", dpi=180)
    plt.close(fig)

    if best_preds is not None:
        cm = confusion_matrix(best_preds["label"], best_preds["pred"], labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{best_key[0]} / {best_key[1]}")
        ax.set_xticks([0, 1], ["pred normal", "pred rogue"])
        ax.set_yticks([0, 1], ["true normal", "true rogue"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(args.results_dir / "rl_confusion_matrix.png", dpi=180)
        plt.close(fig)

    best_rl = metrics[metrics["split"] == "test"].sort_values("f1", ascending=False).head(1)
    best_sup = comparison[comparison["system"] == "supervised"].sort_values("f1", ascending=False).head(1)
    lines = ["Rogue AP DQN short summary", ""]
    if not best_rl.empty:
        r = best_rl.iloc[0]
        lines.append(f"Best RL: {r.agent}/{r.reward_variant}, precision={r.precision:.4f}, recall={r.recall:.4f}, f1={r.f1:.4f}, fpr={r.fpr:.4f}, auroc={r.auroc:.4f}, pr_auc={r.pr_auc:.4f}.")
    if not best_sup.empty:
        s = best_sup.iloc[0]
        lines.append(f"Best supervised tight_event baseline: {s.model}, precision={s.precision:.4f}, recall={s.recall:.4f}, f1={s.f1:.4f}, fpr={s.fpr:.4f}, auroc={s.auroc:.4f}, pr_auc={s.pr_auc:.4f}.")
    lines.append("")
    lines.append("source_file/file_id/event_id are used only for audit grouping and are not included in the RL state.")
    lines.append("RL is useful here only if it improves the precision/recall/FPR trade-off against supervised RF/XGBoost; otherwise supervised hardening remains the better path.")
    (args.results_dir / "short_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
