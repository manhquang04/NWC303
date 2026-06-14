#!/usr/bin/env python3
"""DQN reward ablation on ARP window aggregates."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn


REWARDS = {
    "aggressive_reward": {"tp": 5.0, "tn": 0.5, "fp": -1.5, "fn": -10.0},
    "conservative_reward": {"tp": 4.0, "tn": 1.0, "fp": -3.0, "fn": -6.0},
    "recall_prioritized_reward": {"tp": 6.0, "tn": 0.5, "fp": -2.0, "fn": -12.0},
    "fpr_constrained_reward": {"tp": 4.0, "tn": 1.0, "fp": -5.0, "fn": -7.0, "fpr_target": 0.10, "over": -8.0},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("processed/arp_window_aggregates"))
    p.add_argument("--results-dir", type=Path, default=Path("results/arp_window_dqn"))
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_split(base, split):
    return pd.read_parquet(base / split / "part-00000.parquet")


class QNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x):
        return self.net(x)


@dataclass
class Item:
    s: np.ndarray
    a: int
    r: float
    ns: np.ndarray
    done: bool


def reward(a, y, cfg, fp_count=0, normal_seen=0):
    if y == 1:
        return cfg["tp"] if a == 1 else cfg["fn"]
    if a == 0:
        return cfg["tn"]
    r = cfg["fp"]
    if "fpr_target" in cfg:
        fpr = (fp_count + 1) / max(normal_seen + 1, 1)
        r += max(0, fpr - cfg["fpr_target"]) * cfg["over"]
    return r


def safe_auc(y, s, kind):
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, s) if kind == "roc" else average_precision_score(y, s))


def metrics(y, pred, score):
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": float(pr), "recall": float(rc), "f1": float(f1), "fpr": float(fp / max(fp + tn, 1)), "auroc": safe_auc(y, score, "roc"), "pr_auc": safe_auc(y, score, "pr"), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def predict(model, x):
    with torch.no_grad():
        q = model(torch.tensor(x, dtype=torch.float32)).numpy()
    exp = np.exp(q - q.max(axis=1, keepdims=True))
    prob = exp / exp.sum(axis=1, keepdims=True)
    return q.argmax(axis=1).astype(np.int8), prob[:, 1]


def train_one(xtr, ytr, xv, yv, cfg, epochs):
    q, target = QNet(xtr.shape[1]), QNet(xtr.shape[1])
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=8e-4)
    loss_fn = nn.SmoothL1Loss()
    buf = deque(maxlen=10000)
    curves = []
    best_state, best_f1, steps = None, -1, 0
    for ep in range(epochs):
        eps = max(0.05, 1 - ep / (epochs * 0.75))
        order = np.arange(len(xtr))
        # preserve sequence mostly, but rotate start point by epoch without random split.
        order = np.roll(order, ep % len(order))
        fp_count = normal_seen = 0
        losses = []
        total_r = 0.0
        for j, i in enumerate(order):
            done = j == len(order) - 1
            ni = order[j + 1] if not done else i
            s, ns = xtr[i], xtr[ni]
            a = random.randrange(2) if random.random() < eps else int(q(torch.tensor(s[None, :], dtype=torch.float32)).argmax(1).item())
            r = reward(a, int(ytr[i]), cfg, fp_count, normal_seen)
            if ytr[i] == 0:
                normal_seen += 1
                fp_count += int(a == 1)
            total_r += r
            buf.append(Item(s, a, r, ns, done))
            steps += 1
            if len(buf) >= 64 and steps % 8 == 0:
                batch = random.sample(buf, 64)
                states = torch.tensor(np.stack([b.s for b in batch]), dtype=torch.float32)
                actions = torch.tensor([b.a for b in batch]).long()
                rewards = torch.tensor([b.r for b in batch], dtype=torch.float32)
                nstates = torch.tensor(np.stack([b.ns for b in batch]), dtype=torch.float32)
                dones = torch.tensor([b.done for b in batch], dtype=torch.float32)
                qsa = q(states).gather(1, actions[:, None]).squeeze(1)
                with torch.no_grad():
                    na = q(nstates).argmax(1)
                    nq = target(nstates).gather(1, na[:, None]).squeeze(1)
                    tq = rewards + 0.97 * (1 - dones) * nq
                loss = loss_fn(qsa, tq)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 5); opt.step()
                losses.append(float(loss.item()))
            if steps % 100 == 0:
                target.load_state_dict(q.state_dict())
        vp, vs = predict(q, xv)
        vm = metrics(yv, vp, vs)
        curves.append({"epoch": ep + 1, "epsilon": eps, "train_return": total_r, "loss": np.mean(losses) if losses else np.nan, "val_f1": vm["f1"], "val_fpr": vm["fpr"]})
        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            best_state = {k: v.detach().clone() for k, v in q.state_dict().items()}
    if best_state:
        q.load_state_dict(best_state)
    return q, pd.DataFrame(curves), best_f1


def main():
    args = parse_args(); set_seed(args.seed); args.results_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = [load_split(args.input_dir, s) for s in ["train", "val", "test"]]
    features = [c for c in train.columns if c not in {"split", "window_id", "row_start", "row_end", "attack_packets", "attack_ratio", "label"}]
    scaler = StandardScaler().fit(train[features])
    xtr, xv, xt = [scaler.transform(df[features]).astype(np.float32) for df in [train, val, test]]
    ytr, yv, yt = [df.label.to_numpy(np.int8) for df in [train, val, test]]
    all_metrics, all_curves = [], []
    best = None
    for name, cfg in REWARDS.items():
        model, curves, best_val = train_one(xtr, ytr, xv, yv, cfg, args.epochs)
        curves.insert(0, "reward", name)
        all_curves.append(curves)
        for split, x, y in [("val", xv, yv), ("test", xt, yt)]:
            pred, score = predict(model, x)
            row = {"reward": name, "split": split, "best_val_f1": best_val, **metrics(y, pred, score)}
            all_metrics.append(row)
            if split == "test" and (best is None or row["f1"] > best["f1"]):
                best = {**row, "pred": pred}
    metrics_df = pd.DataFrame(all_metrics)
    curves_df = pd.concat(all_curves, ignore_index=True)
    metrics_df.to_csv(args.results_dir / "arp_dqn_reward_ablation.csv", index=False)
    curves_df.to_csv(args.results_dir / "arp_dqn_training_curves.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    for reward_name, g in curves_df.groupby("reward"):
        ax.plot(g.epoch, g.val_f1, label=reward_name)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation F1"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(args.results_dir / "arp_dqn_training_curves.png", dpi=180); plt.close(fig)
    if best:
        cm = confusion_matrix(yt, best["pred"], labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4)); ax.imshow(cm, cmap="Blues"); ax.set_title(best["reward"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout(); fig.savefig(args.results_dir / "arp_dqn_confusion_matrix.png", dpi=180); plt.close(fig)
    lines = ["ARP window DQN reward ablation", ""]
    for _, r in metrics_df[metrics_df.split == "test"].sort_values("f1", ascending=False).iterrows():
        lines.append(f"- {r.reward}: precision={r.precision:.4f}, recall={r.recall:.4f}, f1={r.f1:.4f}, fpr={r.fpr:.4f}, auroc={r.auroc:.4f}, pr_auc={r.pr_auc:.4f}")
    (args.results_dir / "short_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
