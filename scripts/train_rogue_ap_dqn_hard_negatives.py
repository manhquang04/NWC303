#!/usr/bin/env python3
"""DQN hard-negative and FPR-constrained reward experiments."""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from torch import nn


FEATURES = [
    "n_frames", "ratio_retry", "ratio_protected", "ratio_moredata", "ratio_pwrmgt", "ratio_order",
    "ratio_frag", "ratio_ds_0x00000000", "ratio_ds_0x00000001", "ratio_ds_0x00000002",
    "ratio_ds_0x00000003", "ratio_type_0", "ratio_type_1", "ratio_type_2",
    "ratio_llc_present", "ratio_ip_present", "ratio_tcp_present",
]
AUDIT_COLS = ["source_file", "file_id", "event_id", "window_kind", "row_start", "row_end", "rogue_frames", "label"]
ACTION_NAMES = {0: "allow_ignore", 1: "flag_rogue_ap", 2: "isolate_escalate"}

REWARDS = {
    "fpr_constrained_reward": {
        "tp_flag": 4.5, "tp_isolate": 4.0, "tn": 1.0, "fn": -8.0,
        "fp_flag": -4.0, "fp_isolate": -8.0, "streak": -1.5, "fpr_target": 0.25, "fpr_over": -8.0,
    },
    "recall_prioritized_reward": {
        "tp_flag": 5.0, "tp_isolate": 6.0, "tn": 0.6, "fn": -11.0,
        "fp_flag": -2.5, "fp_isolate": -5.0, "streak": -0.8, "fpr_target": 0.35, "fpr_over": -4.0,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("processed/rogue_ap_event_windows/tight_event"))
    p.add_argument("--supervised-results-dir", type=Path, default=Path("results/rogue_ap_event_windows"))
    p.add_argument("--old-rl-results-dir", type=Path, default=Path("results/rogue_ap_rl_dqn"))
    p.add_argument("--results-dir", type=Path, default=Path("results/rogue_ap_dqn_hard_negative"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--target-sync", type=int, default=100)
    p.add_argument("--train-every", type=int, default=8)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(base: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(base / split / "part-00000.parquet").sort_values(["source_file", "row_start", "row_end", "event_id"]).reset_index(drop=True)


def mine_negative_groups(train: pd.DataFrame, scaler: StandardScaler) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = train.copy()
    df["negative_group"] = "positive"
    pos = df[df.label == 1].copy()
    neg = df[df.label == 0].copy()
    x_pos = scaler.transform(pos[FEATURES])
    x_neg = scaler.transform(neg[FEATURES])
    sim = cosine_similarity(x_neg, x_pos).max(axis=1) if len(pos) and len(neg) else np.zeros(len(neg))
    neg["max_pos_similarity"] = sim

    distances = []
    for _, r in neg.iterrows():
        p = pos[pos.source_file == r.source_file]
        if len(p) == 0:
            distances.append(np.inf)
        else:
            distances.append(float(np.minimum(abs(p.row_start - r.row_end), abs(r.row_start - p.row_end)).clip(lower=0).min()))
    neg["nearest_event_distance"] = distances

    near_cut = np.nanpercentile(neg.replace([np.inf], np.nan)["nearest_event_distance"].dropna(), 35) if np.isfinite(neg["nearest_event_distance"]).any() else 0
    sim_hi = np.nanpercentile(neg["max_pos_similarity"], 85) if len(neg) else 1
    sim_mid = np.nanpercentile(neg["max_pos_similarity"], 65) if len(neg) else 1
    neg["negative_group"] = "easy_negative"
    neg.loc[neg["nearest_event_distance"] <= near_cut, "negative_group"] = "hard_negative"
    neg.loc[(neg["max_pos_similarity"] >= sim_hi) | ((neg["max_pos_similarity"] >= sim_mid) & (neg["nearest_event_distance"] <= near_cut * 2)), "negative_group"] = "borderline_negative"

    # Augment train only: duplicate hard/borderline negatives so replay sees them more often.
    hard = neg[neg.negative_group == "hard_negative"]
    border = neg[neg.negative_group == "borderline_negative"]
    easy = neg[neg.negative_group == "easy_negative"]
    pos = pos.assign(negative_group="positive", max_pos_similarity=np.nan, nearest_event_distance=np.nan)
    augmented = pd.concat([pos, easy, hard, hard, border, border, border], ignore_index=True).sort_values(["source_file", "row_start", "event_id"]).reset_index(drop=True)
    report = {
        "original_train_rows": int(len(train)),
        "augmented_train_rows": int(len(augmented)),
        "groups": {k: int(v) for k, v in neg.negative_group.value_counts().to_dict().items()},
        "positive_windows": int(len(pos)),
        "near_distance_cutoff_rows": float(near_cut),
        "similarity_borderline_cutoff": float(sim_hi),
        "method": "easy=far/low-similarity negatives; hard=normal windows close to positive event windows in the same file; borderline=normal windows most similar to positives in aggregate feature space. Only train split is augmented.",
    }
    return augmented, report


class QNet(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Item:
    s: np.ndarray
    a: int
    r: float
    ns: np.ndarray
    done: bool


def make_eps(df: pd.DataFrame, x: np.ndarray) -> list[dict[str, Any]]:
    meta = df[AUDIT_COLS + [c for c in ["negative_group", "max_pos_similarity", "nearest_event_distance"] if c in df.columns]].copy()
    meta["_idx"] = np.arange(len(meta))
    eps = []
    for sf, g in meta.groupby("source_file", sort=True):
        idx = g._idx.to_numpy()
        eps.append({"source_file": sf, "meta": g.reset_index(drop=True), "x": x[idx].astype(np.float32), "y": g.label.to_numpy(np.int8)})
    return eps


def contextual_reward(action: int, label: int, fp_count: int, normal_seen: int, fp_streak: int, cfg: dict[str, float]) -> float:
    if label == 1:
        return cfg["fn"] if action == 0 else (cfg["tp_flag"] if action == 1 else cfg["tp_isolate"])
    if action == 0:
        return cfg["tn"]
    base = cfg["fp_flag"] if action == 1 else cfg["fp_isolate"]
    fpr_est = (fp_count + 1) / max(normal_seen + 1, 1)
    over = max(0.0, fpr_est - cfg["fpr_target"])
    return base + cfg["streak"] * fp_streak + cfg["fpr_over"] * over


def safe_auc(y, s, kind):
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, s) if kind == "roc" else average_precision_score(y, s))


def eval_metrics(y, pred, score):
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": float(pr), "recall": float(rc), "f1": float(f1), "fpr": float(fp / max(fp + tn, 1)), "auroc": safe_auc(y, score, "roc"), "pr_auc": safe_auc(y, score, "pr"), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def predict(model: QNet, eps: list[dict[str, Any]], cfg: dict[str, float]) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for ep in eps:
            q = model(torch.tensor(ep["x"], dtype=torch.float32)).numpy()
            exp = np.exp(q - q.max(axis=1, keepdims=True))
            prob = exp / exp.sum(axis=1, keepdims=True)
            action = q.argmax(axis=1)
            meta = ep["meta"].copy()
            meta["action"] = action
            meta["action_name"] = [ACTION_NAMES[int(a)] for a in action]
            meta["pred"] = (action > 0).astype(np.int8)
            meta["score"] = prob[:, 1:].sum(axis=1)
            fp_count = normal_seen = fp_streak = 0
            rewards, streaks, fprs = [], [], []
            for a, y in zip(meta.action, meta.label):
                r = contextual_reward(int(a), int(y), fp_count, normal_seen, fp_streak, cfg)
                rewards.append(r)
                if y == 0:
                    normal_seen += 1
                    if a > 0:
                        fp_count += 1
                        fp_streak += 1
                    else:
                        fp_streak = 0
                else:
                    fp_streak = 0
                streaks.append(fp_streak)
                fprs.append(fp_count / max(normal_seen, 1))
            meta["reward"] = rewards
            meta["false_alarm_streak"] = streaks
            meta["episode_fpr_estimate"] = fprs
            rows.append(meta)
    return pd.concat(rows, ignore_index=True)


def train_one(train_eps, val_eps, reward_name, cfg, args):
    q, target = QNet(len(FEATURES)), QNet(len(FEATURES))
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()
    replay = deque(maxlen=12000)
    best, best_f1, curves, steps = None, -1, [], 0
    for epoch in range(args.epochs):
        eps_rate = max(0.05, 1 - epoch / max(args.epochs * 0.75, 1))
        random.shuffle(train_eps)
        losses, returns = [], []
        for ep in train_eps:
            fp_count = normal_seen = fp_streak = 0
            ep_ret = 0.0
            for t, (s, y) in enumerate(zip(ep["x"], ep["y"])):
                done = t == len(ep["x"]) - 1
                ns = ep["x"][t + 1] if not done else np.zeros_like(s)
                a = random.randrange(3) if random.random() < eps_rate else int(q(torch.tensor(s[None, :])).argmax(1).item())
                r = contextual_reward(a, int(y), fp_count, normal_seen, fp_streak, cfg)
                if y == 0:
                    normal_seen += 1
                    fp_streak = fp_streak + 1 if a > 0 else 0
                    fp_count += int(a > 0)
                else:
                    fp_streak = 0
                replay.append(Item(s, a, r, ns, done))
                ep_ret += r
                steps += 1
                if len(replay) >= args.batch_size and steps % args.train_every == 0:
                    batch = random.sample(replay, args.batch_size)
                    states = torch.tensor(np.stack([b.s for b in batch]), dtype=torch.float32)
                    actions = torch.tensor([b.a for b in batch]).long()
                    rewards = torch.tensor([b.r for b in batch], dtype=torch.float32)
                    next_states = torch.tensor(np.stack([b.ns for b in batch]), dtype=torch.float32)
                    dones = torch.tensor([b.done for b in batch], dtype=torch.float32)
                    qsa = q(states).gather(1, actions[:, None]).squeeze(1)
                    with torch.no_grad():
                        na = q(next_states).argmax(1)
                        nq = target(next_states).gather(1, na[:, None]).squeeze(1)
                        tq = rewards + args.gamma * (1 - dones) * nq
                    loss = loss_fn(qsa, tq)
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 5); opt.step()
                    losses.append(float(loss.item()))
                if steps % args.target_sync == 0:
                    target.load_state_dict(q.state_dict())
            returns.append(ep_ret)
        val_pred = predict(q, val_eps, cfg)
        vm = eval_metrics(val_pred.label.to_numpy(), val_pred.pred.to_numpy(), val_pred.score.to_numpy())
        curves.append({"reward_variant": reward_name, "epoch": epoch + 1, "epsilon": eps_rate, "loss": np.mean(losses) if losses else np.nan, "train_return": np.mean(returns), "val_f1": vm["f1"], "val_fpr": vm["fpr"]})
        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            best = {k: v.detach().clone() for k, v in q.state_dict().items()}
    if best:
        q.load_state_dict(best)
    return q, pd.DataFrame(curves), best_f1


def grouped(pred: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = []
    for key, g in pred.groupby(cols, sort=True):
        kt = key if isinstance(key, tuple) else (key,)
        y, p, s = g.label.to_numpy(), g.pred.to_numpy(), g.score.to_numpy()
        row = dict(zip(cols, kt))
        row.update({"windows": len(g), "positive_windows": int(y.sum()), "rogue_frames": int(g.rogue_frames.sum()), "avg_reward": float(g.reward.mean()), "episode_return": float(g.reward.sum()), "max_false_alarm_streak": int(g.false_alarm_streak.max())})
        row.update(eval_metrics(y, p, s))
        out.append(row)
    return pd.DataFrame(out)


def main():
    args = parse_args()
    set_seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = [load_split(args.input_dir, s) for s in ["train", "val", "test"]]
    scaler0 = StandardScaler().fit(train[FEATURES])
    train_aug, mining_report = mine_negative_groups(train, scaler0)
    scaler = StandardScaler().fit(train_aug[FEATURES])
    eps_train = make_eps(train_aug, scaler.transform(train_aug[FEATURES]))
    eps_val = make_eps(val, scaler.transform(val[FEATURES]))
    eps_test = make_eps(test, scaler.transform(test[FEATURES]))

    (args.results_dir / "dqn_hard_negative_design_report.txt").write_text(json.dumps(mining_report, indent=2), encoding="utf-8")
    (args.results_dir / "dqn_reward_design_report.txt").write_text(json.dumps(REWARDS, indent=2), encoding="utf-8")
    (args.results_dir / "dqn_environment_update_report.txt").write_text(
        "Episode remains one source_file ordered by row_start. State uses only the 17 valid aggregate features. "
        "The environment tracks cumulative false positives, false-alarm streak, and per-episode FPR estimate inside reward accounting; these audit counters are not appended to state.\n",
        encoding="utf-8",
    )

    metrics, curves, per_files, per_events = [], [], [], []
    best_pred, best_key, best_f1 = None, None, -1
    for name, cfg in REWARDS.items():
        model, c, best_val = train_one(eps_train, eps_val, name, cfg, args)
        curves.append(c)
        for split, eps in [("val", eps_val), ("test", eps_test)]:
            pred = predict(model, eps, cfg)
            m = eval_metrics(pred.label.to_numpy(), pred.pred.to_numpy(), pred.score.to_numpy())
            metrics.append({"agent": "double_dqn", "training": "hard_negative_augmented", "reward_variant": name, "split": split, "best_val_f1": best_val, "avg_reward": float(pred.reward.mean()), "episode_return_mean": float(pred.groupby("source_file").reward.sum().mean()), "max_false_alarm_streak_mean": float(pred.groupby("source_file").false_alarm_streak.max().mean()), **m})
            if split == "test":
                pf = grouped(pred, ["source_file"]); pf.insert(0, "reward_variant", name); per_files.append(pf)
                pe = pred[["source_file", "event_id", "window_kind", "label", "rogue_frames", "action", "action_name", "pred", "score", "reward", "false_alarm_streak", "episode_fpr_estimate"]].copy()
                pe.insert(0, "reward_variant", name); per_events.append(pe)
                if m["f1"] > best_f1:
                    best_f1, best_pred, best_key = m["f1"], pred, name

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(args.results_dir / "dqn_metrics.csv", index=False)
    pd.concat(per_files, ignore_index=True).to_csv(args.results_dir / "dqn_per_file_metrics.csv", index=False)
    pd.concat(per_events, ignore_index=True).to_csv(args.results_dir / "dqn_per_event_metrics.csv", index=False)
    curves_df = pd.concat(curves, ignore_index=True)
    curves_df.to_csv(args.results_dir / "dqn_training_curves.csv", index=False)

    comparison = []
    sup_path = args.supervised_results_dir / "metrics_by_variant.csv"
    if sup_path.exists():
        sup = pd.read_csv(sup_path)
        sup = sup[(sup.variant == "tight_event") & (sup.split == "test") & (sup.threshold_policy == "val_best_f1")]
        for _, r in sup.iterrows():
            comparison.append({"system": "supervised", "model": r.model, "reward_variant": "", "precision": r.precision, "recall": r.recall, "f1": r.f1, "fpr": r.fpr, "auroc": r.test_auroc, "pr_auc": r.test_pr_auc})
    old = args.old_rl_results_dir / "rl_metrics.csv"
    if old.exists():
        o = pd.read_csv(old)
        o = o[o.split == "test"]
        for _, r in o.iterrows():
            comparison.append({"system": "rl_baseline", "model": r.agent, "reward_variant": r.reward_variant, "precision": r.precision, "recall": r.recall, "f1": r.f1, "fpr": r.fpr, "auroc": r.auroc, "pr_auc": r.pr_auc})
    for _, r in metrics_df[metrics_df.split == "test"].iterrows():
        comparison.append({"system": "rl_hardened", "model": r.agent, "reward_variant": r.reward_variant, "precision": r.precision, "recall": r.recall, "f1": r.f1, "fpr": r.fpr, "auroc": r.auroc, "pr_auc": r.pr_auc})
    pd.DataFrame(comparison).to_csv(args.results_dir / "comparison_with_supervised.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, g in curves_df.groupby("reward_variant"):
        ax.plot(g.epoch, g.val_f1, label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation F1"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(args.results_dir / "dqn_training_curves.png", dpi=180); plt.close(fig)

    if best_pred is not None:
        cm = confusion_matrix(best_pred.label, best_pred.pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4)); ax.imshow(cm, cmap="Blues"); ax.set_title(best_key)
        ax.set_xticks([0, 1], ["pred normal", "pred rogue"]); ax.set_yticks([0, 1], ["true normal", "true rogue"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout(); fig.savefig(args.results_dir / "dqn_confusion_matrix.png", dpi=180); plt.close(fig)

    best = metrics_df[metrics_df.split == "test"].sort_values("f1", ascending=False).iloc[0]
    lines = [
        "DQN hard-negative hardening summary",
        "",
        f"Hard-negative mining: {mining_report}",
        f"Best hardened DQN: {best.reward_variant}, precision={best.precision:.4f}, recall={best.recall:.4f}, f1={best.f1:.4f}, fpr={best.fpr:.4f}, auroc={best.auroc:.4f}, pr_auc={best.pr_auc:.4f}.",
        "The result should be compared against supervised and baseline RL in comparison_with_supervised.csv; do not treat reduced FPR as a win if recall collapses.",
    ]
    (args.results_dir / "short_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
