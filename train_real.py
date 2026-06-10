"""Train DQN on real CSV datasets without Ryu/Mininet."""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import random
import time
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from agent.dqn_agent import DQNAgent
from config import ACTION_ALLOW, ACTION_NAMES, CFG, NUM_ACTIONS
from dataset.data_loader import ARPDataLoader, InSDNDataLoader
from evaluate_real import compute_metrics


def get_device(device_arg: str = "auto") -> torch.device:
    """Resolve the requested torch device without hardcoded CUDA calls."""
    if device_arg == "mps" or (
        device_arg == "auto"
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        device = torch.device("mps")
        print("Using MPS device")
    elif device_arg == "cuda" or (device_arg == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train DQN on real tabular datasets.")
    parser.add_argument("--dataset", choices=["arp", "insdn", "both"], required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--reward", default="v2")
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--profile", action="store_true", help="Print timing breakdown for training.")
    return parser.parse_args()


def load_dataset(name: str):
    """Load one normalized real dataset."""
    if name == "arp":
        return ARPDataLoader().load()
    if name == "insdn":
        return InSDNDataLoader().load()
    raise ValueError(name)


def reward_config_path(name: str) -> Path | None:
    """Resolve reward aliases such as v2 to YAML files."""
    aliases = {
        "v1": "reward_v1.yaml",
        "v2": "reward_v2_fn_penalty.yaml",
        "v3": "reward_v3_isolate_boost.yaml",
        "v4": "reward_v4_balanced.yaml",
        "v5": "reward_v5_conservative.yaml",
        "v6": "reward_v6_logic_fixed.yaml",
    }
    raw = aliases.get(name, name)
    path = Path(raw)
    if not path.exists():
        path = Path("config") / raw
    return path if path.exists() else None


def load_false_positive_penalty(reward_name: str) -> float:
    """Read reward_fp_normal/r_normal_blocked for compute_reward compatibility."""
    path = reward_config_path(reward_name)
    if path is None:
        return CFG.reward.r_normal_blocked
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return CFG.reward.r_normal_blocked
    return float(raw.get("reward_fp_normal", raw.get("r_normal_blocked", CFG.reward.r_normal_blocked)))


def select_action_batch(agent: DQNAgent, states: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Select epsilon-greedy actions for one tensor batch."""
    with torch.no_grad():
        q_values = agent.q_net(states)
        greedy_actions = q_values.argmax(dim=1)
        random_actions = torch.randint(
            low=0,
            high=agent.num_actions,
            size=(states.shape[0],),
            device=states.device,
        )
        random_mask = torch.rand(states.shape[0], device=states.device) < float(epsilon)
        return torch.where(random_mask, random_actions, greedy_actions)


def compute_rewards_vectorized(
    actions: torch.Tensor,
    labels: torch.Tensor,
    fp_penalty: float,
) -> torch.Tensor:
    """Compute batch rewards for actions and binary labels."""
    rewards = torch.full(
        (actions.shape[0],),
        fill_value=float(CFG.reward.r_time_step),
        dtype=torch.float32,
        device=actions.device,
    )
    is_attack = labels == 1
    is_normal = labels == 0
    is_allow = actions == 0
    is_flag = actions == 1
    is_block = actions == 2
    is_isolate = actions == 3

    rewards = torch.where(is_attack & is_isolate, rewards + CFG.reward.r_isolate_correct, rewards)
    rewards = torch.where(is_attack & is_block, rewards + CFG.reward.r_attack_blocked, rewards)
    rewards = torch.where(is_attack & is_flag, rewards + CFG.reward.r_attack_flagged, rewards)
    rewards = torch.where(is_attack & is_allow, rewards + CFG.reward.r_attack_ignored, rewards)
    rewards = torch.where(is_normal & is_allow, rewards + CFG.reward.r_normal_allowed, rewards)
    rewards = torch.where(is_normal & is_flag, rewards + CFG.reward.r_normal_flagged, rewards)
    rewards = torch.where(is_normal & is_isolate, rewards + CFG.reward.r_isolate_wrong, rewards)
    rewards = torch.where(is_normal & is_block, rewards + float(fp_penalty), rewards)
    return rewards


def optimize_batch(
    agent: DQNAgent,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
) -> float:
    """Run one DQN update on the current batch."""
    if len(actions) == 0:
        return 0.0
    a = actions.long().unsqueeze(1)
    r = rewards.float().unsqueeze(1)
    d = dones.float().unsqueeze(1)
    q_sa = agent.q_net(states).gather(1, a)
    with torch.no_grad():
        q_next = agent.target_net(next_states).max(dim=1, keepdim=True).values
        target = r + agent.cfg.gamma * q_next * (1.0 - d)
    loss = agent.loss_fn(q_sa, target)
    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.q_net.parameters(), max_norm=10.0)
    agent.optimizer.step()
    agent.steps_done += 1
    if agent.steps_done % agent.cfg.target_update_freq == 0:
        agent.target_net.load_state_dict(agent.q_net.state_dict())
    return float(loss.item())


def evaluate_agent(agent: DQNAgent, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, object]:
    """Evaluate greedy policy on test features."""
    X_tensor = torch.from_numpy(X_test.astype(np.float32))
    actions_out = []
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            states = X_tensor[start:start + batch_size].to(agent.device)
            actions_out.append(agent.q_net(states).argmax(dim=1).cpu().numpy())
    actions = np.concatenate(actions_out).astype(np.int64) if actions_out else np.array([], dtype=np.int64)
    return compute_metrics(y_test, actions)


def save_checkpoint(
    agent: DQNAgent,
    path: Path,
    episode: int,
    dataset: str,
    feature_names: list[str],
    input_dim: int,
    reward_name: str,
    best_f1: float,
    args: argparse.Namespace,
) -> None:
    """Save a metadata-rich checkpoint compatible with evaluate_real.py."""
    metadata = {
        "episode": int(episode),
        "model_state_dict": agent.q_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "dataset": dataset,
        "feature_names": list(feature_names),
        "input_dim": int(input_dim),
        "n_actions": int(NUM_ACTIONS),
        "reward_config": reward_name,
        "best_f1": float(best_f1),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    agent.save(path, metadata=metadata)
    print(f"Checkpoint saved: {path}")


def make_loader(
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Build a torch DataLoader for tabular training."""
    X_tensor = torch.from_numpy(X_train.astype(np.float32))
    y_tensor = torch.from_numpy(y_train.astype(np.int64))
    workers = max(0, int(num_workers))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(TensorDataset(X_tensor, y_tensor), **kwargs)


def train_one_dataset(dataset_name: str, args: argparse.Namespace) -> Path:
    """Train and evaluate one dataset."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device(args.device)
    batch_size = args.batch_size or (512 if device.type == "mps" else 128)
    save_path = args.save_path or Path(f"runs/real_{dataset_name}_{args.reward}")
    save_path.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names = load_dataset(dataset_name)
    if args.max_train_samples is not None and len(X_train) > args.max_train_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X_train), size=args.max_train_samples, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    if args.eval_limit is not None and len(X_test) > args.eval_limit:
        X_test, y_test = X_test[:args.eval_limit], y_test[:args.eval_limit]

    input_dim = int(X_train.shape[1])
    print(f"Dataset: {dataset_name} | Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {input_dim}")
    agent = DQNAgent(state_dim=input_dim, num_actions=NUM_ACTIONS, device=str(device))
    loader = make_loader(X_train, y_train, batch_size, args.num_workers)
    fp_penalty = load_false_positive_penalty(args.reward)
    log_path = save_path / "training_log.csv"
    best_f1 = -1.0
    best_path = save_path / "dqn_best.pt"
    episode_times: list[float] = []
    timing_rows: list[dict[str, float]] = []

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "episode", "epsilon", "avg_reward", "avg_loss", "accuracy",
            "precision", "recall", "f1", "fpr", "tp", "fp", "tn", "fn",
        ])
        writer.writeheader()
        for episode in tqdm(range(args.episodes), desc=f"Training {dataset_name}", unit="ep"):
            t0 = time.perf_counter()
            t_data = 0.0
            t_forward = 0.0
            t_reward = 0.0
            t_optimize = 0.0
            t_eval = 0.0
            rewards_total = 0.0
            loss_total = 0.0
            updates = 0
            for states_cpu, labels_cpu in loader:
                t1 = time.perf_counter()
                states = states_cpu.to(agent.device)
                labels = labels_cpu.to(agent.device)
                t_data += time.perf_counter() - t1

                t1 = time.perf_counter()
                actions = select_action_batch(agent, states, agent.epsilon)
                t_forward += time.perf_counter() - t1

                t1 = time.perf_counter()
                rewards = compute_rewards_vectorized(actions, labels, fp_penalty)
                t_reward += time.perf_counter() - t1

                next_states = torch.roll(states, shifts=-1, dims=0)
                dones = torch.zeros(len(actions), dtype=torch.float32, device=agent.device)
                t1 = time.perf_counter()
                loss = optimize_batch(agent, states, actions, rewards, next_states, dones)
                t_optimize += time.perf_counter() - t1

                rewards_total += float(rewards.detach().sum().cpu())
                loss_total += loss
                updates += 1
            agent.decay_epsilon()
            elapsed = time.perf_counter() - t0
            episode_times.append(elapsed)

            metrics = {
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "fpr": 0.0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            }
            if episode == 0 or (episode + 1) % args.eval_every == 0 or episode == args.episodes - 1:
                t1 = time.perf_counter()
                metrics = evaluate_agent(agent, X_test, y_test)
                t_eval = time.perf_counter() - t1
                if float(metrics["f1"]) > best_f1:
                    best_f1 = float(metrics["f1"])
                    save_checkpoint(
                        agent, best_path, episode, dataset_name, feature_names,
                        input_dim, args.reward, best_f1, args,
                    )

            avg_reward = rewards_total / max(len(X_train), 1)
            avg_loss = loss_total / max(updates, 1)
            writer.writerow({
                "episode": episode,
                "epsilon": f"{agent.epsilon:.6f}",
                "avg_reward": f"{avg_reward:.6f}",
                "avg_loss": f"{avg_loss:.6f}",
                "accuracy": f"{float(metrics['accuracy']):.6f}",
                "precision": f"{float(metrics['precision']):.6f}",
                "recall": f"{float(metrics['recall']):.6f}",
                "f1": f"{float(metrics['f1']):.6f}",
                "fpr": f"{float(metrics['fpr']):.6f}",
                "tp": int(metrics["tp"]),
                "fp": int(metrics["fp"]),
                "tn": int(metrics["tn"]),
                "fn": int(metrics["fn"]),
            })
            f.flush()
            timing_rows.append({
                "episode": float(episode),
                "total": elapsed,
                "data": t_data,
                "forward": t_forward,
                "reward": t_reward,
                "optimize": t_optimize,
                "eval": t_eval,
                "other": max(0.0, elapsed - t_data - t_forward - t_reward - t_optimize),
            })
            tqdm.write(
                f"Ep {episode:3d} | eps={agent.epsilon:.3f} | "
                f"reward={avg_reward:.3f} | f1={float(metrics['f1']):.3f} | {elapsed:.2f}s/ep"
            )
            if args.profile and (episode == 0 or episode == args.episodes - 1):
                print_timing_breakdown(timing_rows[-1])

    if best_f1 < 0:
        save_checkpoint(agent, best_path, args.episodes - 1, dataset_name, feature_names, input_dim, args.reward, 0.0, args)
    print_estimate(dataset_name, len(X_train), args.episodes, episode_times)
    return best_path


def print_timing_breakdown(row: dict[str, float]) -> None:
    """Print episode timing components."""
    total = max(row["total"], 1e-9)
    print(
        "Timing breakdown "
        f"ep={int(row['episode'])}: "
        f"data={row['data']:.4f}s ({row['data'] / total * 100:.1f}%), "
        f"forward={row['forward']:.4f}s ({row['forward'] / total * 100:.1f}%), "
        f"reward={row['reward']:.4f}s ({row['reward'] / total * 100:.1f}%), "
        f"optimize={row['optimize']:.4f}s ({row['optimize'] / total * 100:.1f}%), "
        f"eval={row['eval']:.4f}s, "
        f"total={row['total']:.4f}s"
    )


def print_estimate(dataset_name: str, train_rows: int, episodes: int, episode_times: Iterable[float]) -> None:
    """Print runtime estimates from the observed episode timing."""
    times = list(episode_times)
    steady_times = times[1:] if len(times) > 1 else times
    steady_ep = float(np.median(steady_times)) if steady_times else 0.0
    per_row = steady_ep / max(train_rows, 1)
    est_arp = per_row * 134000 * 100
    est_insdn = per_row * 205167 * 100
    est_ablation = per_row * 134000 * 50 * 6 / 4
    smoke_time = sum(times)
    print(
        "\nRuntime estimate:\n"
        f"  Current run ({episodes} ep, {train_rows} samples): {smoke_time:.1f}s\n"
        f"  Steady episode median, excluding warmup when possible: {steady_ep:.3f}s\n"
        f"  Full ARP   (100 ep, 134k rows): ~{est_arp:.0f}s (~{est_arp / 60:.1f} min)\n"
        f"  Full InSDN (100 ep, 205k rows): ~{est_insdn:.0f}s (~{est_insdn / 60:.1f} min)\n"
        f"  Ablation   (6 configs x 50 ep, 4 parallel): ~{est_ablation:.0f}s\n"
    )


def main() -> int:
    """Script entry point."""
    args = parse_args()
    datasets = ["arp", "insdn"] if args.dataset == "both" else [args.dataset]
    for name in datasets:
        if args.dataset == "both" and args.save_path is not None:
            child_args = argparse.Namespace(**vars(args))
            child_args.save_path = args.save_path / name
        else:
            child_args = args
        train_one_dataset(name, child_args)
    return 0


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    raise SystemExit(main())
