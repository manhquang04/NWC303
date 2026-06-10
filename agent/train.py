"""Training entry point. Requires sudo (Mininet) and Ryu controller running."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import threading
from pathlib import Path
from typing import Optional

from config import CHECKPOINT_DIR, CFG, LOG_DIR

_TRAIN_STATE_FILE = Path("/tmp/sdnids_train_state.json")


def _write_train_state(action: int, ground_truth: str, reward: float,
                       episode: int, epsilon: float, step: int,
                       cumulative_reward: float, attack_type: str = "none",
                       target: dict | None = None,
                       proposed_action: int | None = None,
                       executed_action: int | None = None) -> None:
    try:
        with open(_TRAIN_STATE_FILE, "w") as f:
            json.dump({
                "action": action,
                "proposed_action": proposed_action if proposed_action is not None else action,
                "executed_action": executed_action if executed_action is not None else action,
                "ground_truth": ground_truth,
                "reward": reward,
                "episode": episode,
                "epsilon": epsilon,
                "step": step,
                "cumulative_reward": cumulative_reward,
                "attack_type": attack_type,
                "target": target,
            }, f)
    except OSError:
        pass

log = logging.getLogger(__name__)


def setup_logging(level: Optional[str] = None) -> None:
    """Configure global logging for training CLI runs."""
    lvl = level or CFG.logging_cfg.log_level
    logging.basicConfig(
        level=getattr(logging, lvl.upper(), logging.INFO),
        format=CFG.logging_cfg.log_format,
    )


class MetricsLogger:
    """Log episode metrics to TensorBoard when available and CSV."""

    def __init__(
        self,
        tb_dir: Path = CFG.logging_cfg.tensorboard_dir,
        csv_path: Path = CFG.logging_cfg.csv_metrics_path,
    ) -> None:
        self.tb_dir = Path(tb_dir)
        self.csv_path = Path(csv_path)
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:  # pragma: no cover
            SummaryWriter = None
        self.writer = SummaryWriter(str(self.tb_dir)) if SummaryWriter is not None else None
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

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        self._csv_file.close()


def plot_reward_curve(
    csv_path: Path = CFG.logging_cfg.csv_metrics_path,
    out_path: Path = CFG.logging_cfg.reward_curve_path,
    smooth_window: int = 20,
) -> Path:
    """Save a reward curve PNG from the training metrics CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

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
        ax.plot(eps[smooth_window - 1:], smoothed, linewidth=2.0, label=f"reward (MA{smooth_window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("DRL training reward curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DRL agent for SDN-IDS.")
    p.add_argument("--algo", choices=["custom", "sb3"], default="custom")
    p.add_argument("--episodes", type=int, default=CFG.dqn.max_episodes)
    p.add_argument("--timesteps", type=int, default=500_000, help="(SB3 only)")
    p.add_argument("--attack-ratio", type=float, default=CFG.attack.attack_ratio)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--seed", type=int, default=CFG.dqn.seed)
    p.add_argument("--max-steps", type=int, default=CFG.dqn.max_steps_per_episode)
    return p.parse_args()


class StepCSVLogger:
    """Append per-step training metadata for experiment analysis."""

    def __init__(self, path: Path = LOG_DIR / "train_steps.csv") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (not self.path.exists()) or self.path.stat().st_size == 0
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._csv = csv.writer(self._file)
        if write_header:
            self._csv.writerow([
                "episode", "step", "epsilon", "loss", "reward",
                "cumulative_reward", "ground_truth", "attack_type",
                "action", "raw_action", "proposed_action", "executed_action",
                "action_gated", "gated_action_count",
                "detection_confidence", "no_target_penalty",
                "normal_only_episode", "target_dpid", "target_port", "target_reason",
            ])
            self._file.flush()

    def log(self, episode: int, step: int, epsilon: float, loss: float, reward: float,
            cumulative_reward: float, info: dict) -> None:
        target = info.get("target") or {}
        self._csv.writerow([
            episode, step, f"{epsilon:.6f}", f"{loss:.6f}", f"{reward:.4f}",
            f"{cumulative_reward:.4f}", info.get("ground_truth", "unknown"),
            info.get("attack_type", "none"), info.get("action", ""),
            info.get("raw_action", ""), info.get("proposed_action", info.get("raw_action", "")),
            info.get("executed_action", info.get("action", "")),
            int(bool(info.get("action_gated", False))),
            info.get("gated_action_count", ""), f"{float(info.get('detection_confidence', 0.0)):.4f}",
            int(bool(info.get("no_target_penalty", False))),
            int(bool(info.get("normal_only_episode", False))),
            target.get("dpid", ""), target.get("port", ""), target.get("reason", ""),
        ])
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class StepDebugLogger:
    """Write a fresh per-step debug trace for action/reward consistency audits."""

    def __init__(self, path: Path = LOG_DIR / "step_debug.csv") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._file, fieldnames=[
            "episode", "step", "scenario", "ground_truth", "proposed_action",
            "executed_action", "gated", "reward", "target", "done",
        ])
        self._csv.writeheader()
        self._file.flush()

    def log(self, episode: int, step: int, reward: float, done: bool, info: dict) -> None:
        self._csv.writerow({
            "episode": episode,
            "step": step,
            "scenario": info.get("scenario", info.get("attack_type", "none")),
            "ground_truth": info.get("ground_truth", "unknown"),
            "proposed_action": info.get("proposed_action", info.get("raw_action", "")),
            "executed_action": info.get("executed_action", info.get("action", "")),
            "gated": int(bool(info.get("action_gated", False))),
            "reward": f"{reward:.4f}",
            "target": json.dumps(info.get("target"), sort_keys=True),
            "done": int(bool(done)),
        })
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def build_env(attack_ratio: float = CFG.attack.attack_ratio,
              max_steps: int = CFG.dqn.max_steps_per_episode):
    """Initialize SDNIDSEnv with topology, attacks, isolator, and flow collector."""
    import random
    from agent.env_wrapper import SDNIDSEnv
    from env.attack_simulator import (
        AttackEventLog, RogueAPAttack, ARPSpoofAttack,
    )
    from env.topology import SDNTopology
    from detection.flow_collector import FlowCollector
    from env.network_env import Isolator

    topology = SDNTopology()
    topology.start()

    attack_log = AttackEventLog(attack_ratio=attack_ratio)
    isolator = Isolator()
    collector = FlowCollector(dpids=[1, 2, 3])
    collector.start()

    # Create attack simulators on Mininet host interfaces
    rogue_host = topology.get_host_by_name(f"h{CFG.topology.rogue_host_idx}")
    spoofer_host = topology.get_host_by_name(f"h{CFG.topology.spoofer_host_idx}")

    rogue_iface = f"{rogue_host.name}-eth0" if rogue_host else "h5-eth0"
    spoofer_iface = f"{spoofer_host.name}-eth0" if spoofer_host else "h6-eth0"
    spoofer_mac = spoofer_host.MAC() if spoofer_host else "de:ad:be:ef:00:06"

    rogue_ap = RogueAPAttack(
        iface=rogue_iface,
        on_event=attack_log.append,
        host=rogue_host,
    )
    arp_spoof = ARPSpoofAttack(
        iface=spoofer_iface,
        attacker_mac=spoofer_mac,
        on_event=attack_log.append,
        host=spoofer_host,
    )

    attacks = [rogue_ap, arp_spoof]

    # Start a background thread that randomly starts/stops attacks
    attack_thread_stop = threading.Event()

    def attack_scheduler():
        while not attack_thread_stop.is_set():
            # Attack active period
            if attack_log.attacks_enabled() and random.random() < attack_ratio:
                atk = random.choice(attacks)
                if not atk.is_running():
                    atk.start(CFG.attack.rogue_beacon_rate_pps
                              if isinstance(atk, RogueAPAttack)
                              else CFG.attack.arp_spoof_rate_pps)
                    duration = random.uniform(
                        CFG.attack.attack_duration_sec_min,
                        CFG.attack.attack_duration_sec_max,
                    )
                    attack_thread_stop.wait(duration)
                    atk.stop()
            attack_thread_stop.wait(random.uniform(2.0, 5.0))

    sched_thread = threading.Thread(target=attack_scheduler, daemon=True)
    sched_thread.start()

    env = SDNIDSEnv(
        topology=topology,
        attack_event_log=attack_log,
        isolator=isolator,
        flow_collector=collector,
        max_steps=max_steps,
        attack_ratio=attack_ratio,
        attacks=attacks,
    )
    return env, topology, collector, attacks, attack_thread_stop


def train_custom(args: argparse.Namespace) -> None:
    from agent.dqn_agent import DQNAgent

    random.seed(args.seed)
    env, topology, collector, attacks, sched_stop = build_env(args.attack_ratio, args.max_steps)
    agent = DQNAgent()
    if args.checkpoint is not None:
        agent.load(args.checkpoint)
    metrics = MetricsLogger()
    step_logger = StepCSVLogger()
    debug_logger = StepDebugLogger()

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_loss = 0.0
            steps = 0
            done = False
            while not done:
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                executed_action = int(info.get("executed_action", info.get("action", action)))
                agent.remember(obs, executed_action, reward, next_obs, done)
                loss = agent.learn()
                ep_loss += loss
                agent.decay_epsilon()
                obs = next_obs
                ep_reward += reward
                steps += 1
                _write_train_state(
                    action=executed_action,
                    ground_truth=info.get("ground_truth", "unknown"),
                    reward=reward,
                    episode=ep,
                    epsilon=agent.epsilon,
                    step=steps,
                    cumulative_reward=ep_reward,
                    attack_type=info.get("attack_type", "none"),
                    target=info.get("target"),
                    proposed_action=action,
                    executed_action=executed_action,
                )
                step_logger.log(ep, steps, agent.epsilon, loss, reward, ep_reward, info)
                debug_logger.log(ep, steps, reward, done, info)

            metrics.log_episode(ep, ep_reward, ep_loss / max(1, steps), agent.epsilon)
            log.info("EP %4d | reward=%+8.2f | eps=%.3f | steps=%d",
                     ep, ep_reward, agent.epsilon, steps)

            if (ep + 1) % CFG.dqn.checkpoint_every == 0:
                ckpt = CHECKPOINT_DIR / f"dqn_ep{ep+1}.pt"
                agent.save(ckpt)

        agent.save(CHECKPOINT_DIR / "dqn_final.pt")
    finally:
        step_logger.close()
        debug_logger.close()
        metrics.close()
        sched_stop.set()
        for atk in attacks:
            atk.stop()
        _TRAIN_STATE_FILE.unlink(missing_ok=True)
        collector.stop()
        topology.stop()
        env.close()


def run_sb3_training(args: argparse.Namespace) -> None:
    from agent.sb3_agent import build_sb3_dqn, train_sb3 as sb3_train

    random.seed(args.seed)
    env, topology, collector, attacks, sched_stop = build_env(args.attack_ratio, args.max_steps)
    try:
        if args.checkpoint is not None:
            from stable_baselines3 import DQN as SB3DQN
            model = SB3DQN.load(str(args.checkpoint), env=env)
        else:
            model = build_sb3_dqn(env, tensorboard_log=str(CFG.logging_cfg.tensorboard_dir))
        sb3_train(model, total_timesteps=args.timesteps,
                  save_path=CHECKPOINT_DIR / "sb3_dqn.zip")
    finally:
        sched_stop.set()
        for atk in attacks:
            atk.stop()
        _TRAIN_STATE_FILE.unlink(missing_ok=True)
        collector.stop()
        topology.stop()
        env.close()


def main() -> int:
    setup_logging()
    args = parse_args()
    log.info("Training algo=%s episodes=%s attack_ratio=%.2f",
             args.algo, args.episodes, args.attack_ratio)
    if args.algo == "custom":
        train_custom(args)
    else:
        run_sb3_training(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
