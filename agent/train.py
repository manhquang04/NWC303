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

from config import CHECKPOINT_DIR, CFG, LOG_DIR
from evaluation.logger import MetricsLogger, setup_logging

_TRAIN_STATE_FILE = Path("/tmp/sdnids_train_state.json")


def _write_train_state(action: int, ground_truth: str, reward: float,
                       episode: int, epsilon: float, step: int,
                       cumulative_reward: float, attack_type: str = "none",
                       target: dict | None = None) -> None:
    try:
        with open(_TRAIN_STATE_FILE, "w") as f:
            json.dump({
                "action": action,
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
                "action", "raw_action", "action_gated", "gated_action_count",
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
            info.get("raw_action", ""), int(bool(info.get("action_gated", False))),
            info.get("gated_action_count", ""), f"{float(info.get('detection_confidence', 0.0)):.4f}",
            int(bool(info.get("no_target_penalty", False))),
            int(bool(info.get("normal_only_episode", False))),
            target.get("dpid", ""), target.get("port", ""), target.get("reason", ""),
        ])
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
    from isolation.isolator import Isolator

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
                agent.remember(obs, action, reward, next_obs, done)
                loss = agent.learn()
                ep_loss += loss
                agent.decay_epsilon()
                obs = next_obs
                ep_reward += reward
                steps += 1
                _write_train_state(
                    action=action,
                    ground_truth=info.get("ground_truth", "unknown"),
                    reward=reward,
                    episode=ep,
                    epsilon=agent.epsilon,
                    step=steps,
                    cumulative_reward=ep_reward,
                    attack_type=info.get("attack_type", "none"),
                    target=info.get("target"),
                )
                step_logger.log(ep, steps, agent.epsilon, loss, reward, ep_reward, info)

            metrics.log_episode(ep, ep_reward, ep_loss / max(1, steps), agent.epsilon)
            log.info("EP %4d | reward=%+8.2f | eps=%.3f | steps=%d",
                     ep, ep_reward, agent.epsilon, steps)

            if (ep + 1) % CFG.dqn.checkpoint_every == 0:
                ckpt = CHECKPOINT_DIR / f"dqn_ep{ep+1}.pt"
                agent.save(ckpt)

        agent.save(CHECKPOINT_DIR / "dqn_final.pt")
    finally:
        step_logger.close()
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
