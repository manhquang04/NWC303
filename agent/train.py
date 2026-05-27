"""Training entry point. Requires sudo (Mininet) and Ryu controller running."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from config import CFG
from evaluation.logger import MetricsLogger, setup_logging

_TRAIN_STATE_FILE = Path("/tmp/sdnids_train_state.json")


def _write_train_state(action: int, ground_truth: str, reward: float,
                       episode: int, epsilon: float, step: int,
                       cumulative_reward: float) -> None:
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
            }, f)
    except OSError:
        pass

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DRL agent for SDN-IDS.")
    p.add_argument("--algo", choices=["custom", "sb3"], default="custom")
    p.add_argument("--episodes", type=int, default=CFG.dqn.max_episodes)
    p.add_argument("--timesteps", type=int, default=500_000, help="(SB3 only)")
    p.add_argument("--attack-ratio", type=float, default=0.4)
    p.add_argument("--checkpoint", type=Path, default=None)
    return p.parse_args()


def build_env():
    """Initialize SDNIDSEnv with topology, attack log, isolator, and flow collector."""
    from agent.env_wrapper import SDNIDSEnv
    from env.attack_simulator import AttackEventLog
    from env.topology import SDNTopology
    from detection.flow_collector import FlowCollector
    from isolation.isolator import Isolator

    topology = SDNTopology()
    topology.start()

    attack_log = AttackEventLog()
    isolator = Isolator()
    collector = FlowCollector(dpids=[1, 2, 3])
    collector.start()

    env = SDNIDSEnv(
        topology=topology,
        attack_event_log=attack_log,
        isolator=isolator,
        flow_collector=collector,
    )
    return env, topology, collector


def train_custom(args: argparse.Namespace) -> None:
    from agent.dqn_agent import DQNAgent

    env, topology, collector = build_env()
    agent = DQNAgent()
    metrics = MetricsLogger()

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
                ep_loss += agent.learn()
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
                )

            metrics.log_episode(ep, ep_reward, ep_loss / max(1, steps), agent.epsilon)
            log.info("EP %4d | reward=%+8.2f | eps=%.3f | steps=%d",
                     ep, ep_reward, agent.epsilon, steps)

            if (ep + 1) % CFG.dqn.checkpoint_every == 0:
                ckpt = CFG.CHECKPOINT_DIR / f"dqn_ep{ep+1}.pt"
                agent.save(ckpt)

        agent.save(CFG.CHECKPOINT_DIR / "dqn_final.pt")
    finally:
        _TRAIN_STATE_FILE.unlink(missing_ok=True)
        collector.stop()
        topology.stop()
        env.close()


def run_sb3_training(args: argparse.Namespace) -> None:
    from agent.sb3_agent import build_sb3_dqn, train_sb3 as sb3_train

    env, topology, collector = build_env()
    try:
        model = build_sb3_dqn(env, tensorboard_log=str(CFG.logging_cfg.tensorboard_dir))
        sb3_train(model, total_timesteps=args.timesteps,
                  save_path=CFG.CHECKPOINT_DIR / "sb3_dqn.zip")
    finally:
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
