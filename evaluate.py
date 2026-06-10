"""Metrics: TP/FP/TN/FN, TPR, FPR, F1, MTTD, MTTI."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import ACTION_ALLOW, ACTION_BLOCK, ACTION_FLAG, ACTION_ISOLATE, CHECKPOINT_DIR, CFG, LOG_DIR

log = logging.getLogger(__name__)


def setup_logging(level: str | None = None) -> None:
    """Configure process-wide logging for CLI evaluation."""
    lvl = level or CFG.logging_cfg.log_level
    logging.basicConfig(
        level=getattr(logging, lvl.upper(), logging.INFO),
        format=CFG.logging_cfg.log_format,
    )


def plot_confusion_matrix(
    tp: int, fp: int, tn: int, fn: int,
    out_path: Path = CFG.logging_cfg.confusion_matrix_path,
) -> Path:
    """Save a simple confusion matrix plot for the latest evaluation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array([[tp, fn], [fp, tn]], dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Detected", "Allowed"])
    ax.set_yticklabels(["Attack (GT)", "Normal (GT)"])
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


@dataclass
class StepRecord:
    timestamp: float
    ground_truth: str         # "attack" | "normal"
    action: int
    reward: float
    proposed_action: int = ACTION_ALLOW
    detected: bool = False
    isolated: bool = False


@dataclass
class MetricsReport:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mttd_sec: float = 0.0
    mtti_sec: float = 0.0
    cumulative_reward: float = 0.0
    episodes: int = 0
    normal_action_dist: Dict[str, int] = field(default_factory=dict)
    attack_action_dist: Dict[str, int] = field(default_factory=dict)
    action_confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "detection_rate": self.detection_rate,
            "false_positive_rate": self.false_positive_rate,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "mttd_sec": self.mttd_sec, "mtti_sec": self.mtti_sec,
            "cumulative_reward": self.cumulative_reward,
            "episodes": self.episodes,
            "normal_action_dist": dict(self.normal_action_dist),
            "attack_action_dist": dict(self.attack_action_dist),
            "action_confusion": dict(self.action_confusion),
        }


class MetricsCalculator:
    """Aggregate StepRecords into a MetricsReport."""

    def __init__(self) -> None:
        self.records: List[StepRecord] = []
        self.attack_starts: List[float] = []
        self.detect_times: List[float] = []
        self.isolate_times: List[float] = []

    def add_record(self, rec: StepRecord) -> None:
        self.records.append(rec)

    def mark_attack_start(self, ts: float) -> None:
        self.attack_starts.append(ts)

    def mark_detected(self, ts: float) -> None:
        self.detect_times.append(ts)

    def mark_isolated(self, ts: float) -> None:
        self.isolate_times.append(ts)

    def compute(self, episodes: int = 0) -> MetricsReport:
        rep = MetricsReport(episodes=episodes)
        for r in self.records:
            is_attack = r.ground_truth == "attack"
            took_action = r.action != ACTION_ALLOW

            if is_attack and took_action:
                rep.tp += 1
            elif is_attack and not took_action:
                rep.fn += 1
            elif (not is_attack) and took_action:
                rep.fp += 1
            else:
                rep.tn += 1
            rep.cumulative_reward += r.reward

        attacks = rep.tp + rep.fn
        normals = rep.tn + rep.fp
        rep.detection_rate = rep.tp / attacks if attacks else 0.0
        rep.false_positive_rate = rep.fp / normals if normals else 0.0
        rep.precision = rep.tp / (rep.tp + rep.fp) if (rep.tp + rep.fp) else 0.0
        rep.recall = rep.detection_rate
        if rep.precision + rep.recall > 0:
            rep.f1 = 2 * rep.precision * rep.recall / (rep.precision + rep.recall)

        rep.mttd_sec = self._mean_delay(self.attack_starts, self.detect_times)
        rep.mtti_sec = self._mean_delay(self.attack_starts, self.isolate_times)
        rep.normal_action_dist = self._action_dist("normal")
        rep.attack_action_dist = self._action_dist("attack")
        rep.action_confusion = self._action_confusion()
        return rep

    def _action_dist(self, ground_truth: str) -> Dict[str, int]:
        """Count actions for one ground-truth class."""
        from config import ACTION_NAMES
        counts = {name: 0 for name in ACTION_NAMES}
        for rec in self.records:
            if rec.ground_truth == ground_truth:
                counts[ACTION_NAMES[int(rec.action)]] += 1
        return counts

    def _action_confusion(self) -> Dict[str, Dict[str, int]]:
        """Build TP/FP/TN/FN buckets for the executed action at each step."""
        from config import ACTION_NAMES
        matrix = {name: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for name in ACTION_NAMES}
        for rec in self.records:
            name = ACTION_NAMES[int(rec.action)]
            is_attack = rec.ground_truth == "attack"
            took_action = rec.action != ACTION_ALLOW
            if is_attack and took_action:
                matrix[name]["tp"] += 1
            elif is_attack and not took_action:
                matrix[name]["fn"] += 1
            elif (not is_attack) and took_action:
                matrix[name]["fp"] += 1
            else:
                matrix[name]["tn"] += 1
        return matrix

    @staticmethod
    def _mean_delay(starts: List[float], events: List[float]) -> float:
        if not starts or not events:
            return 0.0
        delays: List[float] = []
        events_sorted = sorted(events)
        for s in starts:
            after = [e for e in events_sorted if e >= s]
            if after:
                delays.append(after[0] - s)
        return float(sum(delays) / len(delays)) if delays else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained DRL agent.")
    p.add_argument("--checkpoint", "--model", dest="checkpoint", type=Path,
                   default=CHECKPOINT_DIR / "dqn_final.pt")
    p.add_argument("--sb3-checkpoint", type=Path, default=CHECKPOINT_DIR / "sb3_dqn.zip")
    p.add_argument("--algo", choices=["custom", "sb3"], default="custom",
                   help="Backward-compatible alias for --agent when --agent is not set.")
    p.add_argument("--agent", choices=["custom", "sb3", "baseline", "all"], default=None)
    p.add_argument("--scenario", choices=["normal", "arp", "rogue", "mixed", "all"], default="mixed")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=CFG.dqn.max_steps_per_episode)
    p.add_argument("--out-csv", "--output", dest="out_csv", type=Path,
                   default=LOG_DIR / "experiment_results.csv")
    p.add_argument("--reward-config", type=Path, default=None)
    return p.parse_args()


def _scenario_names(name: str) -> List[str]:
    return ["normal", "arp", "rogue", "mixed"] if name == "all" else [name]


def _agent_names(args: argparse.Namespace) -> List[str]:
    agent = args.agent or args.algo
    return ["baseline", "custom", "sb3"] if agent == "all" else [agent]


def _write_report_row(path: Path, scenario: str, agent: str, report: MetricsReport) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["scenario", "agent", *report.as_dict().keys()])
        values = [
            json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
            for v in report.as_dict().values()
        ]
        writer.writerow([scenario, agent, *values])


def _build_eval_env(max_steps: int):
    from agent.env_wrapper import SDNIDSEnv
    from env.attack_simulator import AttackEventLog, ARPSpoofAttack, RogueAPAttack
    from env.topology import SDNTopology
    from detection.flow_collector import FlowCollector
    from env.network_env import Isolator

    topology = SDNTopology()
    topology.start()

    attack_log = AttackEventLog()
    isolator = Isolator()
    collector = FlowCollector(dpids=[1, 2, 3])
    collector.start()

    rogue_host = topology.get_host_by_name(f"h{CFG.topology.rogue_host_idx}")
    spoofer_host = topology.get_host_by_name(f"h{CFG.topology.spoofer_host_idx}")
    rogue_iface = f"{rogue_host.name}-eth0" if rogue_host else "h5-eth0"
    spoofer_iface = f"{spoofer_host.name}-eth0" if spoofer_host else "h6-eth0"
    spoofer_mac = spoofer_host.MAC() if spoofer_host else "de:ad:be:ef:00:06"

    attacks = {
        "rogue": RogueAPAttack(iface=rogue_iface, on_event=attack_log.append,
                               host=rogue_host),
        "arp": ARPSpoofAttack(iface=spoofer_iface, attacker_mac=spoofer_mac,
                              on_event=attack_log.append, host=spoofer_host),
    }

    env = SDNIDSEnv(
        topology=topology,
        attack_event_log=attack_log,
        isolator=isolator,
        flow_collector=collector,
        max_steps=max_steps,
    )
    return env, topology, collector, attacks


def _start_scenario(scenario: str, attacks: Dict[str, object]) -> None:
    _stop_scenario(attacks)
    if scenario in ("arp", "mixed"):
        attacks["arp"].start(CFG.attack.arp_spoof_rate_pps)
    if scenario in ("rogue", "mixed"):
        attacks["rogue"].start(CFG.attack.rogue_beacon_rate_pps)


def _stop_scenario(attacks: Dict[str, object]) -> None:
    for attack in attacks.values():
        attack.stop()


def _load_policy(agent_name: str, args: argparse.Namespace, env):
    if agent_name == "baseline":
        from detection.baseline import BaselineDetector
        return BaselineDetector()
    if agent_name == "custom":
        from agent.dqn_agent import DQNAgent
        agent = DQNAgent()
        agent.load(Path(args.checkpoint))
        agent.epsilon = 0.0
        return agent
    if agent_name == "sb3":
        from stable_baselines3 import DQN as SB3DQN
        return SB3DQN.load(str(args.sb3_checkpoint), env=env)
    raise ValueError(agent_name)


def _choose_action(policy, agent_name: str, obs, env) -> int:
    if agent_name == "baseline":
        action, _ = policy.decide(env._last_features)
        return int(action)
    if agent_name == "custom":
        return int(policy.act(obs, greedy=True))
    action, _ = policy.predict(obs, deterministic=True)
    return int(action)


def _evaluate_policy(policy, agent_name: str, scenario: str, episodes: int,
                     env, attacks: Dict[str, object]) -> MetricsReport:
    calc = MetricsCalculator()
    was_attack = False

    for ep in range(episodes):
        _stop_scenario(attacks)
        obs, _ = env.reset(options={"normal_only_episode": scenario == "normal"})
        if scenario != "normal":
            _start_scenario(scenario, attacks)
            time.sleep(CFG.detection.poll_interval_ms / 1000.0)
            obs = env._get_obs()
        done = False
        while not done:
            raw_action = _choose_action(policy, agent_name, obs, env)
            next_obs, reward, terminated, truncated, info = env.step(raw_action)
            action = int(info.get("executed_action", info.get("action", raw_action)))
            done = terminated or truncated
            ts = time.time()
            gt = info.get("ground_truth", "normal")
            is_attack = gt == "attack"
            if is_attack and not was_attack:
                calc.mark_attack_start(ts)
            if is_attack and action != ACTION_ALLOW:
                calc.mark_detected(ts)
            if is_attack and action in (ACTION_BLOCK, ACTION_ISOLATE):
                calc.mark_isolated(ts)

            calc.add_record(StepRecord(
                timestamp=ts,
                ground_truth=gt,
                action=action,
                reward=reward,
                proposed_action=raw_action,
                detected=action != ACTION_ALLOW,
                isolated=action in (ACTION_BLOCK, ACTION_ISOLATE),
            ))
            was_attack = is_attack
            obs = next_obs

        _stop_scenario(attacks)
        was_attack = False
        if hasattr(env.isolator, "rollback_all"):
            env.isolator.rollback_all()
        log.info("Eval %s/%s episode %d/%d done.", agent_name, scenario, ep + 1, episodes)

    return calc.compute(episodes=episodes)


def evaluate(
    model_path: Path = CHECKPOINT_DIR / "dqn_final.pt",
    scenario: str = "all",
    episodes: int = 10,
    max_steps: int = CFG.dqn.max_steps_per_episode,
    output: Path = LOG_DIR / "experiment_results.csv",
    agent: str = "custom",
    reward_config: Path | None = None,
) -> Path:
    """Programmatic wrapper used by experiment.py and run scripts."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if reward_config is not None and not reward_config.exists():
        raise FileNotFoundError(reward_config)
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)
    if output.exists():
        output.unlink()

    args = argparse.Namespace(
        checkpoint=Path(model_path),
        sb3_checkpoint=CHECKPOINT_DIR / "sb3_dqn.zip",
        algo="custom" if agent in (None, "custom", "baseline", "all") else agent,
        agent=agent,
        scenario=scenario,
        episodes=episodes,
        max_steps=max_steps,
        out_csv=Path(output),
    )
    old_reward = None
    if reward_config is not None:
        import os
        old_reward = os.environ.get("SDNIDS_REWARD_CONFIG")
        os.environ["SDNIDS_REWARD_CONFIG"] = str(reward_config.resolve())
    try:
        _run_evaluation(args)
    finally:
        if reward_config is not None:
            import os
            if old_reward is None:
                os.environ.pop("SDNIDS_REWARD_CONFIG", None)
            else:
                os.environ["SDNIDS_REWARD_CONFIG"] = old_reward
    return Path(output)


def _run_evaluation(args: argparse.Namespace) -> int:
    log.info("Evaluating scenario=%s agent=%s episodes=%d",
             args.scenario, args.agent or args.algo, args.episodes)

    try:
        setup_logging()
        env, topology, collector, attacks = _build_eval_env(args.max_steps)
    except Exception:
        log.exception("Failed to build eval environment.")
        log.info("Skipping evaluation. Use: sudo python evaluate.py --model ...")
        return 1

    try:
        last_report: Optional[MetricsReport] = None
        for scenario in _scenario_names(args.scenario):
            for agent_name in _agent_names(args):
                try:
                    policy = _load_policy(agent_name, args, env)
                except FileNotFoundError as exc:
                    log.warning("Skipping %s: checkpoint not found (%s)", agent_name, exc)
                    continue
                report = _evaluate_policy(policy, agent_name, scenario, args.episodes, env, attacks)
                _write_report_row(args.out_csv, scenario, agent_name, report)
                last_report = report
                log.info("========== Evaluation Report: %s / %s ==========", scenario, agent_name)
                for k, v in report.as_dict().items():
                    log.info("  %-25s = %s", k, f"{v:.4f}" if isinstance(v, float) else v)

        if last_report is not None:
            plot_confusion_matrix(last_report.tp, last_report.fp, last_report.tn, last_report.fn)

    finally:
        _stop_scenario(attacks)
        collector.stop()
        topology.stop()
        env.close()

    return 0


def main() -> int:  # pragma: no cover
    args = parse_args()
    old_reward = None
    if args.reward_config is not None:
        if not args.reward_config.exists():
            raise FileNotFoundError(args.reward_config)
        import os
        old_reward = os.environ.get("SDNIDS_REWARD_CONFIG")
        os.environ["SDNIDS_REWARD_CONFIG"] = str(args.reward_config.resolve())
    try:
        return _run_evaluation(args)
    finally:
        if args.reward_config is not None:
            import os
            if old_reward is None:
                os.environ.pop("SDNIDS_REWARD_CONFIG", None)
            else:
                os.environ["SDNIDS_REWARD_CONFIG"] = old_reward


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(main())
