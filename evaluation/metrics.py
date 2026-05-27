"""Metrics: TP/FP/TN/FN, TPR, FPR, F1, MTTD, MTTI."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import ACTION_ALLOW, ACTION_BLOCK, ACTION_FLAG, ACTION_ISOLATE, CFG

log = logging.getLogger(__name__)


@dataclass
class StepRecord:
    timestamp: float
    ground_truth: str         # "attack" | "normal"
    action: int
    reward: float
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

    def as_dict(self) -> Dict[str, float]:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "detection_rate": self.detection_rate,
            "false_positive_rate": self.false_positive_rate,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "mttd_sec": self.mttd_sec, "mtti_sec": self.mtti_sec,
            "cumulative_reward": self.cumulative_reward,
            "episodes": self.episodes,
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
        return rep

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
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--algo", choices=["custom", "sb3"], default="custom")
    p.add_argument("--episodes", type=int, default=10)
    return p.parse_args()


def main() -> int:  # pragma: no cover
    args = parse_args()
    log.info("Evaluating %s checkpoint=%s episodes=%d",
             args.algo, args.checkpoint, args.episodes)

    try:
        from agent.env_wrapper import SDNIDSEnv
        from env.attack_simulator import AttackEventLog
        from env.topology import SDNTopology
        from detection.flow_collector import FlowCollector
        from isolation.isolator import Isolator
        from evaluation.logger import setup_logging

        setup_logging()

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
    except Exception:
        log.exception("Failed to build eval environment.")
        log.info("Skipping evaluation. Use: sudo python -m evaluation.metrics --checkpoint ...")
        return 1

    calc = MetricsCalculator()

    try:
        if args.algo == "custom":
            from agent.dqn_agent import DQNAgent
            agent = DQNAgent()
            agent.load(Path(args.checkpoint))
            agent.epsilon = 0.0

            for ep in range(args.episodes):
                obs, _ = env.reset()
                done = False
                while not done:
                    action = agent.act(obs, greedy=True)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    ts = time.monotonic()
                    gt = info.get("ground_truth", "normal")
                    rec = StepRecord(
                        timestamp=ts, ground_truth=gt,
                        action=action, reward=reward,
                        detected=action != ACTION_ALLOW,
                        isolated=action in (ACTION_BLOCK, ACTION_ISOLATE),
                    )
                    calc.add_record(rec)
                    obs = next_obs
                log.info("Eval episode %d/%d done.", ep + 1, args.episodes)
        else:
            from stable_baselines3 import DQN as SB3DQN
            model = SB3DQN.load(str(args.checkpoint))

            for ep in range(args.episodes):
                obs, _ = env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    next_obs, reward, terminated, truncated, info = env.step(int(action))
                    done = terminated or truncated
                    ts = time.monotonic()
                    gt = info.get("ground_truth", "normal")
                    a = int(action)
                    rec = StepRecord(
                        timestamp=ts, ground_truth=gt,
                        action=a, reward=reward,
                        detected=a != ACTION_ALLOW,
                        isolated=a in (ACTION_BLOCK, ACTION_ISOLATE),
                    )
                    calc.add_record(rec)
                    obs = next_obs
                log.info("Eval episode %d/%d done.", ep + 1, args.episodes)

        report = calc.compute(episodes=args.episodes)
        log.info("========== Evaluation Report ==========")
        for k, v in report.as_dict().items():
            log.info("  %-25s = %s", k, f"{v:.4f}" if isinstance(v, float) else v)

        from evaluation.visualizer import plot_confusion_matrix
        plot_confusion_matrix(report.tp, report.fp, report.tn, report.fn)

    finally:
        collector.stop()
        topology.stop()
        env.close()

    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(main())
