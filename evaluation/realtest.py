"""Evaluate a trained policy on a real/VM OpenFlow testbed without Mininet."""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from agent.reward import compute_reward
from config import (
    ACTION_ALLOW,
    ACTION_BLOCK,
    ACTION_ISOLATE,
    ACTION_NAMES,
    CHECKPOINT_DIR,
    CFG,
)
from detection.baseline import BaselineDetector
from detection.feature_extractor import FeatureExtractor
from detection.flow_collector import FlowCollector
from detection.state_builder import StateBuilder
from detection.target_selector import TargetSelector, infer_attack_type
from evaluation.logger import setup_logging
from isolation.isolator import Isolator

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run DRL/baseline inference on an external Ryu/OpenFlow testbed."
    )
    p.add_argument("--agent", choices=["custom", "baseline"], default="custom")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "dqn_final.pt")
    p.add_argument("--ryu-url", default=CFG.ryu.rest_base_url)
    p.add_argument("--dpids", default=",".join(str(d) for d in CFG.realtest.dpids),
                   help="Comma-separated datapath IDs, e.g. 1,2")
    p.add_argument("--steps", type=int, default=CFG.realtest.poll_steps)
    p.add_argument("--iface", default=CFG.realtest.default_iface)
    p.add_argument("--apply-actions", action="store_true",
                   help="Actually install block/isolate rules. Default is dry-run.")
    p.add_argument("--ground-truth", choices=["unknown", "normal", "attack"], default="unknown")
    p.add_argument("--out-csv", type=Path, default=CFG.realtest.results_path)
    return p.parse_args()


def parse_dpids(raw: str) -> List[int]:
    dpids = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        dpids.append(int(item, 0))
    if not dpids:
        raise ValueError("At least one DPID is required.")
    return dpids


def load_policy(agent_name: str, checkpoint: Path):
    if agent_name == "baseline":
        return BaselineDetector()
    from agent.dqn_agent import DQNAgent

    agent = DQNAgent()
    agent.load(checkpoint)
    agent.epsilon = 0.0
    return agent


def choose_action(policy, agent_name: str, state_vector, features: Dict[str, float]) -> int:
    if agent_name == "baseline":
        action, _ = policy.decide(features)
        return int(action)
    return int(policy.act(state_vector, greedy=True))


def ensure_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "timestamp", "step", "agent", "ground_truth", "attack_type",
            "action", "action_name", "target_dpid", "target_port", "target_reason",
            "reward", "applied",
        ])


def write_row(path: Path, row: List[object]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def main() -> int:
    setup_logging()
    args = parse_args()
    dpids = parse_dpids(args.dpids)
    out_csv = Path(args.out_csv)
    ensure_header(out_csv)

    collector = FlowCollector(dpids=dpids, base_url=args.ryu_url)
    extractor = FeatureExtractor()
    state_builder = StateBuilder()
    selector = TargetSelector()
    isolator = Isolator(base_url=args.ryu_url)
    policy = load_policy(args.agent, args.checkpoint)

    collector.start()
    log.info("Real-testbed mode started: ryu=%s dpids=%s dry_run=%s",
             args.ryu_url, dpids, not args.apply_actions)

    try:
        for step in range(1, args.steps + 1):
            time.sleep(CFG.detection.poll_interval_ms / 1000.0)
            snap = collector.get_latest()
            if snap is None:
                log.warning("No Ryu snapshot yet at step=%d", step)
                continue

            features = extractor.extract(snap)
            state_vector = state_builder.build(features)
            action = choose_action(policy, args.agent, state_vector, features)
            attack_type = infer_attack_type(features)
            target = selector.select(snap, features) if action in (ACTION_BLOCK, ACTION_ISOLATE) else None

            applied = False
            if target is not None:
                isolator.set_target(target.dpid, target.port)
            if args.apply_actions:
                applied = isolator.apply(action)
            elif action != ACTION_ALLOW:
                log.info("DRY-RUN action=%s target=%s", ACTION_NAMES[action], target)

            reward: Optional[float] = None
            if args.ground_truth in ("normal", "attack"):
                reward = compute_reward(action, args.ground_truth)

            write_row(out_csv, [
                f"{time.time():.6f}",
                step,
                args.agent,
                args.ground_truth,
                attack_type,
                action,
                ACTION_NAMES[action],
                target.dpid if target else "",
                target.port if target else "",
                target.reason if target else "",
                f"{reward:.4f}" if reward is not None else "",
                int(applied),
            ])
            log.info("step=%d action=%s attack_type=%s target=%s applied=%s",
                     step, ACTION_NAMES[action], attack_type, target, applied)
    finally:
        collector.stop()

    log.info("Real-testbed results written to %s", out_csv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
