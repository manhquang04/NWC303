"""Rule-based baseline detector for comparison with DRL agent."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import (
    ACTION_ALLOW,
    ACTION_BLOCK,
    ACTION_FLAG,
    ACTION_ISOLATE,
    CFG,
)

log = logging.getLogger(__name__)


@dataclass
class BaselineState:

    arp_req_window: List[float] = field(default_factory=list)
    arp_reply_window: List[float] = field(default_factory=list)
    mac_ip_map: Dict[str, str] = field(default_factory=dict)      # ip → mac
    seen_ssids: set = field(default_factory=set)
    blocked_ports: set = field(default_factory=lambda: set())      # (dpid, port)
    last_alert_ts: float = 0.0


class BaselineDetector:
    """Rule-based detector using fixed thresholds (no learning)."""

    def __init__(
        self,
        arp_rate_threshold: float = CFG.detection.arp_rate_warn_threshold,
        new_mac_threshold: float = CFG.detection.new_mac_rate_warn_threshold,
        ssid_whitelist: Tuple[str, ...] = CFG.detection.ssid_whitelist,
    ) -> None:
        self.arp_rate_threshold = arp_rate_threshold
        self.new_mac_threshold = new_mac_threshold
        self.ssid_whitelist = set(ssid_whitelist)
        self.state = BaselineState()
        self._window_sec = CFG.detection.feature_window_sec

    def decide(
        self,
        features: Dict[str, float],
        dpid: Optional[int] = None,
        port: Optional[int] = None,
    ) -> Tuple[int, Dict[str, float]]:
        conf: Dict[str, float] = {}

        arp_reply = features.get("arp_reply_rate", 0.0)
        conf["arp_reply_score"] = min(1.0, arp_reply / max(1.0, self.arp_rate_threshold))

        mismatch = features.get("mac_ip_mismatch_count", 0.0)
        conf["mac_mismatch_score"] = 1.0 if mismatch >= 1.0 else 0.0

        unknown_ssid = features.get("unknown_ssid_count", 0.0)
        ssid_beacon = features.get("ssid_beacon_count", 0.0)
        conf["rogue_ssid_score"] = 1.0 if unknown_ssid > 0 else 0.0
        conf["beacon_score"] = min(1.0, ssid_beacon / max(1.0, 10.0))

        new_mac = features.get("new_mac_rate", 0.0)
        conf["new_mac_score"] = min(1.0, new_mac / max(1.0, self.new_mac_threshold))

        suspicious = features.get("suspicious_port_flag", 0.0)
        conf["suspicious_flag"] = suspicious

        is_suspicious = suspicious > 0.5
        high_arp = conf["arp_reply_score"] >= 0.8
        has_mismatch = conf["mac_mismatch_score"] >= 1.0
        has_rogue_ssid = conf["rogue_ssid_score"] >= 1.0
        high_new_mac = conf["new_mac_score"] >= 0.8

        blocked = (dpid, port) in self.state.blocked_ports if dpid is not None and port is not None else False

        if has_mismatch or (high_arp and is_suspicious):
            if not blocked:
                action = ACTION_BLOCK
            else:
                action = ACTION_ISOLATE
        elif has_rogue_ssid and is_suspicious:
            action = ACTION_BLOCK
        elif is_suspicious or high_new_mac:
            action = ACTION_FLAG
        else:
            action = ACTION_ALLOW

        if action in (ACTION_BLOCK, ACTION_ISOLATE) and dpid is not None and port is not None:
            self.state.blocked_ports.add((dpid, port))
        if action == ACTION_ALLOW and dpid is not None and port is not None:
            self.state.blocked_ports.discard((dpid, port))

        if action != ACTION_ALLOW:
            self.state.last_alert_ts = time.time()

        return action, conf

    def reset(self) -> None:
        self.state = BaselineState()


def compare_baseline_vs_drl(
    baseline: BaselineDetector,
    drl_agent,
    env,
    flow_collector,
    feature_extractor,
    state_builder,
    num_episodes: int = 10,
) -> Dict[str, Dict[str, float]]:
    from agent.reward import compute_reward
    from env.attack_simulator import AttackEventLog
    from evaluation.metrics import MetricsCalculator, StepRecord

    baseline_calc = MetricsCalculator()
    drl_calc = MetricsCalculator()
    baseline.reset()

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # DRL action
            drl_action = drl_agent.act(obs, greedy=True)

            snap = flow_collector.get_latest()
            if snap is not None:
                feats = feature_extractor.extract(snap)
                bl_action, _ = baseline.decide(feats)
            else:
                bl_action = ACTION_ALLOW

            next_obs, reward, terminated, truncated, info = env.step(drl_action)
            done = terminated or truncated
            ts = time.time()

            gt = info.get("ground_truth", "normal")

            bl_rec = StepRecord(
                timestamp=ts, ground_truth=gt,
                action=bl_action, reward=compute_reward(bl_action, gt),
                detected=bl_action != ACTION_ALLOW,
                isolated=bl_action in (ACTION_BLOCK, ACTION_ISOLATE),
            )
            baseline_calc.add_record(bl_rec)

            drl_rec = StepRecord(
                timestamp=ts, ground_truth=gt,
                action=drl_action, reward=reward,
                detected=drl_action != ACTION_ALLOW,
                isolated=drl_action in (ACTION_BLOCK, ACTION_ISOLATE),
            )
            drl_calc.add_record(drl_rec)

            obs = next_obs

    return {
        "baseline": baseline_calc.compute(episodes=num_episodes).as_dict(),
        "drl": drl_calc.compute(episodes=num_episodes).as_dict(),
    }
