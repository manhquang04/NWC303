"""Gymnasium Env wrapping Mininet/Ryu for DRL agent.

State:  Box(20,) float32 in [0,1]
Action: Discrete(4) — allow|flag|block|isolate
Modes:  "training" (reward from ground truth) | "evaluation" (reward=0)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Literal, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agent.reward import compute_reward
from config import ACTION_ALLOW, ACTION_BLOCK, ACTION_FLAG, ACTION_ISOLATE, CFG, NUM_ACTIONS
from detection.feature_extractor import FeatureExtractor
from detection.flow_collector import FlowCollector
from detection.state_builder import StateBuilder
from detection.target_selector import IsolationTarget, TargetSelector

log = logging.getLogger(__name__)

EnvMode = Literal["training", "evaluation"]


class SDNIDSEnv(gym.Env):
    """Gymnasium environment wrapping the SDN-IDS detection pipeline."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        topology,
        attack_event_log,
        isolator,
        flow_collector: Optional[FlowCollector] = None,
        max_steps: int = CFG.dqn.max_steps_per_episode,
        mode: EnvMode = "training",
        attack_ratio: float = CFG.attack.attack_ratio,
        attacks: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.topology = topology
        self.attack_log = attack_event_log
        self.isolator = isolator
        self.flow_collector = flow_collector
        self.max_steps = max_steps
        self.mode: EnvMode = mode
        self.attack_ratio = float(attack_ratio)
        self.attacks = attacks or []

        self.feature_extractor = FeatureExtractor()
        self.state_builder = StateBuilder()
        self.target_selector = TargetSelector()

        dim = CFG.detection.state_dim
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._step_count: int = 0
        self._last_obs: Optional[np.ndarray] = None
        self._last_features: Dict[str, float] = {}
        self._last_target: Optional[IsolationTarget] = None
        self._last_detection_confidence: float = 0.0
        self._last_raw_action: int = ACTION_ALLOW
        self._last_effective_action: int = ACTION_ALLOW
        self._last_action_gated: bool = False
        self._last_no_target_penalty: bool = False
        self._gated_action_count: int = 0
        self._normal_only_episode: bool = False
        self._current_scenario: str = "normal"

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._gated_action_count = 0
        self._normal_only_episode = self._sample_normal_only_episode(options)
        if hasattr(self.attack_log, "set_attacks_enabled"):
            self.attack_log.set_attacks_enabled(not self._normal_only_episode)
        if self._normal_only_episode:
            for attack in self.attacks:
                attack.stop()
        self._current_scenario = "normal" if self._normal_only_episode else "scheduled"
        self.feature_extractor = FeatureExtractor()
        self.state_builder = StateBuilder()

        if self.flow_collector is not None and self.flow_collector.get_latest() is None:
            self.flow_collector.start()
        obs = self._get_obs()
        self._last_obs = obs
        return obs, {"normal_only_episode": self._normal_only_episode}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)
        self._step_count += 1

        raw_action = action
        effective_action = self._gate_action(raw_action)
        self._apply_action(effective_action)
        time.sleep(CFG.detection.poll_interval_ms / 1000.0)
        obs = self._get_obs()

        if self.mode == "training":
            now = time.time()
            gt = "attack" if self.attack_log.is_attack_at(now) else "normal"
            active_types = self.attack_log.active_types_at(now) if hasattr(self.attack_log, "active_types_at") else []
            reward = compute_reward(effective_action, gt)
        else:
            gt = "unknown"
            active_types = []
            reward = 0.0

        no_target_penalty = (
            effective_action in (ACTION_BLOCK, ACTION_ISOLATE)
            and self._last_target is None
        )
        if self.mode == "training" and no_target_penalty:
            reward -= CFG.attack.fp_step_penalty

        truncated = self._step_count >= self.max_steps
        terminated = False

        info: Dict[str, Any] = {
            "scenario": self._current_scenario if gt == "attack" else "normal",
            "ground_truth": gt,
            "attack_type": ",".join(active_types) if active_types else "none",
            "action": effective_action,
            "raw_action": raw_action,
            "proposed_action": raw_action,
            "effective_action": effective_action,
            "executed_action": effective_action,
            "action_gated": self._last_action_gated,
            "gated_action_count": self._gated_action_count,
            "detection_confidence": self._last_detection_confidence,
            "no_target_penalty": no_target_penalty,
            "normal_only_episode": self._normal_only_episode,
            "step": self._step_count,
            "target": self._last_target.as_dict() if self._last_target else None,
            "target_dpid": self._last_target.dpid if self._last_target else None,
            "target_port": self._last_target.port if self._last_target else None,
        }
        self._last_obs = obs
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self.flow_collector is not None:
            self.flow_collector.stop()

    def _get_obs(self) -> np.ndarray:
        snap = self.flow_collector.get_latest() if self.flow_collector else None
        if snap is None:
            self._last_features = {}
            return np.zeros(CFG.detection.state_dim, dtype=np.float32)
        feats = self.feature_extractor.extract(snap)
        self._inject_simulated_attack_features(feats)
        self._last_features = feats
        return self.state_builder.build(feats)

    def _inject_simulated_attack_features(self, feats: Dict[str, float]) -> None:
        """Expose simulator attack signals as lab features for Mininet training/eval."""
        if self._normal_only_episode or not hasattr(self.attack_log, "active_types_at"):
            return
        active_types = list(self.attack_log.active_types_at(time.time()))
        if not active_types:
            return

        self._current_scenario = ",".join(active_types)
        if any("arp" in attack_type.lower() for attack_type in active_types):
            feats["arp_reply_rate"] = max(
                float(feats.get("arp_reply_rate", 0.0)),
                CFG.detection.arp_rate_warn_threshold * 1.5,
            )
            feats["mac_ip_mismatch_count"] = max(float(feats.get("mac_ip_mismatch_count", 0.0)), 1.0)
            feats["suspicious_port_flag"] = 1.0
        if any("rogue" in attack_type.lower() for attack_type in active_types):
            feats["ssid_beacon_count"] = max(float(feats.get("ssid_beacon_count", 0.0)), 10.0)
            feats["unknown_ssid_count"] = max(float(feats.get("unknown_ssid_count", 0.0)), 1.0)
            feats["suspicious_port_flag"] = 1.0

    def _sample_normal_only_episode(self, options: Optional[dict]) -> bool:
        """Sample whether this episode should suppress all attack traffic."""
        if self.mode != "training":
            return False
        if options and "normal_only_episode" in options:
            return bool(options["normal_only_episode"])
        return bool(self.np_random.random() >= self.attack_ratio)

    def _gate_action(self, action: int) -> int:
        """Convert unsafe mitigation into allow when confidence is too low."""
        self._last_raw_action = int(action)
        self._last_effective_action = int(action)
        self._last_action_gated = False
        self._last_detection_confidence = detection_confidence(self._last_features)
        if action in (ACTION_BLOCK, ACTION_ISOLATE, ACTION_FLAG):
            if self._last_detection_confidence < CFG.attack.confidence_threshold:
                self._last_action_gated = True
                self._gated_action_count += 1
                self._last_effective_action = ACTION_ALLOW
        return self._last_effective_action

    def _apply_action(self, action: int) -> None:
        self._last_target = None
        if action in (ACTION_BLOCK, ACTION_ISOLATE):
            snap = self.flow_collector.get_latest() if self.flow_collector else None
            target = self.target_selector.select(snap, self._last_features)
            if target is None and self.mode == "training" and hasattr(self.attack_log, "active_types_at"):
                active_types = self.attack_log.active_types_at(time.time())
                for attack_type in active_types:
                    target = self.target_selector.from_attack_type(attack_type)
                    if target is not None:
                        break
            self._last_target = target
            if target is not None:
                self.isolator.set_target(target.dpid, target.port)
        try:
            self.isolator.apply(action)
        except Exception:  # pragma: no cover
            log.exception("apply_action(%d) failed.", action)
        if action != 0:
            self.feature_extractor.mark_alert()


def detection_confidence(features: Dict[str, float]) -> float:
    """Estimate whether features justify block/isolate mitigation."""
    if not features:
        return 0.0
    arp_score = min(1.0, features.get("arp_reply_rate", 0.0) / max(CFG.detection.arp_rate_warn_threshold, 1.0))
    mismatch_score = min(1.0, features.get("mac_ip_mismatch_count", 0.0))
    rogue_score = min(1.0, (
        features.get("unknown_ssid_count", 0.0)
        + features.get("ssid_beacon_count", 0.0) / 10.0
    ))
    suspicious_score = 1.0 if features.get("suspicious_port_flag", 0.0) > 0.5 else 0.0
    return max(arp_score, mismatch_score, rogue_score, suspicious_score)
