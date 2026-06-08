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
from config import ACTION_BLOCK, ACTION_ISOLATE, CFG, NUM_ACTIONS
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
    ) -> None:
        super().__init__()
        self.topology = topology
        self.attack_log = attack_event_log
        self.isolator = isolator
        self.flow_collector = flow_collector
        self.max_steps = max_steps
        self.mode: EnvMode = mode

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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self.feature_extractor = FeatureExtractor()
        self.state_builder = StateBuilder()

        if self.flow_collector is not None and self.flow_collector.get_latest() is None:
            self.flow_collector.start()
        obs = self._get_obs()
        self._last_obs = obs
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)
        self._step_count += 1

        self._apply_action(action)
        time.sleep(CFG.detection.poll_interval_ms / 1000.0)
        obs = self._get_obs()

        if self.mode == "training":
            now = time.time()
            gt = "attack" if self.attack_log.is_attack_at(now) else "normal"
            active_types = self.attack_log.active_types_at(now) if hasattr(self.attack_log, "active_types_at") else []
            reward = compute_reward(action, gt)
        else:
            gt = "unknown"
            active_types = []
            reward = 0.0

        truncated = self._step_count >= self.max_steps
        terminated = False

        info: Dict[str, Any] = {
            "ground_truth": gt,
            "attack_type": ",".join(active_types) if active_types else "none",
            "action": action,
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
        self._last_features = feats
        return self.state_builder.build(feats)

    def _apply_action(self, action: int) -> None:
        self._last_target = None
        if action in (ACTION_BLOCK, ACTION_ISOLATE):
            snap = self.flow_collector.get_latest() if self.flow_collector else None
            target = self.target_selector.select(snap, self._last_features)
            self._last_target = target
            if target is not None:
                self.isolator.set_target(target.dpid, target.port)
        try:
            self.isolator.apply(action)
        except Exception:  # pragma: no cover
            log.exception("apply_action(%d) failed.", action)
        if action != 0:
            self.feature_extractor.mark_alert()
