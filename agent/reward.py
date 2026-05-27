"""Reward function for DRL agent. See config.RewardConfig for values."""

from __future__ import annotations

import logging
from typing import Literal

from config import (
    ACTION_ALLOW,
    ACTION_BLOCK,
    ACTION_FLAG,
    ACTION_ISOLATE,
    CFG,
)

log = logging.getLogger(__name__)

GroundTruth = Literal["attack", "normal"]


def compute_reward(
    action: int,
    ground_truth: GroundTruth,
    false_positive_penalty: float = CFG.reward.r_normal_blocked,
) -> float:
    """Compute reward for one step, including time penalty."""
    r = CFG.reward
    base = r.r_time_step

    if ground_truth == "attack":
        if action in (ACTION_BLOCK, ACTION_ISOLATE):
            return base + r.r_attack_blocked
        if action == ACTION_FLAG:
            return base + r.r_attack_flagged
        if action == ACTION_ALLOW:
            return base + r.r_attack_ignored

    elif ground_truth == "normal":
        if action == ACTION_ALLOW:
            return base + r.r_normal_allowed
        if action == ACTION_FLAG:
            return base + r.r_normal_flagged
        if action in (ACTION_BLOCK, ACTION_ISOLATE):
            return base + false_positive_penalty

    log.warning("Unknown (action=%s, gt=%s) — returning time penalty only.", action, ground_truth)
    return base
