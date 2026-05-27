"""Test agent/reward.py."""

import pytest

from agent.reward import compute_reward
from config import ACTION_ALLOW, ACTION_BLOCK, ACTION_FLAG, ACTION_ISOLATE


def test_attack_blocked_positive():
    assert compute_reward(ACTION_BLOCK, "attack") > 5.0
    assert compute_reward(ACTION_ISOLATE, "attack") > 5.0


def test_attack_ignored_negative():
    assert compute_reward(ACTION_ALLOW, "attack") < 0


def test_normal_blocked_is_false_positive():
    assert compute_reward(ACTION_BLOCK, "normal") < 0
    assert compute_reward(ACTION_ISOLATE, "normal") < 0


def test_normal_allowed_positive():
    assert compute_reward(ACTION_ALLOW, "normal") > 0


def test_attack_flag_partial_credit():
    r_block = compute_reward(ACTION_BLOCK, "attack")
    r_flag = compute_reward(ACTION_FLAG, "attack")
    r_allow = compute_reward(ACTION_ALLOW, "attack")
    assert r_block > r_flag > r_allow
