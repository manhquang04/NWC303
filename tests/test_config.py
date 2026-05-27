"""Test config.py — verify constants are consistent."""

from config import (
    ACTION_ALLOW,
    ACTION_BLOCK,
    ACTION_FLAG,
    ACTION_ISOLATE,
    ACTION_NAMES,
    CFG,
    NUM_ACTIONS,
)


def test_action_constants_are_distinct():
    assert {ACTION_ALLOW, ACTION_FLAG, ACTION_BLOCK, ACTION_ISOLATE} == {0, 1, 2, 3}
    assert NUM_ACTIONS == 4
    assert len(ACTION_NAMES) == NUM_ACTIONS


def test_state_dim_consistency():
    assert CFG.detection.state_dim == 20


def test_reward_signs():
    r = CFG.reward
    assert r.r_attack_blocked > 0
    assert r.r_attack_ignored < 0
    assert r.r_normal_blocked < 0
    assert r.r_normal_allowed > 0
