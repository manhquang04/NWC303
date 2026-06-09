"""Tests for normal-only episode sampling and mitigation action gating."""

from __future__ import annotations

import pytest

from agent.env_wrapper import SDNIDSEnv, detection_confidence
from agent.reward import compute_reward
from config import ACTION_ALLOW, ACTION_BLOCK, ACTION_ISOLATE, CFG


class DummyCollector:
    def __init__(self, snapshot=None) -> None:
        self.snapshot = snapshot
        self.started = False
        self.stopped = False

    def get_latest(self):
        return self.snapshot

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class DummyAttackLog:
    def __init__(self) -> None:
        self.enabled = True

    def set_attacks_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def is_attack_at(self, ts: float) -> bool:
        return False

    def active_types_at(self, ts: float):
        return []


class DummyIsolator:
    def __init__(self) -> None:
        self.actions: list[int] = []
        self.targets: list[tuple[int, int]] = []

    def set_target(self, dpid: int, port: int) -> None:
        self.targets.append((dpid, port))

    def apply(self, action: int) -> bool:
        self.actions.append(int(action))
        return True


class DummyAttack:
    def __init__(self) -> None:
        self.stop_count = 0

    def stop(self) -> None:
        self.stop_count += 1


def _env(**kwargs) -> SDNIDSEnv:
    return SDNIDSEnv(
        topology=None,
        attack_event_log=kwargs.get("attack_log", DummyAttackLog()),
        isolator=kwargs.get("isolator", DummyIsolator()),
        flow_collector=kwargs.get("collector", DummyCollector()),
        max_steps=kwargs.get("max_steps", 1),
        attack_ratio=kwargs.get("attack_ratio", CFG.attack.attack_ratio),
        attacks=kwargs.get("attacks", []),
    )


def test_detection_confidence_uses_attack_features():
    assert detection_confidence({}) == 0.0
    assert detection_confidence({"suspicious_port_flag": 1.0}) == 1.0
    assert detection_confidence({"arp_reply_rate": CFG.detection.arp_rate_warn_threshold}) == 1.0
    assert detection_confidence({"unknown_ssid_count": 1.0}) == 1.0


def test_low_confidence_block_or_isolate_is_gated_to_allow():
    env = _env()
    env._last_features = {}

    assert env._gate_action(ACTION_ISOLATE) == ACTION_ALLOW
    assert env._last_action_gated is True
    assert env._gated_action_count == 1

    assert env._gate_action(ACTION_BLOCK) == ACTION_ALLOW
    assert env._gated_action_count == 2


def test_high_confidence_isolate_is_not_gated():
    env = _env()
    env._last_features = {"suspicious_port_flag": 1.0}

    assert env._gate_action(ACTION_ISOLATE) == ACTION_ISOLATE
    assert env._last_action_gated is False
    assert env._gated_action_count == 0


def test_normal_only_reset_disables_and_stops_attacks():
    attack_log = DummyAttackLog()
    attack = DummyAttack()
    env = _env(attack_log=attack_log, attacks=[attack])

    _, info = env.reset(seed=1, options={"normal_only_episode": True})

    assert info["normal_only_episode"] is True
    assert attack_log.enabled is False
    assert attack.stop_count == 1


def test_no_target_penalty_applies_to_raw_mitigation_without_target():
    isolator = DummyIsolator()
    env = _env(isolator=isolator)
    env.reset(seed=1, options={"normal_only_episode": True})

    _, reward, _, _, info = env.step(ACTION_ISOLATE)

    assert info["raw_action"] == ACTION_ISOLATE
    assert info["action"] == ACTION_ALLOW
    assert info["action_gated"] is True
    assert info["no_target_penalty"] is True
    assert isolator.actions == [ACTION_ALLOW]
    expected = compute_reward(ACTION_ALLOW, "normal") - CFG.attack.fp_step_penalty
    assert reward == pytest.approx(expected)
