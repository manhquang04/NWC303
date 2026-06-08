"""Test detection/target_selector.py."""

import time

from config import CFG
from detection.flow_collector import NetworkSnapshot
from detection.target_selector import TargetSelector, infer_attack_type


def test_infer_attack_type_arp_spoof():
    attack_type = infer_attack_type({
        "arp_reply_rate": 60.0,
        "mac_ip_mismatch_count": 1.0,
        "unknown_ssid_count": 0.0,
        "ssid_beacon_count": 0.0,
    })
    assert attack_type == "arp_spoof"


def test_infer_attack_type_rogue_ap():
    attack_type = infer_attack_type({
        "arp_reply_rate": 0.0,
        "mac_ip_mismatch_count": 0.0,
        "unknown_ssid_count": 2.0,
        "ssid_beacon_count": 10.0,
    })
    assert attack_type == "rogue_ap"


def test_selects_arp_spoofer_target_from_arp_table():
    spoofer_mac = "00:00:00:00:00:06"
    snap = NetworkSnapshot(
        timestamp=time.time(),
        arp_table={CFG.attack.arp_spoof_target_ip: spoofer_mac},
        mac_table={2: {spoofer_mac: 4}},
    )
    features = {
        "arp_reply_rate": 80.0,
        "mac_ip_mismatch_count": 1.0,
        "unknown_ssid_count": 0.0,
        "ssid_beacon_count": 0.0,
        "suspicious_port_flag": 1.0,
    }

    target = TargetSelector().select(snap, features)

    assert target is not None
    assert target.dpid == 2
    assert target.port == 4
    assert target.attack_type == "arp_spoof"


def test_selects_rogue_ap_target_from_known_host_mac():
    rogue_mac = "00:00:00:00:00:05"
    snap = NetworkSnapshot(
        timestamp=time.time(),
        mac_table={3: {rogue_mac: 2}},
    )
    features = {
        "arp_reply_rate": 0.0,
        "mac_ip_mismatch_count": 0.0,
        "unknown_ssid_count": 1.0,
        "ssid_beacon_count": 20.0,
        "suspicious_port_flag": 1.0,
    }

    target = TargetSelector().select(snap, features)

    assert target is not None
    assert target.dpid == 3
    assert target.port == 2
    assert target.attack_type == "rogue_ap"


def test_select_falls_back_to_highest_rx_port_when_suspicious():
    snap = NetworkSnapshot(
        timestamp=time.time(),
        port_stats={
            1: [
                {"port_no": 1, "rx_bytes": 100},
                {"port_no": 2, "rx_bytes": 900},
            ],
        },
    )
    features = {
        "arp_reply_rate": 40.0,
        "mac_ip_mismatch_count": 0.0,
        "unknown_ssid_count": 0.0,
        "ssid_beacon_count": 0.0,
        "suspicious_port_flag": 1.0,
    }

    target = TargetSelector().select(snap, features)

    assert target is not None
    assert target.dpid == 1
    assert target.port == 2
    assert target.reason == "highest_rx_port"


def test_known_mininet_attacker_fallback_for_arp_spoof():
    target = TargetSelector().from_attack_type("ARPSpoofAttack")

    assert target is not None
    assert target.dpid == 3
    assert target.port == 2
    assert target.reason == "known_mininet_attacker"


def test_known_mininet_attacker_fallback_for_rogue_ap():
    target = TargetSelector().from_attack_type("RogueAPAttack")

    assert target is not None
    assert target.dpid == 3
    assert target.port == 1
    assert target.reason == "known_mininet_attacker"
