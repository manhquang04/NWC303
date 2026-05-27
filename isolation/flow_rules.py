"""OpenFlow rule builders for Ryu REST API."""

from __future__ import annotations

from typing import Any, Dict

from config import CFG


def make_drop_rule(
    dpid: int,
    port: int,
    priority: int = CFG.isolation.drop_rule_priority,
    idle_timeout: int = CFG.isolation.rule_idle_timeout_sec,
    hard_timeout: int = CFG.isolation.rule_hard_timeout_sec,
) -> Dict[str, Any]:
    """Build drop rule for all packets entering ``port`` on switch ``dpid``."""
    return {
        "dpid": int(dpid),
        "priority": int(priority),
        "idle_timeout": int(idle_timeout),
        "hard_timeout": int(hard_timeout),
        "match": {"in_port": int(port)},
        "actions": [],
    }


def make_vlan_rule(
    dpid: int,
    port: int,
    vlan_id: int = CFG.isolation.quarantine_vlan_id,
    priority: int = CFG.isolation.vlan_rule_priority,
    idle_timeout: int = CFG.isolation.rule_idle_timeout_sec,
    hard_timeout: int = CFG.isolation.rule_hard_timeout_sec,
) -> Dict[str, Any]:
    """Push packets from ``port`` into quarantine VLAN."""
    return {
        "dpid": int(dpid),
        "priority": int(priority),
        "idle_timeout": int(idle_timeout),
        "hard_timeout": int(hard_timeout),
        "match": {"in_port": int(port)},
        "actions": [
            {"type": "PUSH_VLAN", "ethertype": 0x8100},
            {"type": "SET_FIELD", "field": "vlan_vid", "value": 0x1000 | int(vlan_id)},
            {"type": "OUTPUT", "port": "NORMAL"},
        ],
    }


def make_rate_limit_rule(
    dpid: int,
    port: int,
    kbps: int = CFG.isolation.rate_limit_kbps,
    priority: int = CFG.isolation.drop_rule_priority,
) -> Dict[str, Any]:
    """Rate-limit via meter (requires OF1.3+ meter support)."""
    return {
        "dpid": int(dpid),
        "priority": int(priority),
        "match": {"in_port": int(port)},
        "actions": [
            {"type": "METER", "meter_id": 1},
            {"type": "OUTPUT", "port": "NORMAL"},
        ],
        "_meter_kbps": int(kbps),
    }


def make_match_dict(dpid: int, port: int) -> Dict[str, Any]:
    """Body for DELETE /stats/flowentry/delete."""
    return {"dpid": int(dpid), "match": {"in_port": int(port)}}
