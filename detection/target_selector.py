"""Select the most likely switch port to block or isolate."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

from config import CFG
from detection.flow_collector import NetworkSnapshot

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IsolationTarget:
    """Candidate OpenFlow ingress port for mitigation."""

    dpid: int
    port: int
    score: float
    reason: str
    attack_type: str = "unknown"
    mac: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dpid": self.dpid,
            "port": self.port,
            "score": self.score,
            "reason": self.reason,
            "attack_type": self.attack_type,
            "mac": self.mac,
            "metadata": dict(self.metadata),
        }


def infer_attack_type(features: Dict[str, float]) -> str:
    """Infer attack family from feature evidence."""
    arp_reply = features.get("arp_reply_rate", 0.0)
    mismatch = features.get("mac_ip_mismatch_count", 0.0)
    unknown_ssid = features.get("unknown_ssid_count", 0.0)
    beacon = features.get("ssid_beacon_count", 0.0)

    arp_score = arp_reply + mismatch * 50.0
    rogue_score = unknown_ssid * 50.0 + beacon

    if arp_score <= 0 and rogue_score <= 0:
        return "unknown"
    if arp_score >= rogue_score:
        return "arp_spoof"
    return "rogue_ap"


class TargetSelector:
    """Heuristic mapper from suspicious features to a concrete switch port."""

    def __init__(self) -> None:
        self.rogue_host_idx = CFG.topology.rogue_host_idx
        self.spoofer_host_idx = CFG.topology.spoofer_host_idx
        self.rogue_bssid = "de:ad:be:ef:00:01"

    def select(
        self,
        snap: Optional[NetworkSnapshot],
        features: Dict[str, float],
    ) -> Optional[IsolationTarget]:
        if snap is None:
            return None

        attack_type = infer_attack_type(features)
        suspicious = features.get("suspicious_port_flag", 0.0) > 0.5
        suspicious = suspicious or features.get("mac_ip_mismatch_count", 0.0) > 0
        suspicious = suspicious or features.get("unknown_ssid_count", 0.0) > 0
        suspicious = suspicious or features.get("arp_reply_rate", 0.0) > CFG.detection.arp_rate_warn_threshold
        if not suspicious:
            return None

        candidates: Dict[Tuple[int, int], IsolationTarget] = {}
        arp_table = _normalize_arp_table(snap.arp_table)
        mac_locations = _normalize_mac_table(snap.mac_table)

        def add(
            dpid: int,
            port: int,
            score: float,
            reason: str,
            mac: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> None:
            key = (int(dpid), int(port))
            existing = candidates.get(key)
            total_score = score + (existing.score if existing else 0.0)
            reasons = reason if existing is None else f"{existing.reason},{reason}"
            candidates[key] = IsolationTarget(
                dpid=key[0],
                port=key[1],
                score=total_score,
                reason=reasons,
                attack_type=attack_type,
                mac=mac or (existing.mac if existing else None),
                metadata=metadata or (existing.metadata if existing else {}),
            )

        known_spoofer_mac = _mininet_mac(self.spoofer_host_idx)
        known_rogue_mac = _mininet_mac(self.rogue_host_idx)
        spoofed_target_mac = arp_table.get(CFG.attack.arp_spoof_target_ip)

        spoofer_macs = (spoofed_target_mac, known_spoofer_mac) if attack_type != "rogue_ap" else (spoofed_target_mac,)
        for mac in spoofer_macs:
            if not mac:
                continue
            loc = mac_locations.get(mac.lower())
            if loc:
                add(*loc, score=8.0, reason="arp_spoofer_mac", mac=mac)

        rogue_macs = (known_rogue_mac, self.rogue_bssid) if attack_type != "arp_spoof" else (self.rogue_bssid,)
        for mac in rogue_macs:
            loc = mac_locations.get(mac.lower())
            if loc:
                add(*loc, score=8.0, reason="rogue_ap_mac", mac=mac)

        if features.get("mac_ip_mismatch_count", 0.0) > 0:
            for ip, mac in arp_table.items():
                loc = mac_locations.get(mac.lower())
                if loc:
                    add(*loc, score=4.0, reason="mac_ip_mismatch", mac=mac, metadata={"ip": ip})

        if not candidates:
            top_port = _highest_rx_port(snap.port_stats)
            if top_port is not None:
                add(*top_port, score=1.0, reason="highest_rx_port")

        if not candidates:
            return None

        return max(candidates.values(), key=lambda c: c.score)


def _normalize_arp_table(raw: Dict[str, str]) -> Dict[str, str]:
    return {str(ip): str(mac).lower() for ip, mac in (raw or {}).items()}


def _normalize_mac_table(raw: Any) -> Dict[str, Tuple[int, int]]:
    result: Dict[str, Tuple[int, int]] = {}
    if not isinstance(raw, dict):
        return result
    for dpid_raw, table in raw.items():
        try:
            dpid = int(dpid_raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(table, dict):
            continue
        for mac, port_raw in table.items():
            try:
                port = int(port_raw)
            except (TypeError, ValueError):
                continue
            result[str(mac).lower()] = (dpid, port)
    return result


def _highest_rx_port(raw: Any) -> Optional[Tuple[int, int]]:
    best: Optional[Tuple[int, int, int]] = None
    if not isinstance(raw, dict):
        return None
    for dpid_raw, ports_raw in raw.items():
        try:
            dpid = int(dpid_raw)
        except (TypeError, ValueError):
            continue
        for port in _iter_port_dicts(ports_raw):
            try:
                port_no = int(port.get("port_no", 0))
                rx_bytes = int(port.get("rx_bytes", 0))
            except (TypeError, ValueError):
                continue
            if port_no <= 0:
                continue
            if best is None or rx_bytes > best[2]:
                best = (dpid, port_no, rx_bytes)
    return (best[0], best[1]) if best else None


def _iter_port_dicts(raw: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list):
        return []
    flat = []
    for item in raw:
        if isinstance(item, dict):
            flat.append(item)
        elif isinstance(item, list):
            flat.extend(x for x in item if isinstance(x, dict))
    return flat


def _mininet_mac(host_idx: int) -> str:
    return f"00:00:00:00:00:{int(host_idx):02x}"
