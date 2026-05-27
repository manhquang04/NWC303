"""Feature extraction from NetworkSnapshot into 20-dim feature dict."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from config import CFG
from detection.flow_collector import NetworkSnapshot

log = logging.getLogger(__name__)


@dataclass
class FeatureContext:

    last_arp_table: Dict[str, str] = field(default_factory=dict)
    last_flow_count: int = 0
    last_port_bytes: Dict[str, int] = field(default_factory=dict)   # key=f"{dpid}:{port}:rx"
    last_timestamp: Optional[float] = None
    last_packet_ts: Optional[float] = None
    last_alert_ts: Optional[float] = None

    packet_sizes: Deque[int] = field(default_factory=lambda: deque(maxlen=200))
    inter_arrivals: Deque[float] = field(default_factory=lambda: deque(maxlen=200))

    arp_request_count: int = 0
    arp_reply_count: int = 0
    new_macs: set = field(default_factory=set)
    seen_macs: set = field(default_factory=set)
    beacon_count: int = 0
    unknown_ssid_count: int = 0
    icmp_count: int = 0
    tcp_syn_count: int = 0
    unique_dst_ips: set = field(default_factory=set)


class FeatureExtractor:
    """Compute 20 features from NetworkSnapshot with accumulated context."""

    def __init__(self) -> None:
        self.ctx = FeatureContext()
        self.whitelist = set(CFG.detection.ssid_whitelist)
        self.window = CFG.detection.feature_window_sec

    def extract(self, snap: NetworkSnapshot) -> Dict[str, float]:
        if snap is None:
            return self._zero_features()

        dt = self._dt(snap.timestamp)
        feats: Dict[str, float] = {}

        feats["arp_request_rate"] = self._safe_div(self.ctx.arp_request_count, dt)
        feats["arp_reply_rate"] = self._safe_div(self.ctx.arp_reply_count, dt)

        feats["mac_ip_mismatch_count"] = float(
            self._count_mac_ip_mismatch(snap.arp_table)
        )

        new_macs = self._diff_new_macs(snap.mac_table)
        feats["new_mac_rate"] = self._safe_div(len(new_macs), dt)

        feats["ssid_beacon_count"] = float(self.ctx.beacon_count)
        feats["unknown_ssid_count"] = float(self.ctx.unknown_ssid_count)

        feats["port_rx_rate_s1"], feats["port_tx_rate_s1"] = self._port_rates(snap, dpid=1, dt=dt)
        feats["port_rx_rate_s2"], feats["port_tx_rate_s2"] = self._port_rates(snap, dpid=2, dt=dt)

        cur_flow = sum(len(v) for v in snap.flow_stats.values())
        feats["flow_count_delta"] = float(cur_flow - self.ctx.last_flow_count)
        self.ctx.last_flow_count = cur_flow

        feats["icmp_rate"] = self._safe_div(self.ctx.icmp_count, dt)
        feats["tcp_syn_rate"] = self._safe_div(self.ctx.tcp_syn_count, dt)
        feats["unique_dst_rate"] = self._safe_div(len(self.ctx.unique_dst_ips), dt)

        feats["packet_size_mean"] = self._mean(self.ctx.packet_sizes)
        feats["packet_size_std"] = self._std(self.ctx.packet_sizes)
        feats["inter_arrival_mean"] = self._mean(self.ctx.inter_arrivals)

        active = sum(len(t) for t in snap.mac_table.values())
        feats["active_host_count"] = float(active)

        feats["suspicious_port_flag"] = float(
            feats["arp_reply_rate"] > CFG.detection.arp_rate_warn_threshold
            or feats["new_mac_rate"] > CFG.detection.new_mac_rate_warn_threshold
        )

        if self.ctx.last_alert_ts is None:
            feats["time_since_last_alert"] = 999.0
        else:
            feats["time_since_last_alert"] = float(snap.timestamp - self.ctx.last_alert_ts)

        self._reset_window_counters()
        self.ctx.last_timestamp = snap.timestamp
        return feats

    def mark_alert(self, ts: Optional[float] = None) -> None:
        self.ctx.last_alert_ts = ts if ts is not None else time.time()

    def on_arp_request(self) -> None: self.ctx.arp_request_count += 1
    def on_arp_reply(self) -> None: self.ctx.arp_reply_count += 1
    def on_beacon(self, ssid: str) -> None:
        self.ctx.beacon_count += 1
        if ssid not in self.whitelist:
            self.ctx.unknown_ssid_count += 1
    def on_icmp(self) -> None: self.ctx.icmp_count += 1
    def on_tcp_syn(self) -> None: self.ctx.tcp_syn_count += 1
    def on_packet(self, size: int, dst_ip: Optional[str] = None) -> None:
        now = time.time()
        if self.ctx.last_packet_ts is not None:
            self.ctx.inter_arrivals.append(now - self.ctx.last_packet_ts)
        self.ctx.last_packet_ts = now
        self.ctx.packet_sizes.append(int(size))
        if dst_ip:
            self.ctx.unique_dst_ips.add(dst_ip)

    def _zero_features(self) -> Dict[str, float]:
        from detection.state_builder import FEATURE_ORDER
        return {k: 0.0 for k in FEATURE_ORDER}

    def _dt(self, ts: float) -> float:
        if self.ctx.last_timestamp is None:
            return self.window
        return max(1e-3, ts - self.ctx.last_timestamp)

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        return float(num) / float(den) if den > 0 else 0.0

    @staticmethod
    def _mean(xs) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    @staticmethod
    def _std(xs) -> float:
        if not xs:
            return 0.0
        m = sum(xs) / len(xs)
        return float((sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5)

    def _count_mac_ip_mismatch(self, arp_table: Dict[str, str]) -> int:
        cnt = 0
        for ip, mac in arp_table.items():
            prev = self.ctx.last_arp_table.get(ip)
            if prev is not None and prev != mac:
                cnt += 1
        self.ctx.last_arp_table = dict(arp_table)
        return cnt

    def _diff_new_macs(self, mac_table: Dict[int, Dict[str, int]]) -> List[str]:
        cur: set = set()
        for _dpid, t in mac_table.items():
            cur.update(t.keys())
        new = cur - self.ctx.seen_macs
        self.ctx.seen_macs.update(cur)
        return list(new)

    def _port_rates(self, snap: NetworkSnapshot, dpid: int, dt: float):
        raw = snap.port_stats.get(dpid, [])
        if isinstance(raw, dict):
            ports = list(raw.values())
        else:
            ports = raw
        if not ports:
            return 0.0, 0.0
        best = max(ports, key=lambda p: p.get("rx_bytes", 0))
        port_no = best.get("port_no", 0)
        key_rx = f"{dpid}:{port_no}:rx"
        key_tx = f"{dpid}:{port_no}:tx"
        rx_now = int(best.get("rx_bytes", 0))
        tx_now = int(best.get("tx_bytes", 0))
        rx_prev = self.ctx.last_port_bytes.get(key_rx, rx_now)
        tx_prev = self.ctx.last_port_bytes.get(key_tx, tx_now)
        self.ctx.last_port_bytes[key_rx] = rx_now
        self.ctx.last_port_bytes[key_tx] = tx_now
        return self._safe_div(rx_now - rx_prev, dt), self._safe_div(tx_now - tx_prev, dt)

    def _reset_window_counters(self) -> None:
        self.ctx.arp_request_count = 0
        self.ctx.arp_reply_count = 0
        self.ctx.beacon_count = 0
        self.ctx.unknown_ssid_count = 0
        self.ctx.icmp_count = 0
        self.ctx.tcp_syn_count = 0
        self.ctx.unique_dst_ips.clear()
