"""Normalize feature dict into [0,1] numpy vector for DRL agent."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

from config import CFG

log = logging.getLogger(__name__)

# Feature order — fixed after training.
FEATURE_ORDER: Tuple[str, ...] = (
    "arp_request_rate",
    "arp_reply_rate",
    "mac_ip_mismatch_count",
    "new_mac_rate",
    "ssid_beacon_count",
    "unknown_ssid_count",
    "port_rx_rate_s1",
    "port_tx_rate_s1",
    "port_rx_rate_s2",
    "port_tx_rate_s2",
    "flow_count_delta",
    "icmp_rate",
    "tcp_syn_rate",
    "unique_dst_rate",
    "packet_size_mean",
    "packet_size_std",
    "inter_arrival_mean",
    "active_host_count",
    "suspicious_port_flag",
    "time_since_last_alert",
)

assert len(FEATURE_ORDER) == CFG.detection.state_dim

# Normalization caps per feature.
FEATURE_MAX: Dict[str, float] = {
    "arp_request_rate":      200.0,
    "arp_reply_rate":        200.0,
    "mac_ip_mismatch_count": 50.0,
    "new_mac_rate":          20.0,
    "ssid_beacon_count":     50.0,
    "unknown_ssid_count":    20.0,
    "port_rx_rate_s1":       1e7,
    "port_tx_rate_s1":       1e7,
    "port_rx_rate_s2":       1e7,
    "port_tx_rate_s2":       1e7,
    "flow_count_delta":      100.0,
    "icmp_rate":             500.0,
    "tcp_syn_rate":          1000.0,
    "unique_dst_rate":       100.0,
    "packet_size_mean":      1500.0,
    "packet_size_std":       1500.0,
    "inter_arrival_mean":    1.0,
    "active_host_count":     50.0,
    "suspicious_port_flag":  1.0,
    "time_since_last_alert": 60.0,
}


class StateBuilder:
    """Convert feature dict to normalized [0,1] numpy vector."""

    def __init__(self, feature_max: Dict[str, float] = FEATURE_MAX) -> None:
        self.feature_max = dict(feature_max)
        self._missing_logged: set = set()

    def build(self, features: Dict[str, float]) -> np.ndarray:
        vec = np.zeros(len(FEATURE_ORDER), dtype=np.float32)
        for i, key in enumerate(FEATURE_ORDER):
            if key not in features:
                if key not in self._missing_logged:
                    log.warning("Feature missing: %s — defaulting to 0", key)
                    self._missing_logged.add(key)
                continue
            raw = float(features[key])
            cap = self.feature_max.get(key, 1.0)
            vec[i] = np.clip(raw / cap if cap > 0 else 0.0, 0.0, 1.0)
        return vec

    def feature_names(self) -> Tuple[str, ...]:
        return FEATURE_ORDER
