"""Test detection/feature_extractor.py."""

import time

from detection.feature_extractor import FeatureExtractor
from detection.flow_collector import NetworkSnapshot
from detection.state_builder import FEATURE_ORDER


def _empty_snap(ts: float | None = None) -> NetworkSnapshot:
    return NetworkSnapshot(timestamp=ts or time.time())


def test_extract_returns_all_keys():
    fe = FeatureExtractor()
    snap = _empty_snap()
    feats = fe.extract(snap)
    for key in FEATURE_ORDER:
        assert key in feats, f"Missing key: {key}"


def test_zero_input_zero_output():
    fe = FeatureExtractor()
    snap = _empty_snap()
    feats = fe.extract(snap)
    assert feats["arp_request_rate"] == 0.0
    assert feats["arp_reply_rate"] == 0.0
    assert feats["mac_ip_mismatch_count"] == 0.0


def test_arp_event_increments_counter():
    fe = FeatureExtractor()
    fe.on_arp_request()
    fe.on_arp_request()
    fe.on_arp_reply()
    snap = _empty_snap()
    feats = fe.extract(snap)
    assert feats["arp_request_rate"] > 0
    assert feats["arp_reply_rate"] > 0


def test_mark_alert_resets_time():
    fe = FeatureExtractor()
    fe.mark_alert(ts=time.time())
    snap = _empty_snap()
    feats = fe.extract(snap)
    assert feats["time_since_last_alert"] < 5.0
