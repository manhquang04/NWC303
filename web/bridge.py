"""Bridge between FastAPI and SDN-DRL-IDS modules."""

from __future__ import annotations

import json
import logging
import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    ACTION_NAMES,
    CFG,
)

_TRAIN_STATE_FILE = Path("/tmp/sdnids_train_state.json")
from detection.flow_collector import FlowCollector, NetworkSnapshot
from detection.feature_extractor import FeatureExtractor
from detection.state_builder import StateBuilder, FEATURE_ORDER, FEATURE_MAX
from detection.target_selector import TargetSelector, infer_attack_type
from isolation.isolator import Isolator, IsolationEvent
from evaluation.metrics import MetricsCalculator, StepRecord, MetricsReport

log = logging.getLogger(__name__)

try:
    from scapy.all import sniff as scapy_sniff, IP, ARP, TCP, UDP, ICMP, Ether
    _HAS_SCAPY = True
except ImportError:
    _HAS_SCAPY = False


@dataclass
class DashboardState:

    timestamp: float = 0.0
    state_vector: List[float] = field(default_factory=list)
    raw_features: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)

    current_action: int = 0
    current_action_name: str = "allow"
    ground_truth: str = "unknown"
    attack_type: str = "none"
    step_reward: float = 0.0
    step_count: int = 0
    target: Optional[Dict[str, Any]] = None

    episode: int = 0
    epsilon: float = 0.0
    cumulative_reward: float = 0.0

    metrics: Dict[str, float] = field(default_factory=dict)
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    topology: Dict[str, Any] = field(default_factory=dict)
    arp_table: Dict[str, str] = field(default_factory=dict)
    mac_table: Dict[str, Any] = field(default_factory=dict)


class WebBridge:

    def __init__(
        self,
        flow_collector: Optional[FlowCollector] = None,
        isolator: Optional[Isolator] = None,
    ) -> None:
        self.flow_collector = flow_collector
        self.isolator = isolator
        self.feature_extractor = FeatureExtractor()
        self.state_builder = StateBuilder()
        self.target_selector = TargetSelector()
        self.metrics_calc = MetricsCalculator()

        self._lock = threading.Lock()
        self._current_state = DashboardState()
        self._action_history: List[Dict[str, Any]] = []
        self._event_log: List[Dict[str, Any]] = []
        self._step_count = 0
        self._episode = 0
        self._epsilon = 1.0
        self._cumulative_reward = 0.0
        self._attack_active = False

        self._topology_info = self._build_topology_info()
        self._sniffer_thread: Optional[threading.Thread] = None
        self._sniffer_stop = threading.Event()

    def start_sniffer(self, iface: str = "any") -> None:
        """Start background packet sniffer to feed feature extractor callbacks."""
        if not _HAS_SCAPY:
            log.warning("Scapy not available — sniffer disabled, features will be limited.")
            return
        if self._sniffer_thread is not None and self._sniffer_thread.is_alive():
            return
        self._sniffer_stop.clear()
        self._sniffer_thread = threading.Thread(
            target=self._sniffer_loop, args=(iface,), name="PacketSniffer", daemon=True
        )
        self._sniffer_thread.start()
        log.info("Packet sniffer started on iface=%s", iface)

    def stop_sniffer(self) -> None:
        self._sniffer_stop.set()

    def _sniffer_loop(self, iface: str) -> None:
        try:
            # iface=None means all interfaces on newer scapy
            scapy_sniff(
                iface=iface if iface != "any" else None,
                prn=self._process_packet,
                store=False,
                stop_filter=lambda _: self._sniffer_stop.is_set(),
            )
        except Exception as exc:
            log.warning("Sniffer stopped: %s", exc)

    def _process_packet(self, pkt) -> None:
        try:
            size = len(pkt)
            dst_ip = None
            if pkt.haslayer(IP):
                dst_ip = pkt[IP].dst
            self.feature_extractor.on_packet(size, dst_ip)

            if pkt.haslayer(ARP):
                opcode = pkt[ARP].op
                if opcode == 1:
                    self.feature_extractor.on_arp_request()
                elif opcode == 2:
                    self.feature_extractor.on_arp_reply()

            if pkt.haslayer(ICMP):
                self.feature_extractor.on_icmp()

            if pkt.haslayer(TCP) and pkt[TCP].flags == "S":
                self.feature_extractor.on_tcp_syn()
        except Exception:
            pass

    def _build_topology_info(self) -> Dict[str, Any]:
        num_switches = CFG.topology.num_switches
        num_hosts = CFG.topology.num_hosts

        nodes = []
        edges = []

        for i in range(1, num_switches + 1):
            nodes.append({
                "id": f"s{i}",
                "label": f"Switch {i}",
                "type": "switch",
                "group": "switch",
            })

        for i in range(1, num_hosts + 1):
            role = "normal"
            if i == CFG.topology.rogue_host_idx:
                role = "rogue_ap"
            elif i == CFG.topology.spoofer_host_idx:
                role = "arp_spoofer"

            nodes.append({
                "id": f"h{i}",
                "label": f"Host {i}",
                "type": "host",
                "group": role,
                "ip": f"10.0.0.{i}",
                "role": role,
            })

        for i in range(1, num_switches):
            edges.append({
                "from": f"s{i}",
                "to": f"s{i + 1}",
                "label": "trunk",
            })

        hosts_per_sw = max(1, num_hosts // num_switches)
        for i in range(1, num_hosts + 1):
            sw_idx = min((i - 1) // hosts_per_sw, num_switches - 1) + 1
            edges.append({
                "from": f"h{i}",
                "to": f"s{sw_idx}",
                "label": f"port",
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "config": {
                "num_switches": num_switches,
                "num_hosts": num_hosts,
                "rogue_host": f"h{CFG.topology.rogue_host_idx}",
                "spoofer_host": f"h{CFG.topology.spoofer_host_idx}",
                "link_bw_mbps": CFG.topology.link_bw_mbps,
            },
        }

    def _read_train_state(self) -> Optional[Dict[str, Any]]:
        """Read training state written by agent/train.py."""
        try:
            if not _TRAIN_STATE_FILE.exists():
                return None
            mtime = os.path.getmtime(_TRAIN_STATE_FILE)
            if time.time() - mtime > 5.0:
                return None
            with open(_TRAIN_STATE_FILE) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def update_state(
        self,
        action: int = 0,
        ground_truth: str = "unknown",
        reward: float = 0.0,
    ) -> None:
        # Read training state if available (overrides passed args)
        train = self._read_train_state()
        if train is not None:
            action = train.get("action", action)
            ground_truth = train.get("ground_truth", ground_truth)
            reward = train.get("reward", reward)
            self._episode = train.get("episode", self._episode)
            self._epsilon = train.get("epsilon", self._epsilon)
            train_attack_type = train.get("attack_type", "none")
            train_target = train.get("target")
        else:
            train_attack_type = "none"
            train_target = None

        with self._lock:
            self._step_count += 1
            self._cumulative_reward += reward

            snap = self.flow_collector.get_latest() if self.flow_collector else None

            raw_features = {}
            state_vector = []
            if snap is not None:
                raw_features = self.feature_extractor.extract(snap)
                vec = self.state_builder.build(raw_features)
                state_vector = vec.tolist()

            attack_type = train_attack_type
            if attack_type in ("", "none", "unknown") and raw_features:
                attack_type = infer_attack_type(raw_features)
                if attack_type == "unknown":
                    attack_type = "none"

            target = train_target
            if target is None and action in (2, 3):
                selected = self.target_selector.select(snap, raw_features)
                target = selected.as_dict() if selected else None

            ts = time.time()
            is_attack = ground_truth == "attack"
            took_action = action != 0
            blocked = action in (2, 3)

            if is_attack and not self._attack_active:
                self.metrics_calc.mark_attack_start(ts)
            self._attack_active = is_attack

            rec = StepRecord(
                timestamp=ts,
                ground_truth=ground_truth,
                action=action,
                reward=reward,
                detected=took_action,
                isolated=blocked,
            )
            self.metrics_calc.add_record(rec)

            if is_attack and took_action:
                self.metrics_calc.mark_detected(ts)
            if is_attack and blocked:
                self.metrics_calc.mark_isolated(ts)

            action_entry = {
                "timestamp": ts,
                "step": self._step_count,
                "action": action,
                "action_name": ACTION_NAMES[action],
                "ground_truth": ground_truth,
                "attack_type": attack_type,
                "reward": reward,
                "target": target,
            }
            self._action_history.append(action_entry)
            if len(self._action_history) > 500:
                self._action_history = self._action_history[-500:]

            if action != 0:
                event_msg = self._format_event(action, ground_truth, reward, raw_features, target, attack_type)
                self._event_log.append({
                    "timestamp": ts,
                    "message": event_msg,
                    "action": action,
                    "action_name": ACTION_NAMES[action],
                    "attack_type": attack_type,
                    "target": target,
                })
                if len(self._event_log) > 200:
                    self._event_log = self._event_log[-200:]

            arp_table = dict(snap.arp_table) if snap else {}
            mac_table = {}
            if snap and snap.mac_table:
                for dpid, table in snap.mac_table.items():
                    mac_table[str(dpid)] = table

            self._current_state = DashboardState(
                timestamp=ts,
                state_vector=state_vector,
                raw_features=dict(raw_features),
                feature_names=list(FEATURE_ORDER),
                current_action=action,
                current_action_name=ACTION_NAMES[action],
                ground_truth=ground_truth,
                attack_type=attack_type,
                step_reward=reward,
                step_count=self._step_count,
                target=target,
                episode=self._episode,
                epsilon=self._epsilon,
                cumulative_reward=self._cumulative_reward,
                metrics=self._get_metrics_dict(),
                recent_events=list(self._event_log[-20:]),
                topology=self._topology_info,
                arp_table=arp_table,
                mac_table=mac_table,
            )

    def _format_event(self, action: int, ground_truth: str, reward: float,
                      raw_features: Optional[Dict[str, float]] = None,
                      target: Optional[Dict[str, Any]] = None,
                      attack_type: str = "none") -> str:
        action_name = ACTION_NAMES[action].upper()
        attack_detail = ""
        if ground_truth == "attack":
            if attack_type and attack_type != "none":
                attack_detail = f" [{attack_type}]"
        if ground_truth == "attack":
            if raw_features and not attack_detail:
                arp_rate = raw_features.get("arp_reply_rate", 0)
                beacon = raw_features.get("ssid_beacon_count", 0)
                if arp_rate > 10:
                    attack_detail = " [ARP Spoofing h6]"
                elif beacon > 0:
                    attack_detail = " [Rogue AP h5]"
            if not attack_detail:
                attack_detail = " [Attack]"
        gt_info = f"({ground_truth})" if ground_truth != "unknown" else ""
        if target:
            target_info = f" target=s{target.get('dpid')}:p{target.get('port')}"
        elif action in (2, 3):
            target_info = " target=no_target"
        else:
            target_info = ""
        return f"{action_name}{attack_detail} {gt_info}{target_info} reward={reward:+.1f}"

    def _get_metrics_dict(self) -> Dict[str, float]:
        report = self.metrics_calc.compute(episodes=self._episode)
        return report.as_dict()

    def get_state(self) -> DashboardState:
        with self._lock:
            return self._current_state

    def get_topology(self) -> Dict[str, Any]:
        return self._topology_info

    def get_action_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._action_history[-limit:])

    def get_event_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._event_log[-limit:])

    def get_metrics(self) -> Dict[str, float]:
        with self._lock:
            return self._get_metrics_dict()

    def get_feature_info(self) -> Dict[str, Any]:
        with self._lock:
            state = self._current_state
            features = []
            for i, name in enumerate(FEATURE_ORDER):
                raw = state.raw_features.get(name, 0.0)
                normalized = state.state_vector[i] if i < len(state.state_vector) else 0.0
                features.append({
                    "name": name,
                    "raw_value": raw,
                    "normalized_value": normalized,
                    "max_cap": FEATURE_MAX.get(name, 1.0),
                })
            return {"features": features}

    def get_network_snapshot(self) -> Dict[str, Any]:
        snap = self.flow_collector.get_latest() if self.flow_collector else None
        if snap is None:
            return {"error": "No snapshot available"}
        return {
            "timestamp": snap.timestamp,
            "flow_stats": {
                str(dpid): stats for dpid, stats in snap.flow_stats.items()
            },
            "port_stats": {
                str(dpid): stats for dpid, stats in snap.port_stats.items()
            },
            "arp_table": dict(snap.arp_table),
            "mac_table": {
                str(dpid): table for dpid, table in snap.mac_table.items()
            },
        }

    def set_episode(self, episode: int) -> None:
        with self._lock:
            self._episode = episode

    def set_epsilon(self, epsilon: float) -> None:
        with self._lock:
            self._epsilon = epsilon

    def reset_episode(self) -> None:
        with self._lock:
            self._step_count = 0
            self._cumulative_reward = 0.0
            self.feature_extractor = FeatureExtractor()
            self.state_builder = StateBuilder()

    def mark_attack_start(self, ts: Optional[float] = None) -> None:
        self.metrics_calc.mark_attack_start(ts or time.time())

    def add_event(self, message: str, action_name: str = "info") -> None:
        with self._lock:
            action_map = {"info": 0, "allow": 0, "flag": 1, "block": 2, "isolate": 3}
            self._event_log.append({
                "timestamp": time.time(),
                "message": message,
                "action": action_map.get(action_name, 0),
                "action_name": action_name,
                "attack_type": "none",
                "target": None,
            })
