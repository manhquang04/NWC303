#!/usr/bin/env python3
"""Ryu ARP guard controller for online SDN runtime evaluation.

Detects ARP spoofing by observing ARP sender IP -> MAC changes. On conflict,
it logs an alert and installs a drop rule on the attacker's ingress port.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib.packet import arp, ethernet, packet
from ryu.app.wsgi import ControllerBase, Response, WSGIApplication, route
from ryu.ofproto import ofproto_v1_3


EVENT_LOG = Path(os.environ.get("ARP_GUARD_EVENT_LOG", "/tmp/arp_guard_events.jsonl"))
USE_DQN = os.environ.get("ARP_GUARD_USE_DQN", "1") != "0"
ROGUE_AP_INSTANCE = "rogue_ap_instance"


def append_event(**event):
    event.setdefault("timestamp", time.time())
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


class TinyQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.net(x)


class RuntimeDQNPolicy:
    """Small DQN-style policy for runtime ARP decisions.

    Actions:
      0 allow, 1 flag, 2 block
    State:
      [is_reply, known_ip, mac_conflict, same_ip_mac_count_norm,
       in_port_norm, blocked_port]
    """

    def __init__(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = TinyQNet()
        self._train_synthetic_q()
        self.model.eval()

    @staticmethod
    def reward(action, attack):
        if attack:
            return [ -10.0, 3.0, 10.0 ][action]
        return [ 1.0, -2.0, -8.0 ][action]

    def _train_synthetic_q(self):
        states = []
        targets = []
        rng = np.random.default_rng(42)
        for _ in range(2000):
            is_reply = rng.integers(0, 2)
            known_ip = rng.integers(0, 2)
            conflict = rng.integers(0, 2)
            same_count = rng.choice([0.0, 0.5, 1.0])
            in_port = rng.uniform(0.0, 1.0)
            blocked = rng.integers(0, 2)
            # Treat only observed MAC conflict for known IP as attack.
            attack = bool(is_reply and known_ip and conflict)
            state = np.array([is_reply, known_ip, conflict, same_count, in_port, blocked], dtype=np.float32)
            q = np.array([self.reward(a, attack) for a in range(3)], dtype=np.float32)
            states.append(state)
            targets.append(q)
        x = torch.tensor(np.stack(states), dtype=torch.float32)
        y = torch.tensor(np.stack(targets), dtype=torch.float32)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        for _ in range(400):
            pred = self.model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def act(self, state):
        with torch.no_grad():
            q = self.model(torch.tensor(state[None, :], dtype=torch.float32)).numpy()[0]
        return int(np.argmax(q)), q.tolist()


class ArpGuardController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wsgi = kwargs["wsgi"]
        wsgi.register(RogueApRestController, {ROGUE_AP_INSTANCE: self})
        self.mac_to_port = {}
        self.ip_to_mac = {}
        self.ip_to_macs = {}
        self.datapaths = {}
        self.blocked_ports = set()
        self.policy = RuntimeDQNPolicy() if USE_DQN else None
        EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        EVENT_LOG.write_text("", encoding="utf-8")

    def log_event(self, **event):
        append_event(**event)

    def handle_rogue_ap_alert(self, payload):
        self.log_event(
            event="rogue_ap_alert_received",
            source="wifi_monitor_sensor",
            score=payload.get("score"),
            smoothed_score=payload.get("smoothed_score"),
            threshold=payload.get("threshold"),
            action=payload.get("action"),
            top_bssid=payload.get("top_bssid"),
            top_ssid=payload.get("top_ssid"),
            unique_bssid=payload.get("unique_bssid"),
            unique_ssid=payload.get("unique_ssid"),
            recommended_action=payload.get("recommended_action", "quarantine_or_investigate"),
            raw_payload=payload,
        )
        return {"ok": True, "event": "rogue_ap_alert_received"}

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        self.datapaths[dp.id] = dp
        self.mac_to_port.setdefault(dp.id, {})
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        self.add_flow(dp, 0, match, actions)
        self.log_event(event="switch_connected", dpid=dp.id)

    def add_flow(self, dp, priority, match, actions, idle_timeout=0, hard_timeout=0):
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=dp,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
        )
        dp.send_msg(mod)

    def add_drop_for_port(self, dp, in_port):
        if (dp.id, in_port) in self.blocked_ports:
            return
        parser = dp.ofproto_parser
        match = parser.OFPMatch(in_port=in_port)
        self.add_flow(dp, 200, match, [], idle_timeout=60, hard_timeout=180)
        self.blocked_ports.add((dp.id, in_port))
        self.log_event(event="drop_rule_installed", dpid=dp.id, in_port=int(in_port))

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        in_port = msg.match["in_port"]
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None:
            return

        src = eth.src
        dst = eth.dst
        self.mac_to_port.setdefault(dp.id, {})
        self.mac_to_port[dp.id][src] = in_port

        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt is not None:
            src_ip = arp_pkt.src_ip
            old_mac = self.ip_to_mac.get(src_ip)
            self.log_event(
                event="arp_observed",
                dpid=dp.id,
                in_port=int(in_port),
                src_ip=src_ip,
                src_mac=src,
                opcode=int(arp_pkt.opcode),
            )
            if old_mac is not None and old_mac != src:
                macs = self.ip_to_macs.setdefault(src_ip, set())
                macs.add(old_mac)
                macs.add(src)
                state = np.array([
                    1.0 if int(arp_pkt.opcode) == 2 else 0.0,
                    1.0,
                    1.0,
                    min(1.0, len(macs) / 4.0),
                    min(1.0, float(in_port) / 10.0),
                    1.0 if (dp.id, in_port) in self.blocked_ports else 0.0,
                ], dtype=np.float32)
                if self.policy is not None:
                    action, q_values = self.policy.act(state)
                else:
                    action, q_values = 2, []
                alert_ts = time.time()
                self.log_event(
                    event="policy_decision",
                    dpid=dp.id,
                    in_port=int(in_port),
                    src_ip=src_ip,
                    old_mac=old_mac,
                    new_mac=src,
                    action=int(action),
                    action_name=["allow", "flag", "block"][int(action)],
                    q_values=q_values,
                    state=state.tolist(),
                    policy="runtime_dqn" if self.policy is not None else "rule",
                )
                if action in (1, 2):
                    self.log_event(
                        event="arp_spoof_alert",
                        dpid=dp.id,
                        in_port=int(in_port),
                        src_ip=src_ip,
                        old_mac=old_mac,
                        new_mac=src,
                        alert_timestamp=alert_ts,
                        action=int(action),
                    )
                if action == 2:
                    self.add_drop_for_port(dp, in_port)
                    return
                if action == 1:
                    return
            else:
                self.ip_to_macs.setdefault(src_ip, set()).add(src)
                state = np.array([
                    1.0 if int(arp_pkt.opcode) == 2 else 0.0,
                    1.0 if old_mac is not None else 0.0,
                    0.0,
                    min(1.0, len(self.ip_to_macs.get(src_ip, {src})) / 4.0),
                    min(1.0, float(in_port) / 10.0),
                    1.0 if (dp.id, in_port) in self.blocked_ports else 0.0,
                ], dtype=np.float32)
                if self.policy is not None:
                    action, q_values = self.policy.act(state)
                    self.log_event(
                        event="policy_decision",
                        dpid=dp.id,
                        in_port=int(in_port),
                        src_ip=src_ip,
                        src_mac=src,
                        action=int(action),
                        action_name=["allow", "flag", "block"][int(action)],
                        q_values=q_values,
                        state=state.tolist(),
                        policy="runtime_dqn",
                    )
                    if action == 2:
                        self.log_event(event="false_block_prevented", dpid=dp.id, in_port=int(in_port), src_ip=src_ip, src_mac=src)
                self.ip_to_mac[src_ip] = src

        out_port = self.mac_to_port[dp.id].get(dst, ofp.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]
        if out_port != ofp.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self.add_flow(dp, 10, match, actions, idle_timeout=30, hard_timeout=120)
        out = parser.OFPPacketOut(
            datapath=dp,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=msg.data,
        )
        dp.send_msg(out)


class RogueApRestController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.app = data[ROGUE_AP_INSTANCE]

    @route("rogue_ap", "/rogue-ap-alert", methods=["POST"])
    def rogue_ap_alert(self, req, **kwargs):
        try:
            payload = json.loads(req.body.decode("utf-8") if req.body else "{}")
        except json.JSONDecodeError:
            return Response(status=400, body=json.dumps({"ok": False, "error": "invalid json"}))
        result = self.app.handle_rogue_ap_alert(payload)
        return Response(content_type="application/json", body=json.dumps(result))
