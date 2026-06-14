#!/usr/bin/env python3
"""Run a live Rogue AP beacon-injection test against the monitor-mode sensor."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from scapy.all import Dot11, Dot11Beacon, Dot11Elt, RadioTap, sendp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--interface", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results/rogue_ap_runtime_live_attack_test"))
    p.add_argument("--sensor", type=Path, default=Path("sdn_runtime/rogue_ap_live_sensor.py"))
    p.add_argument("--model", type=Path, default=Path("models/rogue_ap_runtime/rogue_ap_runtime_rf.joblib"))
    p.add_argument("--ssid", default="CODEX_ROGUE_AP")
    p.add_argument("--bssid", default="02:11:22:33:44:55")
    p.add_argument("--channel", type=int, default=1)
    p.add_argument("--beacon-count", type=int, default=240)
    p.add_argument("--beacon-interval", type=float, default=0.03)
    p.add_argument(
        "--authorized-bssid",
        action="append",
        default=[],
        help="Authorized BSSID for runtime policy. Can be repeated or comma-separated.",
    )
    p.add_argument("--flag-unknown-bssid", action="store_true")
    p.add_argument("--warmup-seconds", type=float, default=2.0)
    p.add_argument("--window-frames", type=int, default=20)
    p.add_argument("--stride-frames", type=int, default=10)
    p.add_argument("--max-windows", type=int, default=18)
    return p.parse_args()


def run_checked(cmd: list[str]):
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")
    return proc


def setup_monitor(interface: str, channel: int):
    commands = [
        ["ip", "link", "set", interface, "down"],
        ["iw", "dev", interface, "set", "type", "monitor"],
        ["ip", "link", "set", interface, "up"],
        ["iw", "dev", interface, "set", "channel", str(channel)],
    ]
    results = []
    for cmd in commands:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        results.append({"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")
    return results


def make_beacon(ssid: str, bssid: str, channel: int):
    return (
        RadioTap()
        / Dot11(type=0, subtype=8, addr1="ff:ff:ff:ff:ff:ff", addr2=bssid, addr3=bssid)
        / Dot11Beacon(cap="ESS+privacy")
        / Dot11Elt(ID="SSID", info=ssid.encode("utf-8"))
        / Dot11Elt(ID="Rates", info=b"\x82\x84\x8b\x96\x12\x24\x48\x6c")
        / Dot11Elt(ID="DSset", info=bytes([channel]))
    )


def inject_beacons(interface: str, ssid: str, bssid: str, channel: int, count: int, interval: float):
    frame = make_beacon(ssid, bssid, channel)
    sendp(frame, iface=interface, count=count, inter=interval, verbose=False)


def read_windows(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_results = setup_monitor(args.interface, args.channel)

    sensor_cmd = [
        sys.executable,
        str(args.sensor),
        "--interface",
        args.interface,
        "--model",
        str(args.model),
        "--window-frames",
        str(args.window_frames),
        "--stride-frames",
        str(args.stride_frames),
        "--max-windows",
        str(args.max_windows),
        "--dry-run",
        "--out-dir",
        str(args.out_dir),
    ]
    for bssid in args.authorized_bssid:
        sensor_cmd.extend(["--authorized-bssid", bssid])
    if args.flag_unknown_bssid:
        sensor_cmd.append("--flag-unknown-bssid")
    sensor_stdout = args.out_dir / "sensor_stdout.jsonl"
    sensor_stderr = args.out_dir / "sensor_stderr.txt"
    with sensor_stdout.open("w", encoding="utf-8") as out, sensor_stderr.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(sensor_cmd, stdout=out, stderr=err, text=True)
        time.sleep(args.warmup_seconds)
        inject_started = time.time()
        inject_beacons(args.interface, args.ssid, args.bssid, args.channel, args.beacon_count, args.beacon_interval)
        inject_finished = time.time()
        try:
            proc.wait(timeout=max(15, args.beacon_count * args.beacon_interval + 20))
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    windows = read_windows(args.out_dir / "rogue_ap_runtime_windows.csv")
    attack_bssid = args.bssid.lower()
    attack_windows = 0
    flagged_attack_windows = 0
    if not windows.empty and "observed_bssids" in windows.columns:
        observed = windows["observed_bssids"].fillna("").str.lower().str.contains(attack_bssid, regex=False)
        attack_windows = int(observed.sum())
        flagged_attack_windows = int(((windows["action"] == "flag") & observed).sum()) if "action" in windows else 0

    summary = {
        "test_type": "live_monitor_mode_rogue_ap_beacon_injection",
        "interface": args.interface,
        "ssid": args.ssid,
        "bssid": args.bssid,
        "channel": args.channel,
        "beacon_count": args.beacon_count,
        "beacon_interval": args.beacon_interval,
        "inject_started": inject_started,
        "inject_finished": inject_finished,
        "sensor_returncode": proc.returncode,
        "windows": int(len(windows)),
        "attack_bssid_windows": attack_windows,
        "flagged_attack_bssid_windows": flagged_attack_windows,
        "alerts": int((windows["action"] == "flag").sum()) if not windows.empty and "action" in windows else 0,
        "model_alerts": int(windows["model_flag"].sum()) if not windows.empty and "model_flag" in windows else 0,
        "policy_alerts": int(windows["policy_flag"].sum()) if not windows.empty and "policy_flag" in windows else 0,
        "authorized_bssid": args.authorized_bssid,
        "flag_unknown_bssid": args.flag_unknown_bssid,
        "max_score": float(windows["score"].max()) if not windows.empty and "score" in windows else None,
        "mean_score": float(windows["score"].mean()) if not windows.empty and "score" in windows else None,
        "setup_results": setup_results,
        "interpretation": (
            "attack_frames_observed_by_sensor"
            if attack_windows
            else "attack_frames_not_observed_by_sensor"
        ),
    }
    (args.out_dir / "rogue_ap_live_attack_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out_dir / "rogue_ap_live_attack_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
