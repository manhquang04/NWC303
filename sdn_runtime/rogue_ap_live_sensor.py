#!/usr/bin/env python3
"""Live Rogue AP monitor-mode sensor.

This process reads 802.11 frames from a Wi-Fi adapter in monitor mode using
tshark, aggregates contiguous windows, scores them with an exported model, and
optionally sends Rogue AP alerts to an SDN/Ryu endpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


TSHARK_FIELDS = [
    "frame.time_epoch",
    "wlan.sa",
    "wlan.da",
    "wlan.bssid",
    "wlan.fc.retry",
    "wlan.fc.protected",
    "wlan.fc.moredata",
    "wlan.fc.pwrmgt",
    "wlan.fc.order",
    "wlan.fc.frag",
    "wlan.fc.ds",
    "wlan.fc.type",
    "llc",
    "ip",
    "tcp",
    "wlan.ssid",
    "radiotap.dbm_antsignal",
    "wlan_radio.channel",
    "wlan_radio.data_rate",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--interface", required=True, help="Monitor-mode interface, for example wlan0mon.")
    p.add_argument("--model", type=Path, default=Path("models/rogue_ap_runtime/rogue_ap_runtime_rf.joblib"))
    p.add_argument("--window-frames", type=int, default=100)
    p.add_argument("--stride-frames", type=int, default=50)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--ma-windows", type=int, default=3)
    p.add_argument(
        "--authorized-bssid",
        action="append",
        default=[],
        help="Authorized BSSID. Can be repeated or comma-separated. Used only by the runtime policy layer.",
    )
    p.add_argument("--flag-unknown-bssid", action="store_true", help="Flag windows containing a BSSID outside the authorized list.")
    p.add_argument("--alert-url", default=None, help="Optional POST endpoint, for example http://127.0.0.1:8080/rogue-ap-alert")
    p.add_argument("--out-dir", type=Path, default=Path("results/rogue_ap_runtime_live"))
    p.add_argument("--max-windows", type=int, default=0, help="Stop after N windows; 0 means run until interrupted.")
    p.add_argument("--max-seconds", type=float, default=0.0, help="Stop after N seconds; 0 means run until interrupted or max-windows.")
    p.add_argument("--tshark-monitor-flag", action="store_true", help="Pass -I to tshark. Leave disabled when the interface is already monitor mode.")
    p.add_argument("--setup-monitor", action="store_true", help="Set the interface to monitor mode before starting tshark.")
    p.add_argument("--dry-run", action="store_true", help="Do not send HTTP alerts.")
    return p.parse_args()


def truthy(value: str) -> bool:
    if value is None:
        return False
    v = str(value).strip().lower()
    return v not in ("", "0", "false", "none", "nan")


def first_value(value: str) -> str:
    if value is None:
        return ""
    return str(value).split(",")[0].strip()


def normalize_mac(value: str) -> str:
    return str(value or "").strip().lower()


def parse_authorized_bssids(values: list[str]) -> set[str]:
    result = set()
    for value in values:
        for item in str(value).split(","):
            mac = normalize_mac(item)
            if mac:
                result.add(mac)
    return result


def parse_tshark_line(line: str):
    parts = line.rstrip("\n").split("\t")
    parts += [""] * (len(TSHARK_FIELDS) - len(parts))
    return dict(zip(TSHARK_FIELDS, parts[: len(TSHARK_FIELDS)]))


def run_checked(cmd: list[str]):
    return subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def setup_monitor(interface: str):
    commands = [
        ["ip", "link", "set", interface, "down"],
        ["iw", "dev", interface, "set", "type", "monitor"],
        ["ip", "link", "set", interface, "up"],
    ]
    results = []
    for cmd in commands:
        proc = run_checked(cmd)
        results.append({"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to run {' '.join(cmd)}: {proc.stderr.strip()}")
    return results


def start_tshark(interface: str, monitor_flag: bool = False):
    cmd = ["tshark", "-l"]
    if monitor_flag:
        cmd.append("-I")
    cmd.extend(["-i", interface, "-T", "fields"])
    for field in TSHARK_FIELDS:
        cmd.extend(["-e", field])
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)


def aggregate_window(frames: list[dict], feature_columns: list[str]):
    n = max(1, len(frames))

    def ratio_bool(field):
        return sum(1 for row in frames if truthy(row.get(field, ""))) / n

    ds_counts = {k: 0 for k in ["0x00000000", "0x00000001", "0x00000002", "0x00000003"]}
    type_counts = {k: 0 for k in ["0", "1", "2"]}
    bssid_counts = {}
    ssid_counts = {}
    signal_values = []
    data_rates = []
    channels = set()
    for row in frames:
        ds = first_value(row.get("wlan.fc.ds", ""))
        if ds in ds_counts:
            ds_counts[ds] += 1
        typ = first_value(row.get("wlan.fc.type", ""))
        if typ in type_counts:
            type_counts[typ] += 1
        bssid = first_value(row.get("wlan.bssid", ""))
        ssid = first_value(row.get("wlan.ssid", ""))
        if bssid:
            bssid_counts[bssid] = bssid_counts.get(bssid, 0) + 1
        if ssid:
            ssid_counts[ssid] = ssid_counts.get(ssid, 0) + 1
        try:
            signal_values.append(float(first_value(row.get("radiotap.dbm_antsignal", ""))))
        except ValueError:
            pass
        try:
            data_rates.append(float(first_value(row.get("wlan_radio.data_rate", ""))))
        except ValueError:
            pass
        channel = first_value(row.get("wlan_radio.channel", ""))
        if channel:
            channels.add(channel)

    feature_values = {
        "n_frames": float(n),
        "ratio_retry": ratio_bool("wlan.fc.retry"),
        "ratio_protected": ratio_bool("wlan.fc.protected"),
        "ratio_moredata": ratio_bool("wlan.fc.moredata"),
        "ratio_pwrmgt": ratio_bool("wlan.fc.pwrmgt"),
        "ratio_order": ratio_bool("wlan.fc.order"),
        "ratio_frag": ratio_bool("wlan.fc.frag"),
        "ratio_ds_0x00000000": ds_counts["0x00000000"] / n,
        "ratio_ds_0x00000001": ds_counts["0x00000001"] / n,
        "ratio_ds_0x00000002": ds_counts["0x00000002"] / n,
        "ratio_ds_0x00000003": ds_counts["0x00000003"] / n,
        "ratio_type_0": type_counts["0"] / n,
        "ratio_type_1": type_counts["1"] / n,
        "ratio_type_2": type_counts["2"] / n,
        "ratio_llc_present": ratio_bool("llc"),
        "ratio_ip_present": ratio_bool("ip"),
        "ratio_tcp_present": ratio_bool("tcp"),
    }
    x = {col: float(feature_values.get(col, 0.0)) for col in feature_columns}
    audit = {
        "frames": n,
        "top_bssid": max(bssid_counts, key=bssid_counts.get) if bssid_counts else "",
        "top_ssid": max(ssid_counts, key=ssid_counts.get) if ssid_counts else "",
        "observed_bssids": ";".join(sorted(bssid_counts)),
        "observed_ssids": ";".join(sorted(ssid_counts)),
        "unique_bssid": len(bssid_counts),
        "unique_ssid": len(ssid_counts),
        "mean_signal_dbm": float(np.mean(signal_values)) if signal_values else None,
        "mean_data_rate": float(np.mean(data_rates)) if data_rates else None,
        "unique_channels": len(channels),
    }
    return x, audit


def unknown_bssids_from_audit(audit: dict, authorized_bssids: set[str]) -> list[str]:
    if not authorized_bssids:
        return []
    ignored = {"", "ff:ff:ff:ff:ff:ff", "00:00:00:00:00:00"}
    observed = {normalize_mac(v) for v in str(audit.get("observed_bssids", "")).split(";")}
    return sorted(v for v in observed if v not in ignored and v not in authorized_bssids)


def post_alert(url: str, payload: dict, timeout: float = 2.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"ok": True, "status": resp.status, "body": resp.read().decode("utf-8", errors="replace")}
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc)}


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_results = []
    if args.setup_monitor:
        setup_results = setup_monitor(args.interface)
    bundle = joblib.load(args.model)
    model = bundle["model"]
    features = list(bundle["feature_columns"])
    threshold = float(args.threshold if args.threshold is not None else bundle["threshold"])
    authorized_bssids = parse_authorized_bssids(args.authorized_bssid)
    frame_buffer = deque(maxlen=args.window_frames)
    score_buffer = deque(maxlen=max(1, args.ma_windows))
    rows = []
    alerts = []
    proc = start_tshark(args.interface, monitor_flag=args.tshark_monitor_flag)
    window_id = 0
    frame_count = 0
    started_at = time.time()
    stderr_tail = deque(maxlen=80)
    try:
        if proc.stdout is None:
            raise RuntimeError("tshark stdout is not available")
        for line in proc.stdout:
            if not line.strip():
                continue
            frame_buffer.append(parse_tshark_line(line))
            frame_count += 1
            if len(frame_buffer) < args.window_frames:
                continue
            if (frame_count - args.window_frames) % max(1, args.stride_frames) != 0:
                continue
            window_id += 1
            x, audit = aggregate_window(list(frame_buffer), features)
            score = float(model.predict_proba(pd.DataFrame([x], columns=features))[:, 1][0])
            score_buffer.append(score)
            smoothed_score = float(np.mean(score_buffer))
            model_flag = smoothed_score >= threshold
            unknown_bssids = unknown_bssids_from_audit(audit, authorized_bssids) if args.flag_unknown_bssid else []
            policy_flag = bool(unknown_bssids)
            action = "flag" if (model_flag or policy_flag) else "allow"
            decision_reasons = []
            if model_flag:
                decision_reasons.append("model_score")
            if policy_flag:
                decision_reasons.append("unknown_bssid")
            payload = {
                "event": "rogue_ap_runtime_decision",
                "timestamp": time.time(),
                "window_id": window_id,
                "score": score,
                "smoothed_score": smoothed_score,
                "threshold": threshold,
                "model_flag": model_flag,
                "policy_flag": policy_flag,
                "decision_reasons": ";".join(decision_reasons),
                "unknown_bssids": ";".join(unknown_bssids),
                "action": action,
                **audit,
            }
            rows.append(payload)
            if action == "flag":
                alert_payload = {**payload, "event": "rogue_ap_alert", "recommended_action": "quarantine_or_investigate"}
                if args.alert_url and not args.dry_run:
                    alert_payload["post_result"] = post_alert(args.alert_url, alert_payload)
                alerts.append(alert_payload)
                print(json.dumps(alert_payload), flush=True)
            else:
                print(json.dumps(payload), flush=True)
            if args.max_windows and window_id >= args.max_windows:
                break
            if args.max_seconds and (time.time() - started_at) >= args.max_seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        if proc.stderr is not None:
            for err_line in proc.stderr.readlines():
                if err_line.strip():
                    stderr_tail.append(err_line.rstrip("\n"))

    with (args.out_dir / "rogue_ap_runtime_windows.csv").open("w", newline="", encoding="utf-8") as f:
        keys = sorted({k for row in rows for k in row.keys()}) if rows else ["window_id"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "rogue_ap_runtime_alerts.jsonl").write_text(
        "".join(json.dumps(a, sort_keys=True) + "\n" for a in alerts),
        encoding="utf-8",
    )
    summary = {
        "interface": args.interface,
        "model": str(args.model),
        "window_frames": args.window_frames,
        "stride_frames": args.stride_frames,
        "threshold": threshold,
        "windows": len(rows),
        "alerts": len(alerts),
        "frames_seen": frame_count,
        "elapsed_seconds": time.time() - started_at,
        "setup_monitor": args.setup_monitor,
        "setup_results": setup_results,
        "tshark_monitor_flag": args.tshark_monitor_flag,
        "tshark_stderr_tail": list(stderr_tail),
        "authorized_bssids": sorted(authorized_bssids),
        "flag_unknown_bssid": args.flag_unknown_bssid,
        "uid": os.getuid() if hasattr(os, "getuid") else None,
        "dry_run": args.dry_run,
        "alert_url": args.alert_url,
    }
    (args.out_dir / "rogue_ap_runtime_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
