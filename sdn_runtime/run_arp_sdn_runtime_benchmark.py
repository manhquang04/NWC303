#!/usr/bin/env python3
"""Run repeated ARP SDN runtime episodes and aggregate detector metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    attack_enabled: bool
    attack_start: float
    attack_rate: float
    background_mode: str
    attack_variant: str = "standard"
    benign_variant: str = "standard"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attack-episodes", type=int, default=3)
    p.add_argument("--normal-episodes", type=int, default=3)
    p.add_argument("--duration", type=float, default=14.0)
    p.add_argument("--attack-start", type=float, default=5.0)
    p.add_argument("--attack-rate", type=float, default=0.05)
    p.add_argument("--background-mode", choices=["quiet", "light", "mixed"], default="light")
    p.add_argument("--scenario-grid", action="store_true", help="Vary attack rate/start and benign background traffic across episodes.")
    p.add_argument("--stress-grid", action="store_true", help="Vary attack and benign ARP behavior for robustness stress testing.")
    p.add_argument(
        "--hardcore-grid",
        action="store_true",
        help="Use adversarial hard cases: evasive ARP attacks, benign MAC-conflict churn, and bursty background traffic.",
    )
    p.add_argument("--controller-cmd", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results/sdn_runtime_arp_dqn_benchmark"))
    return p.parse_args()


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def mean(values):
    vals = [v for v in values if v is not None and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else None


def stdev(values):
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(vals) < 2:
        return 0.0 if vals else None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def cleanup_mininet():
    if shutil.which("mn"):
        subprocess.run(["mn", "-c"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def build_scenarios(args):
    if args.hardcore_grid:
        attack_rates = [0.01, 0.02, 0.05, 0.10, 0.20, 1.00]
        attack_starts = [1.5, 2.0, 3.0, 5.0, 7.0, 9.0]
        background_modes = ["quiet", "light", "mixed", "burst"]
        attack_variants = [
            "standard",
            "slow",
            "intermittent",
            "random_mac",
            "request_poison",
            "unicast_reply",
            "burst_then_sleep",
        ]
        benign_variants = [
            "standard",
            "gratuitous",
            "dhcp_like",
            "arp_scan",
            "noisy_gratuitous",
            "gateway_failover",
            "ip_reassignment",
            "arp_storm",
        ]
        scenarios = []
        for i in range(args.attack_episodes):
            variant = attack_variants[i % len(attack_variants)]
            rate = attack_rates[i % len(attack_rates)]
            if variant == "slow":
                rate = max(rate, 1.0)
            if variant == "burst_then_sleep":
                rate = max(rate, 2.0)
            scenarios.append(
                Scenario(
                    True,
                    attack_starts[i % len(attack_starts)],
                    rate,
                    background_modes[i % len(background_modes)],
                    variant,
                    "standard",
                )
            )
        for i in range(args.normal_episodes):
            scenarios.append(
                Scenario(
                    False,
                    attack_starts[i % len(attack_starts)],
                    attack_rates[i % len(attack_rates)],
                    background_modes[i % len(background_modes)],
                    "standard",
                    benign_variants[i % len(benign_variants)],
                )
            )
        return scenarios
    if args.stress_grid:
        attack_rates = [0.02, 0.05, 0.10, 0.20, 1.00]
        attack_starts = [2.0, 3.0, 5.0, 7.0]
        background_modes = ["quiet", "light", "mixed"]
        attack_variants = ["standard", "slow", "intermittent", "random_mac"]
        benign_variants = ["standard", "gratuitous", "dhcp_like", "arp_scan", "noisy_gratuitous"]
        scenarios = []
        for i in range(args.attack_episodes):
            variant = attack_variants[i % len(attack_variants)]
            rate = attack_rates[i % len(attack_rates)]
            if variant == "slow":
                rate = max(rate, 1.0)
            scenarios.append(
                Scenario(
                    True,
                    attack_starts[i % len(attack_starts)],
                    rate,
                    background_modes[i % len(background_modes)],
                    variant,
                    "standard",
                )
            )
        for i in range(args.normal_episodes):
            scenarios.append(
                Scenario(
                    False,
                    attack_starts[i % len(attack_starts)],
                    attack_rates[i % len(attack_rates)],
                    background_modes[i % len(background_modes)],
                    "standard",
                    benign_variants[i % len(benign_variants)],
                )
            )
        return scenarios
    if not args.scenario_grid:
        return (
            [Scenario(True, args.attack_start, args.attack_rate, args.background_mode) for _ in range(args.attack_episodes)]
            + [Scenario(False, args.attack_start, args.attack_rate, args.background_mode) for _ in range(args.normal_episodes)]
        )
    attack_rates = [0.02, 0.05, 0.10, 0.20]
    attack_starts = [3.0, 5.0, 7.0]
    background_modes = ["quiet", "light", "mixed"]
    scenarios = []
    for i in range(args.attack_episodes):
        scenarios.append(
            Scenario(
                True,
                attack_starts[i % len(attack_starts)],
                attack_rates[i % len(attack_rates)],
                background_modes[i % len(background_modes)],
            )
        )
    for i in range(args.normal_episodes):
        scenarios.append(
            Scenario(
                False,
                attack_starts[i % len(attack_starts)],
                attack_rates[i % len(attack_rates)],
                background_modes[i % len(background_modes)],
            )
        )
    return scenarios


def run_episode(args, episode_id: int, scenario: Scenario):
    variant = scenario.attack_variant if scenario.attack_enabled else scenario.benign_variant
    name = f"episode_{episode_id:03d}_{'attack' if scenario.attack_enabled else 'normal'}_{variant}_{scenario.background_mode}_r{scenario.attack_rate:g}_s{scenario.attack_start:g}"
    out_dir = args.out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    event_log = Path(f"/tmp/arp_guard_events_{episode_id:02d}.jsonl")
    cleanup_mininet()
    cmd = [
        sys.executable,
        "sdn_runtime/run_arp_sdn_runtime.py",
        "--duration",
        str(args.duration),
        "--attack-start",
        str(scenario.attack_start),
        "--attack-rate",
        str(scenario.attack_rate),
        "--background-mode",
        scenario.background_mode,
        "--attack-variant",
        scenario.attack_variant,
        "--benign-variant",
        scenario.benign_variant,
        "--controller-cmd",
        args.controller_cmd,
        "--event-log",
        str(event_log),
        "--out-dir",
        str(out_dir),
    ]
    if not scenario.attack_enabled:
        cmd.append("--no-attack")
    print(f"running {name}: {' '.join(cmd)}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    runtime_sec = time.time() - started
    (out_dir / "episode_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "episode_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    summary_path = out_dir / "runtime_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "attack_enabled": scenario.attack_enabled,
            "attack_start": scenario.attack_start,
            "attack_rate": scenario.attack_rate,
            "background_mode": scenario.background_mode,
            "detected": False,
            "mitigated": False,
            "alerts": 0,
            "drop_rules": 0,
            "arp_observed_events": 0,
            "detection_latency_sec": None,
            "mitigation_latency_sec": None,
            "false_alerts_in_benign_episode": 0,
            "runner_failed": True,
        }
    summary.update(
        {
            "episode": name,
            "episode_id": episode_id,
            "scenario_attack_start": scenario.attack_start,
            "scenario_attack_rate": scenario.attack_rate,
            "scenario_background_mode": scenario.background_mode,
            "scenario_attack_variant": scenario.attack_variant,
            "scenario_benign_variant": scenario.benign_variant,
            "return_code": proc.returncode,
            "runtime_sec": runtime_sec,
            "out_dir": str(out_dir),
        }
    )
    return summary


def action_counts(events_csv: Path):
    counts = {"allow": 0, "flag": 0, "block": 0}
    if not events_csv.exists():
        return counts
    with events_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("event") == "policy_decision":
                action = row.get("action_name")
                if action in counts:
                    counts[action] += 1
    return counts


def aggregate(rows, out_dir: Path):
    for row in rows:
        counts = action_counts(Path(row["out_dir"]) / "runtime_events.csv")
        row.update({f"policy_action_{k}": v for k, v in counts.items()})

    tp = sum(1 for r in rows if r["attack_enabled"] and r.get("detected"))
    fn = sum(1 for r in rows if r["attack_enabled"] and not r.get("detected"))
    fp = sum(1 for r in rows if not r["attack_enabled"] and r.get("alerts", 0) > 0)
    tn = sum(1 for r in rows if not r["attack_enabled"] and r.get("alerts", 0) == 0)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    fpr = safe_div(fp, fp + tn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    attack_rows = [r for r in rows if r["attack_enabled"]]
    detected_attack_rows = [r for r in attack_rows if r.get("detected")]
    summary = {
        "episodes": len(rows),
        "attack_episodes": len(attack_rows),
        "normal_episodes": len(rows) - len(attack_rows),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "attack_detection_rate": safe_div(tp, len(attack_rows)),
        "normal_false_alarm_rate": fpr,
        "mitigation_rate_on_attack": safe_div(sum(1 for r in attack_rows if r.get("mitigated")), len(attack_rows)),
        "mean_detection_latency_sec": mean([r.get("detection_latency_sec") for r in detected_attack_rows]),
        "std_detection_latency_sec": stdev([r.get("detection_latency_sec") for r in detected_attack_rows]),
        "mean_mitigation_latency_sec": mean([r.get("mitigation_latency_sec") for r in detected_attack_rows]),
        "std_mitigation_latency_sec": stdev([r.get("mitigation_latency_sec") for r in detected_attack_rows]),
        "total_alerts": sum(int(r.get("alerts", 0)) for r in rows),
        "total_drop_rules": sum(int(r.get("drop_rules", 0)) for r in rows),
        "total_policy_allow": sum(int(r.get("policy_action_allow", 0)) for r in rows),
        "total_policy_flag": sum(int(r.get("policy_action_flag", 0)) for r in rows),
        "total_policy_block": sum(int(r.get("policy_action_block", 0)) for r in rows),
    }
    scenario_rows = []
    for bg in sorted({r.get("scenario_background_mode", "") for r in rows}):
        subset = [r for r in rows if r.get("scenario_background_mode", "") == bg]
        if not subset:
            continue
        bg_tp = sum(1 for r in subset if r["attack_enabled"] and r.get("detected"))
        bg_fn = sum(1 for r in subset if r["attack_enabled"] and not r.get("detected"))
        bg_fp = sum(1 for r in subset if not r["attack_enabled"] and r.get("alerts", 0) > 0)
        bg_tn = sum(1 for r in subset if not r["attack_enabled"] and r.get("alerts", 0) == 0)
        bg_precision = safe_div(bg_tp, bg_tp + bg_fp)
        bg_recall = safe_div(bg_tp, bg_tp + bg_fn)
        bg_fpr = safe_div(bg_fp, bg_fp + bg_tn)
        bg_f1 = safe_div(2 * bg_precision * bg_recall, bg_precision + bg_recall)
        scenario_rows.append(
            {
                "scenario_background_mode": bg,
                "episodes": len(subset),
                "tp": bg_tp,
                "fn": bg_fn,
                "fp": bg_fp,
                "tn": bg_tn,
                "precision": bg_precision,
                "recall": bg_recall,
                "f1": bg_f1,
                "fpr": bg_fpr,
                "mean_detection_latency_sec": mean([r.get("detection_latency_sec") for r in subset if r["attack_enabled"] and r.get("detected")]),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "benchmark_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        keys = sorted({k for row in rows for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "scenario_breakdown.csv").open("w", newline="", encoding="utf-8") as f:
        keys = [
            "scenario_background_mode",
            "episodes",
            "tp",
            "fn",
            "fp",
            "tn",
            "precision",
            "recall",
            "f1",
            "fpr",
            "mean_detection_latency_sec",
        ]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(scenario_rows)
    variant_rows = []
    for variant_key in ["scenario_attack_variant", "scenario_benign_variant"]:
        for variant in sorted({r.get(variant_key, "") for r in rows}):
            subset = [r for r in rows if r.get(variant_key, "") == variant]
            if not subset:
                continue
            v_tp = sum(1 for r in subset if r["attack_enabled"] and r.get("detected"))
            v_fn = sum(1 for r in subset if r["attack_enabled"] and not r.get("detected"))
            v_fp = sum(1 for r in subset if not r["attack_enabled"] and r.get("alerts", 0) > 0)
            v_tn = sum(1 for r in subset if not r["attack_enabled"] and r.get("alerts", 0) == 0)
            v_precision = safe_div(v_tp, v_tp + v_fp)
            v_recall = safe_div(v_tp, v_tp + v_fn)
            v_fpr = safe_div(v_fp, v_fp + v_tn)
            v_f1 = safe_div(2 * v_precision * v_recall, v_precision + v_recall)
            variant_rows.append(
                {
                    "variant_group": variant_key,
                    "variant": variant,
                    "episodes": len(subset),
                    "tp": v_tp,
                    "fn": v_fn,
                    "fp": v_fp,
                    "tn": v_tn,
                    "precision": v_precision,
                    "recall": v_recall,
                    "f1": v_f1,
                    "fpr": v_fpr,
                    "mean_detection_latency_sec": mean([r.get("detection_latency_sec") for r in subset if r["attack_enabled"] and r.get("detected")]),
                }
            )
    with (out_dir / "variant_breakdown.csv").open("w", newline="", encoding="utf-8") as f:
        keys = ["variant_group", "variant", "episodes", "tp", "fn", "fp", "tn", "precision", "recall", "f1", "fpr", "mean_detection_latency_sec"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(variant_rows)
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text = [
        "ARP SDN Runtime DQN Policy Benchmark",
        "",
        f"Episodes: {summary['episodes']} ({summary['attack_episodes']} attack, {summary['normal_episodes']} normal)",
        f"Confusion matrix episodes: TP={tp}, FN={fn}, FP={fp}, TN={tn}",
        f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, FPR={fpr:.4f}",
        f"Mitigation rate on attack={summary['mitigation_rate_on_attack']:.4f}",
        f"Mean detection latency={summary['mean_detection_latency_sec']} sec",
        f"Mean mitigation latency={summary['mean_mitigation_latency_sec']} sec",
        f"Policy actions: allow={summary['total_policy_allow']}, flag={summary['total_policy_flag']}, block={summary['total_policy_block']}",
        "",
        "Scenario breakdown by background mode:",
        *[
            f"- {r['scenario_background_mode']}: episodes={r['episodes']}, TP/FN/FP/TN={r['tp']}/{r['fn']}/{r['fp']}/{r['tn']}, F1={r['f1']:.4f}, FPR={r['fpr']:.4f}"
            for r in scenario_rows
        ],
        "",
        "Scenario breakdown by attack/benign variant:",
        *[
            f"- {r['variant_group']}={r['variant']}: episodes={r['episodes']}, TP/FN/FP/TN={r['tp']}/{r['fn']}/{r['fp']}/{r['tn']}, F1={r['f1']:.4f}, FPR={r['fpr']:.4f}"
            for r in variant_rows
        ],
        "",
        "Note: the runtime DQN policy is used online in the Ryu controller; source labels are not used at runtime.",
    ]
    (out_dir / "benchmark_summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    return summary


def main():
    args = parse_args()
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for episode_id, scenario in enumerate(build_scenarios(args), start=1):
        rows.append(run_episode(args, episode_id, scenario))
    summary = aggregate(rows, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
