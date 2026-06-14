#!/usr/bin/env python3
"""Replay held-out Rogue AP frames as an emulated digital-twin runtime stream.

This is the no-Wi-Fi-adapter path:
- preserve file order and row order
- aggregate contiguous streaming windows
- score with the exported Rogue AP runtime model
- optionally POST alerts to the Ryu /rogue-ap-alert endpoint
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, precision_recall_fscore_support, roc_auc_score


AUDIT_COLUMNS = {"source_file", "file_id"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("processed/rogue_ap_with_source_metadata/test"))
    p.add_argument("--model", type=Path, default=Path("models/rogue_ap_runtime/rogue_ap_runtime_rf.joblib"))
    p.add_argument("--out-dir", type=Path, default=Path("results/rogue_ap_digital_twin_replay"))
    p.add_argument("--window-frames", type=int, default=100)
    p.add_argument("--stride-frames", type=int, default=50)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--tune-data-dir", type=Path, default=None, help="Optional validation replay stream used only to choose threshold.")
    p.add_argument("--target-fpr", type=float, default=0.20)
    p.add_argument("--ma-windows", type=int, default=3)
    p.add_argument("--replay-speed", type=float, default=0.0, help="0 disables sleeping; otherwise simulated frames/sec.")
    p.add_argument("--alert-url", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-windows", type=int, default=0)
    return p.parse_args()


def load_frames(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    if "source_file" in df.columns:
        df["_source_order"] = df["source_file"].str.extract(r"RogueAP_(\d+)").astype(float).fillna(9999).astype(int)
        df["_row_order"] = np.arange(len(df))
        df = df.sort_values(["_source_order", "_row_order"]).drop(columns=["_source_order", "_row_order"]).reset_index(drop=True)
    return df


def any_cols(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    present = [c for c in columns if c in df.columns]
    if not present:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    return df[present].fillna(0).astype(float).max(axis=1)


def aggregate_window(window: pd.DataFrame, feature_columns: list[str]) -> tuple[dict, dict]:
    n = max(1, len(window))

    def mean_col(col: str) -> float:
        if col not in window.columns:
            return 0.0
        return float(window[col].fillna(0).astype(float).mean())

    type_series = window["num__wlan_fc_type"].fillna(-1).astype(float) if "num__wlan_fc_type" in window else pd.Series([], dtype=float)
    ds_cols = {
        "ratio_ds_0x00000000": "cat_wlan_fc_ds_0x00000000",
        "ratio_ds_0x00000001": "cat_wlan_fc_ds_0x00000001",
        "ratio_ds_0x00000002": "cat_wlan_fc_ds_0x00000002",
        "ratio_ds_0x00000003": "cat_wlan_fc_ds_0x00000003",
    }
    ip_present = any_cols(window, [c for c in window.columns if c.startswith("cat_ip_proto_") and not c.endswith("_MISSING")])
    tcp_present = any_cols(window, [c for c in window.columns if c.startswith("cat_tcp_") and not c.endswith("_MISSING")])
    llc_present = any_cols(window, [c for c in window.columns if c.startswith("cat_llc_") and not c.endswith("_MISSING")])

    raw_features = {
        "n_frames": float(n),
        "ratio_retry": mean_col("num__wlan_fc_retry"),
        "ratio_protected": mean_col("num__wlan_fc_protected"),
        "ratio_moredata": mean_col("num__wlan_fc_moredata"),
        "ratio_pwrmgt": mean_col("num__wlan_fc_pwrmgt"),
        "ratio_order": mean_col("num__wlan_fc_order"),
        "ratio_frag": mean_col("num__wlan_fc_frag"),
        "ratio_type_0": float((type_series == 0).mean()) if len(type_series) else 0.0,
        "ratio_type_1": float((type_series == 1).mean()) if len(type_series) else 0.0,
        "ratio_type_2": float((type_series == 2).mean()) if len(type_series) else 0.0,
        "ratio_llc_present": float(llc_present.mean()),
        "ratio_ip_present": float(ip_present.mean()),
        "ratio_tcp_present": float(tcp_present.mean()),
    }
    for out_col, in_col in ds_cols.items():
        raw_features[out_col] = mean_col(in_col)

    x = {col: float(raw_features.get(col, 0.0)) for col in feature_columns}
    labels = window["label"].astype(int) if "label" in window else pd.Series(np.zeros(n, dtype=int))
    audit = {
        "source_file": str(window["source_file"].mode().iloc[0]) if "source_file" in window and not window["source_file"].mode().empty else "",
        "frames": int(n),
        "positive_frames": int(labels.sum()),
        "positive_frame_ratio": float(labels.mean()),
        "window_label": int(labels.max()),
    }
    return x, audit


def post_alert(url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return {"ok": True, "status": resp.status}
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc)}


def metrics(y_true, scores, threshold):
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    p, r, _ = precision_recall_curve(y_true, scores)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / (fp + tn) if (fp + tn) else 0.0),
        "auroc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) == 2 else float("nan"),
        "pr_auc": float(auc(r, p)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_windows(frames: pd.DataFrame, features: list[str], window_frames: int, stride_frames: int, max_windows: int = 0) -> pd.DataFrame:
    rows = []
    window_id = 0
    for end in range(window_frames, len(frames) + 1, stride_frames):
        if max_windows and window_id >= max_windows:
            break
        window = frames.iloc[end - window_frames : end]
        x, audit = aggregate_window(window, features)
        window_id += 1
        rows.append(
            {
                "window_id": window_id,
                "row_start": int(end - window_frames),
                "row_end": int(end),
                **audit,
                **x,
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(y_true, scores, target_fpr: float):
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 501)))
    best = None
    for threshold in candidates:
        m = metrics(y_true, scores, threshold)
        if m["fpr"] <= target_fpr:
            if best is None or m["f1"] > best["f1"] or (m["f1"] == best["f1"] and m["recall"] > best["recall"]):
                best = {**m, "threshold": float(threshold)}
    if best is not None:
        return best
    threshold = float(candidates[-1])
    return {**metrics(y_true, scores, threshold), "threshold": threshold}


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.model)
    model = bundle["model"]
    features = list(bundle["feature_columns"])
    threshold = float(args.threshold if args.threshold is not None else bundle["threshold"])
    threshold_source = "provided_or_model"
    threshold_tuning_metrics = None
    if args.tune_data_dir is not None and args.threshold is None:
        tune_frames = load_frames(args.tune_data_dir)
        tune_windows = build_windows(tune_frames, features, args.window_frames, args.stride_frames, args.max_windows)
        tune_scores = model.predict_proba(tune_windows[features])[:, 1]
        if args.ma_windows > 1:
            tune_scores = pd.Series(tune_scores).rolling(args.ma_windows, min_periods=1).mean().to_numpy()
        threshold_tuning_metrics = choose_threshold(tune_windows["window_label"].astype(int).to_numpy(), tune_scores, args.target_fpr)
        threshold = float(threshold_tuning_metrics["threshold"])
        threshold_source = "validation_replay_stream"
    frames = load_frames(args.data_dir)
    window_df = build_windows(frames, features, args.window_frames, args.stride_frames, args.max_windows)
    raw_scores = model.predict_proba(window_df[features])[:, 1] if len(window_df) else np.asarray([])
    smoothed_scores = pd.Series(raw_scores).rolling(args.ma_windows, min_periods=1).mean().to_numpy() if len(raw_scores) else np.asarray([])
    rows = window_df.drop(columns=features).to_dict(orient="records") if len(window_df) else []
    alerts = []
    start = time.time()

    for idx, event in enumerate(rows):
        score = float(raw_scores[idx])
        smoothed_score = float(smoothed_scores[idx])
        action = "flag" if smoothed_score >= threshold else "allow"
        event.update({
            "event": "rogue_ap_digital_twin_decision",
            "timestamp": time.time(),
            "score": score,
            "smoothed_score": smoothed_score,
            "threshold": threshold,
            "action": action,
        })
        if action == "flag":
            alert = {**event, "event": "rogue_ap_alert", "recommended_action": "quarantine_or_investigate"}
            if args.alert_url and not args.dry_run:
                alert["post_result"] = post_alert(args.alert_url, alert)
            alerts.append(alert)
        if args.replay_speed > 0:
            time.sleep(args.stride_frames / args.replay_speed)

    y_true = np.asarray([r["window_label"] for r in rows], dtype=int)
    scores = np.asarray([r["smoothed_score"] for r in rows], dtype=float)
    result = metrics(y_true, scores, threshold) if len(rows) else {}
    elapsed = time.time() - start
    summary = {
        "mode": "digital_twin_replay",
        "data_dir": str(args.data_dir),
        "model": str(args.model),
        "window_frames": args.window_frames,
        "stride_frames": args.stride_frames,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_tuning_metrics": threshold_tuning_metrics,
        "ma_windows": args.ma_windows,
        "windows": len(rows),
        "alerts": len(alerts),
        "elapsed_sec": elapsed,
        "windows_per_sec": len(rows) / elapsed if elapsed > 0 else None,
        "metrics": result,
        "note": "Replay-based streaming runtime; no live RF capture or monitor-mode adapter used.",
    }

    pd.DataFrame(rows).to_csv(args.out_dir / "rogue_ap_replay_windows.csv", index=False)
    pd.DataFrame(alerts).to_csv(args.out_dir / "rogue_ap_replay_alerts.csv", index=False)
    per_file = []
    if rows:
        df = pd.DataFrame(rows)
        for source_file, group in df.groupby("source_file"):
            m = metrics(group["window_label"].astype(int).to_numpy(), group["smoothed_score"].to_numpy(), threshold)
            m.update({"source_file": source_file, "windows": int(len(group)), "positive_windows": int(group["window_label"].sum())})
            per_file.append(m)
    pd.DataFrame(per_file).to_csv(args.out_dir / "rogue_ap_replay_per_file_metrics.csv", index=False)
    (args.out_dir / "rogue_ap_replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text = [
        "Rogue AP Digital Twin Replay Runtime",
        "",
        f"Windows: {summary['windows']}",
        f"Alerts sent/raised: {summary['alerts']}",
        f"Throughput: {summary['windows_per_sec']:.2f} windows/sec" if summary["windows_per_sec"] is not None else "Throughput: n/a",
        f"Precision: {result.get('precision', float('nan')):.4f}",
        f"Recall: {result.get('recall', float('nan')):.4f}",
        f"F1: {result.get('f1', float('nan')):.4f}",
        f"FPR: {result.get('fpr', float('nan')):.4f}",
        f"AUROC: {result.get('auroc', float('nan')):.4f}",
        f"PR-AUC: {result.get('pr_auc', float('nan')):.4f}",
        "",
        "This is replay-based streaming runtime, not live monitor-mode RF capture.",
    ]
    (args.out_dir / "rogue_ap_replay_summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
