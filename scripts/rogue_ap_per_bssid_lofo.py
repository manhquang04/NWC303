#!/usr/bin/env python3
"""Build per-BSSID/transmitter Rogue AP aggregates and run leave-one-file-out evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve
from xgboost import XGBClassifier


USECOLS = [
    "frame.len", "frame.number", "frame.time_relative",
    "radiotap.channel.freq", "radiotap.datarate", "radiotap.dbm_antsignal",
    "wlan.bssid", "wlan.da", "wlan.fc.ds", "wlan.fc.frag", "wlan.fc.order",
    "wlan.fc.moredata", "wlan.fc.protected", "wlan.fc.pwrmgt", "wlan.fc.type",
    "wlan.fc.retry", "wlan.fc.subtype", "wlan.fixed.beacon",
    "wlan.fixed.capabilities.ess", "wlan.fixed.capabilities.ibss",
    "wlan.fixed.reason_code", "wlan.ra", "wlan.duration", "wlan.sa", "wlan.seq",
    "wlan.ssid", "wlan.ta", "wlan.tag.length", "wlan_radio.channel",
    "wlan_radio.data_rate", "wlan_radio.frequency", "wlan_radio.signal_dbm",
    "wlan_radio.phy", "llc", "arp", "ip.proto", "tcp.flags.syn", "tcp.flags.ack",
    "udp.length", "Label",
]
AUDIT_COLS = {"source_file", "file_id", "entity_id", "entity_hash", "window_id", "row_start", "row_end", "rogue_frames", "normal_frames", "rogue_frame_ratio", "label"}
TARGET_FPRS = [0.10, 0.15, 0.20, 0.30]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("dataset/4.Rogue_AP"))
    p.add_argument("--processed-dir", type=Path, default=Path("processed/rogue_ap_per_bssid_windows"))
    p.add_argument("--results-dir", type=Path, default=Path("results/rogue_ap_per_bssid_lofo"))
    p.add_argument("--window-size", type=int, default=50)
    p.add_argument("--min-window-frames", type=int, default=10)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def ratio_present(s: pd.Series) -> float:
    return float(s.notna().mean()) if len(s) else 0.0


def add_numeric_stats(out: dict[str, Any], prefix: str, s: pd.Series) -> None:
    x = to_num(s).dropna()
    out[f"{prefix}_present_ratio"] = ratio_present(s)
    if len(x) == 0:
        for stat in ["mean", "std", "min", "max", "range", "p25", "p50", "p75"]:
            out[f"{prefix}_{stat}"] = 0.0
        return
    out[f"{prefix}_mean"] = float(x.mean())
    out[f"{prefix}_std"] = float(x.std(ddof=0))
    out[f"{prefix}_min"] = float(x.min())
    out[f"{prefix}_max"] = float(x.max())
    out[f"{prefix}_range"] = float(x.max() - x.min())
    out[f"{prefix}_p25"] = float(x.quantile(0.25))
    out[f"{prefix}_p50"] = float(x.quantile(0.50))
    out[f"{prefix}_p75"] = float(x.quantile(0.75))


def add_binary_ratio(out: dict[str, Any], name: str, s: pd.Series) -> None:
    x = to_num(s).dropna()
    out[f"{name}_present_ratio"] = ratio_present(s)
    out[f"{name}_ratio"] = float((x != 0).mean()) if len(x) else 0.0


def add_value_ratios(out: dict[str, Any], prefix: str, s: pd.Series, values: list[int]) -> None:
    x = to_num(s).dropna()
    out[f"{prefix}_present_ratio"] = ratio_present(s)
    denom = max(len(x), 1)
    for v in values:
        out[f"{prefix}_{v}_ratio"] = float((x == v).sum() / denom)
    out[f"{prefix}_nunique"] = float(x.nunique()) if len(x) else 0.0


def entity_series(df: pd.DataFrame) -> pd.Series:
    e = df["wlan.bssid"].astype("string")
    for c in ["wlan.ta", "wlan.sa", "wlan.ra", "wlan.da"]:
        e = e.fillna(df[c].astype("string"))
    return e.fillna("__missing_entity__")


def window_features(g: pd.DataFrame, source_file: str, file_id: int, entity: str, window_id: int) -> dict[str, Any]:
    g = g.sort_values("frame.number")
    y = (g["Label"].astype(str).str.strip().str.lower() == "rogueap").astype(int)
    out: dict[str, Any] = {
        "source_file": source_file,
        "file_id": file_id,
        "entity_id": entity,
        "entity_hash": abs(hash(entity)) % 1_000_000_007,
        "window_id": window_id,
        "row_start": int(to_num(g["frame.number"]).min()) if to_num(g["frame.number"]).notna().any() else 0,
        "row_end": int(to_num(g["frame.number"]).max()) if to_num(g["frame.number"]).notna().any() else 0,
        "n_frames": int(len(g)),
        "rogue_frames": int(y.sum()),
        "normal_frames": int((y == 0).sum()),
        "rogue_frame_ratio": float(y.mean()) if len(y) else 0.0,
        "label": int(y.any()),
    }

    for col, prefix in [
        ("frame.len", "frame_len"),
        ("wlan.duration", "wlan_duration"),
        ("radiotap.dbm_antsignal", "radiotap_signal_dbm"),
        ("wlan_radio.signal_dbm", "radio_signal_dbm"),
        ("radiotap.datarate", "radiotap_datarate"),
        ("wlan_radio.data_rate", "radio_data_rate"),
        ("wlan.tag.length", "tag_length"),
    ]:
        add_numeric_stats(out, prefix, g[col])

    # Channel/frequency values can be source-correlated; use diversity/consistency stats rather than raw IDs.
    for col, prefix in [
        ("radiotap.channel.freq", "radiotap_freq"),
        ("wlan_radio.frequency", "radio_freq"),
        ("wlan_radio.channel", "radio_channel"),
        ("wlan_radio.phy", "radio_phy"),
    ]:
        x = to_num(g[col]).dropna()
        out[f"{prefix}_present_ratio"] = ratio_present(g[col])
        out[f"{prefix}_nunique"] = float(x.nunique()) if len(x) else 0.0
        out[f"{prefix}_mode_share"] = float(x.value_counts(normalize=True).iloc[0]) if len(x) else 0.0

    for col, name in [
        ("wlan.fc.frag", "frag"), ("wlan.fc.order", "order"), ("wlan.fc.moredata", "moredata"),
        ("wlan.fc.protected", "protected"), ("wlan.fc.pwrmgt", "pwrmgt"), ("wlan.fc.retry", "retry"),
        ("wlan.fixed.beacon", "fixed_beacon"), ("wlan.fixed.capabilities.ess", "cap_ess"),
        ("wlan.fixed.capabilities.ibss", "cap_ibss"), ("llc", "llc"), ("arp", "arp"),
        ("tcp.flags.syn", "tcp_syn"), ("tcp.flags.ack", "tcp_ack"),
    ]:
        if col in {"llc", "arp"}:
            out[f"{name}_present_ratio"] = ratio_present(g[col])
            out[f"{name}_ratio"] = out[f"{name}_present_ratio"]
        else:
            add_binary_ratio(out, name, g[col])

    add_value_ratios(out, "fc_type", g["wlan.fc.type"], [0, 1, 2])
    add_value_ratios(out, "fc_subtype", g["wlan.fc.subtype"], [0, 1, 4, 5, 8, 10, 11, 12, 13])
    # DS values are hex strings; keep as ratios without exposing MAC identities.
    ds = g["wlan.fc.ds"].astype("string").fillna("MISSING")
    denom = max(len(ds), 1)
    for v in ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "MISSING"]:
        out[f"fc_ds_{v.replace('0x', '')}_ratio"] = float((ds == v).sum() / denom)

    seq = to_num(g["wlan.seq"]).dropna().sort_index()
    out["seq_present_ratio"] = ratio_present(g["wlan.seq"])
    if len(seq) >= 2:
        d = seq.diff().dropna()
        out["seq_delta_mean"] = float(d.mean())
        out["seq_delta_std"] = float(d.std(ddof=0))
        out["seq_delta_abs_mean"] = float(d.abs().mean())
        out["seq_gap_gt1_ratio"] = float((d.abs() > 1).mean())
        out["seq_negative_delta_ratio"] = float((d < 0).mean())
    else:
        for k in ["seq_delta_mean", "seq_delta_std", "seq_delta_abs_mean", "seq_gap_gt1_ratio", "seq_negative_delta_ratio"]:
            out[k] = 0.0

    t = to_num(g["frame.time_relative"]).dropna().sort_values()
    if len(t) >= 2:
        dt = t.diff().dropna()
        out["iat_mean"] = float(dt.mean())
        out["iat_std"] = float(dt.std(ddof=0))
        out["iat_p50"] = float(dt.quantile(0.50))
        out["iat_p95"] = float(dt.quantile(0.95))
        out["iat_max"] = float(dt.max())
    else:
        for k in ["iat_mean", "iat_std", "iat_p50", "iat_p95", "iat_max"]:
            out[k] = 0.0
    beacon_t = to_num(g.loc[to_num(g["wlan.fc.subtype"]) == 8, "frame.time_relative"]).dropna().sort_values()
    out["beacon_count"] = int(len(beacon_t))
    out["beacon_ratio"] = float(len(beacon_t) / max(len(g), 1))
    if len(beacon_t) >= 2:
        bd = beacon_t.diff().dropna()
        out["beacon_iat_mean"] = float(bd.mean())
        out["beacon_iat_std"] = float(bd.std(ddof=0))
        out["beacon_iat_jitter"] = float(bd.std(ddof=0) / max(abs(bd.mean()), 1e-9))
    else:
        out["beacon_iat_mean"] = 0.0
        out["beacon_iat_std"] = 0.0
        out["beacon_iat_jitter"] = 0.0

    ssid = g["wlan.ssid"].astype("string")
    out["ssid_present_ratio"] = ratio_present(ssid)
    out["ssid_nunique"] = float(ssid.dropna().nunique())
    out["ip_proto_present_ratio"] = ratio_present(g["ip.proto"])
    out["udp_length_present_ratio"] = ratio_present(g["udp.length"])
    return out


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.processed_dir / "per_bssid_windows.parquet"
    if out_path.exists() and not args.rebuild:
        return pd.read_parquet(out_path)
    rows = []
    files = sorted(args.raw_dir.glob("RogueAP_*.csv"), key=lambda p: int(p.stem.split("_")[-1]))
    for p in files:
        file_id = int(p.stem.split("_")[-1])
        print(f"Building {p.name} ...", flush=True)
        df = pd.read_csv(p, usecols=lambda c: c in USECOLS, low_memory=False)
        missing = [c for c in USECOLS if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        df["entity_id"] = entity_series(df)
        df["_frame_number_num"] = to_num(df["frame.number"])
        df = df.sort_values(["entity_id", "_frame_number_num"]).reset_index(drop=True)
        win_count = 0
        for entity, eg in df.groupby("entity_id", sort=False):
            if entity == "__missing_entity__":
                continue
            eg = eg.reset_index(drop=True)
            for start in range(0, len(eg), args.window_size):
                chunk = eg.iloc[start : start + args.window_size]
                if len(chunk) < args.min_window_frames:
                    continue
                rows.append(window_features(chunk, p.name, file_id, str(entity), win_count))
                win_count += 1
        print(f"  windows={win_count}", flush=True)
    out = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)
    out.to_parquet(out_path, index=False)
    meta = {
        "raw_dir": str(args.raw_dir),
        "rows": int(len(out)),
        "files": int(out.source_file.nunique()) if len(out) else 0,
        "window_size": args.window_size,
        "min_window_frames": args.min_window_frames,
        "label_counts": {str(k): int(v) for k, v in out.label.value_counts().sort_index().to_dict().items()},
        "audit_columns": sorted(AUDIT_COLS),
    }
    (args.processed_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in AUDIT_COLS and pd.api.types.is_numeric_dtype(df[c])]


def safe_auc(y: np.ndarray, s: np.ndarray, kind: str) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s) if kind == "roc" else average_precision_score(y, s))


def metric_row(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (score >= threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / max(fp + tn, 1)),
        "auroc": safe_auc(y, score, "roc"),
        "pr_auc": safe_auc(y, score, "pr"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(y: np.ndarray, score: np.ndarray, mode: str, target: float | None = None) -> float:
    if mode == "best_f1":
        thresholds = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 99)))
        best_th, best_f1 = 0.5, -1.0
        for th in thresholds:
            f1 = metric_row(y, score, float(th))["f1"]
            if f1 > best_f1:
                best_th, best_f1 = float(th), f1
        return best_th
    fpr, tpr, th = roc_curve(y, score)
    valid = np.flatnonzero(fpr <= float(target))
    return float(th[valid[np.argmax(tpr[valid])]]) if len(valid) else 1.0


def run_lofo(df: pd.DataFrame, args: argparse.Namespace) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    features = feature_columns(df)
    (args.results_dir / "feature_columns.txt").write_text("\n".join(features), encoding="utf-8")
    df[["source_file", "label"]].groupby("source_file").agg(rows=("label", "size"), positives=("label", "sum")).reset_index().to_csv(args.results_dir / "lofo_file_audit.csv", index=False)
    metrics, per_file, importance_rows = [], [], []
    all_pred_rows = []
    files = sorted(df.source_file.unique(), key=lambda x: int(Path(x).stem.split("_")[-1]))
    for test_file in files:
        test = df[df.source_file == test_file].reset_index(drop=True)
        train = df[df.source_file != test_file].reset_index(drop=True)
        if train.label.nunique() < 2 or test.label.nunique() < 2:
            continue
        val_file = sorted(train.source_file.unique(), key=lambda x: int(Path(x).stem.split("_")[-1]))[-1]
        val = train[train.source_file == val_file].reset_index(drop=True)
        tr = train[train.source_file != val_file].reset_index(drop=True)
        if tr.label.nunique() < 2 or val.label.nunique() < 2:
            val = train.sample(frac=0.2, random_state=args.random_state)
            tr = train.drop(val.index).reset_index(drop=True)
            val = val.reset_index(drop=True)
        ytr, yv, yt = tr.label.to_numpy(np.int8), val.label.to_numpy(np.int8), test.label.to_numpy(np.int8)
        spw = int((ytr == 0).sum()) / max(int(ytr.sum()), 1)
        models: dict[str, Any] = {
            "rf_per_bssid": RandomForestClassifier(n_estimators=260, max_depth=12, min_samples_leaf=2, class_weight="balanced_subsample", random_state=args.random_state, n_jobs=-1),
            "xgb_per_bssid": XGBClassifier(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", max_depth=3, learning_rate=0.04, n_estimators=220, scale_pos_weight=spw, subsample=0.9, colsample_bytree=0.9, random_state=args.random_state),
        }
        for name, model in models.items():
            model.fit(tr[features], ytr)
            vs = model.predict_proba(val[features])[:, 1]
            ts = model.predict_proba(test[features])[:, 1]
            for policy, target in [("best_f1", None), ("fpr_le_0.20", 0.20), ("fpr_le_0.30", 0.30)]:
                th = choose_threshold(yv, vs, "best_f1" if target is None else "fpr", target)
                row = {"model": name, "test_file": test_file, "val_file": val_file, "policy": policy, "test_windows": len(test), "test_positive_windows": int(yt.sum())}
                row.update(metric_row(yt, ts, th))
                metrics.append(row)
            pred = pd.DataFrame({
                "model": name,
                "test_file": test_file,
                "entity_hash": test.entity_hash,
                "label": yt,
                "score": ts,
            })
            all_pred_rows.append(pred)
            if hasattr(model, "feature_importances_"):
                for f, v in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)[:25]:
                    importance_rows.append({"model": name, "test_file": test_file, "feature": f, "importance": float(v)})

    mdf = pd.DataFrame(metrics)
    mdf.to_csv(args.results_dir / "lofo_metrics_by_file.csv", index=False)
    if all_pred_rows:
        pd.concat(all_pred_rows, ignore_index=True).to_csv(args.results_dir / "lofo_predictions.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(args.results_dir / "feature_importance_lofo.csv", index=False)
    summary_rows = []
    for (model, policy), g in mdf.groupby(["model", "policy"]):
        pooled = {
            "model": model,
            "policy": policy,
            "files": int(g.test_file.nunique()),
            "precision_mean": float(g.precision.mean()),
            "precision_std": float(g.precision.std(ddof=0)),
            "recall_mean": float(g.recall.mean()),
            "recall_std": float(g.recall.std(ddof=0)),
            "f1_mean": float(g.f1.mean()),
            "f1_std": float(g.f1.std(ddof=0)),
            "fpr_mean": float(g.fpr.mean()),
            "fpr_std": float(g.fpr.std(ddof=0)),
            "auroc_mean": float(g.auroc.mean()),
            "pr_auc_mean": float(g.pr_auc.mean()),
            "test_windows_total": int(g.test_windows.sum()),
            "test_positive_windows_total": int(g.test_positive_windows.sum()),
        }
        summary_rows.append(pooled)
    sdf = pd.DataFrame(summary_rows).sort_values("f1_mean", ascending=False)
    sdf.to_csv(args.results_dir / "lofo_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    top = sdf.head(8).iloc[::-1]
    labels = top.model + "/" + top.policy
    ax.barh(labels, top.f1_mean, xerr=top.f1_std, label="F1 mean +/- std")
    ax.scatter(top.fpr_mean, labels, color="tab:red", label="FPR mean")
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(args.results_dir / "lofo_summary.png", dpi=180)
    plt.close(fig)

    lines = [
        "Per-BSSID/per-transmitter Rogue AP LOFO summary",
        "",
        f"Aggregated windows: {len(df)}; files: {df.source_file.nunique()}; positive windows: {int(df.label.sum())}.",
        f"Feature count: {len(features)}. Audit/group IDs are excluded from model features.",
        "",
        "Top LOFO configurations by mean F1:",
    ]
    for _, r in sdf.head(8).iterrows():
        lines.append(
            f"- {r.model}/{r.policy}: F1={r.f1_mean:.4f}+/-{r.f1_std:.4f}, "
            f"precision={r.precision_mean:.4f}, recall={r.recall_mean:.4f}, "
            f"FPR={r.fpr_mean:.4f}, AUROC={r.auroc_mean:.4f}, PR-AUC={r.pr_auc_mean:.4f}"
        )
    (args.results_dir / "short_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def ablation_feature_sets(features: list[str]) -> dict[str, list[str]]:
    def keep_without(patterns: list[str]) -> list[str]:
        return [f for f in features if not any(p in f for p in patterns)]

    mgmt_fingerprint = [
        "fixed_beacon", "cap_ess", "cap_ibss", "ssid", "tag_length", "beacon_",
    ]
    phy_radio = [
        "radiotap_signal", "radio_signal", "radiotap_datarate", "radio_data_rate",
        "radiotap_freq", "radio_freq", "radio_channel", "radio_phy",
    ]
    timing_seq = ["iat_", "seq_", "beacon_iat"]
    return {
        "full": features,
        "drop_mgmt_fingerprint": keep_without(mgmt_fingerprint),
        "drop_phy_radio": keep_without(phy_radio),
        "drop_timing_seq": keep_without(timing_seq),
        "drop_mgmt_phy": keep_without(mgmt_fingerprint + phy_radio),
        "drop_mgmt_phy_timing_seq": keep_without(mgmt_fingerprint + phy_radio + timing_seq),
    }


def run_lofo_ablation(df: pd.DataFrame, args: argparse.Namespace) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    base_features = feature_columns(df)
    sets = ablation_feature_sets(base_features)
    files = sorted(df.source_file.unique(), key=lambda x: int(Path(x).stem.split("_")[-1]))
    rows, imp_rows = [], []
    for set_name, features in sets.items():
        print(f"Ablation {set_name}: {len(features)} features", flush=True)
        for test_file in files:
            test = df[df.source_file == test_file].reset_index(drop=True)
            train = df[df.source_file != test_file].reset_index(drop=True)
            if train.label.nunique() < 2 or test.label.nunique() < 2:
                continue
            val_file = sorted(train.source_file.unique(), key=lambda x: int(Path(x).stem.split("_")[-1]))[-1]
            val = train[train.source_file == val_file].reset_index(drop=True)
            tr = train[train.source_file != val_file].reset_index(drop=True)
            if tr.label.nunique() < 2 or val.label.nunique() < 2:
                val = train.sample(frac=0.2, random_state=args.random_state)
                tr = train.drop(val.index).reset_index(drop=True)
                val = val.reset_index(drop=True)
            ytr, yv, yt = tr.label.to_numpy(np.int8), val.label.to_numpy(np.int8), test.label.to_numpy(np.int8)
            spw = int((ytr == 0).sum()) / max(int(ytr.sum()), 1)
            models: dict[str, Any] = {
                "xgb_per_bssid": XGBClassifier(
                    objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
                    max_depth=3, learning_rate=0.04, n_estimators=220, scale_pos_weight=spw,
                    subsample=0.9, colsample_bytree=0.9, random_state=args.random_state,
                ),
                "rf_per_bssid": RandomForestClassifier(
                    n_estimators=220, max_depth=12, min_samples_leaf=2,
                    class_weight="balanced_subsample", random_state=args.random_state, n_jobs=-1,
                ),
            }
            for model_name, model in models.items():
                model.fit(tr[features], ytr)
                vs = model.predict_proba(val[features])[:, 1]
                ts = model.predict_proba(test[features])[:, 1]
                for policy, target in [("best_f1", None), ("fpr_le_0.20", 0.20)]:
                    th = choose_threshold(yv, vs, "best_f1" if target is None else "fpr", target)
                    row = {
                        "feature_set": set_name,
                        "model": model_name,
                        "test_file": test_file,
                        "val_file": val_file,
                        "policy": policy,
                        "feature_count": len(features),
                        "test_windows": len(test),
                        "test_positive_windows": int(yt.sum()),
                    }
                    row.update(metric_row(yt, ts, th))
                    rows.append(row)
                if hasattr(model, "feature_importances_"):
                    for f, v in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)[:15]:
                        imp_rows.append({"feature_set": set_name, "model": model_name, "test_file": test_file, "feature": f, "importance": float(v)})

    mdf = pd.DataFrame(rows)
    mdf.to_csv(args.results_dir / "lofo_ablation_by_file.csv", index=False)
    pd.DataFrame(imp_rows).to_csv(args.results_dir / "lofo_ablation_feature_importance.csv", index=False)
    summary = []
    for (fs, model, policy), g in mdf.groupby(["feature_set", "model", "policy"]):
        summary.append({
            "feature_set": fs,
            "model": model,
            "policy": policy,
            "feature_count": int(g.feature_count.iloc[0]),
            "files": int(g.test_file.nunique()),
            "precision_mean": float(g.precision.mean()),
            "recall_mean": float(g.recall.mean()),
            "f1_mean": float(g.f1.mean()),
            "f1_std": float(g.f1.std(ddof=0)),
            "fpr_mean": float(g.fpr.mean()),
            "auroc_mean": float(g.auroc.mean()),
            "pr_auc_mean": float(g.pr_auc.mean()),
        })
    sdf = pd.DataFrame(summary).sort_values(["f1_mean", "pr_auc_mean"], ascending=False)
    sdf.to_csv(args.results_dir / "lofo_ablation_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    top = sdf[(sdf.model == "xgb_per_bssid") & (sdf.policy == "fpr_le_0.20")].sort_values("f1_mean")
    labels = top.feature_set
    ax.barh(labels, top.f1_mean, xerr=top.f1_std, label="F1")
    ax.scatter(top.fpr_mean, labels, color="tab:red", label="FPR")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.set_title("Per-BSSID LOFO strict feature ablation")
    fig.tight_layout()
    fig.savefig(args.results_dir / "lofo_ablation_summary.png", dpi=180)
    plt.close(fig)

    lines = ["Per-BSSID LOFO strict feature ablation", ""]
    for _, r in sdf.head(14).iterrows():
        lines.append(
            f"- {r.feature_set}/{r.model}/{r.policy}: F1={r.f1_mean:.4f}+/-{r.f1_std:.4f}, "
            f"precision={r.precision_mean:.4f}, recall={r.recall_mean:.4f}, FPR={r.fpr_mean:.4f}, "
            f"AUROC={r.auroc_mean:.4f}, PR-AUC={r.pr_auc_mean:.4f}, features={int(r.feature_count)}"
        )
    lines += [
        "",
        "Interpretation:",
        "- If full >> drop_mgmt_fingerprint, the high score depends strongly on beacon/capability/SSID fingerprints.",
        "- If drop_mgmt_phy_timing_seq remains strong, the detector has broader behavioral signal.",
    ]
    (args.results_dir / "lofo_ablation_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    args = parse_args()
    df = build_dataset(args)
    run_lofo(df, args)
    run_lofo_ablation(df, args)


if __name__ == "__main__":
    main()
