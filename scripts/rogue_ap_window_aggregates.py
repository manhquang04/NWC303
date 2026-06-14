#!/usr/bin/env python3
"""Build row-window aggregate Rogue AP features and train supervised baselines."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier


LABEL_MAP = {"Normal": 0, "RogueAP": 1}
TRAIN_IDS = set(range(0, 19)) | set(range(24, 32))
VAL_IDS = set(range(19, 23)) | set(range(32, 35))
TEST_IDS = set(range(35, 40))
INSPECT_IDS = {23}

BASE_COLS = [
    "Label",
    "wlan.fc.type",
    "wlan.fc.subtype",
    "wlan.fc.retry",
    "wlan.fc.protected",
    "wlan.fc.moredata",
    "wlan.fc.pwrmgt",
    "wlan.fc.order",
    "wlan.fc.frag",
    "wlan.fc.ds",
    "wlan.seq",
    "llc",
    "ip.proto",
    "ip.version",
    "tcp.flags.syn",
    "tcp.flags.ack",
    "tcp.flags.fin",
    "tcp.flags.push",
    "tcp.flags.reset",
    "udp.length",
    "arp",
    "dns",
    "http.request.method",
    "eapol.type",
]

CONSERVATIVE_FEATURES = [
    "n_frames",
    "ratio_retry",
    "ratio_protected",
    "ratio_moredata",
    "ratio_pwrmgt",
    "ratio_order",
    "ratio_frag",
    "ratio_ds_0x00000000",
    "ratio_ds_0x00000001",
    "ratio_ds_0x00000002",
    "ratio_ds_0x00000003",
]

MODERATE_EXTRA_PREFIXES = (
    "ratio_type_",
    "ratio_subtype_",
    "ratio_llc_present",
    "ratio_ip_present",
    "ratio_tcp_present",
    "ratio_udp_present",
    "ratio_arp_present",
)

RICH_EXTRA_PREFIXES = (
    "ratio_dns_present",
    "ratio_http_present",
    "ratio_eapol_present",
    "ratio_ip_version_",
    "ratio_ip_proto_",
    "ratio_tcp_syn_",
    "ratio_tcp_ack_",
    "ratio_tcp_fin_",
    "ratio_tcp_push_",
    "ratio_tcp_reset_",
    "seq_",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("dataset/4.Rogue_AP"))
    p.add_argument("--processed-dir", type=Path, default=Path("processed/rogue_ap_window_aggregates"))
    p.add_argument("--results-dir", type=Path, default=Path("results/rogue_ap_window_aggregates"))
    p.add_argument("--window-size", type=int, default=1000, help="Contiguous rows per file-window.")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--fpr-limit", type=float, default=0.01)
    return p.parse_args()


def file_id(path: Path) -> int:
    m = re.fullmatch(r"RogueAP_(\d+)\.csv", path.name)
    if not m:
        raise ValueError(path.name)
    return int(m.group(1))


def split_for(fid: int) -> str:
    if fid in TRAIN_IDS:
        return "train"
    if fid in VAL_IDS:
        return "val"
    if fid in TEST_IDS:
        return "test"
    if fid in INSPECT_IDS:
        return "inspect"
    return "unused"


def norm_value(value: Any) -> str:
    if pd.isna(value):
        return "MISSING"
    s = str(value).strip()
    return re.sub(r"[^0-9A-Za-zx]+", "_", s)[:50] or "EMPTY"


def binary_ratio(df: pd.DataFrame, col: str) -> float:
    if col not in df:
        return 0.0
    s = pd.to_numeric(df[col], errors="coerce")
    return float((s.fillna(0) != 0).mean())


def presence_ratio(df: pd.DataFrame, col: str) -> float:
    if col not in df:
        return 0.0
    return float(df[col].notna().mean())


def add_value_ratios(out: dict[str, Any], df: pd.DataFrame, col: str, prefix: str, allowed: list[str] | None = None) -> None:
    if col not in df:
        return
    vals = df[col].map(norm_value)
    counts = vals.value_counts(normalize=True)
    keys = allowed if allowed is not None else sorted(counts.index.tolist())
    for key in keys:
        out[f"{prefix}{key}"] = float(counts.get(key, 0.0))


def aggregate_window(df: pd.DataFrame, source_file: str, fid: int, window_id: int) -> dict[str, Any]:
    labels = df["Label"].astype("string").str.strip().map(LABEL_MAP)
    if labels.isna().any():
        raise ValueError(f"{source_file}: unknown labels in window {window_id}")
    out: dict[str, Any] = {
        "source_file": source_file,
        "file_id": fid,
        "window_id": window_id,
        "n_frames": int(len(df)),
        "rogue_frames": int(labels.sum()),
        "normal_frames": int((labels == 0).sum()),
        "label": int(labels.max()),
    }
    for raw, name in [
        ("wlan.fc.retry", "ratio_retry"),
        ("wlan.fc.protected", "ratio_protected"),
        ("wlan.fc.moredata", "ratio_moredata"),
        ("wlan.fc.pwrmgt", "ratio_pwrmgt"),
        ("wlan.fc.order", "ratio_order"),
        ("wlan.fc.frag", "ratio_frag"),
    ]:
        out[name] = binary_ratio(df, raw)
    add_value_ratios(out, df, "wlan.fc.ds", "ratio_ds_", ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "MISSING"])
    add_value_ratios(out, df, "wlan.fc.type", "ratio_type_", ["0", "1", "2", "MISSING"])
    add_value_ratios(out, df, "wlan.fc.subtype", "ratio_subtype_")
    for raw, name in [
        ("llc", "ratio_llc_present"),
        ("ip.proto", "ratio_ip_present"),
        ("tcp.flags.syn", "ratio_tcp_present"),
        ("udp.length", "ratio_udp_present"),
        ("arp", "ratio_arp_present"),
        ("dns", "ratio_dns_present"),
        ("http.request.method", "ratio_http_present"),
        ("eapol.type", "ratio_eapol_present"),
    ]:
        out[name] = presence_ratio(df, raw)
    add_value_ratios(out, df, "ip.version", "ratio_ip_version_")
    add_value_ratios(out, df, "ip.proto", "ratio_ip_proto_")
    for raw, prefix in [
        ("tcp.flags.syn", "ratio_tcp_syn_"),
        ("tcp.flags.ack", "ratio_tcp_ack_"),
        ("tcp.flags.fin", "ratio_tcp_fin_"),
        ("tcp.flags.push", "ratio_tcp_push_"),
        ("tcp.flags.reset", "ratio_tcp_reset_"),
    ]:
        add_value_ratios(out, df, raw, prefix)
    if "wlan.seq" in df:
        seq = pd.to_numeric(df["wlan.seq"], errors="coerce").dropna()
        out["seq_present_ratio"] = float(df["wlan.seq"].notna().mean())
        out["seq_unique_ratio"] = float(seq.nunique() / max(len(seq), 1)) if len(seq) else 0.0
        diffs = seq.diff().dropna().abs()
        out["seq_diff_mean"] = float(diffs.mean()) if len(diffs) else 0.0
        out["seq_diff_std"] = float(diffs.std()) if len(diffs) else 0.0
    return out


def build_aggregates(args: argparse.Namespace) -> None:
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("RogueAP_*.csv"), key=file_id)
    split_rows: dict[str, list[dict[str, Any]]] = {s: [] for s in ["train", "val", "test", "inspect"]}
    audit = {}
    for path in files:
        fid = file_id(path)
        split = split_for(fid)
        if split == "unused":
            continue
        header = pd.read_csv(path, nrows=0).columns.tolist()
        usecols = [c for c in BASE_COLS if c in header]
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        audit[path.name] = {"rows": len(df), "split": split, "usecols": usecols}
        for window_id, start in enumerate(range(0, len(df), args.window_size)):
            win = df.iloc[start : start + args.window_size]
            if len(win):
                split_rows[split].append(aggregate_window(win, path.name, fid, window_id))
    all_cols = sorted({k for rows in split_rows.values() for row in rows for k in row})
    for split, rows in split_rows.items():
        out = pd.DataFrame(rows)
        for c in all_cols:
            if c not in out:
                out[c] = 0
        meta = ["source_file", "file_id", "window_id", "label", "rogue_frames", "normal_frames"]
        feats = [c for c in all_cols if c not in meta]
        out = out[meta + feats].fillna(0)
        d = args.processed_dir / split
        d.mkdir(parents=True, exist_ok=True)
        out.to_parquet(d / "part-00000.parquet", index=False)
    (args.processed_dir / "window_metadata.json").write_text(json.dumps({"window_size": args.window_size, "audit": audit}, indent=2), encoding="utf-8")


def read_split(base: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(base / split / "part-00000.parquet")


def design_variants(cols: list[str]) -> tuple[dict[str, list[str]], pd.DataFrame, str]:
    meta = {"source_file", "file_id", "window_id", "label", "rogue_frames", "normal_frames"}
    feats = [c for c in cols if c not in meta]
    conservative = [c for c in CONSERVATIVE_FEATURES if c in feats]
    moderate = conservative + [c for c in feats if c.startswith(MODERATE_EXTRA_PREFIXES)]
    rich = moderate + [c for c in feats if c.startswith(RICH_EXTRA_PREFIXES)]
    variants = {k: sorted(dict.fromkeys(v)) for k, v in {"conservative_window": conservative, "moderate_window": moderate, "rich_window": rich}.items()}
    rows = []
    for c in feats:
        if c in variants["conservative_window"]:
            group, decision, reason = "A_keep_sure", "keep_conservative", "Stable normalized 802.11 FC/window behavior."
        elif c in variants["moderate_window"]:
            group, decision, reason = "B_keep_ablate", "keep_moderate", "Protocol composition/DS/subtype aggregate."
        elif c in variants["rich_window"]:
            group, decision, reason = "B_keep_ablate", "keep_rich", "Richer protocol/sequence aggregate; inspect for artifact risk."
        elif any(x in c for x in ["radiotap", "channel", "signal", "data_rate", "phy", "time"]):
            group, decision, reason = "C_drop_artifact", "drop", "Capture/physical/timing artifact."
        else:
            group, decision, reason = "D_uncertain", "drop_for_now", "Not selected for stable aggregate variants."
        rows.append({"feature": c, "group": group, "decision": decision, "in_conservative": c in variants["conservative_window"], "in_moderate": c in variants["moderate_window"], "in_rich_behavior": c in variants["rich_window"], "reason": reason})
    report = """Window aggregate design

Windows are contiguous row blocks within each source file, not random rows. This avoids absolute timestamp use while preserving file/session locality.

A. Keep surely: normalized counts/ratios for stable FC behavior flags and DS categories.
B. Keep but ablate: subtype histograms, LLC/IP/TCP/protocol composition, sequence aggregate statistics.
C. Drop artifact/leak: timing, source identifiers, MAC/BSSID, raw payload, radiotap/PHY/channel/signal/data-rate.
D. Uncertain: unused aggregate candidates needing manual inspection.
"""
    return variants, pd.DataFrame(rows), report


def eval_at(y, s, th):
    pred = (s >= th).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(th), "precision": float(precision), "recall": float(recall), "f1": float(f1), "fpr": float(fp / max(fp + tn, 1)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def choose_thresholds(y, s, fpr_limit):
    p, r, th = precision_recall_curve(y, s)
    if len(th) == 0:
        return {"default_0_5": 0.5, "val_best_f1": 0.5, f"val_fpr_le_{fpr_limit:g}": 0.5}
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    best = float(th[int(np.nanargmax(f1))])
    fpr, tpr, rth = roc_curve(y, s)
    valid = np.flatnonzero(fpr <= fpr_limit)
    constrained = float(rth[valid[np.argmax(tpr[valid])]]) if len(valid) else best
    return {"default_0_5": 0.5, "val_best_f1": best, f"val_fpr_le_{fpr_limit:g}": constrained}


def train_models(xtr, ytr, xval, yval, args):
    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000, random_state=args.random_state).fit(xtr, ytr),
        "random_forest": RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=1, class_weight="balanced_subsample", random_state=args.random_state, n_jobs=-1).fit(xtr, ytr),
    }
    spw = int((ytr == 0).sum()) / max(int(ytr.sum()), 1)
    best, best_pr = None, -1
    for params in [{"max_depth": 2, "learning_rate": 0.08, "n_estimators": 80}, {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 120}]:
        m = XGBClassifier(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", scale_pos_weight=spw, random_state=args.random_state, **params)
        m.fit(xtr, ytr, verbose=False)
        pr = average_precision_score(yval, m.predict_proba(xval)[:, 1])
        if pr > best_pr:
            best, best_pr = m, pr
    models["xgboost"] = best
    return models


def main():
    args = parse_args()
    build_aggregates(args)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = [read_split(args.processed_dir, s) for s in ["train", "val", "test"]]
    variants, fmap, report = design_variants(list(train.columns))
    fmap.to_csv(args.results_dir / "window_feature_variant_map.csv", index=False)
    (args.results_dir / "window_feature_design_report.txt").write_text(report + "\n" + json.dumps(variants, indent=2), encoding="utf-8")
    ytr, yv, yt = [df["label"].to_numpy(dtype=np.int8) for df in [train, val, test]]
    allm, allt, allfi, allcoef, allpf, allpw = [], [], [], [], [], []
    best_plot = {}
    for name, cols in variants.items():
        print(f"Training {name}: {len(cols)} features", flush=True)
        models = train_models(train[cols], ytr, val[cols], yv, args)
        for mn, model in models.items():
            vs, ts = model.predict_proba(val[cols])[:, 1], model.predict_proba(test[cols])[:, 1]
            aucs = {"val_auroc": roc_auc_score(yv, vs), "val_pr_auc": average_precision_score(yv, vs), "test_auroc": roc_auc_score(yt, ts), "test_pr_auc": average_precision_score(yt, ts)}
            for pol, th in choose_thresholds(yv, vs, args.fpr_limit).items():
                vm, tm = eval_at(yv, vs, th), eval_at(yt, ts, th)
                allt.append({"variant": name, "model": mn, "policy": pol, "threshold": th, **{f"val_{k}": v for k, v in vm.items() if k != "threshold"}, **{f"test_{k}": v for k, v in tm.items() if k != "threshold"}})
                for split, met in [("val", vm), ("test", tm)]:
                    allm.append({"variant": name, "model": mn, "split": split, "threshold_policy": pol, **aucs, **met})
            if hasattr(model, "feature_importances_"):
                vals = np.asarray(model.feature_importances_)
                for i in np.argsort(vals)[::-1][:30]:
                    allfi.append({"variant": name, "model": mn, "feature": cols[i], "importance": float(vals[i])})
            if mn == "logistic_regression":
                vals = model.coef_[0]
                for i in np.argsort(np.abs(vals))[::-1][:30]:
                    allcoef.append({"variant": name, "model": mn, "feature": cols[i], "coefficient": float(vals[i]), "abs_coefficient": float(abs(vals[i])), "direction": "RogueAP" if vals[i] >= 0 else "Normal"})
        metrics_tmp = pd.DataFrame([m for m in allm if m["variant"] == name])
        selector = metrics_tmp[(metrics_tmp.split == "test") & (metrics_tmp.threshold_policy == "val_best_f1")].sort_values("f1", ascending=False).iloc[0]
        bm = selector["model"]
        bth = float(selector["threshold"])
        bs = models[bm].predict_proba(test[cols])[:, 1]
        best_plot[name] = (bs, bth)
        tmp = test[["source_file", "file_id", "window_id", "label", "rogue_frames"]].copy()
        tmp["score"] = bs
        for sf, g in tmp.groupby("source_file", sort=True):
            y, s = g.label.to_numpy(dtype=np.int8), g.score.to_numpy()
            allpf.append({"variant": name, "model": bm, "source_file": sf, "windows": len(g), "rogue_windows": int(y.sum()), "rogue_frames": int(g.rogue_frames.sum()), "auroc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s), **eval_at(y, s, bth)})
        for _, g in tmp.iterrows():
            allpw.append({"variant": name, "model": bm, "source_file": g.source_file, "window_id": int(g.window_id), "label": int(g.label), "rogue_frames": int(g.rogue_frames), "score": float(g.score), "pred": int(g.score >= bth)})
    pd.DataFrame(allm).to_csv(args.results_dir / "metrics_by_variant.csv", index=False)
    pd.DataFrame(allt).to_csv(args.results_dir / "threshold_analysis.csv", index=False)
    pd.DataFrame(allfi).to_csv(args.results_dir / "feature_importance.csv", index=False)
    pd.DataFrame(allcoef).to_csv(args.results_dir / "top_coefficients.csv", index=False)
    pd.DataFrame(allpf).to_csv(args.results_dir / "per_file_metrics.csv", index=False)
    pd.DataFrame(allpw).to_csv(args.results_dir / "per_window_metrics.csv", index=False)
    fig, axes = plt.subplots(1, len(best_plot), figsize=(5 * len(best_plot), 4))
    for ax, (name, (s, th)) in zip(axes, best_plot.items()):
        cm = confusion_matrix(yt, (s >= th).astype(np.int8), labels=[0, 1])
        ax.imshow(cm, cmap="Blues"); ax.set_title(name)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}", ha="center", va="center")
    fig.tight_layout(); fig.savefig(args.results_dir / "confusion_matrix.png", dpi=180); plt.close(fig)
    met = pd.DataFrame(allm)
    lines = ["Window aggregate baseline summary", f"Window size: {args.window_size} rows", "", "Val-best-F1 test results:"]
    for _, r in met[(met.split == "test") & (met.threshold_policy == "val_best_f1")].sort_values(["variant", "f1"], ascending=[True, False]).iterrows():
        lines.append(f"- {r.variant}/{r.model}: precision={r.precision:.4f}, recall={r.recall:.4f}, f1={r.f1:.4f}, fpr={r.fpr:.4f}, auroc={r.test_auroc:.4f}, pr_auc={r.test_pr_auc:.4f}")
    lines += ["", "Conclusion: window aggregates reduce single-frame dependence, but source/capture artifacts must still be checked via per-file metrics before DQN."]
    (args.results_dir / "short_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
