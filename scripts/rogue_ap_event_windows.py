#!/usr/bin/env python3
"""Event-centered windowing for Rogue AP detection."""

from __future__ import annotations

import argparse
import json
import re
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
USECOLS = [
    "Label",
    "wlan.fc.retry",
    "wlan.fc.protected",
    "wlan.fc.moredata",
    "wlan.fc.pwrmgt",
    "wlan.fc.order",
    "wlan.fc.frag",
    "wlan.fc.ds",
    "wlan.fc.type",
    "wlan.fc.subtype",
    "llc",
    "ip.proto",
    "tcp.flags.syn",
    "tcp.flags.ack",
]
FEATURES = [
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
    "ratio_type_0",
    "ratio_type_1",
    "ratio_type_2",
    "ratio_llc_present",
    "ratio_ip_present",
    "ratio_tcp_present",
]

VARIANTS = {
    "tight_event": {"before": 25, "after": 75, "neg_per_pos": 2, "merge_gap": 10, "label_rule": "any", "sampling": "distance_balanced"},
    "balanced_event": {"before": 50, "after": 50, "neg_per_pos": 2, "merge_gap": 25, "label_rule": "ratio_ge_0.01", "sampling": "distance_balanced"},
    "wide_event": {"before": 100, "after": 100, "neg_per_pos": 2, "merge_gap": 50, "label_rule": "ratio_ge_0.01", "sampling": "distance_balanced"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("dataset/4.Rogue_AP"))
    p.add_argument("--processed-dir", type=Path, default=Path("processed/rogue_ap_event_windows"))
    p.add_argument("--results-dir", type=Path, default=Path("results/rogue_ap_event_windows"))
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


def norm(v: Any) -> str:
    if pd.isna(v):
        return "MISSING"
    return str(v).strip()


def binary_ratio(df: pd.DataFrame, col: str) -> float:
    if col not in df:
        return 0.0
    return float((pd.to_numeric(df[col], errors="coerce").fillna(0) != 0).mean())


def presence_ratio(df: pd.DataFrame, col: str) -> float:
    if col not in df:
        return 0.0
    return float(df[col].notna().mean())


def value_ratio(df: pd.DataFrame, col: str, value: str) -> float:
    if col not in df:
        return 0.0
    vals = df[col].map(norm)
    return float((vals == value).mean())


def aggregate(df: pd.DataFrame, source_file: str, fid: int, event_id: int, start: int, end: int, kind: str, label_rule: str) -> dict[str, Any]:
    labels = df["Label"].astype("string").str.strip().map(LABEL_MAP)
    rogue = int(labels.sum())
    ratio = rogue / max(len(df), 1)
    if label_rule == "any":
        label = int(rogue >= 1)
    elif label_rule.startswith("ratio_ge_"):
        label = int(ratio >= float(label_rule.replace("ratio_ge_", "")))
    elif label_rule == "majority":
        label = int(ratio >= 0.5)
    else:
        raise ValueError(label_rule)
    out = {
        "source_file": source_file,
        "file_id": fid,
        "event_id": event_id,
        "window_kind": kind,
        "row_start": start,
        "row_end": end,
        "n_frames": int(len(df)),
        "rogue_frames": rogue,
        "normal_frames": int((labels == 0).sum()),
        "rogue_frame_ratio": ratio,
        "label": label,
    }
    for raw, dst in [
        ("wlan.fc.retry", "ratio_retry"),
        ("wlan.fc.protected", "ratio_protected"),
        ("wlan.fc.moredata", "ratio_moredata"),
        ("wlan.fc.pwrmgt", "ratio_pwrmgt"),
        ("wlan.fc.order", "ratio_order"),
        ("wlan.fc.frag", "ratio_frag"),
    ]:
        out[dst] = binary_ratio(df, raw)
    for val in ["0x00000000", "0x00000001", "0x00000002", "0x00000003"]:
        out[f"ratio_ds_{val}"] = value_ratio(df, "wlan.fc.ds", val)
    for val in ["0", "1", "2"]:
        out[f"ratio_type_{val}"] = value_ratio(df, "wlan.fc.type", val)
    out["ratio_llc_present"] = presence_ratio(df, "llc")
    out["ratio_ip_present"] = presence_ratio(df, "ip.proto")
    out["ratio_tcp_present"] = presence_ratio(df, "tcp.flags.syn")
    return out


def rogue_segments(labels: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    pos = np.flatnonzero(labels == 1)
    if len(pos) == 0:
        return []
    segments = []
    start = prev = int(pos[0])
    for idx in map(int, pos[1:]):
        if idx - prev <= merge_gap:
            prev = idx
        else:
            segments.append((start, prev))
            start = prev = idx
    segments.append((start, prev))
    return segments


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def sample_negative_windows(n: int, size: int, forbidden: list[tuple[int, int]], wanted: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if n <= size:
        return []
    candidates = []
    # Deterministic grid gives coverage; rng shuffles for diversity without row-level split.
    step = max(size, size // 2)
    for start in range(0, n - size + 1, step):
        win = (start, start + size - 1)
        if not any(overlaps(win, f) for f in forbidden):
            candidates.append(win)
    rng.shuffle(candidates)
    return candidates[:wanted]


def build_variant(args: argparse.Namespace, name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(args.random_state)
    files = sorted(args.input_dir.glob("RogueAP_*.csv"), key=file_id)
    split_rows = {s: [] for s in ["train", "val", "test", "inspect"]}
    audit = {}
    for path in files:
        fid = file_id(path)
        split = split_for(fid)
        if split == "unused":
            continue
        header = pd.read_csv(path, nrows=0).columns.tolist()
        usecols = [c for c in USECOLS if c in header]
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        labels = df["Label"].astype("string").str.strip().map(LABEL_MAP).to_numpy(dtype=np.int8)
        segments = rogue_segments(labels, cfg["merge_gap"])
        pos_windows = []
        event_id = 0
        for seg_start, seg_end in segments:
            center = (seg_start + seg_end) // 2
            start = max(0, center - cfg["before"])
            end = min(len(df) - 1, center + cfg["after"])
            pos_windows.append((start, end))
            split_rows[split].append(aggregate(df.iloc[start : end + 1], path.name, fid, event_id, start, end, "positive_event", cfg["label_rule"]))
            event_id += 1
        size = cfg["before"] + cfg["after"] + 1
        neg_windows = sample_negative_windows(len(df), size, pos_windows, max(1, len(pos_windows) * cfg["neg_per_pos"]) if pos_windows else 8, rng)
        for nw_id, (start, end) in enumerate(neg_windows):
            split_rows[split].append(aggregate(df.iloc[start : end + 1], path.name, fid, 100000 + nw_id, start, end, "negative_sample", cfg["label_rule"]))
        audit[path.name] = {"split": split, "rows": len(df), "rogue_frames": int(labels.sum()), "events": len(segments), "negative_windows": len(neg_windows)}
    vdir = args.processed_dir / name
    vdir.mkdir(parents=True, exist_ok=True)
    metadata = {"variant": name, "config": cfg, "audit": audit}
    for split, rows in split_rows.items():
        out = pd.DataFrame(rows).fillna(0)
        d = vdir / split
        d.mkdir(parents=True, exist_ok=True)
        out.to_parquet(d / "part-00000.parquet", index=False)
    (vdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


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


def safe_auc(y, s, kind):
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, s) if kind == "roc" else average_precision_score(y, s))


def train_models(xtr, ytr, xval, yval, random_state):
    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000, random_state=random_state).fit(xtr, ytr),
        "random_forest": RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=1, class_weight="balanced_subsample", random_state=random_state, n_jobs=-1).fit(xtr, ytr),
    }
    spw = int((ytr == 0).sum()) / max(int(ytr.sum()), 1)
    best, best_pr = None, -1
    for params in [{"max_depth": 2, "learning_rate": 0.08, "n_estimators": 80}, {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 120}]:
        m = XGBClassifier(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", scale_pos_weight=spw, random_state=random_state, **params)
        m.fit(xtr, ytr, verbose=False)
        pr = safe_auc(yval, m.predict_proba(xval)[:, 1], "pr")
        if pr > best_pr:
            best, best_pr = m, pr
    models["xgboost"] = best
    return models


def main():
    args = parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    variant_rows = []
    for name, cfg in VARIANTS.items():
        meta = build_variant(args, name, cfg)
        variant_rows.append({"variant": name, **cfg})
    pd.DataFrame(variant_rows).to_csv(args.results_dir / "event_window_variant_map.csv", index=False)
    (args.results_dir / "event_window_design_report.txt").write_text(json.dumps(VARIANTS, indent=2), encoding="utf-8")
    metrics, thresholds, per_file, per_event = [], [], [], []
    best_plot = None
    best_f1 = -1
    for name in VARIANTS:
        base = args.processed_dir / name
        train = pd.read_parquet(base / "train" / "part-00000.parquet")
        val = pd.read_parquet(base / "val" / "part-00000.parquet")
        test = pd.read_parquet(base / "test" / "part-00000.parquet")
        ytr, yv, yt = [df["label"].to_numpy(dtype=np.int8) for df in [train, val, test]]
        xtr, xv, xt = [df[FEATURES] for df in [train, val, test]]
        models = train_models(xtr, ytr, xv, yv, args.random_state)
        for mn, model in models.items():
            vs, ts = model.predict_proba(xv)[:, 1], model.predict_proba(xt)[:, 1]
            aucs = {"val_auroc": safe_auc(yv, vs, "roc"), "val_pr_auc": safe_auc(yv, vs, "pr"), "test_auroc": safe_auc(yt, ts, "roc"), "test_pr_auc": safe_auc(yt, ts, "pr")}
            for pol, th in choose_thresholds(yv, vs, args.fpr_limit).items():
                vm, tm = eval_at(yv, vs, th), eval_at(yt, ts, th)
                thresholds.append({"variant": name, "model": mn, "policy": pol, "threshold": th, **{f"val_{k}": v for k, v in vm.items() if k != "threshold"}, **{f"test_{k}": v for k, v in tm.items() if k != "threshold"}})
                for split, met in [("val", vm), ("test", tm)]:
                    metrics.append({"variant": name, "model": mn, "split": split, "threshold_policy": pol, **aucs, **met})
                if pol == "val_best_f1":
                    if tm["f1"] > best_f1:
                        best_f1 = tm["f1"]
                        best_plot = (name, mn, ts, th, yt)
                    tmp = test[["source_file", "file_id", "event_id", "window_kind", "rogue_frames", "label"]].copy()
                    tmp["score"] = ts
                    for sf, g in tmp.groupby("source_file", sort=True):
                        gy, gs = g.label.to_numpy(dtype=np.int8), g.score.to_numpy()
                        per_file.append({"variant": name, "model": mn, "source_file": sf, "windows": len(g), "positive_windows": int(gy.sum()), "rogue_frames": int(g.rogue_frames.sum()), "auroc": safe_auc(gy, gs, "roc"), "pr_auc": safe_auc(gy, gs, "pr"), **eval_at(gy, gs, th)})
                    for _, row in tmp.iterrows():
                        per_event.append({"variant": name, "model": mn, "source_file": row.source_file, "event_id": int(row.event_id), "window_kind": row.window_kind, "label": int(row.label), "rogue_frames": int(row.rogue_frames), "score": float(row.score), "pred": int(row.score >= th)})
    pd.DataFrame(metrics).to_csv(args.results_dir / "metrics_by_variant.csv", index=False)
    pd.DataFrame(thresholds).to_csv(args.results_dir / "threshold_analysis.csv", index=False)
    pd.DataFrame(per_file).to_csv(args.results_dir / "per_file_metrics.csv", index=False)
    pd.DataFrame(per_event).to_csv(args.results_dir / "per_event_metrics.csv", index=False)
    if best_plot:
        name, mn, scores, th, yt = best_plot
        cm = confusion_matrix(yt, (scores >= th).astype(np.int8), labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name}/{mn}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(args.results_dir / "confusion_matrix.png", dpi=180)
        plt.close(fig)
    met = pd.DataFrame(metrics)
    lines = ["Event-centered windowing summary", "", "Val-best-F1 test results:"]
    for _, r in met[(met.split == "test") & (met.threshold_policy == "val_best_f1")].sort_values(["variant", "f1"], ascending=[True, False]).iterrows():
        lines.append(f"- {r.variant}/{r.model}: precision={r.precision:.4f}, recall={r.recall:.4f}, f1={r.f1:.4f}, fpr={r.fpr:.4f}, auroc={r.test_auroc:.4f}, pr_auc={r.test_pr_auc:.4f}")
    lines += ["", "Event windows reduce dilution by construction; negative sampling remains a key source of evaluation bias to audit per file."]
    (args.results_dir / "short_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
