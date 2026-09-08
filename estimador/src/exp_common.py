"""Shared evaluation harness so every experiment arm is scored identically.

A thesis comparison is only meaningful if the arms differ in exactly one thing.
This module fixes everything else: the data, the splits, the metrics, the
conformal calibration procedure, and the results registry. An arm supplies only
a feature representation.

Every record stores the SHA-256 of the dataset file, so a result can always be
traced to the exact data version it was produced from.
"""
import hashlib
import json
import os
import platform
import time

import numpy as np
import pandas as pd

DATA = "data/processed/agix_effort_clean.parquet"
REGISTRY = "results/experiments.json"
LEVELS = [0.50, 0.80, 0.90]


def data_fingerprint(path=DATA):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load(split_col="split"):
    """Return (train, val, test) for the requested split column.

    `split_project` has no validation slice of its own, so we carve a
    deterministic 15% calibration slice out of its training side.
    """
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    if split_col == "split":
        return (sw[sw.split == "train"], sw[sw.split == "val"],
                sw[sw.split == "test"])
    tr_all = sw[sw.split_project == "train"]
    te = sw[sw.split_project == "test"]
    rng = np.random.default_rng(7)
    m = rng.random(len(tr_all)) < 0.15
    return tr_all[~m], tr_all[m], te


def point_metrics(y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred, float), 0.01, None)
    y_true = np.asarray(y_true, float)
    ae = np.abs(y_true - y_pred)
    mre = ae / y_true
    return {
        "MAE_h": float(np.mean(ae)),
        "MdAE_h": float(np.median(ae)),
        "MdMRE": float(np.median(mre)),
        "PRED25": float(np.mean(mre <= .25)),
        "PRED50": float(np.mean(mre <= .50)),
        "RMSE_log": float(np.sqrt(np.mean((np.log(y_true) - np.log(y_pred)) ** 2))),
    }


def interval_metrics(log_pred_val, y_val, log_pred_test, y_test, levels=LEVELS):
    """Split-conformal bands calibrated on val, verified on test."""
    res = np.abs(np.log(np.asarray(y_val, float)) - np.asarray(log_pred_val, float))
    out = {}
    for lv in levels:
        q = float(np.quantile(res, lv))
        lo = np.exp(np.asarray(log_pred_test, float) - q)
        hi = np.exp(np.asarray(log_pred_test, float) + q)
        y = np.asarray(y_test, float)
        out[f"cov{int(lv*100)}"] = float(np.mean((y >= lo) & (y <= hi)))
        out[f"factor{int(lv*100)}"] = float(np.exp(q))
        out[f"width{int(lv*100)}_h"] = float(np.median(hi - lo))
    return out


def record(arm, variant, split_col, metrics, config=None, notes=""):
    os.makedirs("results", exist_ok=True)
    reg = []
    if os.path.exists(REGISTRY):
        with open(REGISTRY, encoding="utf8") as fh:
            reg = json.load(fh)
    entry = {
        "arm": arm,
        "variant": variant,
        "split": split_col,
        "data_sha256": data_fingerprint(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "config": config or {},
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "notes": notes,
    }
    # replace an earlier run of the same arm/variant/split rather than duplicate
    reg = [r for r in reg
           if not (r["arm"] == arm and r["variant"] == variant
                   and r["split"] == split_col)]
    reg.append(entry)
    with open(REGISTRY, "w", encoding="utf8") as fh:
        json.dump(reg, fh, indent=2, ensure_ascii=False)
    return entry


def show(entry):
    m = entry["metrics"]
    print(f"  [{entry['arm']}/{entry['variant']}  split={entry['split']}]")
    print(f"    MdMRE {m['MdMRE']:.3f}  PRED25 {m['PRED25']:.3f}  "
          f"PRED50 {m['PRED50']:.3f}  MdAE {m['MdAE_h']:.2f}h  "
          f"RMSElog {m['RMSE_log']:.3f}")
    if "cov80" in m:
        cov90 = f"{m['cov90']:.3f}" if "cov90" in m else "--"
        print(f"    cobertura 50/80/90: {m.get('cov50', float('nan')):.3f} / "
              f"{m['cov80']:.3f} / {cov90}   factor80 {m['factor80']:.2f}x")


def table():
    with open(REGISTRY, encoding="utf8") as fh:
        reg = json.load(fh)
    rows = []
    for r in reg:
        rows.append(dict(arm=r["arm"], variant=r["variant"], split=r["split"],
                         **r["metrics"]))
    return pd.DataFrame(rows)
