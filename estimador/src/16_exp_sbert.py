"""Arm B: SBERT. Same harness, same splits, same metrics -- only the features change.

Three heads on the sentence embeddings:
  ridge  -- the direct counterpart of the TF-IDF ridge
  knn    -- cosine KNN, the approach the thesis describes
  hgb    -- histogram gradient boosting, matching the SBERT + boosted-trees
            combination reported by Yalciner et al. (2024)
"""
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

sys.path.insert(0, "src")
_enc = __import__("15_encode_sbert")
MODEL = _enc.DEFAULT


def features(tr, others, emb, index, use_meta=True):
    """Embedding rows for each frame, optionally with categorical metadata."""
    def rows(d):
        return emb[[index[t] for t in d.task_id]]

    mats = [rows(tr)] + [rows(d) for d in others]
    if use_meta:
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        mats[0] = np.hstack([mats[0], oh.fit_transform(tr[["task_kind", "source"]])])
        for i, d in enumerate(others, start=1):
            mats[i] = np.hstack([mats[i], oh.transform(d[["task_kind", "source"]])])
    return mats


def run(split_col, emb, index, model_name):
    tr, va, te = E.load(split_col)
    print(f"\n=== split={split_col}  train {len(tr):,} val {len(va):,} test {len(te):,} ===")
    Xtr, Xva, Xte = features(tr, [va, te], emb, index)
    ytr = np.log(tr.effort_hours.to_numpy())
    y_te, y_va = te.effort_hours.to_numpy(), va.effort_hours.to_numpy()

    heads = [
        ("ridge", Ridge(alpha=1.0)),
        ("knn", KNeighborsRegressor(n_neighbors=10, metric="cosine",
                                    weights="distance")),
        ("hgb", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                              max_depth=6, random_state=0)),
    ]
    for name, model in heads:
        model.fit(Xtr, ytr)
        lp_te, lp_va = model.predict(Xte), model.predict(Xva)
        m = E.point_metrics(y_te, np.exp(lp_te))
        m.update(E.interval_metrics(lp_va, y_va, lp_te, y_te))
        E.show(E.record("sbert", name, split_col, m,
                        config={"model": model_name, "dim": int(emb.shape[1]),
                                "meta_features": ["task_kind", "source"]}))


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else MODEL
    emb = _enc.load_embeddings(model_name)
    import pandas as pd
    ids = pd.read_parquet(E.DATA, columns=["task_id"]).task_id.to_numpy()
    index = {t: i for i, t in enumerate(ids)}
    print(f"embeddings: {emb.shape}  modelo: {model_name}")

    for sc in ["split", "split_project"]:
        run(sc, emb, index, model_name)

    print("\n--- comparacion completa ---")
    t = E.table()
    print(t.sort_values(["split", "MdMRE"])[
        ["arm", "variant", "split", "MdMRE", "PRED25", "PRED50",
         "RMSE_log", "cov80", "factor80"]].to_string(index=False))
