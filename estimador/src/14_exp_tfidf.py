"""Arm A: TF-IDF. The reference arm, preserved for the final comparison.

Runs the full evaluation on both splits and writes the results to the registry
so the SBERT arm can be compared against numbers that were actually recorded,
not remembered.
"""
import sys

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402


def features(tr, others, use_meta=True):
    vec = TfidfVectorizer(min_df=2, max_features=50000, ngram_range=(1, 2),
                          sublinear_tf=True, strip_accents="unicode")
    Xtr = vec.fit_transform(tr.text)
    mats = [Xtr] + [vec.transform(d.text) for d in others]
    if use_meta:
        enc = OneHotEncoder(handle_unknown="ignore")
        Ktr = enc.fit_transform(tr[["task_kind", "source"]])
        mats[0] = sparse.hstack([mats[0], Ktr]).tocsr()
        for i, d in enumerate(others, start=1):
            mats[i] = sparse.hstack(
                [mats[i], enc.transform(d[["task_kind", "source"]])]).tocsr()
    return mats


def run(split_col):
    tr, va, te = E.load(split_col)
    print(f"\n=== split={split_col}  train {len(tr):,} val {len(va):,} test {len(te):,} ===")
    Xtr, Xva, Xte = features(tr, [va, te])
    ytr = np.log(tr.effort_hours.to_numpy())
    y_te = te.effort_hours.to_numpy()
    y_va = va.effort_hours.to_numpy()

    # baseline that ignores every feature
    med = float(tr.effort_hours.median())
    m = E.point_metrics(y_te, np.full(len(te), med))
    m.update(E.interval_metrics(np.full(len(va), np.log(med)), y_va,
                                np.full(len(te), np.log(med)), y_te))
    E.show(E.record("baseline", "median", split_col, m,
                    notes="predice siempre la mediana de entrenamiento"))

    for name, model in [("ridge", Ridge(alpha=1.0)),
                        ("knn", KNeighborsRegressor(n_neighbors=10,
                                                    metric="cosine",
                                                    weights="distance"))]:
        model.fit(Xtr, ytr)
        lp_te, lp_va = model.predict(Xte), model.predict(Xva)
        m = E.point_metrics(y_te, np.exp(lp_te))
        m.update(E.interval_metrics(lp_va, y_va, lp_te, y_te))
        E.show(E.record("tfidf", name, split_col, m,
                        config={"max_features": 50000, "ngram": "1-2",
                                "meta_features": ["task_kind", "source"]}))


if __name__ == "__main__":
    for sc in ["split", "split_project"]:
        run(sc)
    print("\n--- registro completo ---")
    print(E.table().to_string(index=False))
