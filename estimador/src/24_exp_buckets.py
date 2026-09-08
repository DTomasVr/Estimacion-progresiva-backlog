"""Experiment H: effort as ordered buckets instead of a continuous number.

With a median relative error near 0.68, a point estimate in hours is hard to
trust. A sprint planner does not need "4.3 h" -- they need to know whether a
task is a morning, a day, or most of a week. Buckets are a coarser question the
data may actually be able to answer.

Two schemes:
  sprint    fixed, planning-relevant boundaries  (<1h, 1-4h, 4-16h, 16-40h, >40h)
  cuartil   data-driven quartiles                (equal-sized classes)

Reported against the majority-class baseline, and with adjacent accuracy --
being one bucket off is a much smaller planning error than being three off.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
import features as F  # noqa: E402

pd.set_option("display.width", 240)

SPRINT_EDGES = [0, 1, 4, 16, 40, np.inf]
SPRINT_NAMES = ["<1h", "1-4h", "4-16h", "16-40h", ">40h"]


def bucketise(tr, others, scheme):
    if scheme == "sprint":
        edges, names = SPRINT_EDGES, SPRINT_NAMES
    else:
        qs = np.array(tr.effort_hours.quantile([0, .25, .5, .75, 1.0]),
                      dtype=float, copy=True)
        qs[0], qs[-1] = 0, np.inf
        edges = list(dict.fromkeys(qs))
        names = [f"Q{i+1}" for i in range(len(edges) - 1)]
    cut = lambda d: pd.cut(d.effort_hours, bins=edges, labels=names,
                           include_lowest=True).astype(str)
    return cut(tr), [cut(d) for d in others], names


def adjacent_accuracy(y_true, y_pred, names):
    idx = {n: i for i, n in enumerate(names)}
    a = np.array([idx.get(v, -99) for v in y_true])
    b = np.array([idx.get(v, -99) for v in y_pred])
    return float(np.mean(np.abs(a - b) <= 1))


def main():
    emb, pos = F.load_embeddings_and_pos()
    for split_col in ["split", "split_project"]:
        tr, va, te = E.load(split_col)
        print(f"\n{'='*78}\nsplit={split_col}\n{'='*78}")
        fb = F.FeatureBuilder(use_kind=False, use_project=False)
        Xtr, Xva, Xte = fb.fit_transform(tr, [va, te], emb, pos)

        for scheme in ["sprint", "cuartil"]:
            b_tr, (b_va, b_te), names = bucketise(tr, [va, te], scheme)
            dist = b_tr.value_counts(normalize=True)
            major = b_te.value_counts(normalize=True).iloc[0]
            print(f"\n--- esquema {scheme} ---")
            print("  distribucion entrenamiento: " +
                  ", ".join(f"{k} {v:.1%}" for k, v in dist.items()))

            for head, mk in [
                ("logreg", lambda: LogisticRegression(max_iter=2000, C=4.0)),
                ("hgb", lambda: HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.08, max_depth=6,
                    random_state=0)),
            ]:
                clf = mk().fit(Xtr, b_tr)
                pred = clf.predict(Xte)
                acc = accuracy_score(b_te, pred)
                f1 = f1_score(b_te, pred, average="macro")
                adj = adjacent_accuracy(b_te, pred, names)
                print(f"  {head:6s} exactitud {acc:.3f} (mayoritaria {major:.3f})"
                      f"  F1 macro {f1:.3f}  a lo sumo un cubo de error {adj:.3f}")
                E.record("buckets", f"{head}_{scheme}", split_col,
                         {"accuracy": float(acc), "f1_macro": float(f1),
                          "adjacent_acc": float(adj), "majority": float(major),
                          "lift_vs_majority": float(acc - major)},
                         config={"esquema": scheme, "clases": names})

            cm = pd.crosstab(b_te, clf.predict(Xte),
                             rownames=["real"], colnames=["predicho"])
            print(f"\n  matriz de confusion ({head}):\n{cm.to_string()}")


if __name__ == "__main__":
    main()
