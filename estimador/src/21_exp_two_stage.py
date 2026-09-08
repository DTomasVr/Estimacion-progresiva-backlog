"""Experiment A: the real two-stage pipeline, with a PREDICTED category.

Every result so far fed the effort model the ground-truth `task_kind`. In
production Agix does not have it -- the category comes from a classifier, and
its errors propagate. This measures the cost of that honestly.

Four conditions for the effort model:
  sin_cat    no category at all           (lower bound)
  cat_pred   the classifier's hard label  (what Agix actually gets)
  cat_prob   the classifier's probability vector (soft, keeps its uncertainty)
  cat_true   the ground-truth label       (optimistic, what we reported before)
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
import features as F  # noqa: E402

pd.set_option("display.width", 240)


def train_classifier(tr, others, emb, pos):
    """text embedding -> task_kind. Returns hard labels and probabilities."""
    Xtr = emb[[pos[t] for t in tr.task_id]]
    clf = LogisticRegression(max_iter=2000, C=4.0, n_jobs=-1)
    clf.fit(Xtr, tr.task_kind)
    out = []
    for d in others:
        X = emb[[pos[t] for t in d.task_id]]
        out.append((clf.predict(X), clf.predict_proba(X)))
    return clf, out


def main():
    emb, pos = F.load_embeddings_and_pos()

    for split_col in ["split", "split_project"]:
        tr, va, te = E.load(split_col)
        print(f"\n{'='*78}\nsplit={split_col}  train {len(tr):,} val {len(va):,} "
              f"test {len(te):,}\n{'='*78}")

        # ---------- stage 1: the categoriser ----------
        clf, preds = train_classifier(tr, [tr, va, te], emb, pos)
        (k_tr, p_tr), (k_va, p_va), (k_te, p_te) = preds
        acc = accuracy_score(te.task_kind, k_te)
        f1 = f1_score(te.task_kind, k_te, average="macro")
        major = te.task_kind.value_counts(normalize=True).iloc[0]
        print(f"\n--- etapa 1: clasificador de categoria ---")
        print(f"  exactitud {acc:.3f}  (clase mayoritaria {major:.3f})  "
              f"F1 macro {f1:.3f}")
        rep = (pd.crosstab(te.task_kind, k_te).pipe(
            lambda c: pd.DataFrame({
                "n": c.sum(axis=1),
                "aciertos": [c.loc[i, i] if i in c.columns else 0 for i in c.index]}))
        )
        rep["recall"] = (rep.aciertos / rep.n).round(3)
        print(rep.sort_values("n", ascending=False).to_string())

        # ---------- stage 2: effort, under four category conditions ----------
        print(f"\n--- etapa 2: efecto sobre la prediccion de esfuerzo ---")
        y_tr = np.log(tr.effort_hours.to_numpy())
        y_va, y_te = va.effort_hours.to_numpy(), te.effort_hours.to_numpy()

        conditions = {
            "sin_cat": dict(use_kind=False, override=None, probs=None),
            "cat_true": dict(use_kind=True, override=None, probs=None),
            "cat_pred": dict(use_kind=True,
                             override={0: k_tr, 1: k_va, 2: k_te}, probs=None),
            "cat_prob": dict(use_kind=False, override=None,
                             probs=(p_tr, p_va, p_te)),
        }

        for name, cfg in conditions.items():
            fb = F.FeatureBuilder(use_kind=cfg["use_kind"])
            Xtr, Xva, Xte = fb.fit_transform(
                tr, [va, te], emb, pos, kind_override=cfg["override"])
            if cfg["probs"] is not None:
                Xtr = np.hstack([Xtr, cfg["probs"][0]])
                Xva = np.hstack([Xva, cfg["probs"][1]])
                Xte = np.hstack([Xte, cfg["probs"][2]])

            for head, mk in [("ridge", lambda: Ridge(alpha=1.0)),
                             ("hgb", lambda: HistGradientBoostingRegressor(
                                 max_iter=400, learning_rate=0.06, max_depth=6,
                                 random_state=0))]:
                mdl = mk().fit(Xtr, y_tr)
                lp_te, lp_va = mdl.predict(Xte), mdl.predict(Xva)
                m = E.point_metrics(y_te, np.exp(lp_te))
                m.update(E.interval_metrics(lp_va, y_va, lp_te, y_te))
                m["clf_accuracy"] = float(acc)
                E.show(E.record("two_stage", f"{head}_{name}", split_col, m,
                                config={"categoria": name}))


if __name__ == "__main__":
    main()
