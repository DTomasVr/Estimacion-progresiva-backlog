"""Fix 3: ask the question the data can answer.

Instead of "how many hours", ask "what is the probability this task exceeds N
hours". Ranking is where the remaining signal lives (Spearman ~0.45 against a
point-estimate MdMRE of 0.66), and a probability is directly usable in sprint
planning: it flags the items likely to overrun.

Reported with AUC, average precision against the base rate, and the precision
and recall a planner would actually get at a usable operating point.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
import features as F  # noqa: E402

pd.set_option("display.width", 250)
E.DATA = "data/processed/agix_effort_v2.parquet"
THRESHOLDS = [4, 8, 16, 40]


def operating_point(y, p, target_recall=0.60):
    prec, rec, thr = precision_recall_curve(y, p)
    ok = rec[:-1] >= target_recall
    if not ok.any():
        return np.nan, np.nan, np.nan
    i = int(np.argmax(prec[:-1] * ok))
    return float(prec[i]), float(rec[i]), float(thr[i])


def main():
    emb, pos = F.load_embeddings_and_pos()
    for split_col in ["split", "split_project"]:
        tr, va, te = E.load(split_col)
        fb = F.FeatureBuilder(use_kind=True, use_source=True, use_priority=False,
                              use_numeric=False, use_project=False)
        Xtr, Xva, Xte = fb.fit_transform(tr, [va, te], emb, pos)
        print(f"\n{'='*78}\nsplit={split_col}   test {len(te):,}\n{'='*78}")
        rows = []
        for thr_h in THRESHOLDS:
            ytr = (tr.effort_hours > thr_h).astype(int).to_numpy()
            yte = (te.effort_hours > thr_h).astype(int).to_numpy()
            base = float(yte.mean())
            if ytr.sum() < 50 or yte.sum() < 20:
                continue
            for name, mk in [
                ("logreg", lambda: LogisticRegression(max_iter=3000, C=1.0)),
                ("hgb", lambda: HistGradientBoostingClassifier(
                    max_iter=300, max_depth=5, learning_rate=0.06,
                    random_state=0)),
            ]:
                clf = mk().fit(Xtr, ytr)
                p = clf.predict_proba(Xte)[:, 1]
                auc = roc_auc_score(yte, p)
                ap = average_precision_score(yte, p)
                brier = brier_score_loss(yte, p)
                prec, rec, cut = operating_point(yte, p)
                rows.append(dict(umbral=f">{thr_h}h", modelo=name,
                                 tasa_base=round(base, 3),
                                 AUC=round(auc, 3), AP=round(ap, 3),
                                 lift_AP=round(ap / base, 2),
                                 Brier=round(brier, 4),
                                 precision=round(prec, 3) if prec == prec else None,
                                 recall=round(rec, 3) if rec == rec else None))
                E.record("riesgo", f"{name}_{thr_h}h", split_col,
                         {"AUC": float(auc), "AP": float(ap),
                          "tasa_base": base, "lift_AP": float(ap / base),
                          "brier": float(brier),
                          "precision_at_r60": float(prec) if prec == prec else 0.0})
        r = pd.DataFrame(rows)
        print(r.to_string(index=False))
        print("\n  AUC 0,5 = azar.  lift_AP = precision media frente a la tasa base.")
        print("  precision/recall en el punto de operacion que alcanza recall 0,60.")


if __name__ == "__main__":
    main()
