"""Experiment B: predict the interval directly with quantile regression.

The conformal band bolted onto a point model has one width for every task. A
quantile model can widen the band for tasks it finds genuinely ambiguous and
narrow it where the text is informative -- which is what a sprint planner
actually wants.

Scored on coverage (does the band contain the truth as often as it claims),
median width, and pinball loss.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
import features as F  # noqa: E402

pd.set_option("display.width", 240)
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


def pinball(y, pred, q):
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def main():
    emb, pos = F.load_embeddings_and_pos()
    for split_col in ["split", "split_project"]:
        tr, va, te = E.load(split_col)
        print(f"\n{'='*78}\nsplit={split_col}\n{'='*78}")
        fb = F.FeatureBuilder(use_project=False, use_numeric=True)
        Xtr, Xva, Xte = fb.fit_transform(tr, [va, te], emb, pos)
        ytr = np.log(tr.effort_hours.to_numpy())
        y_te = te.effort_hours.to_numpy()

        preds = {}
        for q in QUANTILES:
            m = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=300,
                learning_rate=0.06, max_depth=6, random_state=0)
            m.fit(Xtr, ytr)
            preds[q] = m.predict(Xte)
            print(f"  q{int(q*100):02d} entrenado  pinball {pinball(np.log(y_te), preds[q], q):.4f}")

        # point estimate from the median quantile
        m_pt = E.point_metrics(y_te, np.exp(preds[0.50]))
        print(f"\n  estimacion puntual (q50): MdMRE {m_pt['MdMRE']:.4f}  "
              f"PRED25 {m_pt['PRED25']:.4f}")

        rows = []
        for lo_q, hi_q, nominal in [(0.25, 0.75, 0.50), (0.10, 0.90, 0.80)]:
            lo, hi = np.exp(preds[lo_q]), np.exp(preds[hi_q])
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
            cov = float(np.mean((y_te >= lo) & (y_te <= hi)))
            rows.append(dict(nominal=nominal, cobertura=round(cov, 4),
                             ancho_mediano_h=round(float(np.median(hi - lo)), 2),
                             factor_mediano=round(float(np.median(hi / np.maximum(lo, .01))), 2),
                             ancho_p10=round(float(np.quantile(hi - lo, .1)), 2),
                             ancho_p90=round(float(np.quantile(hi - lo, .9)), 2)))
        r = pd.DataFrame(rows)
        print(f"\n  intervalos cuantilicos:\n{r.to_string(index=False)}")
        print("  (ancho_p10 vs ancho_p90 muestra si el modelo adapta la banda "
              "a la dificultad de la tarea)")

        m_pt["cov50"] = rows[0]["cobertura"]
        m_pt["cov80"] = rows[1]["cobertura"]
        m_pt["width80_h"] = rows[1]["ancho_mediano_h"]
        m_pt["factor80"] = rows[1]["factor_mediano"]
        m_pt["width80_p10"] = rows[1]["ancho_p10"]
        m_pt["width80_p90"] = rows[1]["ancho_p90"]
        E.show(E.record("quantile", "hgb", split_col, m_pt,
                        config={"quantiles": QUANTILES}))


if __name__ == "__main__":
    main()
