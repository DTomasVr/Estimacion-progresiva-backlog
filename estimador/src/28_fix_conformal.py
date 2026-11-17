"""Fix 1: make the interval honest across task sizes.

The global conformal band holds at 80% on average but only ~5% for tasks over
40 h -- it is calibrated on a population dominated by small tasks. Two
conditional variants, both using only information available at prediction time:

  mondrian    a separate conformal quantile per bin of the PREDICTED effort
  normalizado residuals divided by a per-task difficulty estimate, so the band
              widens where the model expects to be wrong

Conditioning on the true size would be cheating; a planner does not know it.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
import features as F  # noqa: E402

pd.set_option("display.width", 250)
E.DATA = "data/processed/pbe_effort_v2.parquet"
LEVEL = 0.80
BANDS = [0, 1, 4, 16, 40, np.inf]
NAMES = ["<1h", "1-4h", "4-16h", "16-40h", ">40h"]


def coverage_table(y, lo, hi, label):
    b = pd.cut(y, BANDS, labels=NAMES)
    d = pd.DataFrame({"y": y, "ok": (y >= lo) & (y <= hi),
                      "ancho": hi - lo, "banda": b})
    g = d.groupby("banda", observed=True).agg(
        n=("ok", "size"), cobertura=("ok", "mean"),
        ancho_mediano=("ancho", "median"))
    g.columns = pd.MultiIndex.from_product([[label], g.columns])
    return g


def main():
    emb, pos = F.load_embeddings_and_pos()
    for split_col in ["split", "split_project"]:
        tr, va, te = E.load(split_col)
        fb = F.FeatureBuilder(use_kind=True, use_source=True, use_priority=False,
                              use_numeric=False, use_project=False)
        Xtr, Xva, Xte = fb.fit_transform(tr, [va, te], emb, pos)
        ytr = np.log(tr.effort_hours.to_numpy())
        y_va, y_te = va.effort_hours.to_numpy(), te.effort_hours.to_numpy()
        mdl = Ridge(alpha=1.0).fit(Xtr, ytr)
        lp_va, lp_te = mdl.predict(Xva), mdl.predict(Xte)
        r_va = np.abs(np.log(y_va) - lp_va)
        print(f"\n{'='*78}\nsplit={split_col}   test {len(te):,}\n{'='*78}")

        # ---------- global (what we report today) ----------
        q = float(np.quantile(r_va, LEVEL))
        g_lo, g_hi = np.exp(lp_te - q), np.exp(lp_te + q)

        # ---------- mondrian: one quantile per predicted-size bin ----------
        edges = np.quantile(lp_va, [0, .25, .50, .75, 1.0])
        edges[0], edges[-1] = -np.inf, np.inf
        bin_va = np.digitize(lp_va, edges[1:-1])
        bin_te = np.digitize(lp_te, edges[1:-1])
        m_lo, m_hi = np.empty(len(te)), np.empty(len(te))
        print("  cuantil conforme por cubo de prediccion:")
        for b in range(4):
            res = r_va[bin_va == b]
            qb = float(np.quantile(res, LEVEL)) if len(res) > 20 else q
            sel = bin_te == b
            m_lo[sel], m_hi[sel] = np.exp(lp_te[sel] - qb), np.exp(lp_te[sel] + qb)
            print(f"    cubo {b}: n_val={int((bin_va==b).sum()):4d}  "
                  f"factor {np.exp(qb):5.2f}x  n_test={int(sel.sum()):4d}")

        # ---------- normalised: scale by predicted difficulty ----------
        r_tr_model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=4, learning_rate=0.06, random_state=0)
        lp_tr = mdl.predict(Xtr)
        r_tr_model.fit(Xtr, np.abs(ytr - lp_tr))
        s_va = np.clip(r_tr_model.predict(Xva), 0.05, None)
        s_te = np.clip(r_tr_model.predict(Xte), 0.05, None)
        qn = float(np.quantile(r_va / s_va, LEVEL))
        n_lo, n_hi = np.exp(lp_te - qn * s_te), np.exp(lp_te + qn * s_te)
        print(f"  normalizado: cuantil {qn:.3f} sobre residuo/dificultad; "
              f"factor mediano {np.exp(qn*np.median(s_te)):.2f}x")

        tab = pd.concat([coverage_table(y_te, g_lo, g_hi, "global"),
                         coverage_table(y_te, m_lo, m_hi, "mondrian"),
                         coverage_table(y_te, n_lo, n_hi, "normalizado")], axis=1)
        print(f"\n  cobertura por tamano real (nominal {LEVEL:.0%}):")
        print(tab.round(3).to_string())

        for name, lo, hi in [("global", g_lo, g_hi), ("mondrian", m_lo, m_hi),
                             ("normalizado", n_lo, n_hi)]:
            cov = float(np.mean((y_te >= lo) & (y_te <= hi)))
            b = pd.cut(y_te, BANDS, labels=NAMES)
            per = pd.Series((y_te >= lo) & (y_te <= hi)).groupby(
                b, observed=True).mean()
            peor = float(per.min())
            E.record("conformal", name, split_col,
                     {"cov80": cov, "cov80_peor_banda": peor,
                      "desviacion_max": float(np.max(np.abs(per - LEVEL))),
                      "ancho_mediano_h": float(np.median(hi - lo))})
            print(f"  {name:12s} cobertura global {cov:.3f}  "
                  f"peor banda {peor:.3f}  desviacion max {np.max(np.abs(per-LEVEL)):.3f}")


if __name__ == "__main__":
    main()
