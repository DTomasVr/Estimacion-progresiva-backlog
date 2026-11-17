"""Fix 2: undo the shrinkage and measure what it costs.

Under squared loss the optimal prediction shrinks toward the mean, so the model
uses only ~45% of the true spread and never predicts a large task. Expanding
the predictions around their own centre restores realistic magnitudes at the
cost of average error. This measures that exchange rate so the choice can be
made deliberately rather than by default.

  pred' = centro + k * (pred - centro)

k = 1 is the model as trained; k = 1/compresion restores the full spread.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
import features as F  # noqa: E402

pd.set_option("display.width", 250)
E.DATA = "data/processed/pbe_effort_v2.parquet"


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

        centro = float(lp_va.mean())
        comp0 = float(np.std(lp_te) / np.std(np.log(y_te)))
        k_full = 1.0 / comp0
        print(f"\n{'='*78}\nsplit={split_col}  compresion inicial {comp0:.3f}  "
              f"k para restaurar la dispersion: {k_full:.2f}\n{'='*78}")

        rows = []
        for k in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, round(k_full, 2)]:
            p = np.exp(centro + k * (lp_te - centro))
            m = E.point_metrics(y_te, p)
            comp = float(np.std(centro + k * (lp_te - centro)) / np.std(np.log(y_te)))
            big = int(((y_te > 16) & (p > 16)).sum())
            big_tot = int((y_te > 16).sum())
            fp = int(((y_te <= 16) & (p > 16)).sum())
            rows.append(dict(k=k, MdMRE=round(m["MdMRE"], 4),
                             PRED25=round(m["PRED25"], 4),
                             RMSE_log=round(m["RMSE_log"], 4),
                             compresion=round(comp, 3),
                             grandes_detectadas=f"{big}/{big_tot}",
                             falsos_positivos=fp,
                             pred_max_h=round(float(p.max()), 1)))
            E.record("deshrink", f"k{k}", split_col,
                     {**m, "compresion": comp, "grandes_detectadas": big,
                      "falsos_positivos": fp})
        print(pd.DataFrame(rows).to_string(index=False))
        print("  'grandes_detectadas' = tareas reales > 16 h que el modelo "
              "situa por encima de 16 h")


if __name__ == "__main__":
    main()
