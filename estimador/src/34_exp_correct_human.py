"""Correct the developer's estimate instead of replacing it.

The expert estimate reaches MdMRE 0.27 where the text model reaches 0.66, and
carries no shrinkage. But it is biased in a systematic, measurable way:
small tasks inflated ~2x, large tasks halved. That is a correctable error.

Four conditions, all evaluated on the same held-out tasks:
  humano       the raw estimate, untouched (the baseline Agix must beat)
  recalibrado  a monotone recalibration of the estimate alone
  texto        our text model, no estimate (what we have been building)
  hibrido      estimate + text embedding together
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
E.DATA = "data/processed/agix_effort_v2.parquet"
BANDS = [0, 1, 4, 16, 40, np.inf]
NAMES = ["<1h", "1-4h", "4-16h", "16-40h", ">40h"]


def load_estimates():
    t = pd.read_csv("data/raw/jones/SiP/Sip-task-info.csv", encoding="cp1252")
    t = t.drop_duplicates("TaskNumber")
    t = t[t.HoursEstimate > 0]
    t["task_id"] = "SIP-" + t.TaskNumber.astype(str)
    return t.set_index("task_id").HoursEstimate


def report(y, p, label, rows):
    m = E.point_metrics(y, p)
    comp = float(np.std(np.log(np.clip(p, .01, None))) / np.std(np.log(y)))
    sp = float(pd.Series(y).corr(pd.Series(p), method="spearman"))
    big = int(((y > 16) & (p > 16)).sum())
    big_tot = int((y > 16).sum())
    rows.append(dict(modelo=label, MdMRE=round(m["MdMRE"], 4),
                     PRED25=round(m["PRED25"], 4), PRED50=round(m["PRED50"], 4),
                     RMSE_log=round(m["RMSE_log"], 4), compresion=round(comp, 3),
                     spearman=round(sp, 3),
                     grandes=f"{big}/{big_tot}"))
    return m, comp, sp


def main():
    est = load_estimates()
    df = pd.read_parquet(E.DATA)
    sw = df[(df.is_software_task == 1) & (df.source == "SiP")].copy()
    sw["est"] = sw.task_id.map(est)
    sw = sw[sw.est.notna() & (sw.est > 0)]
    print(f"tareas SiP con estimacion humana: {len(sw):,}")

    emb, pos = F.load_embeddings_and_pos()

    for split_col in ["split", "split_project"]:
        tr = sw[sw[split_col] == "train"]
        te = sw[sw[split_col] == "test"]
        va = sw[sw["split"] == "val"] if split_col == "split" else tr.sample(
            frac=.15, random_state=7)
        if len(te) < 100:
            continue
        print(f"\n{'='*78}\nsplit={split_col}  train {len(tr):,}  test {len(te):,}"
              f"\n{'='*78}")
        y_tr = np.log(tr.effort_hours.to_numpy())
        y_te = te.effort_hours.to_numpy()
        e_tr = np.log(tr.est.to_numpy())
        e_te = np.log(te.est.to_numpy())
        rows = []

        # 1. the raw human estimate
        report(y_te, te.est.to_numpy(), "humano", rows)

        # 2. recalibrate the estimate alone (log-log, captures the bias curve)
        rc = Ridge(alpha=1e-6).fit(e_tr.reshape(-1, 1), y_tr)
        report(y_te, np.exp(rc.predict(e_te.reshape(-1, 1))), "recalibrado", rows)
        print(f"  recalibracion: log(real) = {rc.intercept_:.3f} + "
              f"{rc.coef_[0]:.3f} * log(estimado)")

        # 2b. non-linear recalibration, the estimate is the only input
        rc2 = HistGradientBoostingRegressor(max_iter=200, max_depth=3,
                                            learning_rate=0.06, random_state=0)
        rc2.fit(e_tr.reshape(-1, 1), y_tr)
        report(y_te, np.exp(rc2.predict(e_te.reshape(-1, 1))),
               "recalibrado_nolineal", rows)

        # 3. text only, no estimate
        fb = F.FeatureBuilder(use_kind=True, use_source=False,
                              use_priority=False, use_numeric=False,
                              use_project=False)
        Xtr, Xte = fb.fit_transform(tr, [te], emb, pos)
        txt = Ridge(alpha=1.0).fit(Xtr, y_tr)
        report(y_te, np.exp(txt.predict(Xte)), "texto", rows)

        # 4. hybrid: estimate + text
        Htr = np.hstack([Xtr, e_tr.reshape(-1, 1)])
        Hte = np.hstack([Xte, e_te.reshape(-1, 1)])
        hyb = Ridge(alpha=1.0).fit(Htr, y_tr)
        report(y_te, np.exp(hyb.predict(Hte)), "hibrido_ridge", rows)

        hyb2 = HistGradientBoostingRegressor(max_iter=400, max_depth=6,
                                             learning_rate=0.06, random_state=0)
        hyb2.fit(Htr, y_tr)
        p_h = np.exp(hyb2.predict(Hte))
        report(y_te, p_h, "hibrido_hgb", rows)

        r = pd.DataFrame(rows)
        print(r.to_string(index=False))
        for _, x in r.iterrows():
            E.record("humano", x.modelo, split_col,
                     {"MdMRE": x.MdMRE, "PRED25": x.PRED25, "PRED50": x.PRED50,
                      "RMSE_log": x.RMSE_log, "compresion": x.compresion,
                      "spearman": x.spearman})

        # where does the correction help?
        b = pd.cut(y_te, BANDS, labels=NAMES)
        cmp = pd.DataFrame({
            "real": y_te, "banda": b, "humano": te.est.to_numpy(),
            "hibrido": p_h}).groupby("banda", observed=True).agg(
            n=("real", "size"), real=("real", "median"),
            humano=("humano", "median"), hibrido=("hibrido", "median"))
        cmp["sesgo_humano"] = (cmp.humano / cmp.real).round(2)
        cmp["sesgo_hibrido"] = (cmp.hibrido / cmp.real).round(2)
        print(f"\n  sesgo por banda:\n{cmp.round(2).to_string()}")


if __name__ == "__main__":
    main()
