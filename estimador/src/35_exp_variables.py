"""Retrain with the variables the audit found we had never used.

The estimate is the dominant signal, and its bias differs per developer
(0.89x to 1.95x across the 23 people in SiP). This tests whether knowing WHO
estimated lets us correct it -- which is precisely what PBE can learn from a
team's own history.

Conditions, all on the same held-out tasks:
  humano          the raw estimate
  por_dev         estimate corrected by that developer's learned bias
  est_dev         estimate + developer identity, fitted jointly
  est_dev_wbs     ... plus the work-breakdown code
  est_dev_texto   ... plus the text embedding
Excluded as leakage: TaskPerformance and DeveloperPerformance, both derived
from the actual hours.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
import features as F  # noqa: E402

pd.set_option("display.width", 250)
E.DATA = "data/processed/pbe_effort_v2.parquet"


def load():
    t = pd.read_csv("data/raw/jones/SiP/Sip-task-info.csv", encoding="cp1252")
    t = t.drop_duplicates("TaskNumber")
    t["task_id"] = "SIP-" + t.TaskNumber.astype(str)
    keep = ["task_id", "HoursEstimate", "AssignedToID", "RaisedByID",
            "ProjectBreakdownCode"]
    df = pd.read_parquet(E.DATA)
    sw = df[(df.is_software_task == 1) & (df.source == "SiP")]
    sw = sw.merge(t[keep], on="task_id", how="left")
    return sw[sw.HoursEstimate.notna() & (sw.HoursEstimate > 0)].copy()


def score(y, p, label, rows, split_col):
    m = E.point_metrics(y, p)
    comp = float(np.std(np.log(np.clip(p, .01, None))) / np.std(np.log(y)))
    sp = float(pd.Series(y).corr(pd.Series(p), method="spearman"))
    big = int(((y > 16) & (p > 16)).sum())
    rows.append(dict(modelo=label, MdMRE=round(m["MdMRE"], 4),
                     PRED25=round(m["PRED25"], 4), PRED50=round(m["PRED50"], 4),
                     RMSE_log=round(m["RMSE_log"], 4),
                     compresion=round(comp, 3), spearman=round(sp, 3),
                     grandes=f"{big}/{int((y>16).sum())}"))
    E.record("variables", label, split_col,
             {**m, "compresion": comp, "spearman": sp})


def main():
    sw = load()
    print(f"tareas SiP con estimacion: {len(sw):,}   "
          f"desarrolladores: {sw.AssignedToID.nunique()}")
    emb, pos = F.load_embeddings_and_pos()

    for split_col in ["split", "split_project"]:
        tr = sw[sw[split_col] == "train"]
        te = sw[sw[split_col] == "test"]
        if len(te) < 100:
            continue
        print(f"\n{'='*78}\nsplit={split_col}  train {len(tr):,}  test {len(te):,}"
              f"\n{'='*78}")
        y_tr = np.log(tr.effort_hours.to_numpy())
        y_te = te.effort_hours.to_numpy()
        e_tr = np.log(tr.HoursEstimate.to_numpy())
        e_te = np.log(te.HoursEstimate.to_numpy())
        rows = []

        score(y_te, te.HoursEstimate.to_numpy(), "humano", rows, split_col)

        # per-developer additive correction in log space
        bias = (pd.DataFrame({"d": tr.AssignedToID, "r": y_tr - e_tr})
                .groupby("d").r.median())
        glob = float(np.median(y_tr - e_tr))
        adj = te.AssignedToID.map(bias).fillna(glob).to_numpy()
        score(y_te, np.exp(e_te + adj), "por_dev", rows, split_col)
        print(f"  correccion global: {np.exp(glob):.3f}x   "
              f"por desarrollador de {np.exp(bias.min()):.2f}x a "
              f"{np.exp(bias.max()):.2f}x")

        # joint fits
        def cats(cols):
            oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            return oh.fit_transform(tr[cols]), oh.transform(te[cols])

        d_tr, d_te = cats(["AssignedToID"])
        w_tr, w_te = cats(["AssignedToID", "ProjectBreakdownCode"])
        fb = F.FeatureBuilder(use_kind=True, use_source=False, use_priority=False,
                              use_numeric=False, use_project=False)
        T_tr, T_te = fb.fit_transform(tr, [te], emb, pos)

        blocks = {
            "est_dev": (np.hstack([e_tr[:, None], d_tr]),
                        np.hstack([e_te[:, None], d_te])),
            "est_dev_wbs": (np.hstack([e_tr[:, None], w_tr]),
                            np.hstack([e_te[:, None], w_te])),
            "est_dev_texto": (np.hstack([e_tr[:, None], d_tr, T_tr]),
                              np.hstack([e_te[:, None], d_te, T_te])),
        }
        for name, (A, B) in blocks.items():
            score(y_te, np.exp(Ridge(alpha=1.0).fit(A, y_tr).predict(B)),
                  name, rows, split_col)
        A, B = blocks["est_dev_wbs"]
        h = HistGradientBoostingRegressor(max_iter=400, max_depth=6,
                                          learning_rate=0.06, random_state=0)
        score(y_te, np.exp(h.fit(A, y_tr).predict(B)), "est_dev_wbs_hgb",
              rows, split_col)

        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
