"""Full metric panel on the single test evaluation already performed.

No selection, no tuning: the same frozen configuration, the same test block,
simply scored on every metric the effort-estimation literature reports.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

pd.set_option("display.width", 250)
DATA = "data/processed/pbe_effort_v3.parquet"
NUEVAS = ["n_links", "n_links_estructurales", "n_componentes", "en_sprint",
          "tiene_asignado", "Title_Changed_After_Estimation",
          "Story_Point_Changed_After_Estimation"]
CATS = ["task_kind", "source", "desarrollador"]


def panel(y, p):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), .01, None)
    ae = np.abs(y - p); mre = ae / y
    return {
        "MMRE": float(np.mean(mre)), "MdMRE": float(np.median(mre)),
        "PRED25": float(np.mean(mre <= .25)), "PRED50": float(np.mean(mre <= .50)),
        "MAE_h": float(np.mean(ae)), "MdAE_h": float(np.median(ae)),
        "RMSE_h": float(np.sqrt(np.mean((y - p) ** 2))),
        "RMSE_log": float(np.sqrt(np.mean((np.log(y) - np.log(p)) ** 2))),
        "MAR_log": float(np.mean(np.abs(np.log(y) - np.log(p)))),
        "Spearman": float(pd.Series(y).corr(pd.Series(p), method="spearman")),
    }


def feats(tr, otros, emb, pos):
    rows = lambda d: emb[[pos[t] for t in d.task_id]]
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)
    sc = StandardScaler()
    out = [np.hstack([rows(tr), oh.fit_transform(tr[CATS].astype(str)),
                      sc.fit_transform(np.log1p(tr[NUEVAS].to_numpy(float)))])]
    for d in otros:
        out.append(np.hstack([rows(d), oh.transform(d[CATS].astype(str)),
                              sc.transform(np.log1p(d[NUEVAS].to_numpy(float)))]))
    return out


def main():
    alpha = json.load(open("results/seleccion_val.json", encoding="utf8"))["ridge_alpha"]
    df = pd.read_parquet(DATA); sw = df[df.is_software_task == 1].copy()
    emb = np.load("data/processed/embeddings/paraphrase-multilingual-mpnet-base-v2.npy")
    base = pd.read_parquet("data/processed/pbe_effort_clean.parquet",
                           columns=["task_id"]).task_id.to_numpy()
    pos = {t: i for i, t in enumerate(base)}

    ys, ps, bs = [], [], []
    for proj in sorted(sw[sw.split_t3 == "test"].project.unique()):
        d = sw[sw.project == proj]
        tr, va, te = (d[d.split_t3 == s] for s in ("train", "val", "test"))
        if len(tr) < 50 or len(te) < 12 or len(va) < 8:
            continue
        Xtr, Xva, Xte = feats(tr, [va, te], emb, pos)
        m = Ridge(alpha=alpha).fit(Xtr, np.log(tr.effort_hours.to_numpy()))
        ys.append(te.effort_hours.to_numpy())
        ps.append(np.exp(m.predict(Xte)))
        bs.append(np.full(len(te), float(tr.effort_hours.median())))
    y, p, b = np.concatenate(ys), np.concatenate(ps), np.concatenate(bs)

    est = sw[sw.tiene_estimacion == 1]
    te_e = est[est.split_t3 == "test"]
    ye, pe = te_e.effort_hours.to_numpy(), te_e.HoursEstimate.to_numpy()

    tab = pd.DataFrame({
        "linea base (mediana)": panel(y, b),
        "modelo, arranque frio": panel(y, p),
        "estimacion humana": panel(ye, pe),
    }).T.round(4)
    print(f"tareas de prueba: frio {len(y):,} | con estimacion {len(ye):,}\n")
    print(tab.to_string())
    tab.to_csv("results/panel_metricas.csv")
    print("\nescrito results/panel_metricas.csv")


if __name__ == "__main__":
    main()
