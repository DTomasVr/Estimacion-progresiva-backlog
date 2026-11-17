"""The four acceptance criteria, evaluated on every regime v4 makes available.

The criteria, as stated for the thesis:
  C1  MAE below 20-30% of the mean actual duration
  C2  RMSE no more than twice the MAE
  C3  MAPE below 25-30%, MdAPE below 15-20%
  C4  PRED(25) at or above 0.75

C1 and C2 are computed in raw hours, because that is the space the criteria are
written in. C3 is reported as both the mean (MAPE) and the median (MdAPE) of the
absolute percentage error, since effort is log-normal and the mean of a ratio on
a log-normal target is dominated by a handful of rows.

Every regime is scored the same way, including the human expert, so the answer
to "does the model meet the bar?" comes with the answer to "does anything?".
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
DATA = "data/processed/pbe_effort_v4.parquet"
EMB = "data/processed/embeddings/v4_mpnet_congelado"
NUM = ["n_links", "n_links_estructurales", "n_componentes", "tiene_asignado"]
CAT = ["task_kind", "source"]


def criterios(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 0.01, None)
    ae = np.abs(y - p)
    ape = 100 * ae / y
    mae = float(np.mean(ae))
    rmse = float(np.sqrt(np.mean(ae ** 2)))
    media = float(np.mean(y))
    return dict(
        n=len(y), media_real=round(media, 2),
        MAE_h=round(mae, 2), pct_MAE=round(100 * mae / media, 1),
        RMSE_h=round(rmse, 2), razon_RMSE_MAE=round(rmse / mae, 2),
        MAPE=round(float(np.mean(ape)), 1), MdAPE=round(float(np.median(ape)), 1),
        PRED25=round(float(np.mean(ape <= 25)), 4),
        PRED50=round(float(np.mean(ape <= 50)), 4),
        C1=100 * mae / media <= 30, C2=rmse / mae <= 2.0,
        C3=float(np.median(ape)) <= 20 and float(np.mean(ape)) <= 30,
        C4=float(np.mean(ape <= 25)) >= 0.75)


def main():
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    emb = np.load(f"{EMB}.npy")
    ids = np.load(f"{EMB}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    sw["fila_emb"] = [pos[t] for t in sw.task_id]

    v3 = sw[sw.source != "JiraPublico"]
    jp = sw[sw.source == "JiraPublico"]
    tr3, te3 = v3[v3.split_t3 == "train"], v3[v3.split_t3 == "test"]
    trJ, vaJ, ext = (jp[jp.split_t3 == s] for s in ("train", "val", "externo"))
    pool = pd.concat([tr3, trJ], ignore_index=True)

    def rasgos(tr):
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                           min_frequency=25).fit(tr[CAT].astype(str))
        sc = StandardScaler().fit(np.log1p(tr[NUM].to_numpy(float)))
        return lambda d: np.hstack([emb[d.fila_emb.to_numpy()],
                                    oh.transform(d[CAT].astype(str)),
                                    sc.transform(np.log1p(d[NUM].to_numpy(float)))])

    def hgb():
        return HistGradientBoostingRegressor(max_iter=300, max_depth=6,
                                             learning_rate=0.06,
                                             l2_regularization=1.0,
                                             random_state=0)

    filas = []

    def anotar(escenario, y, p, quien):
        r = criterios(y, p)
        r["escenario"] = escenario
        r["quien"] = quien
        filas.append(r)

    # -------- prueba fija de v3 --------
    R = rasgos(pool)
    m = hgb().fit(R(pool), np.log(pool.effort_hours.to_numpy()))
    anotar("prueba fija de v3", te3.effort_hours.to_numpy(),
           np.exp(m.predict(R(te3))), "modelo v4 (agrupado)")
    med3 = tr3.groupby("project").effort_hours.median()
    anotar("prueba fija de v3", te3.effort_hours.to_numpy(),
           te3.project.map(med3).fillna(tr3.effort_hours.median()).to_numpy(),
           "linea base mediana proyecto")

    # -------- dentro del corpus publico --------
    RJ = rasgos(trJ)
    medJ = trJ.groupby("project").effort_hours.median()
    globJ = float(trJ.effort_hours.median())
    baseJ = vaJ.project.map(medJ).fillna(globJ).to_numpy()
    mr = hgb().fit(RJ(trJ), np.log(trJ.effort_hours.to_numpy())
                   - np.log(trJ.project.map(medJ).fillna(globJ).to_numpy()))
    anotar("dentro del corpus publico", vaJ.effort_hours.to_numpy(),
           np.exp(mr.predict(RJ(vaJ))) * baseJ, "modelo en razon x mediana")
    anotar("dentro del corpus publico", vaJ.effort_hours.to_numpy(), baseJ,
           "linea base mediana proyecto")

    # -------- curva de despliegue sobre el bloque externo --------
    medP = pool.groupby("project").effort_hours.median()
    globP = float(pool.effort_hours.median())
    Rp = rasgos(pool)
    mrp = hgb().fit(Rp(pool), np.log(pool.effort_hours.to_numpy())
                    - np.log(pool.project.map(medP).fillna(globP).to_numpy()))
    e = ext.copy()
    e["orden_p"] = e.created.fillna(pd.Timestamp("2000-01-01"))
    e = e.sort_values(["project", "orden_p", "task_id"]).reset_index(drop=True)
    razon = mrp.predict(Rp(e))
    reg = []
    for proj, d in e.groupby("project", sort=False):
        idx, hrs = d.index.to_numpy(), d.effort_hours.to_numpy()
        for j in range(len(idx)):
            escala = float(np.median(hrs[:j])) if j >= 3 else globP
            reg.append((j, float(np.exp(razon[idx[j]]) * escala), hrs[j]))
    r = pd.DataFrame(reg, columns=["vistas", "pred", "real"])
    for lo, hi, etiq in [(0, 4, "arranque en frio (0-4 tareas)"),
                         (5, 19, "rodaje (5-19 tareas)"),
                         (20, 49, "en marcha (20-49 tareas)"),
                         (50, 10**9, "regimen (50+ tareas)")]:
        s = r[(r.vistas >= lo) & (r.vistas <= hi)]
        anotar(f"proyectos nuevos -- {etiq}", s.real.to_numpy(),
               s.pred.to_numpy(), "modelo v4 + calibracion en linea")

    # -------- la estimacion humana, el mismo raspado --------
    sip = v3[v3.HoursEstimate.notna() & (v3.HoursEstimate > 0)]
    anotar("SiP con estimacion humana", sip.effort_hours.to_numpy(),
           sip.HoursEstimate.to_numpy(), "experto humano")
    pub = jp[jp.HoursEstimate.notna() & (jp.HoursEstimate > 0)]
    anotar("jira publico con estimacion humana", pub.effort_hours.to_numpy(),
           pub.HoursEstimate.to_numpy(), "experto humano")
    Rp2 = rasgos(pool)
    mp = hgb().fit(Rp2(pool), np.log(pool.effort_hours.to_numpy()))
    pm = np.exp(mp.predict(Rp2(pub)))
    anotar("jira publico con estimacion humana", pub.effort_hours.to_numpy(),
           pm, "modelo v4")
    anotar("jira publico con estimacion humana", pub.effort_hours.to_numpy(),
           np.sqrt(pm * pub.HoursEstimate.to_numpy()),
           "media geometrica humano x modelo")

    res = pd.DataFrame(filas)[["escenario", "quien", "n", "media_real", "MAE_h",
                               "pct_MAE", "RMSE_h", "razon_RMSE_MAE", "MAPE",
                               "MdAPE", "PRED25", "PRED50",
                               "C1", "C2", "C3", "C4"]]
    print("=" * 168)
    print("LOS CUATRO CRITERIOS, SOBRE TODOS LOS REGIMENES DE v4")
    print("=" * 168)
    print(res.to_string(index=False))

    print("\n" + "=" * 168)
    print("C1 MAE <= 30% de la duracion media  |  C2 RMSE <= 2x MAE  |  "
          "C3 MdAPE <= 20% y MAPE <= 30%  |  C4 PRED(25) >= 0,75")
    print("=" * 168)
    res["cumple"] = res[["C1", "C2", "C3", "C4"]].sum(axis=1)
    for _, x in res.iterrows():
        marcas = "".join("SI " if x[c] else ".. " for c in ("C1", "C2", "C3", "C4"))
        print(f"  {x.escenario:38s} {x.quien:34s} {marcas}  {x.cumple}/4")
    print(f"\n  filas que cumplen los cuatro: "
          f"{int((res.cumple == 4).sum())} de {len(res)}")
    print(f"  filas que cumplen C4 (PRED25 >= 0,75): {int(res.C4.sum())}")
    res.to_csv("results/criterios_v4.csv", index=False)
    print("\nescrito results/criterios_v4.csv")


if __name__ == "__main__":
    main()
