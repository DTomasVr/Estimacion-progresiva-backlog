"""The deployment curve: what Agix can promise on day 1, day 30 and day 90.

The external block is 150 projects no model has ever seen, which is exactly the
position Agix is in when a customer creates their first project. Tasks arrive in
order; each finished one is a label the system did not have before. So the
honest way to describe the product is not a single MdMRE but a curve: accuracy
as a function of how many tasks that project has already completed.

Calibration is online and causal -- a task is scored using only tasks that
finished BEFORE it in that project, never the project's full median, which would
be reading the future. Intervals are split-conformal, with the quantile taken
from the public validation block and never from anything scored here.

The reported band is exp(2q), the full multiplicative width from the lower to
the upper end -- not exp(q), which is only the half-width.
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
DATA = "data/processed/agix_effort_v4.parquet"
EMB = "data/processed/embeddings/v4_mpnet_congelado"
NUM = ["n_links", "n_links_estructurales", "n_componentes", "tiene_asignado"]
CAT = ["task_kind", "source"]


def cargar():
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    emb = np.load(f"{EMB}.npy")
    ids = np.load(f"{EMB}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    sw["fila_emb"] = [pos[t] for t in sw.task_id]
    return sw, emb


def main():
    sw, emb = cargar()
    v3 = sw[sw.source != "JiraPublico"]
    jp = sw[sw.source == "JiraPublico"]
    tr = pd.concat([v3[v3.split_t3 == "train"], jp[jp.split_t3 == "train"]],
                   ignore_index=True)
    cal = jp[jp.split_t3 == "val"]
    ext = jp[jp.split_t3 == "externo"].copy()
    print(f"entrenamiento {len(tr):,}   calibracion {len(cal):,}   "
          f"externo {len(ext):,} en {ext.project.nunique()} proyectos nuevos\n")

    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                       min_frequency=25).fit(tr[CAT].astype(str))
    sc = StandardScaler().fit(np.log1p(tr[NUM].to_numpy(float)))
    R = lambda d: np.hstack([emb[d.fila_emb.to_numpy()],
                             oh.transform(d[CAT].astype(str)),
                             sc.transform(np.log1p(d[NUM].to_numpy(float)))])

    glob = float(tr.effort_hours.median())
    med_tr = tr.project.map(tr.groupby("project").effort_hours.median()).to_numpy()
    obj = np.log(tr.effort_hours.to_numpy()) - np.log(med_tr)
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=6,
                                      learning_rate=0.06, l2_regularization=1.0,
                                      random_state=0).fit(R(tr), obj)

    # cuantil conformal sobre el bloque de calibracion, escala global
    r_cal = m.predict(R(cal)) + np.log(glob)
    resid = np.abs(np.log(cal.effort_hours.to_numpy()) - r_cal)
    q = {lv: float(np.quantile(resid, lv)) for lv in (0.50, 0.80, 0.90)}
    print("cuantiles conformales (mitad del ancho, en log):")
    for lv, v in q.items():
        print(f"  {int(lv*100)}%  q={v:.3f}  banda completa x{np.exp(2*v):.2f}")

    # orden causal dentro de cada proyecto
    ext["orden_p"] = ext.created.fillna(pd.Timestamp("2000-01-01"))
    ext = ext.sort_values(["project", "orden_p", "task_id"]).reset_index(drop=True)
    razon = m.predict(R(ext))

    filas = []
    for proj, d in ext.groupby("project", sort=False):
        idx = d.index.to_numpy()
        hrs = d.effort_hours.to_numpy()
        for j in range(len(idx)):
            previas = hrs[:j]
            escala = float(np.median(previas)) if j >= 3 else glob
            filas.append((proj, j, float(np.exp(razon[idx[j]]) * escala), hrs[j]))
    r = pd.DataFrame(filas, columns=["project", "vistas", "pred", "real"])

    print("\n" + "=" * 104)
    print("CURVA DE DESPLIEGUE  (cada tarea puntuada solo con las que ya "
          "terminaron antes en su proyecto)")
    print("=" * 104)
    print(f"{'tareas ya terminadas':24s} {'n':>7s} {'MdMRE':>8s} {'PRED25':>8s} "
          f"{'PRED50':>8s} {'MAE h':>8s} {'cob 80%':>9s} {'banda':>8s}")
    tramos = [(0, 0), (1, 4), (5, 9), (10, 19), (20, 49), (50, 10**9)]
    resumen = []
    for lo, hi in tramos:
        s = r[(r.vistas >= lo) & (r.vistas <= hi)]
        if len(s) < 50:
            continue
        mm = E.point_metrics(s.real.to_numpy(), s.pred.to_numpy())
        lp = np.log(np.clip(s.pred.to_numpy(), .01, None))
        y = s.real.to_numpy()
        cob = float(np.mean((y >= np.exp(lp - q[0.80])) & (y <= np.exp(lp + q[0.80]))))
        etiqueta = f"{lo}" if lo == hi else (f"{lo}+" if hi > 10**8 else f"{lo}-{hi}")
        print(f"{etiqueta:24s} {len(s):>7,} {mm['MdMRE']:>8.4f} {mm['PRED25']:>8.4f} "
              f"{mm['PRED50']:>8.4f} {mm['MAE_h']:>8.2f} {cob:>9.3f} "
              f"{np.exp(2*q[0.80]):>7.2f}x")
        resumen.append(dict(tramo=etiqueta, n=len(s), MdMRE=round(mm["MdMRE"], 4),
                            PRED25=round(mm["PRED25"], 4),
                            PRED50=round(mm["PRED50"], 4),
                            MAE_h=round(mm["MAE_h"], 2),
                            cobertura80=round(cob, 3),
                            banda80=round(float(np.exp(2 * q[0.80])), 2)))

    # el mismo corte, pero con bandas recalibradas por tramo
    print("\ncon la banda recalibrada dentro de cada tramo (lo que el producto "
          "mostraria de verdad):")
    for fila, (lo, hi) in zip(resumen, [t for t in tramos
                                        if len(r[(r.vistas >= t[0])
                                                 & (r.vistas <= t[1])]) >= 50]):
        s = r[(r.vistas >= lo) & (r.vistas <= hi)]
        res = np.abs(np.log(s.real.to_numpy())
                     - np.log(np.clip(s.pred.to_numpy(), .01, None)))
        qq = float(np.quantile(res, 0.80))
        print(f"  {fila['tramo']:10s} banda 80% x{np.exp(2*qq):.2f}   "
              f"(mediana {s.pred.median():.2f} h -> "
              f"[{s.pred.median()*np.exp(-qq):.2f}, "
              f"{s.pred.median()*np.exp(qq):.2f}] h)")
        fila["banda80_recal"] = round(float(np.exp(2 * qq)), 2)

    print("\n--- deteccion de tareas grandes (el riesgo que importa) ---")
    for umbral in (8, 16, 40):
        y = (r.real > umbral).astype(int).to_numpy()
        if y.sum() < 30:
            continue
        s = r.pred.to_numpy()
        orden = np.argsort(-s)
        yv = y[orden]
        tpr = np.cumsum(yv) / max(yv.sum(), 1)
        fpr = np.cumsum(1 - yv) / max((1 - yv).sum(), 1)
        auc = float(np.trapezoid(tpr, fpr))
        print(f"  > {umbral:>2} h  positivos {int(y.sum()):>5,}  AUC {auc:.3f}")
        resumen.append(dict(tramo=f"AUC >{umbral}h", n=int(y.sum()),
                            MdMRE=np.nan, PRED25=np.nan, PRED50=np.nan,
                            MAE_h=np.nan, cobertura80=round(auc, 3),
                            banda80=np.nan))

    pd.DataFrame(resumen).to_csv("results/curva_despliegue.csv", index=False)
    print("\nescrito results/curva_despliegue.csv")


if __name__ == "__main__":
    main()
