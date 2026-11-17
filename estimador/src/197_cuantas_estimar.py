"""E84 - ¿Cuantas tareas tiene que estimar el usuario para que valga la pena?

Lo mejor que se ha medido en todo el proyecto es pedirle al usuario SU estimacion:
sobre los proyectos que la tienen, el +-25 % del total pasa del 44,2 % al 65,1 %.
Pero esa cifra supone que estima TODAS las tareas del backlog, que es muchisima
friccion: 30 o 40 casillas.

La pregunta de producto que no se ha medido es cuantas hacen falta. Si con cinco se
lleva la mayor parte, el formulario pide cinco y ya.

El dato se simula con HoursEstimate, un campo que el modelo NUNCA lee y que es
exactamente lo que el usuario escribiria: su estimacion en horas de una tarea.

  A  en frio, sin pedir nada
  B  el usuario estima k tareas          k = 1, 3, 5, 10
  C  el usuario estima TODAS             el techo, los 65,1 % ya medidos
  D  k tareas CERRADAS con horas reales  para comparar friccion contra friccion

La estimacion del usuario esta sesgada, asi que no se usa en crudo: se pasa por un
corrector de Poisson entrenado FUERA de pliegue sobre [vector, log(estimacion),
log(prediccion en frio)] -> horas reales. Con las k corregidas se ancla el resto,
igual que con las cerradas.

Se evalua sobre los proyectos con estimacion en al menos el 80 % de sus tareas, con
GroupKFold por proyecto entero y promediando la METRICA sobre los sorteos.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "10"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/cuantas_estimar")
S = "=" * 104
LAMBDA, SORTEOS, PLIEGUES, COB = 12, 60, 5, .8
KS = (1, 3, 5, 10)


def mt(real, pred):
    real, pred = np.asarray(real, float), np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    m = float(np.median(np.log(pred / real)))
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                MAE_h=float(np.abs(pred - real).mean()),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy")
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap]
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    assert len(pred) == len(sw)
    tam = sw.groupby("project").size()
    m = sw.project.isin(tam[tam >= 8].index).to_numpy()
    sw, V, pred = sw[m].reset_index(drop=True), V[m], pred[m]

    est = sw.HoursEstimate.where(sw.HoursEstimate > 0).to_numpy(float)
    hay = ~np.isnan(est)
    cob = pd.Series(hay).groupby(sw.project.to_numpy()).mean()
    nom = list(cob[cob >= COB].index)
    sel_p = sw.project.isin(nom).to_numpy()
    sw, V, pred, est, hay = (sw[sel_p].reset_index(drop=True), V[sel_p], pred[sel_p],
                             est[sel_p], hay[sel_p])
    proy, real = sw.project.to_numpy(), sw.effort_hours.to_numpy(float)
    idx = {p: np.flatnonzero(proy == p) for p in nom}
    real_tot = np.array([real[idx[p]].sum() for p in nom])
    frio = np.array([pred[idx[p]].sum() for p in nom])
    print(f"{len(nom)} proyectos con estimacion en >={COB:.0%} de sus tareas · "
          f"{len(sw):,} tareas · {hay.mean():.1%} con estimacion · "
          f"{time.time() - t0:.0f} s", flush=True)
    print(f"  en frio: ±25 % {mt(real_tot, frio)['PRED25']:.1%}", flush=True)

    # ---- corrector fuera de pliegue -------------------------------------
    X = np.hstack([V, np.log(np.where(hay, est, 1))[:, None],
                   np.log(pred)[:, None]]).astype(np.float32)
    corr = np.full(len(sw), np.nan)
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(X, real, proy)):
        a = a[hay[a]]
        mo = HistGradientBoostingRegressor(max_iter=250, max_depth=6, learning_rate=.06,
                                           l2_regularization=1., loss="poisson",
                                           random_state=0).fit(X[a], real[a])
        bb = b[hay[b]]
        corr[bb] = mo.predict(X[bb])
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s", flush=True)
    corr = np.clip(corr, 1e-6, None)

    filas = [dict(brazo="A · en frio, no se pide nada", pedidas=0,
                  **mt(real_tot, frio))]

    # C: el usuario estima TODAS
    tot_todas = np.array([np.where(hay[idx[p]], corr[idx[p]], pred[idx[p]]).sum()
                          for p in nom])
    filas.append(dict(brazo="C · el usuario estima TODAS", pedidas=-1,
                      **mt(real_tot, tot_todas)))

    # B: el usuario estima solo k, y con esas se ancla el resto
    for k in KS:
        ms = []
        for s in range(SORTEOS):
            rng = np.random.default_rng(300 + s)
            t_ = np.empty(len(nom))
            for j, p in enumerate(nom):
                i = idx[p]
                con = i[hay[i]]
                if len(con) <= k or len(i) <= k:
                    t_[j] = pred[i].sum()
                    continue
                sel = rng.choice(con, k, replace=False)
                resto = np.setdiff1d(i, sel)
                off = np.log(corr[sel].sum() / pred[sel].sum()) * k / (k + LAMBDA)
                t_[j] = corr[sel].sum() + (pred[resto] * np.exp(off)).sum()
            ms.append(mt(real_tot, t_))
        filas.append(dict(brazo=f"B · el usuario estima {k}", pedidas=k,
                          **{c: float(np.mean([x[c] for x in ms])) for c in ms[0]}))

    # D: k tareas CERRADAS, con horas reales
    for k in (3, 5):
        ms = []
        for s in range(SORTEOS):
            rng = np.random.default_rng(700 + s)
            t_ = np.empty(len(nom))
            for j, p in enumerate(nom):
                i = idx[p]
                if len(i) <= k:
                    t_[j] = pred[i].sum()
                    continue
                sel = rng.choice(i, k, replace=False)
                resto = np.setdiff1d(i, sel)
                off = np.log(real[sel].sum() / pred[sel].sum()) * k / (k + LAMBDA)
                t_[j] = real[sel].sum() + (pred[resto] * np.exp(off)).sum()
            ms.append(mt(real_tot, t_))
        filas.append(dict(brazo=f"D · {k} tareas CERRADAS (horas reales)", pedidas=k,
                          **{c: float(np.mean([x[c] for x in ms])) for c in ms[0]}))

    t = pd.DataFrame(filas)
    print(f"\n{S}\n¿CUANTAS TAREAS TIENE QUE ESTIMAR EL USUARIO? ({len(nom)} proyectos)"
          f"\n{S}")
    print(t[["brazo", "PRED25", "MdMRE", "factor2", "MAE_h", "SSPB_pc"]].round(4)
          .to_string(index=False))

    a = float(t.iloc[0].PRED25)
    todas = float(t[t.brazo.str.startswith("C")].PRED25.iloc[0])
    print(f"\n{S}\nLECTURA\n{S}")
    print(f"  no pedir nada            ±25 % {a:.1%}")
    print(f"  estimarlas TODAS         ±25 % {todas:.1%}  ({(todas - a) * 100:+.1f} pts)")
    for k in KS:
        v = float(t[t.brazo.eq(f"B · el usuario estima {k}")].PRED25.iloc[0])
        pct = (v - a) / (todas - a) * 100 if todas > a else 0
        print(f"  estimar solo {k:2d}          ±25 % {v:.1%}  "
              f"({(v - a) * 100:+.1f} pts · {pct:.0f} % de lo que dan todas)")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"proyectos": len(nom), "sorteos": SORTEOS, "cobertura": COB},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
