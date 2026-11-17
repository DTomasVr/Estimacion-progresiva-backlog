"""E116 - Si el usuario solo estima unas pocas, ¿cuales le pedimos?

E84 midio que pedirle al usuario que estime k tareas AL AZAR no sirve: con 5 de
cada proyecto el +-25 % del total apenas se mueve, y la ganancia completa
-de 46,3 % a 73,2 %- exige que estime practicamente todo el backlog. De ahi
salio el criterio HU-46-2, que pide el 80 % de cobertura.

Pero nadie ha probado a ELEGIR cuales. Y hay una razon de peso para pensar que
importa: para el TOTAL, una tarea prevista de 40 h pesa ochenta veces mas que
una de 0,5 h. Estimar al azar reparte el esfuerzo del usuario por igual entre
tareas que no valen lo mismo.

Criterios, todos calculables ANTES de pedir nada:

  azar            lo de E84
  mas horas       las de mayor prediccion: las que mandan en la suma
  mas inciertas   mayor sigma del modelo. CONTROL: E115 midio que sigma
                  correlaciona 0,04 con el error real, asi que deberia fallar,
                  y si falla confirma E115 desde otro angulo
  mas atipicas    las mas lejos de sus vecinos del corpus
  oraculo         las que de verdad mas reducen el error del total: la cota

Se mide el +-25 % del total y que fraccion de las horas previstas cubren las
tareas elegidas, que es lo que explica el resultado.

Y se repite SIN las filas en que el real es la estimacion copiada (E93), porque
si no, elegir tareas podria estar eligiendo regalos.
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
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/que_tareas_estimar")
S = "=" * 112
PLIEGUES, COB, SORTEOS, K_VEC, LOTE = 5, .8, 30, 25, 1024
KS = (1, 3, 5, 10, 20)
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)


def p25(real, pred):
    return float(np.mean(np.abs(np.clip(pred, 1e-6, None) - real) / real <= .25))


def mdmre(real, pred):
    return float(np.median(np.abs(np.clip(pred, 1e-6, None) - real) / real))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy").astype(np.float32)
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap]
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    tam = sw.groupby("project").size()
    m = sw.project.isin(tam[tam >= 8].index).to_numpy()
    sw, V, pred = sw[m].reset_index(drop=True), V[m], pred[m]
    est = sw.HoursEstimate.where(sw.HoursEstimate > 0).to_numpy(float)
    hay = ~np.isnan(est)
    cob = pd.Series(hay).groupby(sw.project.to_numpy()).mean()
    nom = list(cob[cob >= COB].index)
    s_p = sw.project.isin(nom).to_numpy()
    sw, V, pred, est, hay = (sw[s_p].reset_index(drop=True), V[s_p], pred[s_p],
                             est[s_p], hay[s_p])
    y = sw.effort_hours.to_numpy(float)
    proy = sw.project.to_numpy()
    idx = {p: np.flatnonzero(proy == p) for p in nom}
    real_tot = np.array([y[idx[p]].sum() for p in nom])
    frio = np.array([pred[idx[p]].sum() for p in nom])
    limpio = hay & (np.abs(np.where(hay, est, 1) - y) / y > .01)
    print(f"{len(nom)} proyectos · {len(sw):,} tareas · "
          f"limpias {limpio.sum() / hay.sum():.1%} de las que traen estimacion · "
          f"{time.time() - t0:.0f} s", flush=True)

    # corrector fuera de pliegue, igual que E84
    X = np.hstack([V, np.log(np.where(hay, est, 1))[:, None],
                   np.log(pred)[:, None]]).astype(np.float32)
    corr = pred.copy()
    sig = np.zeros(len(y))
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    atip = np.zeros(len(y))
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(X, y, proy)):
        aa = a[hay[a]]
        corr[b[hay[b]]] = HistGradientBoostingRegressor(
            **dict(PAR, loss="poisson")).fit(X[aa], y[aa]).predict(X[b[hay[b]]])
        # sigma del modelo en frio y atipicidad, para los criterios
        ly = np.log(y)
        pm = HistGradientBoostingRegressor(**PAR).fit(V[a], ly[a])
        r_ = ly[a] - pm.predict(V[a])
        sg = HistGradientBoostingRegressor(
            **dict(PAR, max_iter=150, max_depth=4)).fit(
            V[a], np.log(np.abs(r_) + 1e-3))
        sig[b] = np.exp(sg.predict(V[b]))
        for i in range(0, len(b), LOTE):
            bb = b[i:i + LOTE]
            s_ = Vn[bb] @ Vn[a].T
            atip[bb] = -np.partition(-s_, K_VEC, axis=1)[:, :K_VEC].mean(1)
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s",
              flush=True)
    corr = np.clip(corr, 1e-6, None)

    def total_con(sel_mask):
        return np.array([np.where(sel_mask[idx[p]], corr[idx[p]],
                                  pred[idx[p]]).sum() for p in nom])

    CRIT = {"azar": None, "mas horas": pred, "mas inciertas": sig,
            "mas atipicas": atip}
    filas = []
    for ambito, elegible in (("todas", hay), ("solo limpias", limpio)):
        tot_todas = total_con(elegible)
        filas.append(dict(ambito=ambito, criterio="-- en frio --", k=0,
                          PRED25=p25(real_tot, frio), MdMRE=mdmre(real_tot, frio),
                          frac_horas=0.))
        filas.append(dict(ambito=ambito, criterio="estima TODAS", k=-1,
                          PRED25=p25(real_tot, tot_todas),
                          MdMRE=mdmre(real_tot, tot_todas), frac_horas=1.))
        for cr, clave in CRIT.items():
            for k in KS:
                ms, fr = [], []
                reps = SORTEOS if cr == "azar" else 1
                for s in range(reps):
                    rng = np.random.default_rng(2600 + s)
                    sel = np.zeros(len(y), bool)
                    for p in nom:
                        i = idx[p][elegible[idx[p]]]
                        if len(i) == 0:
                            continue
                        kk = min(k, len(i))
                        j = (rng.choice(i, kk, replace=False) if clave is None
                             else i[np.argsort(-clave[i])[:kk]])
                        sel[j] = True
                    ms.append(p25(real_tot, total_con(sel)))
                    fr.append(np.mean([pred[idx[p]][sel[idx[p]]].sum() /
                                       pred[idx[p]].sum() for p in nom]))
                filas.append(dict(ambito=ambito, criterio=cr, k=k,
                                  PRED25=float(np.mean(ms)),
                                  MdMRE=mdmre(real_tot, total_con(sel)),
                                  frac_horas=float(np.mean(fr))))
        # oraculo: las k que mas acercan el total al real
        for k in KS:
            sel = np.zeros(len(y), bool)
            for p in nom:
                i = idx[p][elegible[idx[p]]]
                if len(i) == 0:
                    continue
                # cuanto acerca cada tarea el total, por si sola
                base = pred[idx[p]].sum()
                gan = -np.abs(base - pred[i] + corr[i] - real_tot[nom.index(p)])
                sel[i[np.argsort(-gan)[:min(k, len(i))]]] = True
            filas.append(dict(ambito=ambito, criterio="ORACULO", k=k,
                              PRED25=p25(real_tot, total_con(sel)),
                              MdMRE=mdmre(real_tot, total_con(sel)),
                              frac_horas=float(np.mean(
                                  [pred[idx[p]][sel[idx[p]]].sum() /
                                   pred[idx[p]].sum() for p in nom]))))
        print(f"  {ambito} listo · {time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(filas)
    for amb in ("todas", "solo limpias"):
        d = t[t.ambito == amb]
        print(f"\n{S}\n±25 % DEL TOTAL · {amb.upper()}\n{S}")
        print(d[d.k > 0].pivot(index="k", columns="criterio",
                               values="PRED25").round(4).to_string())
        f0 = float(d[d.k == 0].PRED25.iloc[0])
        ft = float(d[d.k == -1].PRED25.iloc[0])
        print(f"  en frio {f0:.1%} · estimando TODAS {ft:.1%}")
        print("\n  fraccion de las horas previstas que cubren las elegidas:")
        print(d[d.k > 0].pivot(index="k", columns="criterio",
                               values="frac_horas").round(3).to_string())

    print(f"\n{S}\nLECTURA\n{S}")
    d = t[t.ambito == "solo limpias"]
    f0 = float(d[d.k == 0].PRED25.iloc[0])
    ft = float(d[d.k == -1].PRED25.iloc[0])
    print(f"  en frio {f0:.1%} · estimando TODAS {ft:.1%} "
          f"({(ft - f0) * 100:+.1f} puntos)")
    for k in KS:
        az = float(d[(d.criterio == "azar") & (d.k == k)].PRED25.iloc[0])
        mh = float(d[(d.criterio == "mas horas") & (d.k == k)].PRED25.iloc[0])
        print(f"  k={k:2d}: al azar {az:.1%} ({(az - f0) / max(ft - f0, 1e-9):.0%} "
              f"de lo que dan todas) · por MAS HORAS {mh:.1%} "
              f"({(mh - f0) / max(ft - f0, 1e-9):.0%})")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"proyectos": len(nom), "frio": f0, "todas": ft},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
