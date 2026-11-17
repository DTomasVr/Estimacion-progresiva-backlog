"""E93 - ¿Cuanto de lo que gana la estimacion del usuario es un artefacto?

En 11.546 tareas con estimacion, el 34,6 % tiene la estimacion EXACTAMENTE
igual a las horas reales. No es fuga en el sentido de ML -la estimacion se
conoce antes de hacer la tarea- pero si es un artefacto de medicion: en esos
proyectos de Jira la tarea se cierra imputando el tiempo estimado en vez del
medido, asi que para esas filas effort_hours NO es una medicion independiente,
es la propia estimacion copiada.

Eso afecta a dos cifras que ya estan escritas:

  E92   meter la estimacion como variable sube el ±25 % por tarea de 14,5 %
        a 41,3 %
  HU-51 el ±25 % del TOTAL sube de 44,2 % a 65,1 % en 43 proyectos

Las dos hay que repetirlas sin esas filas. Si la ventaja aguanta, la conclusion
vale y ademas queda blindada frente a un revisor. Si se desploma, hay que
reescribir las dos.

El reparto por fuente ya avisa de que el artefacto no esta repartido: Spring
47,4 % de filas identicas, Jira 17,0 %, Sonatype 10,5 %.
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
OUT = Path("results/estimacion_igual_al_real")
S = "=" * 100
PLIEGUES, LAMBDA, COB = 5, 12, .8
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)


def p25(y, p):
    return float(np.mean(np.abs(np.clip(p, 1e-6, None) - y) / y <= .25))


def mdmre(y, p):
    return float(np.median(np.abs(np.clip(p, 1e-6, None) - y) / y))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy")
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap].astype(np.float32)
    est_all = sw.HoursEstimate.where(sw.HoursEstimate > 0).to_numpy(float)
    hay = ~np.isnan(est_all)

    # ---------------- (1) por tarea ----------------
    swe, Ve, est = sw[hay].reset_index(drop=True), V[hay], est_all[hay]
    y = swe.effort_hours.to_numpy(float)
    proy = swe.project.to_numpy()
    igual = np.abs(est - y) / y <= .01
    print(f"{len(y):,} tareas con estimacion · identicas {igual.mean():.1%} · "
          f"{time.time() - t0:.0f} s", flush=True)

    ly, le = np.log(y), np.log(est)
    Xe = np.hstack([Ve, le[:, None]]).astype(np.float32)
    p_txt, p_est = np.zeros(len(y)), np.zeros(len(y))
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(Ve, y, proy)):
        p_txt[b] = np.exp(HistGradientBoostingRegressor(**PAR)
                          .fit(Ve[a], ly[a]).predict(Ve[b]))
        p_est[b] = np.exp(HistGradientBoostingRegressor(**PAR)
                          .fit(Xe[a], ly[a]).predict(Xe[b]))
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s", flush=True)

    filas = []
    print(f"\n{S}\n(1) POR TAREA: CON Y SIN LAS FILAS IDENTICAS\n{S}")
    print(f"  {'subconjunto':34s}{'n':>8}{'solo texto':>13}{'+estimacion':>13}"
          f"{'est. cruda':>13}")
    for etq, s in (("todas", np.ones(len(y), bool)),
                   ("SIN las identicas", ~igual),
                   ("solo las identicas", igual)):
        r = dict(subconjunto=etq, n=int(s.sum()), texto=p25(y[s], p_txt[s]),
                 con_estimacion=p25(y[s], p_est[s]), cruda=p25(y[s], est[s]),
                 texto_MdMRE=mdmre(y[s], p_txt[s]),
                 con_estimacion_MdMRE=mdmre(y[s], p_est[s]))
        filas.append(r)
        print(f"  {etq:34s}{r['n']:>8,}{r['texto']:>12.1%}"
              f"{r['con_estimacion']:>13.1%}{r['cruda']:>13.1%}")
    t1 = pd.DataFrame(filas)
    t1.to_csv(OUT / "por_tarea.csv", index=False)

    # ---------------- (2) por proyecto: la cifra de HU-51 ----------------
    print(f"\n{S}\n(2) POR PROYECTO: LA CIFRA DE HU-51 (44,2 % -> 65,1 %)\n{S}")
    pred_frio = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    tam = sw.groupby("project").size()
    m8 = sw.project.isin(tam[tam >= 8].index).to_numpy()
    cob = pd.Series(hay).groupby(sw.project.to_numpy()).mean()
    nom = [p for p in cob[cob >= COB].index if p in set(sw.project[m8])]
    sel = sw.project.isin(nom).to_numpy() & m8
    sw2 = sw[sel].reset_index(drop=True)
    pr2, y2 = sw2.project.to_numpy(), sw2.effort_hours.to_numpy(float)
    fr2 = pred_frio[sel]
    e2 = est_all[sel]
    h2 = ~np.isnan(e2)
    ig2 = np.zeros(len(y2), bool)
    ig2[h2] = np.abs(e2[h2] - y2[h2]) / y2[h2] <= .01
    # el corrector, fuera de pliegue, igual que E84
    V2 = V[sel]
    X2 = np.hstack([V2, np.log(np.where(h2, e2, 1))[:, None],
                    np.log(fr2)[:, None]]).astype(np.float32)
    corr = fr2.copy()
    for a, b in GroupKFold(PLIEGUES).split(X2, y2, pr2):
        a = a[h2[a]]
        mo = HistGradientBoostingRegressor(max_iter=250, max_depth=6,
                                           learning_rate=.06, l2_regularization=1.,
                                           loss="poisson", random_state=0)
        mo.fit(X2[a], y2[a])
        bb = b[h2[b]]
        corr[bb] = mo.predict(X2[bb])
    corr = np.clip(corr, 1e-6, None)

    idx = {p: np.flatnonzero(pr2 == p) for p in nom}
    frac_ig = np.array([ig2[idx[p]].mean() for p in nom])
    real_t = np.array([y2[idx[p]].sum() for p in nom])
    frio_t = np.array([fr2[idx[p]].sum() for p in nom])
    corr_t = np.array([np.where(h2[idx[p]], corr[idx[p]], fr2[idx[p]]).sum()
                       for p in nom])
    print(f"  {len(nom)} proyectos · fraccion de tareas identicas: mediana "
          f"{np.median(frac_ig):.1%} · max {frac_ig.max():.1%}")
    filas2 = []
    for etq, s in (("todos", np.ones(len(nom), bool)),
                   ("proyectos con <25 % de identicas", frac_ig < .25),
                   ("proyectos con <10 % de identicas", frac_ig < .10)):
        if s.sum() < 8:
            continue
        r = dict(subconjunto=etq, proyectos=int(s.sum()),
                 frio=p25(real_t[s], frio_t[s]),
                 con_estimacion=p25(real_t[s], corr_t[s]),
                 frio_MdMRE=mdmre(real_t[s], frio_t[s]),
                 con_estimacion_MdMRE=mdmre(real_t[s], corr_t[s]))
        filas2.append(r)
        print(f"  {etq:34s} n={r['proyectos']:3d}  en frio {r['frio']:.1%} -> "
              f"con estimacion {r['con_estimacion']:.1%}  "
              f"({(r['con_estimacion'] - r['frio']) * 100:+.1f} pts)")
    t2 = pd.DataFrame(filas2)
    t2.to_csv(OUT / "por_proyecto.csv", index=False)

    print(f"\n{S}\nLECTURA\n{S}")
    sinid = t1[t1.subconjunto == "SIN las identicas"].iloc[0]
    tod = t1[t1.subconjunto == "todas"].iloc[0]
    print(f"  POR TAREA: la ventaja de la estimacion pasa de "
          f"{(tod.con_estimacion - tod.texto) * 100:+.1f} puntos a "
          f"{(sinid.con_estimacion - sinid.texto) * 100:+.1f} al quitar las filas "
          f"identicas.")
    if sinid.con_estimacion - sinid.texto > .05:
        print("  Aguanta: la estimacion del usuario aporta de verdad, no solo por")
        print("  las filas en que el real es la propia estimacion copiada.")
    else:
        print("  NO aguanta: casi todo lo que ganaba la estimacion eran las filas")
        print("  identicas. Hay que reescribir la conclusion.")
    if len(filas2) > 1:
        a0, a1 = filas2[0], filas2[-1]
        print(f"\n  POR PROYECTO: la ganancia pasa de "
              f"{(a0['con_estimacion'] - a0['frio']) * 100:+.1f} puntos (todos) a "
              f"{(a1['con_estimacion'] - a1['frio']) * 100:+.1f} en los "
              f"{a1['proyectos']} proyectos menos contaminados.")
    json.dump({"identicas_por_tarea": float(igual.mean()),
               "por_tarea": filas, "por_proyecto": filas2},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/ · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
