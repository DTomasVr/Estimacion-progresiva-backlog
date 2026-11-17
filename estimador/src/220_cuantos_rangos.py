"""E105 - ¿Y si en vez de dos rangos son varios? El techo sube, pero ¿cuanto?

E90 midio la mezcla de expertos con DOS regimenes (<=2 h y >2 h): el enrutado
oraculo vale +5,8 puntos de +-25 % y la compuerta necesita AUC 0,80, pero el
texto solo da 0,646.

La idea de partir en MAS rangos tira de dos fuerzas opuestas y nadie las ha
separado:

  A FAVOR    cada experto cubre un intervalo mas estrecho, asi que predecir su
             centro acierta mas. El techo del oraculo sube
  EN CONTRA  cada experto entrena con 1/N de los datos. E102 ya midio que el
             experto binario, con 8.500 tareas, sobreajusta con brecha 0,3064.
             Con 8 rangos tendria ~2.000 y la brecha seria peor
  Y ADEMAS   acertar el rango es mas dificil cuantos mas hay: el clasificador
             de 4 clases que ya existe saca F1 0,32 y recall 0,057 en >16 h

Se mide todo junto para N = 1, 2, 3, 4, 6, 8 rangos definidos por cuantiles:

  el TECHO       enrutado oraculo, que es la cota de toda la familia
  la BRECHA      media de los expertos, para ver si se desbocan
  el ACIERTO     que porcentaje de rango habria que acertar para empatar con el
                 modelo global, simulando compuertas de acierto creciente

Si el techo sube mucho mas rapido que el acierto exigido, la via revive. Si no,
queda cerrada con un numero por N y no hay que volver.
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
OUT = Path("results/cuantos_rangos")
S = "=" * 100
PLIEGUES = 3
NS = (1, 2, 3, 4, 6, 8)
ACIERTOS = (.4, .5, .6, .7, .8, .9, 1.0)
SORTEOS = 20
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)


def p25(y, p):
    return float(np.mean(np.abs(np.clip(p, 1e-6, None) - y) / y <= .25))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy").astype(np.float32)
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap]
    y = sw.effort_hours.to_numpy(float)
    ly = np.log(y)
    proy = sw.project.to_numpy()
    part = list(GroupKFold(PLIEGUES).split(V, y, proy))
    print(f"{len(y):,} tareas · {sw.project.nunique()} proyectos · "
          f"{time.time() - t0:.0f} s", flush=True)

    filas, det = [], []
    for N in NS:
        # rangos por cuantiles: todos con el mismo numero de tareas
        if N == 1:
            cortes = np.array([])
        else:
            cortes = np.quantile(y, np.arange(1, N) / N)
        g = np.digitize(y, cortes)
        pred_or = np.zeros(len(y))
        brechas, tamanos = [], []
        for a, b in part:
            for t_ in range(N):
                tr = a[g[a] == t_]
                ev = b[g[b] == t_]
                if len(tr) < 40 or len(ev) == 0:
                    if len(ev):
                        pred_or[ev] = np.exp(np.median(ly[a]))
                    continue
                mo = HistGradientBoostingRegressor(**PAR).fit(V[tr], ly[tr])
                pred_or[ev] = np.exp(mo.predict(V[ev]))
                dentro = float(np.median(np.abs(mo.predict(V[tr]) - ly[tr])))
                fuera = float(np.median(np.abs(mo.predict(V[ev]) - ly[ev])))
                brechas.append(fuera - dentro)
                tamanos.append(len(tr))
        techo = p25(y, pred_or)
        filas.append(dict(N=N, techo_oraculo=techo,
                          brecha_media=float(np.mean(brechas)) if brechas else 0.,
                          n_por_experto=int(np.mean(tamanos)) if tamanos else 0,
                          limites=" · ".join(f"{c:.2f}" for c in cortes)))
        print(f"  N={N} · techo {techo:.1%} · brecha "
              f"{np.mean(brechas) if brechas else 0:.4f} · "
              f"n/experto {int(np.mean(tamanos)) if tamanos else 0:,} · "
              f"{time.time() - t0:.0f} s", flush=True)

        # que acierto de rango haria falta: se enruta mal una fraccion al azar
        if N > 1:
            for ac in ACIERTOS:
                vs = []
                for s in range(SORTEOS):
                    rng = np.random.default_rng(6100 + s)
                    gg = g.copy()
                    mal = rng.random(len(y)) > ac
                    gg[mal] = rng.integers(0, N, mal.sum())
                    # se usa el experto del rango ADIVINADO sobre cada tarea
                    p_ = np.zeros(len(y))
                    for a, b in part:
                        for t_ in range(N):
                            tr = a[g[a] == t_]
                            ev = b[gg[b] == t_]
                            if len(tr) < 40 or len(ev) == 0:
                                continue
                            p_[ev] = np.exp(np.median(ly[tr]))
                    v = p_ > 0
                    vs.append(p25(y[v], p_[v]))
                det.append(dict(N=N, acierto=ac, PRED25=float(np.mean(vs))))
            print(f"    curva de acierto lista · {time.time() - t0:.0f} s",
                  flush=True)

    t = pd.DataFrame(filas)
    d = pd.DataFrame(det)
    print(f"\n{S}\n¿CUANTOS RANGOS?\n{S}")
    print(t[["N", "techo_oraculo", "brecha_media", "n_por_experto",
             "limites"]].round(4).to_string(index=False))
    print(f"\n{S}\n±25 % SEGUN LO BIEN QUE SE ACIERTE EL RANGO\n{S}")
    if len(d):
        print(d.pivot(index="acierto", columns="N",
                      values="PRED25").round(4).to_string())

    print(f"\n{S}\nLECTURA\n{S}")
    base = float(t[t.N == 1].techo_oraculo.iloc[0])
    print(f"  un solo modelo (N=1): {base:.1%}")
    for _, r in t[t.N > 1].iterrows():
        print(f"  N={int(r.N)}: techo {r.techo_oraculo:.1%} "
              f"({(r.techo_oraculo - base) * 100:+.1f} pts) · brecha "
              f"{r.brecha_media:.4f} · {int(r.n_por_experto):,} tareas por experto")
    mejor = t.loc[t.techo_oraculo.idxmax()]
    print(f"\n  el techo mas alto es N={int(mejor.N)} con {mejor.techo_oraculo:.1%}")
    cre = float(t[t.N > 1].brecha_media.iloc[-1] - t[t.N > 1].brecha_media.iloc[0])
    print(f"  la brecha crece {cre:+.4f} al pasar de {int(NS[1])} a "
          f"{int(NS[-1])} rangos: los expertos SI se sobreajustan mas")
    t.to_csv(OUT / "rangos.csv", index=False)
    d.to_csv(OUT / "curva_acierto.csv", index=False)
    json.dump({"base": base, "mejor_N": int(mejor.N),
               "mejor_techo": float(mejor.techo_oraculo)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/ · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
