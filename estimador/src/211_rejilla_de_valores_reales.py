"""E96 - Afinar el ajuste a la rejilla, y medirlo contra el estado del arte.

E95 dejo dos cosas. La incomoda: predecir una constante da mas PRED(25) que
nuestro modelo, porque PRED(25) mide la curtosis de estimado/real y la
distribucion del esfuerzo es un peine -diez valores cubren el 46,8 %-. Y la
util: AJUSTAR la prediccion al valor frecuente mas cercano sube el +-25 % de
15,06 % a 16,49 % y el factor 2 de 0,390 a 0,431 sin tocar el MAE, el SA ni el
orden entre tareas.

La regla agresiva -buscar la moda de la ventana relativa- daba algo mas de
PRED(25) pero hundia el spearman de 0,32 a 0,12 y el SA a negativo: mostraria
casi el mismo numero para toda tarea. Inservible como producto.

Asi que aqui se afina solo el ajuste a la rejilla, que es gratis:

  cuantos valores tiene que tener la rejilla, elegido FUERA de pliegue
  rejilla por frecuencia frente a rejilla geometrica
  y si conviene ajustar en log o en lineal

Y se mide donde se puede comparar: el bloque split_t3 == test, que es el que
usan las reimplementaciones de Deep-SE y GPT2SP en 46_baselines_literatura.py.

AVISO DE COMPARABILIDAD: nuestras predicciones son fuera de pliegue con
GroupKFold por PROYECTO ENTERO, asi que el modelo nunca vio ni la tarea ni su
proyecto. Las de la literatura se entrenan en el bloque train, que SI contiene
tareas de los mismos proyectos. La comparacion nos perjudica a nosotros, no al
reves; conviene decirlo tal cual.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
OUT = Path("results/rejilla_valores")
S = "=" * 104
PLIEGUES, TOL = 5, .25
TAMANOS = (4, 6, 8, 10, 12, 16, 20, 30, 50)


def met(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, None)
    r = np.abs(p - y) / y
    mar = np.abs(np.median(y) - y).mean()
    return dict(PRED25=float(np.mean(r <= TOL)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                MAE_h=float(np.abs(p - y).mean()),
                SA=float(1 - np.abs(p - y).mean() / mar),
                spearman=float(spearmanr(p, y).statistic))


def ajusta(p, rej):
    """Al valor de la rejilla mas cercano EN LOG: lo que importa es el cociente."""
    return rej[np.argmin(np.abs(np.log(p[:, None] / rej[None])), 1)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    sw = todo[~todo.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy(float)
    proy = sw.project.to_numpy()
    p_act = np.load("results/experto_cortas/predicciones.npz")["p_glob"]
    assert len(p_act) == len(y)
    print(f"{len(y):,} tareas · {time.time() - t0:.0f} s", flush=True)

    # ---- 1. que tamano de rejilla, elegido fuera de pliegue ----
    filas = []
    for g in TAMANOS:
        for tipo in ("frecuencia", "geometrica"):
            p = np.empty(len(y))
            for a, b in GroupKFold(PLIEGUES).split(y[:, None], y, proy):
                if tipo == "frecuencia":
                    v, c = np.unique(y[a], return_counts=True)
                    rej = np.sort(v[np.argsort(-c)[:g]])
                else:
                    rej = np.geomspace(np.quantile(y[a], .02),
                                       np.quantile(y[a], .98), g)
                p[b] = ajusta(p_act[b], rej)
            filas.append(dict(tamano=g, rejilla=tipo, **met(y, p)))
    t1 = pd.DataFrame(filas)
    print(f"\n{S}\n¿CUANTOS VALORES DEBE TENER LA REJILLA?\n{S}")
    print(t1.pivot(index="tamano", columns="rejilla",
                   values="PRED25").round(4).to_string())
    mejor = t1.loc[t1.PRED25.idxmax()]
    print(f"\n  mejor: {int(mejor.tamano)} valores, rejilla por "
          f"{mejor.rejilla} · ±25 % {mejor.PRED25:.2%} · factor2 "
          f"{mejor.factor2:.4f} · MAE {mejor.MAE_h:.3f} · SA {mejor.SA:.4f}")

    # la rejilla final, ajustada en todo el corpus, para dejarla escrita
    v, c = np.unique(y, return_counts=True)
    rej_fin = np.sort(v[np.argsort(-c)[:int(mejor.tamano)]])
    print(f"  la rejilla: {' · '.join(f'{x:g}' for x in rej_fin)} h")

    # ---- 2. donde se compara con la literatura ----
    te = (sw.split_t3 == "test").to_numpy()
    p_sn = np.empty(len(y))
    for a, b in GroupKFold(PLIEGUES).split(y[:, None], y, proy):
        vv, cc = np.unique(y[a], return_counts=True)
        p_sn[b] = ajusta(p_act[b], np.sort(vv[np.argsort(-cc)[:int(mejor.tamano)]]))
    cte = np.empty(len(y))
    for a, b in GroupKFold(PLIEGUES).split(y[:, None], y, proy):
        cand = np.unique(y[a])
        cte[b] = cand[int(np.argmax([np.mean(np.abs(x - y[a]) / y[a] <= TOL)
                                     for x in cand]))]
    print(f"\n{S}\nEN EL BLOQUE split_t3 == test ({int(te.sum()):,} tareas)\n{S}")
    f2 = [dict(modelo="nuestro, sin ajustar", **met(y[te], p_act[te])),
          dict(modelo="nuestro, AJUSTADO a la rejilla", **met(y[te], p_sn[te])),
          dict(modelo="la mejor constante", **met(y[te], cte[te]))]
    t2 = pd.DataFrame(f2)
    print(t2[["modelo", "PRED25", "MdMRE", "factor2", "MAE_h", "SA",
              "spearman"]].round(4).to_string(index=False))
    lit = pd.read_csv("results/baselines_literatura.csv")
    print(f"\n  lo publicado sobre el mismo bloque:")
    for _, x in lit.iterrows():
        print(f"    {x.modelo:42s} ±25 % {x.PRED25:.4f} · MdMRE {x.MdMRE:.4f}")

    print(f"\n{S}\nLECTURA\n{S}")
    a = float(t2[t2.modelo.str.startswith("nuestro, sin")].PRED25.iloc[0])
    s_ = float(t2[t2.modelo.str.contains("AJUSTADO")].PRED25.iloc[0])
    cc_ = float(t2[t2.modelo.str.startswith("la mejor")].PRED25.iloc[0])
    top = float(lit.PRED25.max())
    nom = lit.loc[lit.PRED25.idxmax(), "modelo"]
    print(f"  sin ajustar {a:.2%} · AJUSTADO {s_:.2%} ({(s_ - a) * 100:+.1f} pts)")
    print(f"  el mejor publicado: {nom} con {top:.2%}")
    print(f"  la mejor constante: {cc_:.2%}")
    if cc_ >= top:
        print(f"\n  OJO: una constante le gana a TODO lo publicado en esta metrica.")
        print("  No es un resultado sobre nuestro modelo, es sobre la metrica:")
        print("  PRED(25) mide la curtosis de estimado/real, no la capacidad de")
        print("  ordenar. Hay que reportarlo junto al SA y al spearman, donde la")
        print("  constante saca 0 y nosotros no.")
    t1.to_csv(OUT / "tamano_rejilla.csv", index=False)
    t2.to_csv(OUT / "contra_literatura.csv", index=False)
    json.dump({"tamano": int(mejor.tamano), "tipo": mejor.rejilla,
               "rejilla": [float(x) for x in rej_fin],
               "test_sin_ajustar": a, "test_ajustado": s_,
               "test_constante": cc_, "mejor_publicado": top},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/ · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
