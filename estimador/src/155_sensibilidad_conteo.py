"""E49 - El total depende de cuantas tareas genere la pipeline, no solo de acertar cada una.

Toda la metrica de proyecto se apoya en una suma sobre las tareas que la pipeline
de generacion produce. Pero esa pipeline no sabe cuantas tareas va a tener el
proyecto de verdad: genera las que le parecen. Si genera 40 cuando el proyecto
acaba teniendo 55, el total se queda corto un 27 % por mucho que cada tarea este
perfectamente estimada.

Eso no es un detalle: es una fuente de error que NO esta en ninguna de las cifras
que hemos reportado, porque en la evaluacion las tareas son las reales. Antes de
presentarlo hay que saber cuanto pesa, y decirlo.

Se mide de dos formas:

  A  CON EL MODELO         se sortea una fraccion f de las tareas de cada
                           proyecto, se suman sus estimaciones y se compara con el
                           total real ENTERO. Es lo que pasaria si la pipeline
                           generase de mas o de menos.
  B  CON HORAS PERFECTAS   lo mismo pero sumando las horas REALES de las tareas
                           sorteadas. Aisla la parte del error que viene solo del
                           conteo, sin mezclarla con el error del modelo. Es la
                           cota: ni con un modelo perfecto se baja de ahi.

La comparacion entre A y B dice cual de los dos problemas conviene atacar.
"""
import json
import os
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "4"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
OUT = Path("results/sensibilidad_conteo")
S = "=" * 112
MIN_TAREAS, SORTEOS = 8, 60
FRACCIONES = (.5, .7, .8, .9, 1.0, 1.1, 1.3, 1.5)


def tasas(real, pred):
    real = np.asarray(real, float)
    pred = np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    q = np.log(pred / real)
    m = float(np.median(q))
    return dict(MdMRE=float(np.median(r)), PRED25=float(np.mean(r <= .25)),
                PRED50=float(np.mean(r <= .50)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    proy = sw.project.to_numpy()
    lp = np.load("results/analogia_cv/pred.npy")[:, 0]
    e = np.exp(ly - lp)
    tramo = pd.qcut(lp, 6, labels=False, duplicates="drop")
    ft = pd.DataFrame({"t": tramo, "e": e}).groupby("t").e.mean()
    est = np.exp(lp) * ft.reindex(tramo).to_numpy()

    d = pd.DataFrame({"p": proy, "real": y, "est": est})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)].reset_index(drop=True)
    nombres = list(pd.unique(d.p))
    idx = {p: np.flatnonzero(d.p.to_numpy() == p) for p in nombres}
    total = d.groupby("p").real.sum().reindex(nombres).to_numpy()
    print(f"{len(nombres)} proyectos · {SORTEOS} sorteos por fraccion")

    rng = np.random.default_rng(0)
    filas = []
    for f in FRACCIONES:
        acA = np.zeros(len(nombres))
        acB = np.zeros(len(nombres))
        for _ in range(SORTEOS):
            a, b = [], []
            for p in nombres:
                i = idx[p]
                m = max(1, int(round(f * len(i))))
                s = (rng.choice(i, m, replace=False) if m <= len(i)
                     else np.concatenate([i, rng.choice(i, m - len(i), replace=True)]))
                a.append(d.est.to_numpy()[s].sum())
                b.append(d.real.to_numpy()[s].sum())
            acA += np.array(a)
            acB += np.array(b)
        filas.append(dict(fraccion=f, fuente="A · con el modelo",
                          **tasas(total, acA / SORTEOS)))
        filas.append(dict(fraccion=f, fuente="B · con horas perfectas",
                          **tasas(total, acB / SORTEOS)))
    t = pd.DataFrame(filas)
    for fuente in ("B · con horas perfectas", "A · con el modelo"):
        v = t[t.fuente == fuente]
        print(f"\n{S}\n{fuente}\n{S}")
        print(v[["fraccion", "MdMRE", "PRED25", "PRED50", "factor2",
                 "SSPB_pc"]].round(4).to_string(index=False))
    def q(fu, f):
        return t[(t.fuente == fu) & (t.fraccion == f)].iloc[0]
    A, B = "A · con el modelo", "B · con horas perfectas"
    print(f"\n{S}\nLECTURA\n{S}")
    print(f"  con horas PERFECTAS y un 20 % de tareas de menos, el "
          f"{q(B, .8).PRED25:.1%} de los proyectos sigue dentro del +-25 %:")
    print(f"  un error del 20 % en el conteo se traduce en un 20 % en el total, y eso")
    print(f"  cabe dentro de la ventana. Con el modelo real y el conteo EXACTO, solo el")
    print(f"  {q(A, 1.).PRED25:.1%}.")
    print()
    print(f"  o sea que el error del modelo por tarea pesa MUCHO mas que el del conteo.")
    print(f"  La pipeline de generacion tendria que equivocarse en mas de un 25 % del")
    print(f"  numero de tareas para que el conteo fuese el problema principal.")
    print()
    print(f"  robustez practica del +-25 %: {q(A, .9).PRED25:.1%} con un 10 % de tareas")
    print(f"  de menos · {q(A, 1.).PRED25:.1%} con el conteo exacto · "
          f"{q(A, 1.1).PRED25:.1%} con un 10 % de mas")
    print(f"  y el factor 2 apenas se mueve entre 0,8 y 1,1 "
          f"({q(A, .8).factor2:.1%} a {q(A, 1.1).factor2:.1%})")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"sorteos": SORTEOS, "fracciones": list(FRACCIONES)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
