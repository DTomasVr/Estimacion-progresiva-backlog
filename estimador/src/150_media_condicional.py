"""E45 - Para un TOTAL hace falta la media condicional. Mejor estimarla que corregirla.

E43 llega al total sumando medianas y multiplicando por un factor de smearing de
2,254. Funciona -- el sesgo pasa del -130 % al -2 % -- pero es un apano: se estima
la mediana y luego se la empuja hacia la media con un factor UNICO para todas las
tareas. Si la dispersion condicional cambia de una tarea a otra, y hay motivos
para pensar que si (E31: las tareas grandes se ven, las microtareas no), ese
factor esta mal tarea por tarea aunque acierte de media.

La alternativa es estimar E[y | texto] directamente, sin pasar por logaritmos.
Hay tres formas, cada una con su supuesto sobre la cola:

  cuadratica en horas   ningun supuesto, pero se la comen los valores extremos
  Poisson               varianza proporcional a la media
  gamma                 desviacion proporcional a la media, que es lo que suele
                        pasar con duraciones

Se comparan contra el apano actual sobre los mismos 114 proyectos, con
validacion cruzada por proyecto entero. La unidad de analisis es el proyecto.

Nota de ejecucion: se limita a 6 hilos a proposito, porque esto corre en paralelo
con un ajuste fino en la GPU y saturar los 12 nucleos deja sin CPU al hilo que la
alimenta. Eso ya costo tres horas una vez.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "6"

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/media_condicional")
S = "=" * 120
MIN_TAREAS, PLIEGUES = 8, 5


def mt(real, pred):
    real = np.asarray(real, float)
    pred = np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    q = np.log(pred / real)
    m = float(np.median(q))
    lr = np.log(real)
    return dict(MAE_h=float(np.abs(pred - real).mean()),
                RMSE_h=float(np.sqrt(np.mean((pred - real) ** 2))),
                MdMRE=float(np.median(r)), MMRE=float(r.mean()),
                R2_log=float(1 - ((lr - np.log(pred)) ** 2).mean() / lr.var()),
                PRED25=float(np.mean(r <= .25)), PRED50=float(np.mean(r <= .50)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def hgb(**kw):
    return HistGradientBoostingRegressor(max_iter=250, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=0, **kw)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = joblib.load("models/estimador_pbe_v3.joblib")
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    proy = sw.project.to_numpy()
    X = np.hstack([np.load(CACHE / "e5-large.npy")] +
                  [sv.transform(tf.transform(tit)).astype(np.float32)
                   for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    lp = np.load("results/analogia_cv/pred.npy")[:, 0]     # log-mediana, fuera de muestra
    print(f"{len(sw):,} tareas · {X.shape[1]} rasgos · {time.time() - t0:.0f} s", flush=True)

    perdidas = {"media directa, cuadratica en horas": dict(),
                "media directa, Poisson": dict(loss="poisson"),
                "media directa, gamma": dict(loss="gamma")}
    pred = {k: np.zeros(len(sw)) for k in perdidas}
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(X, y, proy)):
        for k, kw in perdidas.items():
            pred[k][b] = hgb(**kw).fit(X[a], y[a]).predict(X[b])
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s", flush=True)

    # el apano actual: mediana x smearing, dejando fuera el propio proyecto
    e = np.exp(ly - lp)
    g = pd.DataFrame({"p": proy, "e": e}).groupby("p").e.agg(["sum", "count"])
    fac = (e.sum() - g["sum"]) / (len(e) - g["count"])
    tramo = pd.qcut(lp, 6, labels=False, duplicates="drop")
    ft = pd.DataFrame({"t": tramo, "e": e}).groupby("t").e.mean()
    pred["mediana x smearing global"] = np.exp(lp) * pd.Series(proy).map(fac).to_numpy()
    pred["mediana x smearing por tramo"] = np.exp(lp) * ft.reindex(tramo).to_numpy()
    pred["media de las dos mejores"] = None          # se rellena abajo

    d = pd.DataFrame({"p": proy, "real": y})
    for k, v in pred.items():
        if v is not None:
            d[k] = np.clip(v, 1e-6, None)
    d["media de las dos mejores"] = np.sqrt(d["mediana x smearing por tramo"] *
                                            d["media directa, gamma"])
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)]
    tot = d.groupby("p").sum(numeric_only=True)
    print(f"{len(tot)} proyectos con al menos {MIN_TAREAS} tareas")

    cols = [c for c in d.columns if c not in ("p", "real")]
    r = pd.DataFrame([dict(estimador=c, **mt(tot.real, tot[c])) for c in cols])
    r = r.sort_values("MdMRE")
    print(f"\n{S}\nTOTAL DEL PROYECTO · que estimador de la media funciona mejor\n{S}")
    print(r[["estimador", "MAE_h", "RMSE_h", "MdMRE", "MMRE", "R2_log", "PRED25",
             "PRED50", "factor2", "SSPB_pc"]].round(4).to_string(index=False))

    print(f"\n{S}\nLOS MISMOS ESTIMADORES, TAREA A TAREA\n{S}")
    rt = pd.DataFrame([dict(estimador=c, **mt(d.real, d[c])) for c in cols])
    print(rt[["estimador", "MdMRE", "MMRE", "PRED25", "factor2",
              "SSPB_pc"]].round(4).to_string(index=False))
    print("  ojo: tarea a tarea estos estimadores tienen que salir PEOR que la mediana,")
    print("  porque estiman la media y la media no es el mejor punto para una tarea suelta")

    r.to_csv(OUT / "resumen.csv", index=False)
    rt.to_csv(OUT / "por_tarea.csv", index=False)
    json.dump({"min_tareas": MIN_TAREAS, "pliegues": PLIEGUES},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
