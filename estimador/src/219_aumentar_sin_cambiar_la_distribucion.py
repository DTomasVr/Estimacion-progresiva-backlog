"""E104 - El experto corto sobreajusta. Las tres curas, sin traer datos de fuera.

Lo que se sabe hasta aqui sobre el experto en tareas cortas:

  E102  SOBREAJUSTA: brecha 0,3064 en log. Con iter=800 el error de
        entrenamiento cae a 0,07 y el de fuera no se mueve
  E102  la curva NO esta plana: del 50 % al 100 % de los proyectos el spearman
        se dobla, de 0,050 a 0,100
  E103  pero traer las 51.215 tareas cortas de Apache lo EMPEORA: spearman
        0,100 -> 0,054, porque su mediana es 0,50 h contra 1,00 h de las
        nuestras. Mas datos de otra distribucion hacen dano

O sea que el problema es varianza, y la solucion tiene que venir de los datos
que YA tenemos, sin cambiar su distribucion. Tres formas, de menos a mas
sofisticada, y las tres son lo que de verdad significa "datos sinteticos" aqui:

  RUIDO    perturbar el vector del texto. Equivale a decir "una tarea parecida
           cuesta parecido", que es la hipotesis del modelo
  MIXUP    interpolar pares de tareas y sus etiquetas en log. Genera ejemplos
           nuevos que viven entre dos reales, sin salirse de la distribucion
  BOLSA    promediar varios expertos con semillas y submuestras distintas. Es
           la cura clasica de la varianza y ya se usa en la tuberia principal

Se mide el spearman FUERA de pliegue, que es lo que al experto le falta, y la
brecha, para ver si de verdad baja la varianza o solo se mueve el error.
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
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/aumentar_experto")
S = "=" * 100
PLIEGUES, CORTE, N_BOLSA = 3, 2.0, 7
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)


def met(y, p, dentro=None):
    p = np.clip(np.asarray(p, float), 1e-6, None)
    r = np.abs(p - y) / y
    d = dict(err_log=float(np.median(np.abs(np.log(p) - np.log(y)))),
             rho=float(spearmanr(p, y).statistic),
             PRED25=float(np.mean(r <= .25)),
             sd_log_pred=float(np.std(np.log(p))))
    if dentro is not None:
        d["brecha"] = d["err_log"] - dentro
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy").astype(np.float32)
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap]
    y = sw.effort_hours.to_numpy(float)
    ly = np.log(y)
    proy = sw.project.to_numpy()
    corto = y <= CORTE
    esc = float(np.mean(np.linalg.norm(V, axis=1)))
    print(f"{int(corto.sum()):,} tareas cortas · {sw.project.nunique()} "
          f"proyectos · norma media del vector {esc:.2f} · "
          f"{time.time() - t0:.0f} s", flush=True)

    part = list(GroupKFold(PLIEGUES).split(V, y, proy))

    def corre(hacer, etq):
        """hacer(X, Y, rng) -> (X, Y) ya aumentados. Devuelve la fila medida."""
        ms, dentro = [], []
        for nf, (a, b) in enumerate(part):
            tr = a[corto[a]]
            rng = np.random.default_rng(2200 + nf)
            X, Y = hacer(V[tr], ly[tr], rng)
            mo = HistGradientBoostingRegressor(**PAR).fit(X, Y)
            bb = b[corto[b]]
            dentro.append(float(np.median(np.abs(
                mo.predict(V[tr]) - ly[tr]))))
            ms.append(met(y[bb], np.exp(mo.predict(V[bb]))))
        d = {c: float(np.mean([x[c] for x in ms])) for c in ms[0]}
        d["brecha"] = d["err_log"] - float(np.mean(dentro))
        print(f"  {etq} · {time.time() - t0:.0f} s", flush=True)
        return dict(brazo=etq, **d)

    filas = [corre(lambda X, Y, r: (X, Y), "A · el experto tal cual")]

    for s_ in (.02, .05, .10):
        def ruido(X, Y, r, s_=s_):
            Z = X + r.normal(0, s_ * esc / np.sqrt(X.shape[1]), X.shape
                             ).astype(np.float32)
            return np.vstack([X, Z]), np.concatenate([Y, Y])
        filas.append(corre(ruido, f"B · ruido en el vector sd={s_:g}"))

    for al in (.2, .4):
        def mixup(X, Y, r, al=al):
            n = len(X)
            j = r.permutation(n)
            lam = r.beta(al, al, n).astype(np.float32)[:, None]
            Z = (lam * X + (1 - lam) * X[j]).astype(np.float32)
            W = lam[:, 0] * Y + (1 - lam[:, 0]) * Y[j]
            return np.vstack([X, Z]), np.concatenate([Y, W])
        filas.append(corre(mixup, f"C · mixup alfa={al:g}"))

    # D: bolsa de expertos
    ms, dentro = [], []
    for nf, (a, b) in enumerate(part):
        tr = a[corto[a]]
        bb = b[corto[b]]
        preds, pin = [], []
        for s_ in range(N_BOLSA):
            rng = np.random.default_rng(3300 + 100 * nf + s_)
            sub = rng.choice(len(tr), len(tr), replace=True)
            mo = HistGradientBoostingRegressor(
                **dict(PAR, random_state=s_, max_features=.8)).fit(
                V[tr][sub], ly[tr][sub])
            preds.append(mo.predict(V[bb]))
            pin.append(mo.predict(V[tr]))
        dentro.append(float(np.median(np.abs(np.mean(pin, 0) - ly[tr]))))
        ms.append(met(y[bb], np.exp(np.mean(preds, 0))))
    d = {c: float(np.mean([x[c] for x in ms])) for c in ms[0]}
    d["brecha"] = d["err_log"] - float(np.mean(dentro))
    filas.append(dict(brazo=f"D · bolsa de {N_BOLSA} expertos", **d))
    print(f"  bolsa · {time.time() - t0:.0f} s", flush=True)

    # E: lo mejor de cada cosa, si procede
    t = pd.DataFrame(filas)
    print(f"\n{S}\nCURAR LA VARIANZA DEL EXPERTO CORTO\n{S}")
    print(t[["brazo", "rho", "err_log", "brecha", "PRED25",
             "sd_log_pred"]].round(4).to_string(index=False))

    print(f"\n{S}\nLECTURA\n{S}")
    a = t.iloc[0]
    mr = t.loc[t.rho.idxmax()]
    me = t.loc[t.err_log.idxmin()]
    print(f"  spearman: {a.rho:.4f} (tal cual) -> {mr.rho:.4f} ({mr.brazo})")
    print(f"  error:    {a.err_log:.4f} -> {me.err_log:.4f} ({me.brazo})")
    print(f"  brecha:   {a.brecha:.4f} -> {t.brecha.min():.4f} "
          f"({t.loc[t.brecha.idxmin()].brazo})")
    if mr.rho - a.rho > .015:
        print(f"\n  {mr.brazo} SI mejora el orden. Es la respuesta a lo sintetico:")
        print("  generar a partir de los datos que ya hay, sin importar")
        print("  distribuciones de fuera.")
    else:
        print("\n  Ninguna cura mueve el orden. El sobreajuste es real pero")
        print("  quitarlo no destapa senal: debajo no hay nada que aprender")
        print("  dentro del regimen corto, y por eso Apache tampoco servia.")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"rho_base": float(a.rho), "rho_mejor": float(mr.rho),
               "mejor": str(mr.brazo), "brecha_base": float(a.brecha),
               "brecha_min": float(t.brecha.min())},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
