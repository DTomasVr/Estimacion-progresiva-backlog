"""E113 - El intervalo CONTIGUO mas corto por tarea, en frio.

Lo que queda sin estrechar es la banda por tarea cuando no hay ni estimacion
del usuario ni tareas cerradas. E108 probo cuatro formas y ninguna bajo del
x32 del modelo limpio; E110 anadio el intervalo asimetrico (x31,6, nada) y el
conjunto HPD, que baja a x10 pero sale en 43 trozos.

Falta la via de enmedio, que es la que Chernozhukov et al. llaman la variante
OPTIMA de la prediccion conforme distribucional: en vez de tomar los cuantiles
de colas iguales del residuo GLOBAL, se busca para CADA tarea la ventana
CONTIGUA mas corta que acumula la masa pedida de SU densidad condicional. Sigue
siendo un intervalo -se puede ensenar- pero su posicion y su anchura se adaptan.

La diferencia con lo ya probado:

  E108 D   agrupaba por quintil del valor predicho: cinco anchuras distintas
  E110 B   buscaba UN desplazamiento optimo, el mismo para todas
  aqui     una ventana propia por tarea, de la densidad condicional

Y como la densidad es lo que manda, se barre su nitidez: cuantos vecinos y
cuanto se suaviza hacia la marginal. Menos vecinos y menos suavizado dan una
densidad mas afilada, que es lo que puede estrechar la ventana.

El nivel de masa tau se calibra por busqueda binaria en los pliegues de
calibracion para que la cobertura salga donde se pide, asi que la comparacion
con la banda de hoy es a cobertura igual.
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
OUT = Path("results/ventana_mas_corta")
S = "=" * 112
PLIEGUES, NIVEL, LOTE, N_BIN = 3, .8, 1024, 240
CONFIGS = [(200, .25), (200, .05), (100, .10), (50, .10), (50, .02), (25, .05)]
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)
CORTES_Y = [1, 2, 4, 8, 16]
ETIQ_Y = ["<=1", "1-2", "2-4", "4-8", "8-16", ">16"]


def qconf(x, p, arriba):
    s = np.sort(np.asarray(x, float))
    m = len(s)
    j = (int(np.ceil((m + 1) * p)) if arriba else int(np.floor((m + 1) * p))) - 1
    return float(s[int(np.clip(j, 0, m - 1))])


def ventanas(dens, tau):
    """Por fila, la ventana contigua mas corta con masa >= tau. Devuelve (ini, fin)."""
    n, B = dens.shape
    ini = np.zeros(n, int)
    fin = np.full(n, B - 1)
    c = np.cumsum(dens, axis=1)
    for i in range(n):
        ci = c[i]
        antes = np.r_[0.0, ci[:-1]]
        obj = antes + tau
        j = np.searchsorted(ci, obj, side="left")
        val = j < B
        if not val.any():
            continue
        anch = np.where(val, j - np.arange(B), B + 1)
        k = int(np.argmin(anch))
        ini[i], fin[i] = k, min(j[k], B - 1)
    return ini, fin


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
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    part = list(GroupKFold(PLIEGUES).split(V, y, proy))
    fold = np.zeros(len(y), int)
    for nf, (_, b) in enumerate(part):
        fold[b] = nf
    lo_g, hi_g = np.quantile(ly, .001) - .05, np.quantile(ly, .999) + .05
    borde = np.linspace(lo_g, hi_g, N_BIN + 1)
    paso = float(borde[1] - borde[0])
    b_y = np.clip(np.digitize(ly, borde) - 1, 0, N_BIN - 1)
    tr_y = np.digitize(y, CORTES_Y)
    print(f"{len(y):,} tareas · {N_BIN} bins de x{np.exp(paso):.3f} · "
          f"{time.time() - t0:.0f} s", flush=True)

    p_med = np.zeros(len(y))
    for a, b in part:
        p_med[b] = HistGradientBoostingRegressor(**PAR).fit(V[a], ly[a]).predict(V[b])
    filas = []

    def anota(nombre, dentro, med_log, extra=None):
        w = np.exp(med_log)
        d = dict(conjunto=nombre, cobertura=float(dentro.mean()),
                 anchura_mediana=float(np.median(w)),
                 anchura_p10=float(np.quantile(w, .1)),
                 anchura_p90=float(np.quantile(w, .9)))
        d.update(extra or {})
        for g, e in enumerate(ETIQ_Y):
            d[f"cob_{e}"] = float(dentro[tr_y == g].mean())
        filas.append(d)

    r = ly - p_med
    loA, hiA = np.zeros(len(y)), np.zeros(len(y))
    for nf in range(PLIEGUES):
        cal, ev = fold != nf, fold == nf
        loA[ev] = qconf(r[cal], (1 - NIVEL) / 2, False)
        hiA[ev] = qconf(r[cal], 1 - (1 - NIVEL) / 2, True)
    anota("A · simetrica en log (hoy)", (ly >= p_med + loA) & (ly <= p_med + hiA),
          hiA - loA, dict(K=0, suave=0.))
    print(f"  banda de hoy: x{np.exp(np.median(hiA - loA)):.2f} · "
          f"{filas[0]['cobertura']:.1%}", flush=True)

    for K, SU in CONFIGS:
        dens = np.zeros((len(y), N_BIN), np.float32)
        for nf, (a, b) in enumerate(part):
            marg = np.bincount(b_y[a], minlength=N_BIN).astype(np.float32)
            marg /= marg.sum()
            for i in range(0, len(b), LOTE):
                bb = b[i:i + LOTE]
                sim = Vn[bb] @ Vn[a].T
                vec = np.argpartition(-sim, K, axis=1)[:, :K]
                d = np.zeros((len(bb), N_BIN), np.float32)
                fil = np.repeat(np.arange(len(bb)), K)
                np.add.at(d, (fil, b_y[a][vec].ravel()), 1.0)
                d /= d.sum(1, keepdims=True)
                dens[bb] = (1 - SU) * d + SU * marg
        ini = np.zeros(len(y), int)
        fin = np.zeros(len(y), int)
        taus = []
        for nf in range(PLIEGUES):
            cal = np.flatnonzero(fold != nf)
            ev = np.flatnonzero(fold == nf)
            lo_t, hi_t = .30, .9995
            for _ in range(10):
                md = (lo_t + hi_t) / 2
                i0, i1 = ventanas(dens[cal], md)
                cob = ((b_y[cal] >= i0) & (b_y[cal] <= i1)).mean()
                if cob < NIVEL:
                    lo_t = md
                else:
                    hi_t = md
            taus.append(hi_t)
            ini[ev], fin[ev] = ventanas(dens[ev], hi_t)
        anota(f"B · ventana mas corta · K={K} suave={SU:g}",
              (b_y >= ini) & (b_y <= fin), (fin - ini + 1) * paso,
              dict(K=K, suave=SU, tau=float(np.mean(taus))))
        print(f"  K={K} suave={SU:g}: x{filas[-1]['anchura_mediana']:.2f} · "
              f"{filas[-1]['cobertura']:.1%} · tau={np.mean(taus):.4f} · "
              f"{time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(filas)
    print(f"\n{S}\nLA VENTANA CONTIGUA MAS CORTA, EN FRIO ({len(y):,} tareas)\n{S}")
    print(t[["conjunto", "cobertura", "anchura_mediana", "anchura_p10",
             "anchura_p90"]].round(3).to_string(index=False))
    print(f"\n{S}\nCOBERTURA POR TRAMO\n{S}")
    print(t[["conjunto"] + [f"cob_{e}" for e in ETIQ_Y]].round(3).to_string(index=False))

    print(f"\n{S}\nLECTURA\n{S}")
    a0 = t.iloc[0]
    mej = t.iloc[1:].loc[t.iloc[1:].anchura_mediana.idxmin()]
    print(f"  hoy          x{a0.anchura_mediana:6.2f} (cubre {a0.cobertura:.1%})")
    print(f"  lo mejor     x{mej.anchura_mediana:6.2f} (cubre {mej.cobertura:.1%}) "
          f"· {mej.conjunto}")
    g = 1 - mej.anchura_mediana / a0.anchura_mediana
    if g > .05:
        print(f"\n  Estrecha un {g:.0%} siendo UN SOLO INTERVALO, asi que se puede")
        print("  ensenar tal cual. Y la densidad mas afilada ayuda: ver el barrido.")
    else:
        print(f"\n  No estrecha ({g:+.0%}). La densidad condicional no esta mas")
        print("  concentrada que el residuo global: el modelo no sabe que tareas")
        print("  son mas ciertas, solo donde esta la masa del corpus.")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"n": int(len(y)), "nivel": NIVEL, "bins": N_BIN,
               "mejor": str(mej.conjunto),
               "anchura_hoy": float(a0.anchura_mediana),
               "anchura_mejor": float(mej.anchura_mediana)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
