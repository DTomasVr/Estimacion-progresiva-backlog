"""E112 - La curva de canje del conjunto HPD: cuantos trozos contra cuanto estrecha.

E110 midio que el conjunto de densidad predictiva maxima estrecha un 69 % -de
x31,99 a x10,04- con la misma cobertura, que es lo que anticipa la literatura
para distribuciones multimodales (Izbicki, Shimizu y Stern). Pero sale en 43
trozos.

E111 intento recortarlo y salio erratico: truncar a M trozos rompe la
monotonia del umbral sobre la densidad, asi que la busqueda del umbral no
converge. Aqui se arregla el procedimiento:

  el conjunto se construye acumulando MASA, no cortando por densidad: se
  ordenan los bins por densidad, se van sumando hasta llegar a tau, se agrupan
  en trozos y se dejan los M de mas masa. Con M fijo, la cobertura crece de
  forma monotona con tau, asi que tau se encuentra por BUSQUEDA BINARIA sobre
  los pliegues de calibracion y la cobertura sale donde se pide

Se barre M para ver la curva de canje entre cuantos trozos puede ensenar la
interfaz y cuanto se estrecha de verdad.
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
OUT = Path("results/hpd_curva")
S = "=" * 112
PLIEGUES, NIVEL, K_VEC, LOTE = 3, .8, 200, 1024
N_BIN, SUAVE = 240, .25
MS = (1, 2, 3, 5, 10, 999)
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)
CORTES_Y = [1, 2, 4, 8, 16]
ETIQ_Y = ["<=1", "1-2", "2-4", "4-8", "8-16", ">16"]


def qconf(x, p, arriba):
    s = np.sort(np.asarray(x, float))
    m = len(s)
    j = (int(np.ceil((m + 1) * p)) if arriba else int(np.floor((m + 1) * p))) - 1
    return float(s[int(np.clip(j, 0, m - 1))])


def construye(dens, tau, M):
    """Bins ordenados por densidad hasta acumular tau, y los M trozos de mas masa."""
    n, B = dens.shape
    ordn = np.argsort(-dens, axis=1)
    acum = np.cumsum(np.take_along_axis(dens, ordn, 1), axis=1)
    # cuantos bins hacen falta por fila
    cuantos = (acum < tau).sum(1) + 1
    inc = np.zeros((n, B), bool)
    cols = np.arange(B)[None, :]
    marca = cols < cuantos[:, None]
    inc[np.repeat(np.arange(n), B).reshape(n, B)[marca], ordn[marca]] = True
    if M >= B:
        return inc
    out = np.zeros_like(inc)
    for i in range(n):
        j = np.flatnonzero(inc[i])
        if len(j) == 0:
            continue
        cor = np.flatnonzero(np.diff(j) > 1)
        ini, fin = np.r_[0, cor + 1], np.r_[cor, len(j) - 1]
        if len(ini) <= M:
            out[i] = inc[i]
            continue
        masa = np.array([dens[i, j[a]:j[b] + 1].sum() for a, b in zip(ini, fin)])
        for o in np.argsort(-masa)[:M]:
            out[i, j[ini[o]]:j[fin[o]] + 1] = True
    return out


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
    paso = borde[1] - borde[0]
    b_y = np.clip(np.digitize(ly, borde) - 1, 0, N_BIN - 1)
    print(f"{len(y):,} tareas · {N_BIN} bins de x{np.exp(paso):.3f} · "
          f"{time.time() - t0:.0f} s", flush=True)

    p_med = np.zeros(len(y))
    dens = np.zeros((len(y), N_BIN), np.float32)
    for nf, (a, b) in enumerate(part):
        p_med[b] = HistGradientBoostingRegressor(**PAR).fit(V[a], ly[a]).predict(V[b])
        marg = np.bincount(b_y[a], minlength=N_BIN).astype(np.float32)
        marg /= marg.sum()
        for i in range(0, len(b), LOTE):
            bb = b[i:i + LOTE]
            sim = Vn[bb] @ Vn[a].T
            vec = np.argpartition(-sim, K_VEC, axis=1)[:, :K_VEC]
            d = np.zeros((len(bb), N_BIN), np.float32)
            fil = np.repeat(np.arange(len(bb)), K_VEC)
            np.add.at(d, (fil, b_y[a][vec].ravel()), 1.0)
            d /= d.sum(1, keepdims=True)
            dens[bb] = (1 - SUAVE) * d + SUAVE * marg
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s",
              flush=True)

    tr_y = np.digitize(y, CORTES_Y)
    filas = []

    def anota(nombre, dentro, med_log, trozos, tau=np.nan):
        w = np.exp(med_log)
        d = dict(conjunto=nombre, tau=float(tau), cobertura=float(dentro.mean()),
                 anchura_mediana=float(np.median(w)),
                 anchura_p25=float(np.quantile(w, .25)),
                 anchura_p75=float(np.quantile(w, .75)),
                 trozos_mediana=float(np.median(trozos)),
                 trozos_p90=float(np.quantile(trozos, .9)))
        for g, e in enumerate(ETIQ_Y):
            d[f"cob_{e}"] = float(dentro[tr_y == g].mean())
        filas.append(d)

    r = ly - p_med
    loA, hiA = np.zeros(len(y)), np.zeros(len(y))
    for nf in range(PLIEGUES):
        cal, ev = fold != nf, fold == nf
        loA[ev] = qconf(r[cal], (1 - NIVEL) / 2, False)
        hiA[ev] = qconf(r[cal], 1 - (1 - NIVEL) / 2, True)
    anota("A · la banda de hoy", (ly >= p_med + loA) & (ly <= p_med + hiA),
          hiA - loA, np.ones(len(y)))

    def trozos_de(inc):
        return np.maximum((np.abs(np.diff(inc.astype(np.int8), 1)).sum(1) +
                           inc[:, 0] + inc[:, -1]) // 2, 1)

    for M in MS:
        fin = np.zeros((len(y), N_BIN), bool)
        taus = []
        for nf in range(PLIEGUES):
            cal, ev = np.flatnonzero(fold != nf), np.flatnonzero(fold == nf)
            lo_t, hi_t = .50, .9995
            for _ in range(12):                      # busqueda binaria en tau
                md = (lo_t + hi_t) / 2
                ic = construye(dens[cal], md, M)
                cob = ic[np.arange(len(cal)), b_y[cal]].mean()
                if cob < NIVEL:
                    lo_t = md
                else:
                    hi_t = md
            taus.append(hi_t)
            fin[ev] = construye(dens[ev], hi_t, M)
        et = f"HPD · maximo {M} trozos" if M < 999 else "HPD · sin limite de trozos"
        anota(et, fin[np.arange(len(y)), b_y], fin.sum(1) * paso,
              trozos_de(fin), float(np.mean(taus)))
        print(f"  M={M} listo (tau={np.mean(taus):.4f}) · "
              f"{time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(filas)
    print(f"\n{S}\nCURVA DE CANJE: TROZOS CONTRA ANCHURA, AL {NIVEL:.0%}\n{S}")
    print(t[["conjunto", "tau", "cobertura", "anchura_mediana", "anchura_p25",
             "anchura_p75", "trozos_mediana", "trozos_p90"]].round(3)
          .to_string(index=False))
    print(f"\n{S}\nCOBERTURA POR TRAMO\n{S}")
    print(t[["conjunto"] + [f"cob_{e}" for e in ETIQ_Y]].round(3).to_string(index=False))

    print(f"\n{S}\nLECTURA\n{S}")
    a0 = t.iloc[0]
    print(f"  hoy: x{a0.anchura_mediana:.2f} en 1 trozo, cubre "
          f"{a0.cobertura:.1%}, y el reparto por tramo va de "
          f"{min(a0[f'cob_{e}'] for e in ETIQ_Y):.2f} a "
          f"{max(a0[f'cob_{e}'] for e in ETIQ_Y):.2f}")
    for _, x in t.iloc[1:].iterrows():
        rep = (f"{min(x[f'cob_{e}'] for e in ETIQ_Y):.2f}-"
               f"{max(x[f'cob_{e}'] for e in ETIQ_Y):.2f}")
        print(f"  {x.conjunto:30s} x{x.anchura_mediana:6.2f} · "
              f"{1 - x.anchura_mediana / a0.anchura_mediana:+6.0%} · "
              f"cubre {x.cobertura:.1%} · {x.trozos_mediana:.0f} trozos "
              f"(p90 {x.trozos_p90:.0f}) · reparto {rep}")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"n": int(len(y)), "nivel": NIVEL, "bins": N_BIN, "k_vec": K_VEC,
               "suave": SUAVE}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
