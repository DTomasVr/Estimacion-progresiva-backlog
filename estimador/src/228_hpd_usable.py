"""E111 - El conjunto HPD, hecho usable: pocos trozos y garantia conforme.

E110 midio que el conjunto de densidad predictiva maxima estrecha un 69 % -de
x31,99 a x10,04 con la misma cobertura- y ademas reparte mucho mejor la
cobertura por tramo. Era lo que anticipaba la literatura (Izbicki, Shimizu y
Stern con CD-split/HPD-split): la region HPD gana justo cuando la distribucion
es multimodal, y la nuestra es un peine.

Pero salia en 43 trozos, que no se puede ensenar. Aqui se hace usable sin
perder la garantia:

  1  UNIR los trozos separados por huecos pequenos
  2  QUEDARSE con los M trozos de mas masa
  3  recalibrar el umbral DESPUES de 1 y 2, buscandolo en los pliegues de
     calibracion para que la cobertura sea la pedida. Mientras el procedimiento
     sea una funcion determinista de x y del umbral, calibrar el umbral fuera
     de muestra conserva la cobertura

Y una variante que encaja con el producto: el conjunto sobre la REJILLA de
valores que la gente registra, que se lee como "lo mas probable: 1 h, 2 h o
4 h" en vez de como un intervalo.

  A  la banda de hoy
  C  HPD sin tocar (E110)
  E  HPD unido y recortado a M trozos, con M = 1, 2, 3
  G  HPD sobre la rejilla de 30 valores
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
OUT = Path("results/hpd_usable")
S = "=" * 112
PLIEGUES, NIVEL, K_VEC, LOTE = 3, .8, 200, 1024
N_BIN, SUAVE = 240, .25
HUECO = 0          # NO se rellenan huecos: la ganancia del HPD esta en dejarlos
MAXTROZOS = (2, 3, 5, 8)
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)
CORTES_Y = [1, 2, 4, 8, 16]
ETIQ_Y = ["<=1", "1-2", "2-4", "4-8", "8-16", ">16"]


def qconf(x, p, arriba):
    s = np.sort(np.asarray(x, float))
    m = len(s)
    j = (int(np.ceil((m + 1) * p)) if arriba else int(np.floor((m + 1) * p))) - 1
    return float(s[int(np.clip(j, 0, m - 1))])


def arregla(inc, dens, hueco, maxtr):
    """Une huecos pequenos y deja los maxtr trozos de mas masa."""
    out = np.zeros_like(inc)
    for i in range(len(inc)):
        v = inc[i].copy()
        if not v.any():
            v[int(np.argmax(dens[i]))] = True
        # unir huecos
        j = np.flatnonzero(v)
        if len(j) > 1:
            d = np.diff(j)
            for a, b in zip(j[:-1][d <= hueco + 1], j[1:][d <= hueco + 1]):
                v[a:b + 1] = True
        # trozos por masa
        j = np.flatnonzero(v)
        cortes = np.flatnonzero(np.diff(j) > 1)
        ini = np.r_[0, cortes + 1]
        fin = np.r_[cortes, len(j) - 1]
        masa = [dens[i][j[a]:j[b] + 1].sum() for a, b in zip(ini, fin)]
        ordw = np.argsort(masa)[::-1][:maxtr]
        w = np.zeros_like(v)
        for o in ordw:
            w[j[ini[o]]:j[fin[o]] + 1] = True
        out[i] = w
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
    sc = dens[np.arange(len(y)), b_y]
    a_ = (1 - NIVEL) / 2
    tr_y = np.digitize(y, CORTES_Y)

    filas = []

    def anota(nombre, dentro, med_log, trozos):
        w = np.exp(med_log)
        d = dict(conjunto=nombre, cobertura=float(dentro.mean()),
                 anchura_mediana=float(np.median(w)),
                 anchura_p25=float(np.quantile(w, .25)),
                 anchura_p75=float(np.quantile(w, .75)),
                 trozos_mediana=float(np.median(trozos)))
        for g, e in enumerate(ETIQ_Y):
            s_ = tr_y == g
            d[f"cob_{e}"] = float(dentro[s_].mean())
        filas.append(d)

    # A: la de hoy
    loA, hiA = np.zeros(len(y)), np.zeros(len(y))
    for nf in range(PLIEGUES):
        cal, ev = fold != nf, fold == nf
        loA[ev], hiA[ev] = qconf(r_ := (ly - p_med)[cal], a_, False), qconf(r_, 1 - a_, True)
    anota("A · la banda de hoy", (ly >= p_med + loA) & (ly <= p_med + hiA),
          hiA - loA, np.ones(len(y)))

    # C: HPD crudo
    umb = np.zeros(len(y))
    for nf in range(PLIEGUES):
        cal, ev = fold != nf, fold == nf
        umb[ev] = qconf(sc[cal], 1 - NIVEL, False)
    inc = dens >= umb[:, None]
    anota("C · HPD crudo (E110)", inc[np.arange(len(y)), b_y],
          inc.sum(1) * paso, np.maximum(
              (np.abs(np.diff(inc.astype(np.int8), 1)).sum(1) + inc[:, 0] +
               inc[:, -1]) // 2, 1))

    # E: unido y recortado, con el umbral REcalibrado para cada M
    for M in MAXTROZOS:
        fin = np.zeros_like(inc)
        for nf in range(PLIEGUES):
            cal, ev = fold != nf, fold == nf
            # buscar el umbral que da la cobertura pedida DESPUES de arreglar
            cands = np.quantile(sc[cal], np.linspace(.02, .45, 16))
            mejor, mejor_c = cands[0], -1
            for c in cands:
                ic = arregla(dens[cal] >= c, dens[cal], HUECO, M)
                cob = ic[np.arange(ic.shape[0]), b_y[cal]].mean()
                if cob >= NIVEL and (mejor_c < 0 or c > mejor):
                    mejor, mejor_c = c, cob
            fin[ev] = arregla(dens[ev] >= mejor, dens[ev], HUECO, M)
        tz = np.maximum((np.abs(np.diff(fin.astype(np.int8), 1)).sum(1) +
                         fin[:, 0] + fin[:, -1]) // 2, 1)
        anota(f"E · HPD unido, maximo {M} trozo{'s' if M > 1 else ''}",
              fin[np.arange(len(y)), b_y], fin.sum(1) * paso, tz)
        print(f"  M={M} listo · {time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(filas)
    print(f"\n{S}\nEL CONJUNTO AL {NIVEL:.0%}, HECHO USABLE ({len(y):,} tareas)\n{S}")
    print(t[["conjunto", "cobertura", "anchura_mediana", "anchura_p25",
             "anchura_p75", "trozos_mediana"]].round(3).to_string(index=False))
    print(f"\n{S}\nCOBERTURA POR TRAMO\n{S}")
    print(t[["conjunto"] + [f"cob_{e}" for e in ETIQ_Y]].round(3).to_string(index=False))

    print(f"\n{S}\nLECTURA\n{S}")
    a0 = t.iloc[0]
    for _, x in t.iloc[1:].iterrows():
        print(f"  {x.conjunto:38s} x{x.anchura_mediana:6.2f} "
              f"({x.cobertura:.1%}, {x.trozos_mediana:.0f} trozos) · "
              f"{1 - x.anchura_mediana / a0.anchura_mediana:+.0%} de anchura")
    us = t[t.conjunto.str.startswith("E")]
    us = us[us.cobertura >= NIVEL - .01]
    if len(us):
        b = us.loc[us.anchura_mediana.idxmin()]
        print(f"\n  USABLE: {b.conjunto} · x{b.anchura_mediana:.2f} cubriendo "
              f"{b.cobertura:.1%}, un {1 - b.anchura_mediana / a0.anchura_mediana:.0%} "
              f"mas estrecho que hoy.")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"n": int(len(y)), "nivel": NIVEL, "bins": N_BIN,
               "hueco": HUECO, "k_vec": K_VEC},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
