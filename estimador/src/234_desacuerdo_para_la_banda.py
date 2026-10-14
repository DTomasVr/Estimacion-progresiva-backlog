"""E115 - El desacuerdo del ensemble como senal de incertidumbre por tarea.

Van seis vias medidas para estrechar la banda por tarea en frio y todas
negativas: acuerdo de vecinos, CQR y quintil del valor predicho (E108),
intervalo asimetrico y HPD contiguo (E110), y la ventana mas corta (E113).

Queda una senal que el artefacto YA calcula y que nunca se ha usado para la
banda: el DESACUERDO entre los cinco modelos de la bolsa. Es una senal distinta
de las probadas -no mira la densidad del objetivo sino la estabilidad del
predictor- y en la literatura de incertidumbre es de las mas usadas.

Y dos cosas que E108 no hizo:

  COMBINAR   alli cada senal se uso por separado y en quintiles. Aqui se
             aprende sigma(x) de todas juntas, que es la forma correcta de
             prediccion conforme normalizada
  IGUALAR    E114 enseno que comparar a nivel nominal engana cuando un metodo
             cubre menos. Aqui el nivel nominal de CADA metodo se ajusta hasta
             que la cobertura REAL sea 0,800 exacta, y solo entonces se comparan
             anchuras

  A  banda fija (lo de hoy)
  B  normalizada por el desacuerdo de la bolsa
  C  normalizada por sigma aprendido de TODAS las senales
  D  sigma aprendido, y ademas con el residuo en valor absoluto (banda
     simetrica) por si la asimetria metia ruido

Si C no gana aqui, la banda por tarea en frio esta en su limite: su anchura es
exactamente la que implica la dispersion del residuo, y esa no baja sin un
modelo mejor.
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
OUT = Path("results/desacuerdo_banda")
S = "=" * 112
PLIEGUES, NIVEL, N_BOLSA, K_VEC, LOTE = 3, .8, 5, 25, 1024
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)
PAR_S = dict(max_iter=200, max_depth=4, learning_rate=.06, l2_regularization=1.,
             random_state=0)
CORTES_Y = [1, 2, 4, 8, 16]
ETIQ_Y = ["<=1", "1-2", "2-4", "4-8", "8-16", ">16"]


def qconf(x, p, arriba):
    s = np.sort(np.asarray(x, float))
    m = len(s)
    j = (int(np.ceil((m + 1) * p)) if arriba else int(np.floor((m + 1) * p))) - 1
    return float(s[int(np.clip(j, 0, m - 1))])


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
    tit = sw.title.fillna("").astype(str)
    nw = tit.str.split().str.len().to_numpy(float)
    part = list(GroupKFold(PLIEGUES).split(V, y, proy))
    fold = np.zeros(len(y), int)
    for nf, (_, b) in enumerate(part):
        fold[b] = nf
    print(f"{len(y):,} tareas · {time.time() - t0:.0f} s", flush=True)

    p_med = np.zeros(len(y))
    desac = np.zeros(len(y))
    disp = np.zeros(len(y))
    for nf, (a, b) in enumerate(part):
        p_med[b] = HistGradientBoostingRegressor(**PAR).fit(V[a], ly[a]).predict(V[b])
        pr = []
        for s_ in range(N_BOLSA):
            rng = np.random.default_rng(700 + 10 * nf + s_)
            sub = rng.choice(len(a), len(a), replace=True)
            mo = HistGradientBoostingRegressor(
                **dict(PAR, random_state=s_, max_features=.8)).fit(
                V[a][sub], ly[a][sub])
            pr.append(mo.predict(V[b]))
        desac[b] = np.std(pr, axis=0)
        for i in range(0, len(b), LOTE):
            bb = b[i:i + LOTE]
            sim = Vn[bb] @ Vn[a].T
            vec = np.argpartition(-sim, K_VEC, axis=1)[:, :K_VEC]
            disp[bb] = np.std(ly[a][vec], axis=1)
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s",
              flush=True)

    r = ly - p_med
    print(f"\n  correlacion con |residuo|:  desacuerdo "
          f"{np.corrcoef(np.abs(r), desac)[0, 1]:+.4f} · dispersion de vecinos "
          f"{np.corrcoef(np.abs(r), disp)[0, 1]:+.4f}", flush=True)

    X = np.column_stack([desac, disp, p_med, nw]).astype(np.float32)
    sig = np.zeros(len(y))
    for nf, (a, b) in enumerate(part):
        sig[b] = np.exp(HistGradientBoostingRegressor(**PAR_S).fit(
            X[a], np.log(np.abs(r[a]) + 1e-3)).predict(X[b]))
    sig = np.clip(sig, 1e-3, None)
    print(f"  sigma aprendido · correlacion con |residuo| "
          f"{np.corrcoef(np.abs(r), sig)[0, 1]:+.4f}", flush=True)

    tr_y = np.digitize(y, CORTES_Y)

    def banda(etq, nivel_nom):
        aa = (1 - nivel_nom) / 2
        lo, hi = np.zeros(len(y)), np.zeros(len(y))
        for nf in range(PLIEGUES):
            cal, ev = fold != nf, fold == nf
            if etq == "A":
                lo[ev], hi[ev] = qconf(r[cal], aa, False), qconf(r[cal], 1 - aa, True)
            elif etq == "D":
                q = qconf(np.abs(r[cal] / sig[cal]), nivel_nom, True)
                lo[ev], hi[ev] = -q * sig[ev], q * sig[ev]
            else:
                z = r[cal] / (desac[cal] if etq == "B" else sig[cal])
                e = desac[ev] if etq == "B" else sig[ev]
                lo[ev], hi[ev] = qconf(z, aa, False) * e, qconf(z, 1 - aa, True) * e
        return (ly >= p_med + lo) & (ly <= p_med + hi), np.exp(hi - lo)

    NOM = {"A": "A · banda fija (hoy)", "B": "B · por el desacuerdo de la bolsa",
           "C": "C · sigma aprendido de todas", "D": "D · sigma aprendido, simetrica"}
    filas = []
    for etq in ("A", "B", "C", "D"):
        lo_n, hi_n = .50, .9995
        for _ in range(14):
            md = (lo_n + hi_n) / 2
            d_, _ = banda(etq, md)
            if d_.mean() < NIVEL:
                lo_n = md
            else:
                hi_n = md
        dentro, w = banda(etq, hi_n)
        d = dict(banda=NOM[etq], nivel_nominal=float(hi_n),
                 cobertura=float(dentro.mean()),
                 anchura_mediana=float(np.median(w)),
                 anchura_p10=float(np.quantile(w, .1)),
                 anchura_p90=float(np.quantile(w, .9)))
        for g, e in enumerate(ETIQ_Y):
            d[f"cob_{e}"] = float(dentro[tr_y == g].mean())
        filas.append(d)
        print(f"  {NOM[etq]} listo · {time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(filas)
    print(f"\n{S}\nLA BANDA POR TAREA EN FRIO, TODAS A COBERTURA REAL 0,800\n{S}")
    print(t[["banda", "nivel_nominal", "cobertura", "anchura_mediana",
             "anchura_p10", "anchura_p90"]].round(3).to_string(index=False))
    print(f"\n{S}\nCOBERTURA POR TRAMO\n{S}")
    print(t[["banda"] + [f"cob_{e}" for e in ETIQ_Y]].round(3).to_string(index=False))

    print(f"\n{S}\nLECTURA\n{S}")
    a0 = t.iloc[0]
    mej = t.iloc[1:].loc[t.iloc[1:].anchura_mediana.idxmin()]
    for _, x in t.iloc[1:].iterrows():
        print(f"  {x.banda:36s} x{x.anchura_mediana:6.2f} · "
              f"{1 - x.anchura_mediana / a0.anchura_mediana:+.0%} · "
              f"p10 x{x.anchura_p10:.2f} · p90 x{x.anchura_p90:.2f}")
    g = 1 - mej.anchura_mediana / a0.anchura_mediana
    if g > .05:
        print(f"\n  {mej.banda} estrecha un {g:.0%} a la misma cobertura.")
    else:
        print(f"\n  Nada estrecha ({g:+.0%}). La anchura de la banda por tarea es")
        print("  exactamente la que implica la dispersion del residuo, y esa no")
        print("  baja sin un modelo mejor. Es el limite, no una mala calibracion.")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"n": int(len(y)), "nivel": NIVEL,
               "corr_desacuerdo": float(np.corrcoef(np.abs(r), desac)[0, 1]),
               "corr_sigma": float(np.corrcoef(np.abs(r), sig)[0, 1]),
               "anchura_hoy": float(a0.anchura_mediana),
               "anchura_mejor": float(mej.anchura_mediana)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
