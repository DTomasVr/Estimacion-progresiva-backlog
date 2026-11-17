"""E110 - El conjunto MAS CORTO con la misma cobertura, explotando el peine.

E108 probo cuatro formas de estrechar la banda por tarea variando su ANCHURA
segun la tarea (acuerdo de vecinos, CQR, valor predicho) y ninguna bajo de x32.
El motivo es que el modelo no sabe que tareas son mas inciertas.

Pero queda la otra mitad del problema, que la literatura de prediccion conforme
trata aparte: la FORMA del conjunto.

  Izbicki, Shimizu y Stern (CD-split / HPD-split) construyen la region de
  DENSIDAD PREDICTIVA MAXIMA -el conjunto mas pequeno con la cobertura pedida-
  y obtienen regiones bastante menores que los metodos de intervalo. La ventaja
  aparece justo cuando la distribucion es ASIMETRICA o MULTIMODAL.

  Chernozhukov et al. (distributional conformal prediction) y Romano et al.
  (CQR) senalan ademas que el intervalo de colas iguales NO es el mas corto
  cuando la distribucion es asimetrica.

Nuestro objetivo es un peine: diez valores cubren el 46,8 % del corpus. Es el
caso de libro donde el conjunto HPD gana, porque puede quedarse con los dientes
y tirar los huecos.

  A  la banda de hoy: simetrica en log, colas iguales
  B  el INTERVALO mas corto, buscando el desplazamiento optimo (asimetrico)
  C  el conjunto HPD: union de trozos, ordenando por densidad
  D  el intervalo mas corto que CONTIENE al HPD, por si la interfaz quiere algo
     contiguo

La densidad condicional se estima con los K vecinos en el espacio del
codificador, suavizada hacia la marginal. Todo conforme: el umbral se calibra
en pliegues que no contienen la tarea, asi que la cobertura esta garantizada.

La anchura de un conjunto con huecos se mide como exp(su medida en log), que es
lo comparable con el x32 de un intervalo.
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
OUT = Path("results/conjunto_mas_corto")
S = "=" * 112
PLIEGUES, NIVEL, K_VEC, LOTE = 3, .8, 200, 1024
N_BIN, SUAVE = 240, .25
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
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
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
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

    # rejilla en log, uniforme: asi la medida de un trozo es su numero de bins
    lo_g, hi_g = np.quantile(ly, .001) - .05, np.quantile(ly, .999) + .05
    borde = np.linspace(lo_g, hi_g, N_BIN + 1)
    paso = borde[1] - borde[0]
    b_y = np.clip(np.digitize(ly, borde) - 1, 0, N_BIN - 1)
    print(f"{len(y):,} tareas · {N_BIN} bins de {paso:.4f} en log "
          f"(x{np.exp(paso):.3f} cada uno) · {time.time() - t0:.0f} s", flush=True)

    # prediccion central y densidad condicional por vecinos, fuera de pliegue
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

    r = ly - p_med
    a_ = (1 - NIVEL) / 2

    # ---------- A: simetrica en log, colas iguales ----------
    loA, hiA = np.zeros(len(y)), np.zeros(len(y))
    for nf in range(PLIEGUES):
        cal, ev = fold != nf, fold == nf
        loA[ev], hiA[ev] = qconf(r[cal], a_, False), qconf(r[cal], 1 - a_, True)
    setA = (p_med + loA, p_med + hiA)

    # ---------- B: el intervalo mas corto, desplazamiento optimo ----------
    # se busca el desplazamiento d que minimiza la anchura del intervalo conforme
    # [q(d), q(d+NIVEL)] sobre los residuos de calibracion
    loB, hiB = np.zeros(len(y)), np.zeros(len(y))
    for nf in range(PLIEGUES):
        cal, ev = fold != nf, fold == nf
        rc = np.sort(r[cal])
        n = len(rc)
        w = int(np.ceil((n + 1) * NIVEL))
        if w >= n:
            lo_, hi_ = rc[0], rc[-1]
        else:
            anch = rc[w:] - rc[:n - w]
            j = int(np.argmin(anch))
            lo_, hi_ = rc[j], rc[j + w]
        loB[ev], hiB[ev] = lo_, hi_
    setB = (p_med + loB, p_med + hiB)

    # ---------- C: HPD, union de trozos ----------
    # score = densidad estimada en el bin verdadero; umbral = cuantil conforme
    sc = dens[np.arange(len(y)), b_y]
    umb = np.zeros(len(y))
    for nf in range(PLIEGUES):
        cal, ev = fold != nf, fold == nf
        # el conjunto incluye los bins con densidad >= umbral; para cubrir el 80 %
        # el umbral es el cuantil 20 % de los scores de calibracion
        umb[ev] = qconf(sc[cal], 1 - NIVEL, False)
    inc = dens >= umb[:, None]
    # ---------- D: el intervalo mas corto que contiene al HPD ----------
    filas = []

    def mide(nombre, dentro, med_log, trozos):
        w = np.exp(med_log)
        d = dict(conjunto=nombre, cobertura=float(dentro.mean()),
                 anchura_mediana=float(np.median(w)),
                 anchura_media=float(w.mean()),
                 anchura_p25=float(np.quantile(w, .25)),
                 trozos_mediana=float(np.median(trozos)))
        for g, e in enumerate(ETIQ_Y):
            s_ = np.digitize(y, CORTES_Y) == g
            d[f"cob_{e}"] = float(dentro[s_].mean())
        return d

    for nombre, (L, H) in (("A · simetrica en log (hoy)", setA),
                           ("B · intervalo mas corto (asimetrico)", setB)):
        filas.append(mide(nombre, (ly >= L) & (ly <= H), H - L,
                          np.ones(len(y))))
    n_in = inc.sum(1)
    cambios = np.abs(np.diff(inc.astype(np.int8), axis=1)).sum(1)
    trozos = (cambios + inc[:, 0] + inc[:, -1]) // 2
    filas.append(mide("C · HPD (union de trozos)",
                      inc[np.arange(len(y)), b_y], n_in * paso,
                      np.maximum(trozos, 1)))
    prim = np.argmax(inc, 1)
    ult = N_BIN - 1 - np.argmax(inc[:, ::-1], 1)
    filas.append(mide("D · menor intervalo que contiene al HPD",
                      (b_y >= prim) & (b_y <= ult), (ult - prim + 1) * paso,
                      np.ones(len(y))))

    t = pd.DataFrame(filas)
    print(f"\n{S}\nEL CONJUNTO AL {NIVEL:.0%} ({len(y):,} tareas)\n{S}")
    print(t[["conjunto", "cobertura", "anchura_mediana", "anchura_media",
             "anchura_p25", "trozos_mediana"]].round(3).to_string(index=False))
    print(f"\n{S}\nCOBERTURA POR TRAMO DE HORAS REALES\n{S}")
    print(t[["conjunto"] + [f"cob_{e}" for e in ETIQ_Y]].round(3).to_string(index=False))

    print(f"\n{S}\nLECTURA\n{S}")
    a0 = t.iloc[0]
    for _, x in t.iloc[1:].iterrows():
        print(f"  {x.conjunto:40s} x{x.anchura_mediana:6.2f} "
              f"(cubre {x.cobertura:.1%}, {x.trozos_mediana:.0f} trozos) · "
              f"frente a x{a0.anchura_mediana:.2f}")
    mej = t.loc[t.anchura_mediana.idxmin()]
    if mej.anchura_mediana < a0.anchura_mediana * .9:
        print(f"\n  {mej.conjunto} estrecha un "
              f"{1 - mej.anchura_mediana / a0.anchura_mediana:.0%} con la misma "
              f"cobertura.")
    else:
        print("\n  Ninguna forma de conjunto estrecha de verdad: la masa no esta")
        print("  concentrada donde el modelo cree, aunque lo este en el corpus.")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"n": int(len(y)), "nivel": NIVEL, "bins": N_BIN, "k_vec": K_VEC,
               "suave": SUAVE}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
