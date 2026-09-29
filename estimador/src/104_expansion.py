"""E9 · Descomprimir dentro del proyecto, aprovechando que el backlog llega entero.

El defecto de fondo esta medido desde hace dias: el modelo comprime. Sus salidas
caben entre 1,3 y 7 h mientras la realidad va de 0,07 a mas de 40. La compresion
es la respuesta optima de una perdida cuadratica cuando el rasgo es poco
informativo -- no es un fallo de implementacion -- pero se puede DESHACER si uno
esta dispuesto a cambiar error cuadratico por calibracion de la dispersion, que
es justo el compromiso que describe la literatura de pronostico probabilistico
("sharpness subject to calibration") y el de regresion desbalanceada
(Dist Loss, 2025: alinear la distribucion predicha con la real).

Aqui se hace de la forma mas barata posible, en la salida y no en la perdida:

    log p' = media_proyecto(log p) + s x (log p - media_proyecto(log p))

con s > 1. Agix genera el backlog entero de golpe, asi que la media del proyecto
sobre las predicciones esta disponible sin coste y sin fuga: no usa ninguna hora
real, solo las predicciones del propio lote.

s se elige en VALIDACION. Se prueba tambien la version global (expandir contra
la media del entrenamiento) para separar el efecto de "expandir" del efecto de
"expandir dentro del proyecto".

Se combina con las anclas contraidas de E8b, que resuelven el otro termino: la
expansion arregla la DISPERSION y el ancla arregla el NIVEL.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
DATA = "data/processed/agix_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/expansion")
SEP = "=" * 116
LAMBDA = 12.0


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly, lp = np.log(y), np.log(p)
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lp - ly))),
                PRED25=m["PRED25"], MAE_h=m["MAE_h"],
                R2=float(1 - ((ly - lp) ** 2).mean() / ly.var()),
                sesgo=float(np.exp(np.median(lp - ly))),
                compresion=float(lp.std() / ly.std()),
                spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def expandir(lp, proy, s, global_=False):
    if s == 1.0:
        return lp.copy()
    if global_:
        m = lp.mean()
        return m + s * (lp - m)
    d = pd.DataFrame({"p": proy, "lp": lp})
    m = d.groupby("p").lp.transform("mean").to_numpy()
    return m + s * (lp - m)


def anclar(lp, ly, proy, n_anc, lam, semilla):
    """Desplazamiento contraido de E8b; devuelve el log-pred desplazado y la mascara."""
    if n_anc == 0:
        return lp, np.ones(len(lp), bool)
    rng = np.random.default_rng(semilla)
    out = lp.copy()
    usar = np.zeros(len(lp), bool)
    for p in np.unique(proy):
        idx = np.flatnonzero(proy == p)
        if len(idx) <= n_anc:
            continue
        anc = rng.choice(idx, n_anc, replace=False)
        resto = np.setdiff1d(idx, anc)
        desp = float(np.median(ly[anc] - lp[anc])) * n_anc / (n_anc + lam)
        out[resto] += desp
        usar[resto] = True
    return out, usar


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sw = sw[~sw.apache].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    tr_m = (sw.split_t3 == "train").to_numpy()

    X = [np.load(CACHE / "e5-large.npy")]
    for kw in (dict(analyzer="char_wb", ngram_range=(3, 5)), dict(ngram_range=(1, 2))):
        tf = TfidfVectorizer(min_df=3, max_features=200_000, sublinear_tf=True,
                             **kw).fit(tit[tr_m])
        M = tf.transform(tit)
        X.append(TruncatedSVD(256, random_state=0).fit(M[tr_m]).transform(M).astype(np.float32))
    X = np.hstack(X)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                      l2_regularization=1., random_state=0).fit(
        X[tr_m], ly[tr_m])
    lp = m.predict(X)
    va = (sw.split_t3 == "val").to_numpy()
    proy = sw.project.to_numpy()
    print(f"compresión en validación: {lp[va].std() / ly[va].std():.3f} "
          f"(1,0 sería calibrada)")

    print("\nelección de s en validación:")
    mejor = {}
    for etq, glob in (("por proyecto", False), ("global", True)):
        best, bv = 1.0, 1e9
        for s in np.arange(1.0, 3.01, .1):
            v = metricas(y[va], np.exp(expandir(lp[va], proy[va], s, glob)))
            if v["MdAElog"] < bv:
                best, bv = round(float(s), 1), v["MdAElog"]
        mejor[etq] = best
        print(f"  expansión {etq:13s} s = {best} (MdAElog val {bv:.4f})")

    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    filas = []
    for b, k in bloques.items():
        lpb, lyb, yb, pb = lp[k], ly[k], y[k], proy[k]
        variantes = {
            "0 · sin nada": lpb,
            f"1 · expansión global s={mejor['global']}":
                expandir(lpb, pb, mejor["global"], True),
            f"2 · expansión por proyecto s={mejor['por proyecto']}":
                expandir(lpb, pb, mejor["por proyecto"]),
        }
        for nombre, v in variantes.items():
            filas.append(dict(bloque=b, variante=nombre, anclas=0, **metricas(yb, np.exp(v))))
        # con anclas, promediando sorteos
        for n_anc in (3, 10, 20):
            for etq, s in (("sin expansión", 1.0),
                           (f"con expansión s={mejor['por proyecto']}",
                            mejor["por proyecto"])):
                acum = []
                for semilla in range(20):
                    la, usar = anclar(lpb, lyb, pb, n_anc, LAMBDA, semilla)
                    if not usar.any():
                        continue
                    le = expandir(la, pb, s)
                    acum.append(metricas(yb[usar], np.exp(le[usar])))
                if acum:
                    filas.append(dict(bloque=b, variante=f"3 · {n_anc} anclas, {etq}",
                                      anclas=n_anc,
                                      **{kk: float(np.mean([a[kk] for a in acum]))
                                         for kk in acum[0]}))
    t = pd.DataFrame(filas)
    cols = ["variante", "MdMRE", "MdAElog", "PRED25", "R2", "sesgo", "compresion", "spearman"]
    for b in bloques:
        print(f"\n{SEP}\n{b}\n{SEP}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"s_por_proyecto": mejor["por proyecto"], "s_global": mejor["global"],
               "lambda_anclas": LAMBDA},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
