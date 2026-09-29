"""E8b · Anclas con contraccion: el ancla sola es peor que no anclar.

E8 salio con un resultado que parecia absurdo y no lo es: anclar con UNA tarea
empeora el MdMRE de 0,726 a 0,830 en proyectos nuevos. La razon es que el
desplazamiento se estima con el residuo de esa unica tarea, y el residuo tiene
una desviacion tipica de mas de 1 en log. Se le esta metiendo ese ruido entero a
las otras cien tareas del proyecto.

La correccion es la de toda la vida para medias de grupos pequeños, contraccion
tipo James-Stein / Bayes empirico:

    desplazamiento_contraido = k / (k + lambda) x desplazamiento_observado

Con k anclas el ruido del desplazamiento baja como 1/k, asi que el peso optimo
crece con k. lambda se elige en VALIDACION, sobre proyectos distintos de los de
prueba, y sale de la relacion entre la varianza entre proyectos y la varianza
dentro de proyecto -- que las dos estan medidas.

Se compara contra no anclar y contra anclar sin contraer.
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
pd.set_option("display.width", 205)
DATA = "data/processed/agix_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/anclas")
SEP = "=" * 112
ANCLAS = [0, 1, 2, 3, 5, 10, 20]
SORTEOS = 30
LAMBDAS = [0, .5, 1, 2, 3, 5, 8, 12, 20, 40]


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly = np.log(y)
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(np.log(p) - ly))),
                PRED25=m["PRED25"], MAE_h=m["MAE_h"],
                R2=float(1 - ((ly - np.log(p)) ** 2).mean() / ly.var()),
                sesgo=float(np.exp(np.median(np.log(p) - ly))))


def evaluar(lp, ly, yv, proy, n_anc, lam, sorteos):
    """Ancla k tareas por proyecto, contrae el desplazamiento y lee el resto."""
    idx_por_proy = {p: np.flatnonzero(proy == p) for p in np.unique(proy)}
    acum = []
    for s in range(sorteos if n_anc else 1):
        rng = np.random.default_rng(s)
        pred, real = [], []
        for p, idx in idx_por_proy.items():
            if n_anc == 0:
                resto, desp = idx, 0.0
            elif len(idx) > n_anc:
                anc = rng.choice(idx, n_anc, replace=False)
                resto = np.setdiff1d(idx, anc)
                bruto = float(np.median(ly[anc] - lp[anc]))
                desp = bruto * n_anc / (n_anc + lam)
            else:
                continue
            pred.append(np.exp(lp[resto] + desp))
            real.append(yv[resto])
        if pred:
            acum.append(metricas(np.concatenate(real), np.concatenate(pred)))
    if not acum:
        return None
    return {k: float(np.mean([a[k] for a in acum])) for k in acum[0]}


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
    res = ly[tr_m] - lp[tr_m]
    g = pd.DataFrame({"p": sw.project[tr_m], "r": res}).groupby("p").r
    var_entre = float(g.mean().var())
    var_dentro = float(res.var())
    print(f"varianza del residuo entre proyectos {var_entre:.4f} · dentro {var_dentro:.4f}")
    print(f"  lambda teórico (var_dentro / var_entre) = {var_dentro / var_entre:.1f}")

    # lambda se elige en VALIDACION
    print("\nelección de lambda en validación (MdAElog medio sobre k=1..10):")
    mejor, mejor_v = None, 1e9
    for lam in LAMBDAS:
        vs = [evaluar(lp[va], ly[va], y[va], sw.project.to_numpy()[va], k, lam, 10)
              for k in (1, 2, 3, 5, 10)]
        v = float(np.mean([x["MdAElog"] for x in vs if x]))
        print(f"  lambda {lam:5.1f} -> {v:.4f}")
        if v < mejor_v:
            mejor, mejor_v = lam, v
    print(f"  elegido lambda = {mejor}")

    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    filas = []
    for b, k in bloques.items():
        print(f"\n{b}")
        for n_anc in ANCLAS:
            for etq, lam in (("sin contraer", 0.0), (f"contraído λ={mejor}", float(mejor))):
                if n_anc == 0 and etq != "sin contraer":
                    continue
                r = evaluar(lp[k], ly[k], y[k], sw.project.to_numpy()[k], n_anc, lam, SORTEOS)
                if r is None:
                    continue
                filas.append(dict(bloque=b, anclas=n_anc, variante=etq, **r))
                print(f"  {n_anc:2d} anclas {etq:16s} MdMRE {r['MdMRE']:.4f} · "
                      f"MdAElog {r['MdAElog']:.4f} · PRED25 {r['PRED25']:.4f} · "
                      f"R² {r['R2']:+.4f}", flush=True)

    t = pd.DataFrame(filas)
    t.to_csv(OUT / "contraidas.csv", index=False)
    print(f"\n{SEP}\nRESUMEN: CONTRAER FRENTE A NO ANCLAR\n{SEP}")
    for b in bloques:
        v = t[t.bloque == b]
        base = v[v.anclas == 0].iloc[0]
        print(f"\n{b} (sin anclas: MdMRE {base.MdMRE:.4f} · R² {base.R2:+.4f})")
        for _, r in v[v.variante.str.startswith("contraído")].iterrows():
            print(f"  {int(r.anclas):2d} anclas · MdMRE {r.MdMRE:.4f} "
                  f"({r.MdMRE - base.MdMRE:+.4f}) · R² {r.R2:+.4f} "
                  f"({r.R2 - base.R2:+.4f}) · PRED25 {r.PRED25:.3f}")
    json.dump({"lambda": mejor, "var_entre": var_entre, "var_dentro": var_dentro},
              open(OUT / "contraidas.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/contraidas.csv")


if __name__ == "__main__":
    main()
