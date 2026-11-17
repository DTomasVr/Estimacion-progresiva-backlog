"""E21 · La confianza que de verdad funciona: cuanto se parece al historial del proyecto.

E20 dejo una diferencia enorme y util. En SiP, sobre las tareas que tienen un
gemelo en el historial del propio proyecto con coseno >= 0,97, el modelo saca
PRED(25) 0,688 y MdMRE 0,158. Sobre las que no tienen gemelo, PRED(25) 0,157 y
MdMRE 0,693. Es el mismo modelo: lo que cambia es si el proyecto ya ha hecho
algo parecido.

Eso convierte la similitud con el historial del proyecto en una señal de
confianza mucho mejor que las que usa hoy el artefacto (anchura de la banda,
confianza del clasificador, cercania al corpus ENTERO de entrenamiento). Y a
diferencia de aquellas, esta mejora sola con el tiempo: cuantas mas tareas cierra
el proyecto, mas cobertura tiene.

  1  curva riesgo-cobertura con la señal nueva frente a las viejas
  2  el diseño por niveles: "ya hice esto" / "he hecho algo parecido" / "esto es
     nuevo", con la cifra de cada nivel, que es lo que el producto puede enseñar
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import LinearSVR

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/confianza")
SEP = "=" * 118
COBERTURAS = [1.0, .8, .6, .5, .4, .3, .2, .1]


def met(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    return dict(MdMRE=m["MdMRE"],
                MdAElog=float(np.median(np.abs(np.log(p) - np.log(y)))),
                PRED25=m["PRED25"], MAE_h=m["MAE_h"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sw = sw[~sw.apache].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    tr_m = (sw.split_t3 == "train").to_numpy()
    va_m = (sw.split_t3 == "val").to_numpy()
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    proy = sw.project.to_numpy()
    Xe = np.load(CACHE / "e5-large.npy")
    Xn = Xe / np.linalg.norm(Xe, axis=1, keepdims=True)
    tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                          sublinear_tf=True).fit(tit[tr_m])
    tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tit[tr_m])
    Xs = sparse.hstack([tfc.transform(tit), tfw.transform(tit)]).tocsr()
    partes = [Xe]
    for tf in (tfc, tfw):
        M = tf.transform(tit)
        partes.append(TruncatedSVD(256, random_state=0).fit(M[tr_m]).transform(M).astype(np.float32))
    Xd = np.hstack(partes)

    def hgb(s=0, **kw):
        return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                             l2_regularization=1., random_state=s, **kw)
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(tr_m)
    acum = []
    for s in range(7):
        sub = rng.choice(idx, int(.85 * len(idx)), replace=False)
        acum.append(hgb(s, max_features=.8).fit(Xd[sub], ly[sub]).predict(Xd))
    Z = np.column_stack([np.mean(acum, 0),
                         Ridge(alpha=2.0).fit(Xs[tr_m], ly[tr_m]).predict(Xs),
                         LinearSVR(C=.3, max_iter=4000, random_state=0).fit(
                             Xs[tr_m], ly[tr_m]).predict(Xs)])
    lp = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Z[va_m], ly[va_m]).predict(Z)
    q10 = hgb(loss="quantile", quantile=.1).fit(Xd[tr_m], ly[tr_m]).predict(Xd)
    q90 = hgb(loss="quantile", quantile=.9).fit(Xd[tr_m], ly[tr_m]).predict(Xd)
    print("modelo y cuantiles listos")

    def sim_historial(k):
        """Coseno maximo con el historial de ENTRENAMIENTO del mismo proyecto."""
        ib = np.flatnonzero(k)
        s = np.full(len(ib), -1.0)
        for p in np.unique(proy[k]):
            hist = np.flatnonzero(tr_m & (proy == p))
            if not len(hist):
                continue
            t = np.flatnonzero(proy[k] == p)
            s[t] = (Xn[ib[t]] @ Xn[hist].T).max(1)
        return s

    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy()}
    filas, niveles = [], []
    for b, k in bloques.items():
        yb, pb = y[k], np.exp(lp[k])
        err = np.abs(np.log(np.clip(pb, .01, None)) - np.log(yb))
        sh = sim_historial(k)
        senales = {
            "similitud con el historial del proyecto": sh,
            "anchura de la banda": -(q90[k] - q10[k]),
            "cercanía al corpus entero": (Xn[k] @ Xn[tr_m].T).max(1),
            "azar (referencia)": np.random.default_rng(0).random(int(k.sum())),
        }
        print(f"\n{b}: n={int(k.sum()):,} · historial disponible para "
              f"{float((sh > -1).mean()):.1%}")
        for nombre, s in senales.items():
            orden = np.argsort(-s)
            v = []
            for c in COBERTURAS:
                kk = orden[:max(20, int(round(c * len(yb))))]
                v.append(float(np.median(err[kk])))
                m_ = met(yb[kk], pb[kk])
                m_["MdAElog"] = v[-1]
                filas.append(dict(bloque=b, senal=nombre, cobertura=c, n=len(kk), **m_))
            print(f"  {nombre:40s} AURC {np.mean(v):.4f} · "
                  f"al 30 % {v[COBERTURAS.index(.3)]:.4f}")
        for lo, hi, etq in ((.97, 1.01, "ya hice esto (≥ 0,97)"),
                            (.93, .97, "algo parecido (0,93–0,97)"),
                            (.85, .93, "de la misma familia (0,85–0,93)"),
                            (-2, .85, "nuevo (< 0,85)")):
            m = (sh >= lo) & (sh < hi)
            if m.sum() < 20:
                continue
            niveles.append(dict(bloque=b, nivel=etq, n=int(m.sum()),
                                fraccion=float(m.mean()), **met(yb[m], pb[m])))
    t = pd.DataFrame(filas)
    n = pd.DataFrame(niveles)
    print(f"\n{SEP}\nDISEÑO POR NIVELES DE FAMILIARIDAD\n{SEP}")
    for b in bloques:
        print(f"\n{b}")
        print(n[n.bloque == b].drop(columns=["bloque"]).round(4).to_string(index=False))
    t.to_csv(OUT / "curvas.csv", index=False)
    n.to_csv(OUT / "niveles.csv", index=False)
    json.dump({"coberturas": COBERTURAS}, open(OUT / "meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/curvas.csv y niveles.csv")


if __name__ == "__main__":
    main()
