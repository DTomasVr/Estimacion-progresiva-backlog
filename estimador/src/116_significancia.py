"""E18 · ¿Es significativa la mejora de la noche?

Todo lo de esta noche son diferencias de dos o tres centesimas, que es
exactamente el tamaño en el que uno se engaña. Antes de escribirlas en la
memoria hay que compararlas como se comparan las cosas pareadas: sobre las
MISMAS tareas, con el error de cada tarea, prueba de Wilcoxon por la direccion y
un intervalo bootstrap por la magnitud.

Se comparan tres cosas:

  A  el punto de partida de la noche   mpnet + HGB
  B  el modelo final                   bolsa + lineales dispersos apilados
  C  el mismo final con 20 anclas      para ver si esa ganancia tambien aguanta

El estadistico es el error logaritmico absoluto por tarea, que es simetrico y no
premia subestimar, y la diferencia de MdMRE por bootstrap sobre las mismas
tareas. Se hace en los tres bloques.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wilcoxon
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import LinearSVR

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/final")
SEP = "=" * 118
LAMBDA = 12.0


def comparar(y, a, b, n=4000):
    """a frente a b sobre las mismas tareas. Negativo = a mejor."""
    ea = np.abs(np.log(np.clip(a, .01, None)) - np.log(y))
    eb = np.abs(np.log(np.clip(b, .01, None)) - np.log(y))
    p = wilcoxon(ea, eb).pvalue if np.any(ea != eb) else 1.0
    rng = np.random.default_rng(1)
    ra = np.abs(y - a) / y
    rb = np.abs(y - b) / y
    d = [np.median(ra[i]) - np.median(rb[i])
         for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    lo, hi = np.percentile(d, [2.5, 97.5])
    dl = [np.median(ea[i]) - np.median(eb[i])
          for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    llo, lhi = np.percentile(dl, [2.5, 97.5])
    ver = "mejor" if hi < 0 else "peor" if lo > 0 else "empate"
    return dict(veredicto=ver, dif_MdMRE=float(np.median(d)),
                IC95_MdMRE=f"[{lo:+.4f}, {hi:+.4f}]",
                dif_MdAElog=float(np.median(dl)),
                IC95_MdAElog=f"[{llo:+.4f}, {lhi:+.4f}]", p_wilcoxon=float(p))


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

    def hgb(s=0, **kw):
        return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                             l2_regularization=1., random_state=s, **kw)

    MP = np.load(CACHE / "mpnet.npy")
    lp_a = hgb().fit(MP[tr_m], ly[tr_m]).predict(MP)
    print("  A · punto de partida ajustado")

    tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                          sublinear_tf=True).fit(tit[tr_m])
    tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tit[tr_m])
    Xs = sparse.hstack([tfc.transform(tit), tfw.transform(tit)]).tocsr()
    partes = [np.load(CACHE / "e5-large.npy")]
    for tf in (tfc, tfw):
        M = tf.transform(tit)
        partes.append(TruncatedSVD(256, random_state=0).fit(M[tr_m]).transform(M).astype(np.float32))
    Xd = np.hstack(partes)
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(tr_m)
    acum = []
    for s in range(7):
        sub = rng.choice(idx, int(.85 * len(idx)), replace=False)
        acum.append(hgb(s, max_features=.8).fit(Xd[sub], ly[sub]).predict(Xd))
    Z = np.column_stack([np.mean(acum, 0), Ridge(alpha=2.0).fit(Xs[tr_m], ly[tr_m]).predict(Xs),
                         LinearSVR(C=.3, max_iter=4000, random_state=0).fit(
                             Xs[tr_m], ly[tr_m]).predict(Xs)])
    lp_b = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Z[va_m], ly[va_m]).predict(Z)
    print("  B · modelo final ajustado")

    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    filas = []
    for b, k in bloques.items():
        yb = y[k]
        pa, pb = np.exp(lp_a[k]), np.exp(lp_b[k])
        filas.append(dict(bloque=b, n=int(k.sum()),
                          comparacion="final frente al punto de partida",
                          **comparar(yb, pb, pa)))
        # con 20 anclas contraidas, promediando sorteos
        difs = []
        for s in range(20):
            r2 = np.random.default_rng(s)
            la, usar = lp_b[k].copy(), np.zeros(k.sum(), bool)
            for p in np.unique(proy[k]):
                i = np.flatnonzero(proy[k] == p)
                if len(i) <= 20:
                    continue
                anc = r2.choice(i, 20, replace=False)
                resto = np.setdiff1d(i, anc)
                la[resto] += float(np.median(ly[k][anc] - lp_b[k][anc])) * 20 / (20 + LAMBDA)
                usar[resto] = True
            if usar.sum() > 30:
                difs.append(comparar(yb[usar], np.exp(la[usar]), pb[usar], n=800))
        if difs:
            filas.append(dict(bloque=b, n=int(usar.sum()),
                              comparacion="20 anclas frente al final sin anclas",
                              veredicto=pd.Series([d["veredicto"] for d in difs]).mode()[0],
                              dif_MdMRE=float(np.mean([d["dif_MdMRE"] for d in difs])),
                              IC95_MdMRE="(media de 20 sorteos)",
                              dif_MdAElog=float(np.mean([d["dif_MdAElog"] for d in difs])),
                              IC95_MdAElog="(media de 20 sorteos)",
                              p_wilcoxon=float(np.median([d["p_wilcoxon"] for d in difs]))))
    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nCONTRASTES PAREADOS SOBRE LAS MISMAS TAREAS\n{SEP}")
    print(t.round(5).to_string(index=False))
    t.to_csv(OUT / "significancia.csv", index=False)
    json.dump({"nota": "negativo = la primera es mejor"},
              open(OUT / "significancia.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/significancia.csv")


if __name__ == "__main__":
    main()
