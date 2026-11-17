"""E16 · tf-idf disperso completo con modelo lineal, y apilado con el arbol.

Hasta ahora el tf-idf ha entrado siempre reducido a 256 componentes por SVD,
porque era lo comodo para meterlo en el HGB. Pero en texto corto un modelo
lineal sobre la matriz dispersa ENTERA suele ser muy competitivo: no pierde los
terminos raros, que en titulos de siete palabras son justo los informativos
("regression", "typo", "migrate", "refactor").

  A  ridge sobre char 3-5 + palabras 1-2, sin reducir
  B  SVR lineal sobre lo mismo (perdida epsilon-insensible, mas robusta)
  C  apilado del lineal con el HGB sobre e5-large, con los pesos en validacion

Si el lineal disperso aporta algo que el arbol sobre vectores densos no tiene,
el apilado lo recoge; si no, el apilado se quedara con el arbol y sabremos que
la reduccion a 256 componentes no estaba tirando nada.
"""
import json
import sys
import time
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
pd.set_option("display.width", 210)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/lineal")
SEP = "=" * 118


def metricas(y, p, proyecto=None):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly, lp = np.log(y), np.log(p)
    d = dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lp - ly))),
             PRED25=m["PRED25"], MAE_h=m["MAE_h"],
             R2=float(1 - ((ly - lp) ** 2).mean() / ly.var()),
             sesgo=float(np.exp(np.median(lp - ly))),
             spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))
    if proyecto is not None:
        g = pd.DataFrame({"p": proyecto, "ly": ly, "lp": lp})
        dy = ly - g.groupby("p").ly.transform("mean").to_numpy()
        dp = lp - g.groupby("p").lp.transform("mean").to_numpy()
        d["rho_intra"] = float(pd.Series(dy).corr(pd.Series(dp), method="spearman"))
    return d


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
    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "externo no Apache": (sw.split_t3 == "externo").to_numpy()}

    t0 = time.time()
    tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                          sublinear_tf=True).fit(tit[tr_m])
    tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tit[tr_m])
    Xs = sparse.hstack([tfc.transform(tit), tfw.transform(tit)]).tocsr()
    print(f"matriz dispersa {Xs.shape} · {Xs.nnz / 1e6:.1f} M no nulos · "
          f"{time.time() - t0:.0f} s")

    partes = [np.load(CACHE / "e5-large.npy")]
    for tf in (tfc, tfw):
        M = tf.transform(tit)
        partes.append(TruncatedSVD(256, random_state=0).fit(M[tr_m]).transform(M).astype(np.float32))
    Xd = np.hstack(partes)

    modelos = {}
    for etq, m in (("A · ridge disperso", Ridge(alpha=2.0)),
                   ("B · SVR lineal disperso", LinearSVR(C=.3, max_iter=4000,
                                                          random_state=0))):
        t1 = time.time()
        m.fit(Xs[tr_m], ly[tr_m])
        modelos[etq] = m.predict(Xs)
        print(f"  {etq} en {time.time() - t1:.0f} s", flush=True)
    hgb = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                        l2_regularization=1., random_state=0).fit(
        Xd[tr_m], ly[tr_m])
    modelos["C · HGB sobre e5 + tf-idf reducido"] = hgb.predict(Xd)

    Z = np.column_stack([modelos[k] for k in modelos])
    ap = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Z[va_m], ly[va_m])
    modelos["D · apilado (pesos en validación)"] = ap.predict(Z)
    print("  pesos del apilado: " + " · ".join(
        f"{k.split(' · ')[0]} {w:+.3f}" for k, w in zip(list(modelos)[:3], ap.coef_)))

    filas = []
    for etq, lp in modelos.items():
        val = metricas(y[va_m], np.exp(lp[va_m]))["MdAElog"]
        for b, k in bloques.items():
            filas.append(dict(modelo=etq, bloque=b, n=int(k.sum()), val_MdAElog=val,
                              **metricas(y[k], np.exp(lp[k]), proy[k])))
    t = pd.DataFrame(filas)
    cols = ["modelo", "val_MdAElog", "MdMRE", "MdAElog", "PRED25", "R2", "sesgo",
            "spearman", "rho_intra"]
    for b in bloques:
        print(f"\n{SEP}\n{b}\n{SEP}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"dim_dispersa": list(Xs.shape), "pesos": ap.coef_.tolist()},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
