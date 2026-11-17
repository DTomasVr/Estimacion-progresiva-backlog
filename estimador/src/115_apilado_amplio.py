"""E17 · Ampliar el apilado: lo que gana un apilado es la DIVERSIDAD, no la potencia.

En el modelo final el ridge disperso aporta (+0,386 de peso) y el SVR disperso
no (-0,032): son casi el mismo modelo. Si lo que hace falta es diversidad,
conviene meter familias que fallen en sitios distintos:

  bolsa HGB         arboles sobre e5-large + tf-idf reducido        (ya estaba)
  ridge disperso    lineal sobre 55.706 columnas de tf-idf          (ya estaba)
  analogia          mediana de log-horas de los 15 vecinos mas
                    cercanos en el espacio de e5-large. Es el metodo
                    clasico del area (Shepperd y Schofield, 1997) y
                    falla donde el arbol acierta y al reves
  mpnet             el codificador viejo: peor por su cuenta, pero
                    representa el texto de otra manera
  xgboost           otra implementacion de arboles, con otro sesgo
  k-vecinos ridge   ridge sobre los vectores densos, sin arboles

Los pesos se ajustan en validacion con un ridge no negativo, para que el
apilado no se invente compensaciones raras entre modelos correlacionados.
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
pd.set_option("display.width", 215)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/apilado_amplio")
SEP = "=" * 120


def metricas(y, p, proyecto=None):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly, lp = np.log(y), np.log(p)
    d = dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lp - ly))),
             PRED25=m["PRED25"], R2=float(1 - ((ly - lp) ** 2).mean() / ly.var()),
             sesgo=float(np.exp(np.median(lp - ly))),
             spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))
    if proyecto is not None:
        g = pd.DataFrame({"p": proyecto, "ly": ly, "lp": lp})
        dy = ly - g.groupby("p").ly.transform("mean").to_numpy()
        dp = lp - g.groupby("p").lp.transform("mean").to_numpy()
        d["rho_intra"] = float(pd.Series(dy).corr(pd.Series(dp), method="spearman"))
    return d


def analogia(Xtr, ytr, X, k=15):
    A = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
    B = X / np.linalg.norm(X, axis=1, keepdims=True)
    out = np.empty(len(B))
    for i in range(0, len(B), 2048):
        s = B[i:i + 2048] @ A.T
        idx = np.argpartition(-s, k, axis=1)[:, :k]
        out[i:i + 2048] = np.median(ytr[idx], axis=1)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
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

    E5 = np.load(CACHE / "e5-large.npy")
    MP = np.load(CACHE / "mpnet.npy")
    tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                          sublinear_tf=True).fit(tit[tr_m])
    tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tit[tr_m])
    Xs = sparse.hstack([tfc.transform(tit), tfw.transform(tit)]).tocsr()
    partes = [E5]
    for tf in (tfc, tfw):
        M = tf.transform(tit)
        partes.append(TruncatedSVD(256, random_state=0).fit(M[tr_m]).transform(M).astype(np.float32))
    Xd = np.hstack(partes)
    print(f"denso {Xd.shape} · disperso {Xs.shape} · {time.time() - t0:.0f} s")

    def hgb(s=0, **kw):
        return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                             l2_regularization=1., random_state=s, **kw)

    base = {}
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(tr_m)
    acum = []
    for s in range(7):
        sub = rng.choice(idx, int(.85 * len(idx)), replace=False)
        acum.append(hgb(s, max_features=.8).fit(Xd[sub], ly[sub]).predict(Xd))
    base["bolsa HGB"] = np.mean(acum, 0)
    print("  bolsa lista", flush=True)
    base["ridge disperso"] = Ridge(alpha=2.0).fit(Xs[tr_m], ly[tr_m]).predict(Xs)
    base["analogía k=15"] = analogia(E5[tr_m], ly[tr_m], E5)
    base["HGB sobre mpnet"] = hgb().fit(MP[tr_m], ly[tr_m]).predict(MP)
    base["ridge denso"] = RidgeCV(alphas=np.logspace(-1, 3, 15)).fit(
        Xd[tr_m], ly[tr_m]).predict(Xd)
    base["SVR denso"] = LinearSVR(C=.3, max_iter=3000, random_state=0).fit(
        Xd[tr_m], ly[tr_m]).predict(Xd)
    try:
        from xgboost import XGBRegressor
        base["xgboost"] = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=.06,
                                       subsample=.85, colsample_bytree=.8,
                                       tree_method="hist", random_state=0).fit(
            Xd[tr_m], ly[tr_m]).predict(Xd)
    except Exception as e:
        print(f"  xgboost no disponible: {type(e).__name__}")
    print(f"  {len(base)} modelos base · {time.time() - t0:.0f} s", flush=True)

    Z = np.column_stack(list(base.values()))
    ap = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Z[va_m], ly[va_m])
    pos = Ridge(alpha=1.0, positive=True).fit(Z[va_m], ly[va_m])
    print("  pesos (ridge):    " + " · ".join(f"{k} {w:+.3f}" for k, w in zip(base, ap.coef_)))
    print("  pesos (≥0):       " + " · ".join(f"{k} {w:+.3f}" for k, w in zip(base, pos.coef_)))

    modelos = dict(base)
    modelos["APILADO (ridge)"] = ap.predict(Z)
    modelos["APILADO (pesos ≥ 0)"] = pos.predict(Z)
    modelos["media simple"] = Z.mean(1)
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
        print(t[t.bloque == b][cols].sort_values("R2", ascending=False).round(4)
              .to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"modelos": list(base), "pesos_ridge": ap.coef_.tolist(),
               "pesos_positivos": pos.coef_.tolist()},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
