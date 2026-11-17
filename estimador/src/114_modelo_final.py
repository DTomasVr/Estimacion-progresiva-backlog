"""El modelo final de la noche: bolsa de arboles + lineales dispersos, apilados.

E16 mostro que el tf-idf disperso ENTERO aporta algo que el arbol sobre vectores
densos no tiene: apilando un ridge y un SVR lineal dispersos con el HGB, el R²
de la prueba fija sube de 0,146 a 0,177. Aqui se junta eso con la bolsa de siete
arboles de E11, que por si sola daba 0,164.

  base      7 x HGB sobre e5-large + tf-idf reducido (submuestreo 85 %)
  lineales  ridge y SVR lineal sobre char 3-5 + palabras 1-2 sin reducir
  apilado   ridge con los pesos ajustados en validacion

Se lee ademas con las dos palancas que el artefacto puede usar -- anclas
contraidas y correccion de la estimacion del usuario -- para tener la tabla
completa sobre el modelo definitivo.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVR

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("m71", "src/71_estimacion_usuario.py")
M71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M71)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 225)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/final")
PESOS = Path("models/modelo_final.joblib")
LAMBDA, N_BOLSA, SORTEOS = 12.0, 7, 20
SEP = "=" * 132


def metricas(y, p, proyecto=None):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly, lp = np.log(y), np.log(p)
    d = dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lp - ly))),
             PRED25=m["PRED25"], MAE_h=m["MAE_h"],
             R2=float(1 - ((ly - lp) ** 2).mean() / ly.var()),
             sesgo=float(np.exp(np.median(lp - ly))), SA=M71.sa(y, p),
             spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))
    if proyecto is not None:
        g = pd.DataFrame({"p": proyecto, "ly": ly, "lp": lp})
        dy = ly - g.groupby("p").ly.transform("mean").to_numpy()
        dp = lp - g.groupby("p").lp.transform("mean").to_numpy()
        d["rho_intra"] = float(pd.Series(dy).corr(pd.Series(dp), method="spearman"))
    return d


def hgb(semilla=0, **kw):
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=semilla, **kw)


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

    tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                          sublinear_tf=True).fit(tit[tr_m])
    tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tit[tr_m])
    Xs = sparse.hstack([tfc.transform(tit), tfw.transform(tit)]).tocsr()
    partes = [np.load(CACHE / "e5-large.npy")]
    svds = []
    for tf in (tfc, tfw):
        M = tf.transform(tit)
        sv = TruncatedSVD(256, random_state=0).fit(M[tr_m])
        svds.append(sv)
        partes.append(sv.transform(M).astype(np.float32))
    Xd = np.hstack(partes)
    print(f"denso {Xd.shape} · disperso {Xs.shape} · {time.time() - t0:.0f} s")

    rng = np.random.default_rng(0)
    idx = np.flatnonzero(tr_m)
    bolsa = []
    for s in range(N_BOLSA):
        sub = rng.choice(idx, int(.85 * len(idx)), replace=False)
        bolsa.append(hgb(s, max_features=.8).fit(Xd[sub], ly[sub]))
        print(f"  bolsa {s + 1}/{N_BOLSA}", flush=True)
    lin_r = Ridge(alpha=2.0).fit(Xs[tr_m], ly[tr_m])
    lin_s = LinearSVR(C=.3, max_iter=4000, random_state=0).fit(Xs[tr_m], ly[tr_m])
    print("  lineales dispersos ajustados")

    # el codificador ajustado de extremo a extremo (script 105). Por su cuenta
    # ordena mejor (Spearman 0,364 -> 0,393) y estima peor la escala
    # (R2 0,149 -> 0,091); dentro del apilado esa diferencia es justo lo que
    # aporta, y en proyectos nuevos casi duplica el R2 del conjunto.
    AJ = np.load(CACHE / "e5-ajustado.npy")
    hgb_aj = hgb().fit(AJ[tr_m], ly[tr_m])
    print("  HGB sobre el codificador ajustado")

    base = {"bolsa": np.mean([m.predict(Xd) for m in bolsa], 0),
            "ridge disperso": lin_r.predict(Xs),
            "SVR disperso": lin_s.predict(Xs),
            "e5 ajustado": hgb_aj.predict(AJ)}
    Z = np.column_stack(list(base.values()))
    ap = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Z[va_m], ly[va_m])
    lp = ap.predict(Z)
    print("  pesos: " + " · ".join(f"{k} {w:+.3f}" for k, w in zip(base, ap.coef_)))

    # corrector de la estimacion del usuario, con frio fuera de pliegue
    sw["est"] = sw.HoursEstimate.where(sw.HoursEstimate > 0)
    pt = sw[(sw.source == "TAWOS") & (sw.story_point > 0)]
    ptr = pt[pt.split_t3 == "train"]
    pend, corte = np.polyfit(np.log(ptr.story_point.to_numpy(float)),
                             np.log(ptr.effort_hours.to_numpy(float)), 1)
    sw.loc[pt.index, "est"] = np.exp(corte) * pt.story_point.to_numpy(float) ** pend
    hay = sw.est.notna().to_numpy()
    oof = lp.copy()
    for a, b in KFold(5, shuffle=True, random_state=0).split(idx):
        oof[idx[b]] = hgb().fit(Xd[idx[a]], ly[idx[a]]).predict(Xd[idx[b]])
    Zc = np.column_stack([np.log(np.where(hay, sw.est, 1).astype(float)), oof])
    corrector = hgb().fit(Zc[tr_m & hay], (ly - np.log(sw.est.fillna(1).to_numpy(float)))[tr_m & hay])

    def anclar(lpb, lyb, pb, n_anc, semilla):
        if n_anc == 0:
            return lpb, np.ones(len(lpb), bool)
        r2 = np.random.default_rng(semilla)
        out, usar = lpb.copy(), np.zeros(len(lpb), bool)
        for p in np.unique(pb):
            i = np.flatnonzero(pb == p)
            if len(i) <= n_anc:
                continue
            anc = r2.choice(i, n_anc, replace=False)
            resto = np.setdiff1d(i, anc)
            out[resto] += float(np.median(lyb[anc] - lpb[anc])) * n_anc / (n_anc + LAMBDA)
            usar[resto] = True
        return out, usar

    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "prueba TAWOS": ((sw.split_t3 == "test") & (sw.source == "TAWOS")).to_numpy(),
               "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    filas = []
    for b, k in bloques.items():
        lpb, lyb, yb, pb = lp[k], ly[k], y[k], proy[k]
        estb, hayb = sw.est.to_numpy()[k], hay[k]
        for n_anc in (0, 3, 10, 20):
            for con_est in (False, True):
                if con_est and not hayb.any():
                    continue
                acum = []
                for s in range(SORTEOS if n_anc else 1):
                    la, usar = anclar(lpb, lyb, pb, n_anc, s)
                    p = np.exp(la)
                    if con_est:
                        z = np.column_stack([np.log(np.where(hayb, estb, 1).astype(float)), la])
                        p = np.where(hayb, np.where(hayb, estb, 1) * np.exp(corrector.predict(z)), p)
                    acum.append(metricas(yb[usar], p[usar], pb[usar]))
                filas.append(dict(bloque=b, anclas=n_anc, estimacion=con_est,
                                  **{kk: float(np.mean([a[kk] for a in acum]))
                                     for kk in acum[0]}))
    t = pd.DataFrame(filas)
    cols = ["anclas", "estimacion", "MdMRE", "MdAElog", "PRED25", "MAE_h", "R2", "SA",
            "sesgo", "spearman", "rho_intra"]
    for b in bloques:
        print(f"\n{SEP}\n{b}\n{SEP}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    joblib.dump({"bolsa": bolsa, "ridge": lin_r, "svr": lin_s, "hgb_ajustado": hgb_aj,
                 "codificador_ajustado": "models/e5_ajustado", "apilado": ap,
                 "tfc": tfc, "tfw": tfw, "svds": svds, "corrector": corrector,
                 "lambda": LAMBDA, "codificador": "intfloat/multilingual-e5-large",
                 "dim_emb": int(partes[0].shape[1]), "corpus": "v4 sin Apache",
                 "sha_datos": E.data_fingerprint(DATA),
                 "fecha": time.strftime("%Y-%m-%d")}, PESOS, compress=3)
    json.dump({"pesos_apilado": ap.coef_.tolist(), "lambda": LAMBDA},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv y {PESOS} "
          f"({PESOS.stat().st_size / 1e6:.0f} MB) en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
