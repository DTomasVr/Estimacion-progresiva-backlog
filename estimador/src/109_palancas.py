"""E14 · La tabla definitiva: que aporta cada palanca sobre el MEJOR modelo.

Las palancas se midieron en momentos distintos y sobre configuraciones
distintas: la estimacion del usuario sobre mpnet con titulo + descripcion, las
anclas sobre e5-large con titulos. Esta tabla las pone todas sobre el mismo
modelo, el mismo corpus y los mismos bloques congelados, para que se puedan
comparar entre si y para que la memoria tenga una sola cifra por palanca.

Modelo: bolsa de 7 HGB sobre e5-large + tf-idf de caracteres + tf-idf de
palabras, corpus sin Apache, entrenada con train (+val donde se indica).

Palancas, de menos a mas friccion para el usuario:

  0  nada                        el arranque en frio
  1  k anclas contraidas         el usuario da las horas de k tareas del backlog
  2  estimacion por tarea        el usuario estima TODAS las tareas y el modelo
                                 las corrige
  3  anclas + estimacion         las dos

Las anclas usan la contraccion de E8b (lambda = 12, elegido en validacion). El
corrector de la estimacion es el brazo C2 de 94: estimacion + prediccion en
frio, sin texto, que fue el que eligio el SA.
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
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("m71", "src/71_estimacion_usuario.py")
M71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M71)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/palancas")
SEP = "=" * 128
LAMBDA, SORTEOS = 12.0, 20


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly, lp = np.log(y), np.log(p)
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lp - ly))),
                PRED25=m["PRED25"], MAE_h=m["MAE_h"],
                R2=float(1 - ((ly - lp) ** 2).mean() / ly.var()),
                sesgo=float(np.exp(np.median(lp - ly))),
                SA=M71.sa(y, p),
                spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def hgb(semilla=0, **kw):
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=semilla, **kw)


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
    sw["est"] = sw.HoursEstimate.where(sw.HoursEstimate > 0)
    pt = sw[(sw.source == "TAWOS") & (sw.story_point > 0)]
    ptr = pt[pt.split_t3 == "train"]
    pend, corte = np.polyfit(np.log(ptr.story_point.to_numpy(float)),
                             np.log(ptr.effort_hours.to_numpy(float)), 1)
    sw.loc[pt.index, "est"] = np.exp(corte) * pt.story_point.to_numpy(float) ** pend
    hay_est = sw.est.notna().to_numpy()
    print(f"corpus {len(sw):,} · con estimación {hay_est.mean():.1%}")

    # modelo en frio: bolsa de 7, y version fuera de pliegue para el corrector
    rng = np.random.default_rng(0)
    idx_tr = np.flatnonzero(tr_m)
    acum = []
    for s in range(7):
        sub = rng.choice(idx_tr, int(.85 * len(idx_tr)), replace=False)
        acum.append(hgb(s, max_features=.8).fit(X[sub], ly[sub]).predict(X))
        print(f"  bolsa {s + 1}/7", flush=True)
    lp = np.mean(acum, 0)

    oof = np.zeros(len(sw))
    for a, b in KFold(5, shuffle=True, random_state=0).split(idx_tr):
        oof[idx_tr[b]] = hgb().fit(X[idx_tr[a]], ly[idx_tr[a]]).predict(X[idx_tr[b]])
    frio_rasgo = lp.copy()
    frio_rasgo[tr_m] = oof[tr_m]

    # corrector C2: estimacion + frio, sin texto
    k_tr = tr_m & hay_est
    Zc = np.column_stack([np.log(np.where(hay_est, sw.est, 1).astype(float)), frio_rasgo])
    c2 = hgb().fit(Zc[k_tr], (ly - np.log(sw.est.fillna(1).to_numpy(float)))[k_tr])
    print("  corrector C2 ajustado")

    def con_anclas(lpb, lyb, proy, n_anc, semilla):
        if n_anc == 0:
            return lpb, np.ones(len(lpb), bool)
        rng2 = np.random.default_rng(semilla)
        out, usar = lpb.copy(), np.zeros(len(lpb), bool)
        for p in np.unique(proy):
            idx = np.flatnonzero(proy == p)
            if len(idx) <= n_anc:
                continue
            anc = rng2.choice(idx, n_anc, replace=False)
            resto = np.setdiff1d(idx, anc)
            out[resto] += float(np.median(lyb[anc] - lpb[anc])) * n_anc / (n_anc + LAMBDA)
            usar[resto] = True
        return out, usar

    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    filas = []
    for b, k in bloques.items():
        lpb, lyb, yb = lp[k], ly[k], y[k]
        proy = sw.project.to_numpy()[k]
        estb, hayb = sw.est.to_numpy()[k], hay_est[k]
        cob = float(hayb.mean())
        for n_anc in (0, 3, 10, 20):
            for con_est in (False, True):
                if con_est and cob == 0:
                    continue
                acum2 = []
                for semilla in range(SORTEOS if n_anc else 1):
                    la, usar = con_anclas(lpb, lyb, proy, n_anc, semilla)
                    p = np.exp(la)
                    if con_est:
                        z = np.column_stack([np.log(np.where(hayb, estb, 1).astype(float)), la])
                        corr = np.where(hayb, estb * np.exp(c2.predict(z)), p)
                        p = corr
                    acum2.append(metricas(yb[usar], p[usar]))
                prom = {kk: float(np.mean([a[kk] for a in acum2])) for kk in acum2[0]}
                filas.append(dict(bloque=b, anclas=n_anc, estimacion=con_est,
                                  cobertura_est=cob, **prom))
                print(f"  {b:20s} {n_anc:2d} anclas · estimación {str(con_est):5s} -> "
                      f"MdMRE {prom['MdMRE']:.4f} · R² {prom['R2']:+.4f}", flush=True)

    t = pd.DataFrame(filas)
    t.to_csv(OUT / "resumen.csv", index=False)
    cols = ["anclas", "estimacion", "MdMRE", "MdAElog", "PRED25", "MAE_h", "R2", "SA",
            "sesgo", "spearman"]
    for b in bloques:
        print(f"\n{SEP}\n{b}\n{SEP}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
    json.dump({"lambda": LAMBDA, "sorteos": SORTEOS},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
