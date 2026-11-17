"""E6 · La perdida y los hiperparametros del regresor, que nunca se tocaron.

Todo el proyecto ha usado el mismo HGB desde el principio: perdida cuadratica
sobre log-horas, 300 iteraciones, profundidad 6, lr 0,06. Se eligio al empezar y
nunca se reviso, pese a dos cosas que deberian haber levantado la sospecha:

  1  las metricas del trabajo son MEDIANAS (MdMRE, MdAElog) y la perdida
     cuadratica estima la MEDIA condicional. La perdida absoluta estima la
     mediana, que es el estadistico que luego se puntua.
  2  el objetivo en log tiene colas largas; la perdida cuadratica las persigue y
     eso es justo lo que produce la compresion.

Se barre perdida, profundidad, iteraciones y regularizacion sobre las vistas que
sobrevivieron a E5, se elige en validacion por MdAElog -- simetrico, por la
leccion del experimento 77 -- y se lee en los cuatro bloques congelados.

Al final, apilado: un ridge sobre validacion que combina las predicciones de los
mejores modelos de cada vista, porque E5 mostro que vistas distintas ganan en
bloques distintos (e5-large dentro de dominio, tf-idf de palabras fuera).
"""
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/barrido")
SEP = "=" * 130


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly = np.log(y)
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(np.log(p) - ly))),
                PRED25=m["PRED25"], MAE_h=m["MAE_h"],
                R2=float(1 - ((ly - np.log(p)) ** 2).mean() / ly.var()),
                sesgo=float(np.exp(np.median(np.log(p) - ly))),
                spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def vistas(sw, tit, tr_m):
    V = {k: np.load(CACHE / f"{k}.npy") for k in ("mpnet", "e5-large", "bge-m3")
         if (CACHE / f"{k}.npy").exists()}
    for etq, kw in (("tfidf", dict(analyzer="char_wb", ngram_range=(3, 5))),
                    ("tfidf_pal", dict(ngram_range=(1, 2)))):
        tf = TfidfVectorizer(min_df=3, max_features=200_000, sublinear_tf=True,
                             **kw).fit(tit[tr_m])
        X = tf.transform(tit)
        V[etq] = TruncatedSVD(256, random_state=0).fit(X[tr_m]).transform(X).astype(np.float32)
    return {
        "e5-large": V["e5-large"],
        "e5 + tfidf car": np.hstack([V["e5-large"], V["tfidf"]]),
        "e5 + tfidf car + pal": np.hstack([V["e5-large"], V["tfidf"], V["tfidf_pal"]]),
        "tfidf pal": V["tfidf_pal"],
        "mpnet (referencia)": V["mpnet"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sw = sw[~sw.apache].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    tr_m = (sw.split_t3 == "train").to_numpy()
    va_m = (sw.split_t3 == "val").to_numpy()
    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "prueba TAWOS": ((sw.split_t3 == "test") & (sw.source == "TAWOS")).to_numpy(),
               "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    V = vistas(sw, tit, tr_m)
    print(f"corpus {len(sw):,} · vistas: " + " · ".join(f"{k} {v.shape[1]}"
                                                        for k, v in V.items()))

    rejilla = list(itertools.product(
        ["squared_error", "absolute_error"], [4, 6, 10], [300, 800], [1.0]))
    filas, guardados = [], {}
    for vista, X in V.items():
        for perdida, prof, iters, l2 in rejilla:
            t0 = time.time()
            m = HistGradientBoostingRegressor(
                loss=perdida, max_iter=iters, max_depth=prof, learning_rate=.06,
                l2_regularization=l2, early_stopping=True, validation_fraction=.1,
                n_iter_no_change=30, random_state=0).fit(X[tr_m], ly[tr_m])
            pv = np.exp(m.predict(X[va_m]))
            val = metricas(y[va_m], pv)
            clave = (vista, perdida, prof, iters)
            guardados[clave] = m
            fila = dict(vista=vista, perdida=perdida, prof=prof, iters=iters,
                        val_MdAElog=val["MdAElog"], val_MdMRE=val["MdMRE"],
                        arboles=m.n_iter_, seg=round(time.time() - t0, 1))
            for b, k in bloques.items():
                for mk, mv in metricas(y[k], np.exp(m.predict(X[k]))).items():
                    fila[f"{b}|{mk}"] = mv
            filas.append(fila)
            print(f"  {vista:22s} {perdida[:8]:8s} prof {prof:2d} it {iters:3d} "
                  f"-> val {val['MdAElog']:.4f} · fija "
                  f"{fila['prueba fija|MdAElog']:.4f}", flush=True)

    t = pd.DataFrame(filas)
    t.to_csv(OUT / "rejilla.csv", index=False)
    print(f"\n{SEP}\nMEJORES 12 POR VALIDACIÓN (MdAElog simétrico)\n{SEP}")
    cols = ["vista", "perdida", "prof", "iters", "arboles", "val_MdAElog",
            "prueba fija|MdMRE", "prueba fija|MdAElog", "prueba fija|R2",
            "prueba fija|spearman", "externo no Apache|MdAElog"]
    print(t.sort_values("val_MdAElog").head(12)[cols].round(4).to_string(index=False))

    print(f"\n{SEP}\nEFECTO DE LA PÉRDIDA, PROMEDIADO SOBRE LA REJILLA\n{SEP}")
    print(t.groupby("perdida")[["val_MdAElog", "prueba fija|MdAElog", "prueba fija|MdMRE",
                                "prueba fija|R2", "prueba fija|spearman",
                                "prueba fija|sesgo"]].mean().round(4).to_string())

    # ------------------------------------------------ apilado sobre validación
    mejores = t.sort_values("val_MdAElog").groupby("vista").head(1)
    claves = [(r.vista, r.perdida, r.prof, r.iters) for _, r in mejores.iterrows()]
    P = {}
    for c in claves:
        X = V[c[0]]
        P[c] = guardados[c].predict(X)
    Z = np.column_stack([P[c] for c in claves])
    apil = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Z[va_m], ly[va_m])
    print(f"\n{SEP}\nAPILADO DE {len(claves)} MODELOS (pesos ajustados en validación)\n{SEP}")
    print("  pesos: " + " · ".join(f"{c[0]} {w:+.3f}" for c, w in zip(claves, apil.coef_)))
    filas2 = []
    mejor_indiv = mejores.sort_values("val_MdAElog").iloc[0]
    for b, k in bloques.items():
        filas2.append(dict(modelo="apilado", bloque=b, n=int(k.sum()),
                           **metricas(y[k], np.exp(apil.predict(Z[k])))))
        filas2.append(dict(modelo=f"mejor individual ({mejor_indiv.vista})", bloque=b,
                           n=int(k.sum()),
                           **{m: mejor_indiv[f"{b}|{m}"] for m in
                              ("MdMRE", "MdAElog", "PRED25", "MAE_h", "R2", "sesgo",
                               "spearman")}))
        filas2.append(dict(modelo="mpnet + cuadrática (punto de partida)", bloque=b,
                           n=int(k.sum()),
                           **{m: t[(t.vista == "mpnet (referencia)") &
                                   (t.perdida == "squared_error") & (t.prof == 6) &
                                   (t.iters == 300)].iloc[0][f"{b}|{m}"] for m in
                              ("MdMRE", "MdAElog", "PRED25", "MAE_h", "R2", "sesgo",
                               "spearman")}))
    t2 = pd.DataFrame(filas2)
    for b in bloques:
        print(f"\n{b}")
        print(t2[t2.bloque == b].drop(columns=["bloque"]).round(4).to_string(index=False))
    t2.to_csv(OUT / "apilado.csv", index=False)
    json.dump({"claves": [list(c) for c in claves],
               "pesos": apil.coef_.tolist(), "intercepto": float(apil.intercept_)},
              open(OUT / "apilado.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/rejilla.csv y apilado.csv")


if __name__ == "__main__":
    main()
