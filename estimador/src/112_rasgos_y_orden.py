"""E15 · Rasgos de superficie, y ordenar en vez de regresionar.

Dos ideas distintas contra el mismo hueco. El modelo explica el 3,6 % de la
varianza DENTRO del proyecto en proyectos nuevos, y el techo por ruido deja
sitio para bastante mas. O sea que queda señal intra-proyecto sin explotar.

  RASGOS DE SUPERFICIE   cosas que el vector semantico puede estar ignorando
                         porque no son semanticas: cuantas palabras tiene el
                         titulo, si lleva numeros, siglas en mayusculas,
                         identificadores tipo codigo, signos de interrogacion.
                         Un "fix typo" y un "refactor authentication module"
                         se distinguen por el contenido, pero "add X" y
                         "add X, Y, Z and update docs" se distinguen por forma.

  ORDENAR                el R² intra y el Spearman miden ORDEN, y estamos
                         optimizando error cuadratico. Un modelo de ranking con
                         el proyecto como grupo optimiza directamente el orden
                         dentro del proyecto, que es justo lo que el texto puede
                         saber y lo que combinado con un ancla da la prediccion.
                         Se usa XGBRanker con los proyectos como consultas.

Se lee con las metricas de siempre y, sobre todo, con rho_intra y R2_intra.
"""
import json
import re
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
pd.set_option("display.width", 215)
DATA = "data/processed/agix_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/rasgos_orden")
SEP = "=" * 120


def superficie(tit):
    """Rasgos de forma del titulo, nada semanticos."""
    s = tit.astype(str)
    return np.column_stack([
        s.str.split().str.len(),
        s.str.len(),
        s.str.count(r"\d"),
        s.str.count(r"[A-Z]{2,}"),
        s.str.count(r"[\.\-_/]"),
        s.str.count(r"[,;]"),
        s.str.contains(r"\?").astype(int),
        s.str.contains(r"[a-z]+[A-Z]", regex=True).astype(int),        # camelCase
        s.str.contains(r"\b(and|y)\b", case=False, regex=True).astype(int),
        s.str.split().str.len().div(1 + s.str.count(r"[,;]")),
    ]).astype(np.float32)


def metricas(y, p, proyecto=None):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly, lp = np.log(y), np.log(p)
    d = dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lp - ly))),
             PRED25=m["PRED25"], R2=float(1 - ((ly - lp) ** 2).mean() / ly.var()),
             spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))
    if proyecto is not None:
        g = pd.DataFrame({"p": proyecto, "ly": ly, "lp": lp})
        dy = ly - g.groupby("p").ly.transform("mean").to_numpy()
        dp = lp - g.groupby("p").lp.transform("mean").to_numpy()
        d["R2_intra"] = float(1 - ((dy - dp) ** 2).mean() / dy.var())
        d["rho_intra"] = float(pd.Series(dy).corr(pd.Series(dp), method="spearman"))
    return d


def hgb():
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sw = sw[~sw.apache].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    tr_m = (sw.split_t3 == "train").to_numpy()
    partes = [np.load(CACHE / "e5-large.npy")]
    for kw in (dict(analyzer="char_wb", ngram_range=(3, 5)), dict(ngram_range=(1, 2))):
        tf = TfidfVectorizer(min_df=3, max_features=200_000, sublinear_tf=True,
                             **kw).fit(tit[tr_m])
        M = tf.transform(tit)
        partes.append(TruncatedSVD(256, random_state=0).fit(M[tr_m]).transform(M).astype(np.float32))
    X = np.hstack(partes)
    S = superficie(tit)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    print(f"vista {X.shape} · superficie {S.shape}")

    filas = []
    vistas = {"1 · texto (referencia)": X,
              "2 · texto + superficie": np.hstack([X, S]),
              "3 · solo superficie": S}
    for nombre, XX in vistas.items():
        m = hgb().fit(XX[tr_m], ly[tr_m])
        for b, k in bloques.items():
            filas.append(dict(modelo=nombre, bloque=b, n=int(k.sum()),
                              **metricas(y[k], np.exp(m.predict(XX[k])),
                                         sw.project.to_numpy()[k])))
        print(f"  {nombre:26s} fija R² {filas[-3]['R2']:+.4f} · "
              f"R²_intra {filas[-3]['R2_intra']:+.4f}", flush=True)

    # ---------------------------------------------- ordenacion
    try:
        from xgboost import XGBRanker
        Xr = np.hstack([X, S])
        orden = np.argsort(sw.project.to_numpy()[tr_m], kind="stable")
        idx_tr = np.flatnonzero(tr_m)[orden]
        grupos = pd.Series(sw.project.to_numpy()[idx_tr]).groupby(
            pd.Series(sw.project.to_numpy()[idx_tr]), sort=False).size().to_numpy()
        # relevancia graduada: decil de log-horas DENTRO del proyecto
        rel = (pd.DataFrame({"p": sw.project.to_numpy()[idx_tr], "ly": ly[idx_tr]})
               .groupby("p").ly.rank(pct=True) * 9).round().astype(int).to_numpy()
        rk = XGBRanker(objective="rank:pairwise", n_estimators=400, max_depth=6,
                       learning_rate=.06, tree_method="hist", random_state=0)
        rk.fit(Xr[idx_tr], rel, group=grupos)
        print("  ranker ajustado")
        for b, k in bloques.items():
            s = rk.predict(Xr[k])
            proy = sw.project.to_numpy()[k]
            dy = ly[k] - pd.DataFrame({"p": proy, "ly": ly[k]}).groupby(
                "p").ly.transform("mean").to_numpy()
            ds = s - pd.DataFrame({"p": proy, "s": s}).groupby("p").s.transform("mean").to_numpy()
            filas.append(dict(modelo="4 · ranker (orden dentro del proyecto)", bloque=b,
                              n=int(k.sum()), MdMRE=np.nan, MdAElog=np.nan, PRED25=np.nan,
                              R2=np.nan,
                              spearman=float(pd.Series(y[k]).corr(pd.Series(s),
                                                                  method="spearman")),
                              R2_intra=np.nan,
                              rho_intra=float(pd.Series(dy).corr(pd.Series(ds),
                                                                 method="spearman"))))
            print(f"    ranker {b:20s} ρ_intra {filas[-1]['rho_intra']:+.4f}", flush=True)
    except Exception as e:
        print(f"  ranker no disponible: {type(e).__name__}: {str(e)[:100]}")

    t = pd.DataFrame(filas)
    cols = ["modelo", "MdMRE", "MdAElog", "PRED25", "R2", "spearman", "R2_intra", "rho_intra"]
    for b in bloques:
        print(f"\n{SEP}\n{b}\n{SEP}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"n_superficie": int(S.shape[1])},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
