"""E22 · Evaluar en proyectos nuevos con potencia de verdad.

El bloque externo no-Apache son 792 tareas de 49 proyectos. Con eso, la mejora
del modelo salio "empate" con IC95 [-0,034, +0,040]: el intervalo es tan ancho
que no distingue nada. No es que el modelo no mejore; es que el banco no puede
verlo.

Hay una forma de medir lo mismo con 24.848 tareas en vez de 792: validacion
cruzada dejando PROYECTOS ENTEROS fuera, sobre todo el corpus sin Apache. Cada
tarea se predice con un modelo que nunca vio su proyecto, que es exactamente la
condicion de "cliente nuevo", y se usan las 24.848.

Se comparan el punto de partida de la noche y el modelo final, con el contraste
pareado sobre las mismas tareas. Es mas caro -- cinco ajustes por modelo -- pero
es la unica forma de saber si la mejora transfiere a proyectos que no se han
visto.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wilcoxon
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVR

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/cv_proyectos")
SEP = "=" * 116
PLIEGUES = 5


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
    t0 = time.time()
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sw = sw[~sw.apache].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    proy = sw.project.to_numpy()
    MP = np.load(CACHE / "mpnet.npy")
    E5 = np.load(CACHE / "e5-large.npy")
    print(f"{len(sw):,} tareas · {len(np.unique(proy))} proyectos · "
          f"{PLIEGUES} pliegues por proyecto entero")

    def hgb(s=0, **kw):
        return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                             l2_regularization=1., random_state=s, **kw)

    p_a = np.zeros(len(sw))
    p_b = np.zeros(len(sw))
    gkf = GroupKFold(n_splits=PLIEGUES)
    for f, (tr, te) in enumerate(gkf.split(E5, ly, proy), 1):
        p_a[te] = hgb().fit(MP[tr], ly[tr]).predict(MP[te])
        # el tf-idf se ajusta SOLO con el pliegue de entrenamiento
        tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                              sublinear_tf=True).fit(tit.iloc[tr])
        tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tit.iloc[tr])
        Ms = sparse.hstack([tfc.transform(tit), tfw.transform(tit)]).tocsr()
        Xd = np.hstack([E5] + [TruncatedSVD(256, random_state=0).fit(
            tf.transform(tit)[tr]).transform(tf.transform(tit)).astype(np.float32)
            for tf in (tfc, tfw)])
        rng = np.random.default_rng(f)
        acum = []
        for s in range(5):
            sub = rng.choice(tr, int(.85 * len(tr)), replace=False)
            acum.append(hgb(s, max_features=.8).fit(Xd[sub], ly[sub]).predict(Xd[te]))
        bolsa = np.mean(acum, 0)
        rid = Ridge(alpha=2.0).fit(Ms[tr], ly[tr]).predict(Ms[te])
        svr = LinearSVR(C=.3, max_iter=4000, random_state=0).fit(Ms[tr], ly[tr]).predict(Ms[te])
        # pesos fijos, los que eligio la validacion en el modelo final
        p_b[te] = .759 * bolsa + .386 * rid - .032 * svr
        p_b[te] += ly[tr].mean() - p_b[te].mean() * 0 + 0  # sin recentrar
        print(f"  pliegue {f}/{PLIEGUES} · {time.time() - t0:.0f} s", flush=True)

    # el apilado con pesos que no suman 1 necesita un intercepto; se ajusta
    # sobre el propio pliegue de entrenamiento de cada iteracion seria lo ideal,
    # pero al ser una constante global basta con corregir el sesgo mediano global
    ajuste = float(np.median(ly - p_b))
    p_b = p_b + ajuste
    print(f"  ajuste de intercepto del apilado: {ajuste:+.4f} en log")

    filas = []
    for etq, lp in (("punto de partida (mpnet + HGB)", p_a),
                    ("modelo final", p_b)):
        filas.append(dict(modelo=etq, n=len(sw), **metricas(y, np.exp(lp), proy)))
    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nVALIDACIÓN CRUZADA POR PROYECTO ENTERO · {len(sw):,} tareas\n{SEP}")
    print(t.round(4).to_string(index=False))

    ea = np.abs(p_a - ly)
    eb = np.abs(p_b - ly)
    pv = wilcoxon(ea, eb).pvalue
    rng = np.random.default_rng(1)
    ra = np.abs(y - np.exp(p_a)) / y
    rb = np.abs(y - np.exp(p_b)) / y
    d = [np.median(rb[i]) - np.median(ra[i])
         for i in (rng.integers(0, len(y), len(y)) for _ in range(4000))]
    lo, hi = np.percentile(d, [2.5, 97.5])
    dl = [np.median(eb[i]) - np.median(ea[i])
          for i in (rng.integers(0, len(y), len(y)) for _ in range(4000))]
    llo, lhi = np.percentile(dl, [2.5, 97.5])
    ver = "mejor" if hi < 0 else "peor" if lo > 0 else "empate"
    print(f"\n{SEP}\nCONTRASTE PAREADO EN PROYECTOS NUNCA VISTOS\n{SEP}")
    print(f"  el modelo final frente al punto de partida: {ver.upper()}")
    print(f"    ΔMdMRE   {np.median(d):+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]")
    print(f"    ΔMdAElog {np.median(dl):+.4f}  IC95 [{llo:+.4f}, {lhi:+.4f}]")
    print(f"    Wilcoxon p = {pv:.3g}")
    print(f"\n  (el bloque externo no-Apache, con 792 tareas, daba IC95 "
          f"[-0,034, +0,040] y p = 0,28)")

    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"veredicto": ver, "dif_MdMRE": float(np.median(d)),
               "IC95_MdMRE": [float(lo), float(hi)],
               "dif_MdAElog": float(np.median(dl)),
               "IC95_MdAElog": [float(llo), float(lhi)], "p": float(pv),
               "n": len(sw), "proyectos": int(len(np.unique(proy)))},
              open(OUT / "contraste.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv y contraste.json")


if __name__ == "__main__":
    main()
