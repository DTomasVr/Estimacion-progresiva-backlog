"""E11 · Lo que queda por exprimir del mismo corpus: ensamblado y expertos por fuente.

Tres cosas baratas que no se han probado y que suelen dar una decima cada una:

  BOLSA        promediar varios HGB con semillas y submuestras distintas. Es la
               unica forma de reducir varianza sin cambiar el sesgo.
  EXPERTOS     un modelo por fuente (SiP, TAWOS, Jira sin Apache) y una mezcla.
               El corpus son tres organizaciones con costumbres distintas; un
               modelo unico tiene que promediarlas. La mezcla se aprende en
               validacion, no se elige la fuente real, porque en produccion no
               se sabe de que organizacion viene la tarea.
  MAS DATOS    reentrenar con train + val para el artefacto final. No se puede
               leer en validacion, asi que se lee solo en prueba y se reporta
               como lo que es: el modelo que se desplegaria.

Sobre la mejor vista de E5-E6: e5-large + tf-idf de caracteres + tf-idf de
palabras.
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
from sklearn.linear_model import RidgeCV

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 205)
DATA = "data/processed/agix_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/ensamblado")
SEP = "=" * 114


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly, lp = np.log(y), np.log(p)
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lp - ly))),
                PRED25=m["PRED25"], MAE_h=m["MAE_h"],
                R2=float(1 - ((ly - lp) ** 2).mean() / ly.var()),
                sesgo=float(np.exp(np.median(lp - ly))),
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
    va_m = (sw.split_t3 == "val").to_numpy()
    X = [np.load(CACHE / "e5-large.npy")]
    for kw in (dict(analyzer="char_wb", ngram_range=(3, 5)), dict(ngram_range=(1, 2))):
        tf = TfidfVectorizer(min_df=3, max_features=200_000, sublinear_tf=True,
                             **kw).fit(tit[tr_m])
        M = tf.transform(tit)
        X.append(TruncatedSVD(256, random_state=0).fit(M[tr_m]).transform(M).astype(np.float32))
    X = np.hstack(X)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "prueba TAWOS": ((sw.split_t3 == "test") & (sw.source == "TAWOS")).to_numpy(),
               "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    print(f"vista {X.shape} · train {tr_m.sum():,} · val {va_m.sum():,}")

    pred = {}
    base = hgb().fit(X[tr_m], ly[tr_m])
    pred["1 · base (referencia)"] = base.predict(X)

    # ---------------------------------------------- bolsa
    rng = np.random.default_rng(0)
    idx_tr = np.flatnonzero(tr_m)
    acum = []
    for s in range(7):
        sub = rng.choice(idx_tr, int(.85 * len(idx_tr)), replace=False)
        acum.append(hgb(semilla=s, max_features=.8).fit(X[sub], ly[sub]).predict(X))
        print(f"  bolsa {s + 1}/7", flush=True)
    pred["2 · bolsa de 7"] = np.mean(acum, 0)

    # ---------------------------------------------- expertos por fuente
    expertos, cols = {}, []
    for f in ("SiP", "TAWOS", "JiraPublico"):
        k = tr_m & (sw.source == f).to_numpy()
        if k.sum() < 500:
            continue
        expertos[f] = hgb().fit(X[k], ly[k])
        cols.append(expertos[f].predict(X))
        print(f"  experto {f} ({k.sum():,} filas)", flush=True)
    Z = np.column_stack(cols)
    mezcla = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Z[va_m], ly[va_m])
    pred["3 · expertos por fuente, mezclados en val"] = mezcla.predict(Z)
    pred["4 · media de los expertos"] = Z.mean(1)
    print("  pesos de la mezcla: " + " · ".join(
        f"{f} {w:+.3f}" for f, w in zip(expertos, mezcla.coef_)))

    # ---------------------------------------------- base + expertos apilados
    Z2 = np.column_stack([pred["1 · base (referencia)"], pred["2 · bolsa de 7"], Z])
    ap = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Z2[va_m], ly[va_m])
    pred["5 · apilado de todo (pesos en val)"] = ap.predict(Z2)

    # ---------------------------------------------- reentrenado con train + val
    trva = tr_m | va_m
    pred["6 · base reentrenada con train+val"] = hgb().fit(X[trva], ly[trva]).predict(X)
    acum2 = []
    idx2 = np.flatnonzero(trva)
    for s in range(7):
        sub = rng.choice(idx2, int(.85 * len(idx2)), replace=False)
        acum2.append(hgb(semilla=s, max_features=.8).fit(X[sub], ly[sub]).predict(X))
    pred["7 · bolsa de 7 con train+val"] = np.mean(acum2, 0)

    filas = []
    for nombre, lp in pred.items():
        val = metricas(y[va_m], np.exp(lp[va_m]))["MdAElog"] if "train+val" not in nombre else np.nan
        for b, k in bloques.items():
            filas.append(dict(modelo=nombre, bloque=b, n=int(k.sum()), val_MdAElog=val,
                              **metricas(y[k], np.exp(lp[k]))))
    t = pd.DataFrame(filas)
    cols_m = ["modelo", "val_MdAElog", "MdMRE", "MdAElog", "PRED25", "R2", "sesgo", "spearman"]
    for b in bloques:
        print(f"\n{SEP}\n{b}\n{SEP}")
        print(t[t.bloque == b][cols_m].round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"pesos_mezcla": dict(zip(expertos, mezcla.coef_.tolist())),
               "pesos_apilado": ap.coef_.tolist()},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
