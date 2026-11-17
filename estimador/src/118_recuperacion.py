"""E20 · Si la tarea ya se hizo antes en este proyecto, copiar sus horas.

E19 dejo un numero que pide ser explotado: dos tareas con coseno >= 0,95 DENTRO
del mismo proyecto difieren ×1,00 en la mediana. Si eso es cierto, para las
tareas que tienen un gemelo en el historial del proyecto no hace falta modelo
ninguno: basta con mirar cuanto tardo el gemelo.

Hay que decir el reparo antes de medir: las tareas que tienen gemelo no son una
muestra cualquiera. Son las formularias y repetitivas, que probablemente sean
tambien las mas predecibles. Asi que hay que medir dos cosas por separado:

  1  que fraccion de las tareas de prueba tiene un gemelo en el entrenamiento
     del MISMO proyecto, y a que umbral
  2  en ese subconjunto, quien acierta mas: el modelo o copiar al gemelo. Y en
     el subconjunto sin gemelo, cuanto empeora el modelo -- que es la prueba de
     si el techo de E19 valia para todas las tareas o solo para esas

Si copiar gana, es una funcion de producto ademas de un resultado: "esta tarea
se parece a esta otra que hiciste, que te llevo X horas".
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
OUT = Path("results/recuperacion")
SEP = "=" * 122
UMBRALES = [.90, .93, .95, .97]


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly, lp = np.log(y), np.log(p)
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lp - ly))),
                PRED25=m["PRED25"], MAE_h=m["MAE_h"],
                R2=float(1 - ((ly - lp) ** 2).mean() / ly.var()),
                sesgo=float(np.exp(np.median(lp - ly))))


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
    print("modelo final ajustado")

    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    filas = []
    for b, k in bloques.items():
        ib = np.flatnonzero(k)
        # el historial disponible: entrenamiento del MISMO proyecto
        mejor_s = np.full(len(ib), -1.0)
        mejor_h = np.full(len(ib), np.nan)
        for p in np.unique(proy[k]):
            hist = np.flatnonzero(tr_m & (proy == p))
            if not len(hist):
                continue
            tgt = np.flatnonzero(proy[k] == p)
            s = Xn[ib[tgt]] @ Xn[hist].T
            j = s.argmax(1)
            mejor_s[tgt] = s[np.arange(len(tgt)), j]
            mejor_h[tgt] = y[hist[j]]
        yb, pb = y[k], np.exp(lp[k])
        for u in UMBRALES:
            tiene = mejor_s >= u
            if tiene.sum() < 25:
                continue
            filas.append(dict(bloque=b, umbral=u, subconjunto="con gemelo",
                              n=int(tiene.sum()), fraccion=float(tiene.mean()),
                              metodo="modelo", **metricas(yb[tiene], pb[tiene])))
            filas.append(dict(bloque=b, umbral=u, subconjunto="con gemelo",
                              n=int(tiene.sum()), fraccion=float(tiene.mean()),
                              metodo="copiar al gemelo",
                              **metricas(yb[tiene], mejor_h[tiene])))
            filas.append(dict(bloque=b, umbral=u, subconjunto="con gemelo",
                              n=int(tiene.sum()), fraccion=float(tiene.mean()),
                              metodo="media geométrica",
                              **metricas(yb[tiene], np.sqrt(pb[tiene] * mejor_h[tiene]))))
            if (~tiene).sum() > 25:
                filas.append(dict(bloque=b, umbral=u, subconjunto="sin gemelo",
                                  n=int((~tiene).sum()),
                                  fraccion=float((~tiene).mean()),
                                  metodo="modelo", **metricas(yb[~tiene], pb[~tiene])))
        print(f"  {b}: cobertura de gemelo " + " · ".join(
            f"≥{u} {float((mejor_s >= u).mean()):.1%}" for u in UMBRALES), flush=True)

    t = pd.DataFrame(filas)
    cols = ["umbral", "subconjunto", "metodo", "n", "fraccion", "MdMRE", "MdAElog",
            "PRED25", "R2", "sesgo"]
    for b in bloques:
        print(f"\n{SEP}\n{b}\n{SEP}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"umbrales": UMBRALES}, open(OUT / "meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
