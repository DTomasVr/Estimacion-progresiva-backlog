"""E13 · Higiene del entrenamiento y forma del objetivo.

Cuatro cosas baratas que nunca se probaron sobre la configuracion actual y que
atacan el ruido del entrenamiento en vez del modelo:

  DEDUPLICAR    el corpus tiene textos repetidos con horas distintas. Repetirlos
                no aporta informacion y sesga el ajuste hacia los titulos
                genericos, que son justamente los mas ruidosos.
  RECORTAR      winsorizar el objetivo en el entrenamiento (no en la prueba).
                Una tarea de 400 h con un titulo de seis palabras es una
                etiqueta equivocada, no una señal.
  PUNTUACION    transformar el objetivo a puntuaciones normales (rank-gauss)
  NORMAL        antes de ajustar y deshacer la transformacion al predecir.
                Quita las colas del objetivo sin tocar el orden.
  RECIENTE      pesar mas las tareas recientes: las costumbres de un equipo
                cambian y el corpus abarca mas de una decada.

Todo se lee sobre los mismos bloques congelados; nada de esto toca la prueba.
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

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/higiene")
SEP = "=" * 116


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
    X = [np.load(CACHE / "e5-large.npy")]
    for kw in (dict(analyzer="char_wb", ngram_range=(3, 5)), dict(ngram_range=(1, 2))):
        tf = TfidfVectorizer(min_df=3, max_features=200_000, sublinear_tf=True,
                             **kw).fit(tit[tr_m])
        M = tf.transform(tit)
        X.append(TruncatedSVD(256, random_state=0).fit(M[tr_m]).transform(M).astype(np.float32))
    X = np.hstack(X)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    va = (sw.split_t3 == "val").to_numpy()
    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    idx_tr = np.flatnonzero(tr_m)
    print(f"entrenamiento {len(idx_tr):,} · duplicados de texto "
          f"{int(sw.text_hash[tr_m].duplicated().sum()):,}")

    from scipy.stats import norm

    def rank_gauss(v):
        r = pd.Series(v).rank(method="average").to_numpy()
        return norm.ppf((r - .5) / len(r))

    variantes = {}
    variantes["0 · como está"] = (idx_tr, ly[idx_tr], None, None)
    # deduplicar por texto: una fila por texto, con la mediana de sus horas
    d = pd.DataFrame({"h": sw.text_hash[tr_m].to_numpy(), "i": idx_tr, "ly": ly[idx_tr]})
    primeros = d.groupby("h").agg(i=("i", "first"), ly=("ly", "median"))
    variantes["1 · deduplicado por texto"] = (primeros.i.to_numpy(),
                                              primeros.ly.to_numpy(), None, None)
    for lo, hi in ((.02, .98), (.01, .99)):
        a, b = np.quantile(ly[idx_tr], [lo, hi])
        variantes[f"2 · recortado {int(lo * 100)}–{int(hi * 100)} %"] = (
            idx_tr, np.clip(ly[idx_tr], a, b), None, None)
    # puntuaciones normales: se ajusta sobre la transformada y se deshace por interpolacion
    orden = np.argsort(ly[idx_tr])
    ly_ord = ly[idx_tr][orden]
    z_ord = rank_gauss(ly[idx_tr])[orden]
    variantes["3 · puntuaciones normales"] = (idx_tr, rank_gauss(ly[idx_tr]),
                                              (z_ord, ly_ord), None)
    if "created" in sw.columns:
        t = pd.to_datetime(sw.created, errors="coerce")
        edad = (t.max() - t).dt.days.fillna(3650).to_numpy()
        for vida in (1500, 3000):
            w = np.exp(-edad[idx_tr] / vida)
            variantes[f"4 · peso por recencia (vida {vida} d)"] = (idx_tr, ly[idx_tr],
                                                                    None, w)

    filas = []
    for nombre, (idx, obj, inv, peso) in variantes.items():
        m = hgb().fit(X[idx], obj, sample_weight=peso)
        def pred(k):
            v = m.predict(X[k])
            return np.exp(np.interp(v, inv[0], inv[1])) if inv else np.exp(v)
        val = metricas(y[va], pred(va))
        for b, k in bloques.items():
            filas.append(dict(variante=nombre, bloque=b, n=int(k.sum()),
                              n_train=len(idx), val_MdAElog=val["MdAElog"],
                              **metricas(y[k], pred(k))))
        print(f"  {nombre:34s} n={len(idx):6,} val {val['MdAElog']:.4f} · "
              f"fija {filas[-3]['MdAElog']:.4f} · R² {filas[-3]['R2']:+.4f}", flush=True)

    t = pd.DataFrame(filas)
    cols = ["variante", "n_train", "val_MdAElog", "MdMRE", "MdAElog", "PRED25", "R2",
            "sesgo", "spearman"]
    for b in bloques:
        print(f"\n{SEP}\n{b}\n{SEP}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"variantes": list(variantes)}, open(OUT / "meta.json", "w",
                                                   encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
