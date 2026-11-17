"""E5 · Combinar representaciones, y probar la estimacion por analogia.

E4 dejo dos pistas. La primera: e5-large mejora de verdad la prueba fija
(R² 0,113 -> 0,149, ρ 0,320 -> 0,364). La segunda, menos esperada: tf-idf de
n-gramas de caracteres, con 256 componentes, queda a la par del mpnet y es el
MEJOR en error logaritmico sobre el bloque externo. Las dos representaciones
fallan en sitios distintos, asi que conviene juntarlas.

Ademas entra un metodo que la literatura de estimacion de esfuerzo lleva usando
desde Shepperd y Schofield (1997) y que aqui nunca se probo: estimacion por
ANALOGIA, o sea la mediana de los k vecinos mas parecidos. Es el baseline
clasico del area y sobre un espacio semantico bueno puede no ser ridiculo.

  vistas       e5-large · tf-idf · mpnet · combinaciones
  modelos      HGB sobre la vista · kNN por analogia · media geometrica de ambos
  seleccion    en validacion, y lectura sobre los cuatro bloques congelados
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
pd.set_option("display.width", 220)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/codificadores")
SEP = "=" * 126


def hgb():
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=0)


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


def analogia(Xtr, ytr, X, k=15):
    """Estimacion por analogia: mediana de log-horas de los k vecinos mas cercanos."""
    A = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
    B = X / np.linalg.norm(X, axis=1, keepdims=True)
    salida = np.empty(len(B))
    paso = 2048
    for i in range(0, len(B), paso):
        s = B[i:i + paso] @ A.T
        idx = np.argpartition(-s, k, axis=1)[:, :k]
        salida[i:i + paso] = np.median(ytr[idx], axis=1)
    return salida


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
    print(f"corpus sin Apache {len(sw):,} · train {tr_m.sum():,} · val {va_m.sum():,}")

    V = {k: np.load(CACHE / f"{k}.npy") for k in ("mpnet", "e5-large", "bge-m3")
         if (CACHE / f"{k}.npy").exists()}
    tfv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                          max_features=200_000, sublinear_tf=True).fit(tit[tr_m])
    Xt = tfv.transform(tit)
    V["tfidf"] = TruncatedSVD(256, random_state=0).fit(Xt[tr_m]).transform(Xt).astype(np.float32)
    # tf-idf de palabras, por si los n-gramas de caracteres solo capturan morfologia
    tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                          sublinear_tf=True).fit(tit[tr_m])
    V["tfidf_palabras"] = TruncatedSVD(256, random_state=0).fit(
        tfw.transform(tit)[tr_m]).transform(tfw.transform(tit)).astype(np.float32)
    print("  vistas: " + " · ".join(f"{k} {v.shape[1]}" for k, v in V.items()))

    combis = {
        "e5-large": ["e5-large"],
        "mpnet (actual)": ["mpnet"],
        "tf-idf caracteres": ["tfidf"],
        "tf-idf palabras": ["tfidf_palabras"],
        "e5-large + tf-idf car.": ["e5-large", "tfidf"],
        "e5-large + tf-idf car. + pal.": ["e5-large", "tfidf", "tfidf_palabras"],
        "mpnet + tf-idf car.": ["mpnet", "tfidf"],
        "e5-large + bge-m3 + tf-idf car.": ["e5-large", "bge-m3", "tfidf"],
    }
    filas = []
    for nombre, partes in combis.items():
        if any(p not in V for p in partes):
            continue
        X = np.hstack([V[p] for p in partes])
        m = hgb().fit(X[tr_m], ly[tr_m])
        pr = lambda k: np.exp(m.predict(X[k]))
        an = lambda k: np.exp(analogia(X[tr_m], ly[tr_m], X[k]))
        val = E.point_metrics(y[va_m], pr(va_m))["MdMRE"]
        val_an = E.point_metrics(y[va_m], an(va_m))["MdMRE"]
        for b, k in bloques.items():
            p_h, p_a = pr(k), an(k)
            for met, p in (("HGB", p_h), ("analogía k=15", p_a),
                           ("HGB × analogía", np.sqrt(p_h * p_a))):
                filas.append(dict(vista=nombre, dim=X.shape[1], metodo=met, bloque=b,
                                  n=int(k.sum()), val_MdMRE=val if met == "HGB" else val_an,
                                  **metricas(y[k], p)))
        print(f"  {nombre:32s} val {val:.4f} (analogía {val_an:.4f}) · "
              f"fija {filas[-9]['MdMRE']:.4f}", flush=True)

    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nPRUEBA FIJA\n{SEP}")
    v = t[t.bloque == "prueba fija"].sort_values("MdAElog")
    print(v[["vista", "dim", "metodo", "MdMRE", "MdAElog", "PRED25", "R2", "sesgo",
             "spearman"]].head(14).round(4).to_string(index=False))
    print(f"\n{SEP}\nEXTERNO NO APACHE\n{SEP}")
    v = t[t.bloque == "externo no Apache"].sort_values("MdAElog")
    print(v[["vista", "dim", "metodo", "MdMRE", "MdAElog", "PRED25", "R2", "sesgo",
             "spearman"]].head(14).round(4).to_string(index=False))

    print(f"\n{SEP}\nMEJOR POR BLOQUE, ELIGIENDO EN VALIDACIÓN\n{SEP}")
    mejor = t[t.metodo == "HGB"].sort_values("val_MdMRE").vista.iloc[0]
    print(f"  la validación elige la vista «{mejor}»")
    for b in bloques:
        v = t[(t.bloque == b) & (t.vista == mejor)]
        print(f"\n  {b}")
        print(v[["metodo", "MdMRE", "MdAElog", "PRED25", "R2", "spearman"]].round(4)
              .to_string(index=False))
    t.to_csv(OUT / "combinaciones.csv", index=False)
    json.dump({"mejor_vista_val": mejor}, open(OUT / "combinaciones.json", "w",
                                               encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/combinaciones.csv")


if __name__ == "__main__":
    main()
