"""E4 · ¿Y si el cuello de botella es el codificador?

Todo el trabajo se ha hecho sobre `paraphrase-multilingual-mpnet-base-v2`, que
es de 2021 y se eligio al principio del proyecto sin comparar con nada. Desde
entonces han salido codificadores multilingues bastante mejores en las pruebas
de recuperacion y de similitud. Nunca se ha comprobado si el techo de 0,68 es
del problema o del codificador.

Es una comparacion limpia: mismo corpus, mismas particiones congeladas, mismo
regresor, misma vista de titulos. Lo unico que cambia es quien convierte el
texto en vectores.

  paraphrase-multilingual-mpnet-base-v2   278M · 768d · el actual
  intfloat/multilingual-e5-base           278M · 768d · necesita el prefijo "query: "
  intfloat/multilingual-e5-large          560M · 1024d
  BAAI/bge-m3                             568M · 1024d
  Qwen/Qwen3-Embedding-0.6B               595M · 1024d · 2025

Tambien se prueba la concatenacion de los dos mejores, por si aportan cosas
distintas, y TF-IDF de caracteres como control: si un modelo de 2021 o de 2025
no le gana a n-gramas de caracteres, el problema no es la representacion.

    python src/98_codificadores.py [solo-tfidf]
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/codificadores")
SEP = "=" * 122
MODELOS = {
    "mpnet (el actual)": ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", ""),
    "e5-base": ("intfloat/multilingual-e5-base", "query: "),
    "e5-large": ("intfloat/multilingual-e5-large", "query: "),
    "bge-m3": ("BAAI/bge-m3", ""),
    "qwen3-0.6B": ("Qwen/Qwen3-Embedding-0.6B", ""),
}


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


def cargar_datos():
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    return sw[~sw.apache].reset_index(drop=True)


def codificar(sw):
    """Codifica los titulos con cada modelo y deja la cache en disco."""
    import torch
    from sentence_transformers import SentenceTransformer

    CACHE.mkdir(parents=True, exist_ok=True)
    textos = sw.title.fillna("").astype(str).tolist()
    vectores = {}
    for etq, (nombre, prefijo) in MODELOS.items():
        f = CACHE / f"{etq.split(' ')[0]}.npy"
        if f.exists():
            vectores[etq] = np.load(f)
            print(f"  {etq:20s} en caché {vectores[etq].shape}")
            continue
        t0 = time.time()
        try:
            st = SentenceTransformer(nombre, device="cuda" if torch.cuda.is_available() else "cpu")
            st.max_seq_length = min(getattr(st, "max_seq_length", 128) or 128, 128)
            v = st.encode([prefijo + t for t in textos], batch_size=48,
                          show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
            np.save(f, v)
            vectores[etq] = v
            print(f"  {etq:20s} {v.shape} en {time.time() - t0:.0f} s", flush=True)
            del st
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {etq:20s} FALLÓ: {type(e).__name__}: {str(e)[:120]}", flush=True)
    return vectores


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sw = cargar_datos()
    print(f"corpus sin Apache: {len(sw):,} tareas")
    V = codificar(sw)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    tr_m = (sw.split_t3 == "train").to_numpy()
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                            max_features=200_000, sublinear_tf=True)
    Xt = tfidf.fit(sw.title.fillna("").astype(str)[tr_m]).transform(
        sw.title.fillna("").astype(str))
    V["tf-idf de caracteres (control)"] = TruncatedSVD(
        256, random_state=0).fit(Xt[tr_m]).transform(Xt).astype(np.float32)
    print(f"  tf-idf {V['tf-idf de caracteres (control)'].shape}")

    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "prueba TAWOS": ((sw.split_t3 == "test") & (sw.source == "TAWOS")).to_numpy(),
               "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    va_m = (sw.split_t3 == "val").to_numpy()
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)

    if len(V) >= 3:
        mejores = [k for k in ("e5-large", "bge-m3", "qwen3-0.6B") if k in V][:2]
        if len(mejores) == 2:
            V["concatenación de los dos mejores"] = np.hstack([V[mejores[0]], V[mejores[1]]])

    filas = []
    for etq, X in V.items():
        t0 = time.time()
        m = hgb().fit(X[tr_m], ly[tr_m])
        pv = np.exp(m.predict(X[va_m]))
        val = E.point_metrics(y[va_m], pv)["MdMRE"]
        for b, k in bloques.items():
            filas.append(dict(codificador=etq, dim=X.shape[1], bloque=b, n=int(k.sum()),
                              val_MdMRE=val, segundos=round(time.time() - t0, 1),
                              **metricas(y[k], np.exp(m.predict(X[k])))))
        print(f"  {etq:32s} val {val:.4f} · fija "
              f"{filas[-4]['MdMRE']:.4f} · externo {filas[-1]['MdMRE']:.4f}", flush=True)

    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nCODIFICADORES, MISMO CORPUS Y MISMAS PARTICIONES\n{SEP}")
    for b in bloques:
        print(f"\n{b}")
        v = t[t.bloque == b].sort_values("MdAElog")
        print(v[["codificador", "dim", "val_MdMRE", "MdMRE", "MdAElog", "PRED25",
                 "R2", "sesgo", "spearman"]].round(4).to_string(index=False))

    base = t[(t.codificador.str.startswith("mpnet")) & (t.bloque == "prueba fija")].iloc[0]
    print(f"\n{SEP}\nFRENTE AL ACTUAL, EN LA PRUEBA FIJA\n{SEP}")
    for _, r in t[t.bloque == "prueba fija"].iterrows():
        if r.codificador.startswith("mpnet"):
            continue
        print(f"  {r.codificador:32s} ΔMdMRE {r.MdMRE - base.MdMRE:+.4f} · "
              f"ΔR² {r.R2 - base.R2:+.4f} · Δρ {r.spearman - base.spearman:+.4f}")

    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"modelos": {k: v[0] for k, v in MODELOS.items()}},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
