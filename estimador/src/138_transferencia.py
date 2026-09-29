"""E38 - Recuperar las 65.000 tareas de Apache sin que contaminen la escala.

E17 decidio tirar Apache y acerto: es el 72 % del corpus, su mediana es 0,67 h
contra 2,0 h del resto, y mezclarlo crudo desplazaba todas las predicciones
hacia abajo. Desde entonces el modelo se entrena con 24.848 filas y se ignoran
65.003. Con una senal tan debil como esta, tirar tres cuartas partes de los
datos es caro, y conviene mirar si habia otra forma.

La hay, y sale del propio diagnostico. E30 mide que un desplazamiento por
proyecto -- conocido de oraculo -- sube el R2 de 0,182 a 0,297, o sea que buena
parte de lo que distingue una tarea de otra es EN QUE PROYECTO esta, no que dice.
Eso parte el problema en dos:

    log y  =  nivel del proyecto  +  desviacion dentro del proyecto

El nivel es justo lo que no transfiere de Apache: sus proyectos van a otra
escala. Pero la DESVIACION -- que dentro de un mismo proyecto una tarea sea
grande o pequena, y que eso se note en el texto -- no tiene razon para no
transferir, y para aprenderla se pueden usar las 89.851 tareas enteras.

  A  lo actual                     regresion sobre log y, solo sin Apache
  B  desviacion, solo sin Apache    mismo tamano de muestra, para aislar el
                                    efecto de centrar del efecto de los datos
  C  desviacion, CORPUS ENTERO      3,6 veces mas datos para la misma tarea
  D  C con Apache pesando la mitad
  E  C como rasgo anadido al modelo actual

Y dos formas de poner el nivel de vuelta: la mediana global (que es lo que hay en
arranque en frio, sin historial) y la mediana del proyecto sacada del
entrenamiento (que es lo que habra cuando el proyecto lleve un tiempo). La
segunda no aplica al bloque externo, donde los proyectos no se han visto nunca.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/transferencia")
S = "=" * 128


def metricas(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), .01, None)
    q = np.log(p / y)
    r = np.abs(p - y) / y
    m = float(np.median(q))
    return dict(MAE_h=float(np.abs(p - y).mean()),
                MdAElog=float(np.median(np.abs(q))),
                zeta_pc=float((np.exp(np.median(np.abs(q))) - 1) * 100),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100),
                R2log=float(1 - ((np.log(y) - np.log(p)) ** 2).mean() / np.log(y).var()),
                PRED25=float(np.mean(r <= .25)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                rho=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def tfidf(textos, apto):
    """Vista de CPU, para probar la hipotesis sin pelearse por la GPU.

    Si el corpus entero no aporta con n-gramas de caracteres y palabras, que
    sobre titulos cortos rinden casi tanto como un codificador neuronal, tampoco
    va a aportar con e5 y no merece la pena codificar 90.000 titulos."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    t0 = time.time()
    partes = []
    for kw in (dict(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True),
               dict(ngram_range=(1, 2), min_df=3, sublinear_tf=True)):
        tf = TfidfVectorizer(**kw).fit(pd.Series(textos)[apto])
        M = tf.transform(textos)
        partes.append(TruncatedSVD(256, random_state=0).fit(M[apto])
                      .transform(M).astype(np.float32))
    print(f"  TF-IDF + SVD listo en {time.time() - t0:.0f} s", flush=True)
    return np.hstack(partes)


def codificar(textos):
    f = CACHE / "e5-large-todo.npy"
    if f.exists():
        return np.load(f)
    import torch
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    st = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")
    st.max_seq_length = 128
    v = st.encode(["query: " + t for t in textos], batch_size=48,
                  show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    np.save(f, v)
    print(f"  corpus entero codificado {v.shape} en {time.time() - t0:.0f} s", flush=True)
    del st
    torch.cuda.empty_cache()
    return v


def hgb(**kw):
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=0, **kw)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy().reset_index(drop=True)
    sw["apache"] = sw.fuente_jira.eq("Apache")
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    # filas utilizables para aprender: todo Apache (nunca se evalua) mas el
    # entrenamiento del resto. Ni una fila de validacion o prueba entra aqui,
    # tampoco para calcular las medianas por proyecto ni para ajustar el TF-IDF.
    tr_r = (~sw.apache) & sw.split_t3.eq("train")
    apto = (sw.apache | tr_r).to_numpy()
    tit = sw.title.fillna("").astype(str).tolist()
    vista = "tfidf" if "tfidf" in sys.argv else "e5"
    X = tfidf(tit, apto) if vista == "tfidf" else codificar(tit)
    print(f"vista {vista} {X.shape} · {len(sw):,} tareas · Apache {int(sw.apache.sum()):,} · "
          f"resto {int((~sw.apache).sum()):,} · {time.time() - t0:.0f} s", flush=True)
    va = ((~sw.apache) & sw.split_t3.eq("val")).to_numpy()
    bl = {"prueba fija": ((~sw.apache) & sw.split_t3.eq("test")).to_numpy(),
          "prueba SiP": ((~sw.apache) & sw.split_t3.eq("test") & sw.source.eq("SiP")).to_numpy(),
          "externo no Apache": ((~sw.apache) & sw.split_t3.eq("externo")).to_numpy()}

    med = pd.Series(ly[apto]).groupby(sw.project[apto].to_numpy()).median()
    # el nivel que hay que devolver es el del DOMINIO donde se despliega, no el
    # del corpus de entrenamiento: Apache tiene mediana 0,67 h y el resto 2,0 h,
    # asi que usar la mediana global dejaria todas las predicciones a la mitad.
    glob = float(np.median(ly[tr_r.to_numpy()]))
    nivel_proy = sw.project.map(med).fillna(glob).to_numpy(float)
    vistos = sw.project.isin(med.index).to_numpy()
    d = ly - nivel_proy
    print(f"proyectos con nivel estimable: {int(vistos.sum()):,} tareas · "
          f"nivel global {np.exp(glob):.2f} h")
    print(f"desviacion dentro de proyecto: sd {d[apto].std():.3f} frente a "
          f"sd de log y {ly[apto].std():.3f}  -> el proyecto explica el "
          f"{1 - d[apto].var() / ly[apto].var():.1%} de la varianza")

    tr_solo = tr_r.to_numpy()
    modelos = {}
    modelos["A · log y, solo sin Apache (lo actual)"] = (hgb().fit(X[tr_solo], ly[tr_solo]), None)
    modelos["B · desviacion, solo sin Apache"] = (hgb().fit(X[tr_solo], d[tr_solo]), "d")
    modelos["C · desviacion, CORPUS ENTERO"] = (hgb().fit(X[apto], d[apto]), "d")
    w = np.where(sw.apache[apto], .5, 1.)
    modelos["D · C con Apache pesando la mitad"] = (
        hgb().fit(X[apto], d[apto], sample_weight=w), "d")
    for n, _ in modelos.items():
        print(f"  {n} ajustado · {time.time() - t0:.0f} s", flush=True)

    dC = modelos["C · desviacion, CORPUS ENTERO"][0].predict(X)
    Xe = np.hstack([X, dC[:, None]])
    modelos["E · el actual + la desviacion del corpus entero"] = (
        hgb().fit(Xe[tr_solo], ly[tr_solo]), "e")
    print(f"  E ajustado · {time.time() - t0:.0f} s", flush=True)

    filas = []
    for n, (m, tipo) in modelos.items():
        if tipo is None:
            p = {"nivel global": np.exp(m.predict(X))}
        elif tipo == "e":
            p = {"nivel global": np.exp(m.predict(Xe))}
        else:
            q = m.predict(X)
            p = {"nivel global": np.exp(glob + q),
                 "nivel del proyecto": np.exp(np.where(vistos, nivel_proy, glob) + q)}
        for niv, pr in p.items():
            for b, k in bl.items():
                if niv == "nivel del proyecto" and b == "externo no Apache":
                    continue
                filas.append(dict(bloque=b, nivel=niv, modelo=n, n=int(k.sum()),
                                  **metricas(y[k], pr[k])))
    t = pd.DataFrame(filas)
    cols = ["modelo", "nivel", "MAE_h", "MdAElog", "zeta_pc", "SSPB_pc", "R2log",
            "PRED25", "factor2", "rho"]
    for b in bl:
        print(f"\n{S}\n{b}\n{S}")
        print(t[t.bloque == b][cols].sort_values(["nivel", "modelo"]).round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
