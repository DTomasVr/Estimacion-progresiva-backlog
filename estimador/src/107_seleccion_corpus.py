"""E12 · Seleccionar el corpus por ESCALA, no por nombre; y el valor de la prioridad.

Hasta ahora la seleccion del corpus ha sido por fuente: "sin Apache". Es una
regla cruda. Lo que hace daño de Apache no es llamarse Apache, es que su escala
horaria (mediana 0,83 h) no es la del trabajo de producto (2,00 h). Entre los
487 proyectos del bloque publico hay muchos que no son Apache y algunos de
Apache que si trabajan a escala de producto.

  SELECCION POR ESCALA   quedarse con los proyectos cuya mediana de horas EN
                         ENTRENAMIENTO cae en una banda, sin mirar de quien son.
                         La banda se elige en validacion. Puede recuperar miles
                         de filas de Apache que hoy se tiran.

  PRIORIDAD              la prioridad sola explica un R² de 0,048 y en Agix es
                         una pregunta de un clic, mucho mas barata que pedir
                         horas. Se prueba armonizada a seis niveles, porque en
                         crudo mezcla P1..P10 de SiP con Major/Minor de Jira y
                         seria un sustituto de la fuente.
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
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
DATA = "data/processed/agix_effort_v4.parquet"
EMB_TIT = "data/processed/embeddings/v4_mpnet_titulos"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/seleccion_corpus")
SEP = "=" * 116
PRIO = {"blocker": "1 bloqueante", "critical": "2 crítica", "p1": "2 crítica",
        "high": "3 alta", "p2": "3 alta", "major": "3 alta",
        "medium": "4 media", "p3": "4 media", "normal": "4 media",
        "minor": "5 baja", "low": "5 baja", "p4": "5 baja", "p5": "5 baja",
        "trivial": "6 trivial", "optional": "6 trivial"}


def armonizar(p):
    s = str(p).strip().lower()
    return PRIO.get(s, "0 sin prioridad" if s in ("", "nan", "(none)", "none") else "4 media")


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
    sw = df[df.is_software_task == 1].copy().reset_index(drop=True)
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sw["prio"] = sw.priority.map(armonizar)
    emb = np.load(f"{EMB_TIT}.npy")
    ids = np.load(f"{EMB_TIT}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    T = emb[[pos[t] for t in sw.task_id]].astype(np.float32)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    tr_all = (sw.split_t3 == "train").to_numpy()
    va = (sw.split_t3 == "val").to_numpy()
    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "externo no Apache": ((sw.split_t3 == "externo") & (~sw.apache)).to_numpy()}
    print(f"corpus completo {len(sw):,} · entrenamiento {tr_all.sum():,} "
          f"(sin Apache {(tr_all & ~sw.apache.to_numpy()).sum():,})")

    med = pd.DataFrame({"p": sw.project, "h": y})[tr_all].groupby("p").h.median()
    filas = []
    bandas = [(0, 1e9, "todo"), (1.0, 1e9, "mediana ≥ 1 h"), (1.0, 12.0, "mediana 1–12 h"),
              (1.5, 12.0, "mediana 1,5–12 h"), (0.8, 20.0, "mediana 0,8–20 h")]
    for lo, hi, etq in bandas:
        ok = sw.project.map(lambda p: lo <= med.get(p, -1) < hi).to_numpy()
        k = tr_all & ok
        if k.sum() < 3000:
            continue
        m = hgb().fit(T[k], ly[k])
        val = metricas(y[va], np.exp(m.predict(T[va])))
        fila = dict(seleccion=etq, n_train=int(k.sum()),
                    apache_dentro=float(sw.apache.to_numpy()[k].mean()),
                    val_MdAElog=val["MdAElog"])
        for b, kk in bloques.items():
            for mk, mv in metricas(y[kk], np.exp(m.predict(T[kk]))).items():
                fila[f"{b}|{mk}"] = mv
        filas.append(fila)
        print(f"  {etq:18s} {k.sum():6,} filas ({fila['apache_dentro']:.0%} Apache) "
              f"val {val['MdAElog']:.4f} · fija {fila['prueba fija|MdAElog']:.4f} · "
              f"externo {fila['externo no Apache|MdAElog']:.4f}", flush=True)
    sin_ap = tr_all & (~sw.apache.to_numpy())
    m = hgb().fit(T[sin_ap], ly[sin_ap])
    fila = dict(seleccion="sin Apache (lo actual)", n_train=int(sin_ap.sum()),
                apache_dentro=0.0,
                val_MdAElog=metricas(y[va], np.exp(m.predict(T[va])))["MdAElog"])
    for b, kk in bloques.items():
        for mk, mv in metricas(y[kk], np.exp(m.predict(T[kk]))).items():
            fila[f"{b}|{mk}"] = mv
    filas.append(fila)
    print(f"  {'sin Apache (actual)':18s} {sin_ap.sum():6,} filas · "
          f"val {fila['val_MdAElog']:.4f} · fija {fila['prueba fija|MdAElog']:.4f}")

    t = pd.DataFrame(filas)
    print(f"\n{SEP}\n1 · SELECCIONAR EL CORPUS POR ESCALA DEL PROYECTO\n{SEP}")
    cols = ["seleccion", "n_train", "apache_dentro", "val_MdAElog",
            "prueba fija|MdMRE", "prueba fija|MdAElog", "prueba fija|R2",
            "externo no Apache|MdAElog", "externo no Apache|R2"]
    print(t[cols].round(4).to_string(index=False))

    # ---------------------------------------------- 2 · prioridad
    print(f"\n{SEP}\n2 · ¿CUÁNTO VALE PREGUNTAR LA PRIORIDAD?\n{SEP}")
    swn = sw[~sw.apache].reset_index(drop=True)
    tit = swn.title.fillna("").astype(str)
    trn = (swn.split_t3 == "train").to_numpy()
    X = [np.load(CACHE / "e5-large.npy")]
    for kw in (dict(analyzer="char_wb", ngram_range=(3, 5)), dict(ngram_range=(1, 2))):
        tf = TfidfVectorizer(min_df=3, max_features=200_000, sublinear_tf=True,
                             **kw).fit(tit[trn])
        M = tf.transform(tit)
        X.append(TruncatedSVD(256, random_state=0).fit(M[trn]).transform(M).astype(np.float32))
    Xb = np.hstack(X)
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        swn.loc[trn, ["prio"]])
    P = oh.transform(swn[["prio"]])
    yn = swn.effort_hours.to_numpy()
    lyn = np.log(yn)
    bl = {"prueba fija": (swn.split_t3 == "test").to_numpy(),
          "prueba SiP": ((swn.split_t3 == "test") & (swn.source == "SiP")).to_numpy(),
          "externo no Apache": (swn.split_t3 == "externo").to_numpy()}
    filas2 = []
    for etq, XX in (("solo el título", Xb), ("título + prioridad", np.hstack([Xb, P])),
                    ("solo la prioridad", P)):
        m = hgb().fit(XX[trn], lyn[trn])
        for b, k in bl.items():
            filas2.append(dict(rasgos=etq, bloque=b, n=int(k.sum()),
                               **metricas(yn[k], np.exp(m.predict(XX[k])))))
        print(f"  {etq:22s} fija R² {filas2[-3]['R2']:+.4f} · "
              f"MdAElog {filas2[-3]['MdAElog']:.4f}", flush=True)
    t2 = pd.DataFrame(filas2)
    for b in bl:
        print(f"\n{b}")
        print(t2[t2.bloque == b].drop(columns=["bloque"]).round(4).to_string(index=False))

    t.to_csv(OUT / "por_escala.csv", index=False)
    t2.to_csv(OUT / "prioridad.csv", index=False)
    json.dump({"bandas": [list(b) for b in bandas]},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/por_escala.csv y prioridad.csv")


if __name__ == "__main__":
    main()
